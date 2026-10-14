"""Обучение нейросетей: finetune (с нуля или из чекпойнта), pretrain (MLM на train-клиентах), embed.

    python -m src.train finetune --data prepared/mbd --model gru --name gru_s0 --seed 0
    python -m src.train pretrain --data prepared/mbd --name mlm --steps 6000
    python -m src.train finetune --data prepared/mbd --model transformer --init runs/mlm/encoder.pt --lr 1e-4 --name tr_mlm_s0
    python -m src.train embed --data prepared/mbd --ckpt runs/mlm/encoder.pt --out runs/mlm/embeddings.npz
"""
import argparse
import copy
import os
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import load
from .metrics import mean_roc, save_run
from .models import EventTransformer, build_model
from .torch_data import PretrainDataset, SupervisedDataset, collate, to_device

ARCH_KEYS = ("d_model", "layers", "heads", "dropout", "max_len", "hidden")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_optimizer(model, lr, wd, total_steps, warmup_frac=0.05):
    def no_wd(n, p):  # без weight decay для bias, LayerNorm и эмбеддингов
        return p.ndim < 2 or any(k in n for k in ("emb.cat.", "emb.dow.", "pos.", "cls", "mask_token"))

    decay = [p for n, p in model.named_parameters() if p.requires_grad and not no_wd(n, p)]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and no_wd(n, p)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.98))
    warm = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total_steps - warm))))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def loader(ds, batch, shuffle, workers):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle, persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, dl, device, amp, fn="forward"):
    model.eval()
    outs = []
    for b in dl:
        b = to_device(b, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            o = model(b) if fn == "forward" else model.pooled(b)
        outs.append((torch.sigmoid(o.float()) if fn == "forward" else o.float()).cpu().numpy())
    return np.concatenate(outs)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def finetune(a):
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp
    P = load(a.data)
    cfg = {k: getattr(a, k) for k in ARCH_KEYS}
    init_state = None
    if a.init and a.model != "transformer":
        raise SystemExit("--init поддерживается только для transformer")
    if a.init:
        ckpt = torch.load(a.init, map_location="cpu")
        cfg.update(ckpt["config"])  # архитектура из чекпойнта
        init_state = ckpt["state"]
    model = build_model(a.model, P.meta, cfg)
    if init_state is not None:
        missing, unexpected = model.load_state_dict({k: v for k, v in init_state.items()
                                                     if not k.startswith("head.")}, strict=False)
        assert all(k.startswith("head.") for k in missing), missing
        print(f"загружен предобученный энкодер; не загружены (новая голова): {len(missing)} тензоров")
    model.to(device)

    tr = P.subsample_clients(P.split("train"), a.label_frac, a.seed)
    va, te = P.split("valid"), P.split("test")
    dl_tr = loader(SupervisedDataset(P, tr, cfg["max_len"]), a.batch, True, a.workers)
    dl_va = loader(SupervisedDataset(P, va, cfg["max_len"]), a.eval_batch, False, a.workers)
    dl_te = loader(SupervisedDataset(P, te, cfg["max_len"]), a.eval_batch, False, a.workers)
    steps_per_epoch = min(len(dl_tr), a.steps_per_epoch or len(dl_tr))
    opt, sched = make_optimizer(model, a.lr, a.wd, a.epochs * steps_per_epoch)
    scaler = make_scaler(amp)
    y_tr = P.samples["y"][tr].astype(np.float64)
    pos_weight = None
    if a.pos_weight == "sqrt":
        pr = y_tr.mean(0).clip(1e-4, 1 - 1e-4)
        pos_weight = torch.tensor(np.sqrt((1 - pr) / pr), dtype=torch.float32, device=device)
    print(f"{a.name}: {a.model}, параметров {count_params(model):,}, train {len(tr):,} сэмплов "
          f"({len(np.unique(P.samples['client'][tr])):,} клиентов), {steps_per_epoch} шагов/эпоху, device={device}")

    best, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for step, b in enumerate(dl_tr):
            if step >= steps_per_epoch:
                break
            b = to_device(b, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(b)
            loss = F.binary_cross_entropy_with_logits(logits.float(), b["y"], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val = mean_roc(P.samples["y"][va], predict(model, dl_va, device, amp))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_mean_roc_auc": val,
                        "seconds": round(time.time() - t0)})
        print(f"  эпоха {epoch}: loss {np.mean(losses):.4f}, valid mean ROC-AUC {val:.4f}, {time.time() - t0:.0f} с")
        score = val if np.isfinite(val) else -1.0
        if best_state is None or score > best + 1e-4:
            best, best_state, bad = score, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    p = predict(model, dl_te, device, amp)
    res = save_run(Path(a.runs) / a.name, a.name, {**vars(a), "arch": cfg}, P.samples["y"][te], p,
                   P.samples["client"][te],
                   {"model": a.model + ("+mlm" if a.init else ""), "params": count_params(model),
                    "best_valid_mean_roc_auc": best, "history": history, "seconds": round(time.time() - t0)})
    print(f"{a.name}: test mean ROC-AUC {res['mean_roc_auc']:.4f} (95% CI {res['mean_roc_auc_ci95'][0]:.4f}–"
          f"{res['mean_roc_auc_ci95'][1]:.4f}), mean PR-AUC {res['mean_pr_auc']:.4f}")


def pretrain(a):
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp
    P = load(a.data)
    cfg = {k: getattr(a, k) for k in ARCH_KEYS}
    model = build_model("transformer", P.meta, cfg).to(device)
    train_clients = np.unique(P.samples["client"][P.split("train")])
    valid_clients = np.unique(P.samples["client"][P.split("valid")])
    ds = PretrainDataset(P, train_clients, cfg["max_len"], n_items=a.steps * a.batch, seed=a.seed)
    ds_val = PretrainDataset(P, valid_clients, cfg["max_len"], n_items=a.val_items, seed=10_000 + a.seed)
    dl = loader(ds, a.batch, True, a.workers)
    dl_val = loader(ds_val, a.batch, False, a.workers)
    opt, sched = make_optimizer(model, a.lr, a.wd, a.steps)
    scaler = make_scaler(amp)
    print(f"MLM-предобучение: параметров {count_params(model):,}, клиентов {len(ds.clients):,}, шагов {a.steps}")

    def val_loss():
        # фиксированные маски на валидации, RNG обучения не трогаем
        model.eval()
        devices = [device.index or 0] if device.type == "cuda" else []
        tot, n = 0.0, 0
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(0)
            for b in dl_val:
                b = to_device(b, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss, _ = model.mlm_loss(b, a.mask_prob)
                tot, n = tot + loss.item(), n + 1
        model.train()
        return tot / max(n, 1)

    log, t0 = [], time.time()
    model.train()
    for step, b in enumerate(dl):
        b = to_device(b, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss, parts = model.mlm_loss(b, a.mask_prob)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "train_mlm": loss.item(), **{f"train_{k}": v for k, v in parts.items()},
                   "seconds": round(time.time() - t0)}
            if step % a.val_every == 0 or step == a.steps - 1:
                rec["valid_mlm"] = val_loss()
            log.append(rec)
            print(f"  шаг {step}: train {loss.item():.4f}" + (f", valid {rec['valid_mlm']:.4f}" if "valid_mlm" in rec
                                                              else "") + f", {time.time() - t0:.0f} с")
    out = Path(a.runs) / a.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": cfg}, out / "encoder.pt")
    (out / "pretrain_log.json").write_text(json.dumps({"args": vars(a), "log": log}, indent=1))
    print(f"сохранено: {out / 'encoder.pt'}")


def embed(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp
    P = load(a.data)
    ckpt = torch.load(a.ckpt, map_location="cpu")
    model = EventTransformer(P.meta, d_model=ckpt["config"]["d_model"], n_layers=ckpt["config"]["layers"],
                             n_heads=ckpt["config"]["heads"], dropout=0.0, max_len=ckpt["config"]["max_len"])
    model.load_state_dict(ckpt["state"])
    model.to(device)
    idx = np.arange(len(P.samples["cut"]))
    dl = loader(SupervisedDataset(P, idx, ckpt["config"]["max_len"]), a.eval_batch, False, a.workers)
    E = predict(model, dl, device, amp, fn="pooled")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, emb=E.astype(np.float32))
    print(f"эмбеддинги {E.shape} -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True)
    common.add_argument("--runs", default="runs")
    common.add_argument("--seed", type=int, default=0)
    # на Windows воркеры стартуют через spawn и копируют все массивы, поэтому там 0
    common.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    common.add_argument("--eval-batch", type=int, default=512)
    common.add_argument("--no-amp", action="store_true")
    arch = argparse.ArgumentParser(add_help=False)
    arch.add_argument("--d-model", dest="d_model", type=int, default=128)
    arch.add_argument("--layers", type=int, default=4)
    arch.add_argument("--heads", type=int, default=4)
    arch.add_argument("--hidden", type=int, default=256, help="размер скрытого состояния GRU")
    arch.add_argument("--dropout", type=float, default=0.1)
    arch.add_argument("--max-len", dest="max_len", type=int, default=256)

    f = sub.add_parser("finetune", parents=[common, arch])
    f.add_argument("--model", choices=["transformer", "gru"], default="transformer")
    f.add_argument("--name", required=True)
    f.add_argument("--init", help="чекпойнт предобученного энкодера")
    f.add_argument("--batch", type=int, default=256)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--wd", type=float, default=0.01)
    f.add_argument("--epochs", type=int, default=12)
    f.add_argument("--patience", type=int, default=3)
    f.add_argument("--steps-per-epoch", dest="steps_per_epoch", type=int, default=None)
    f.add_argument("--label-frac", dest="label_frac", type=float, default=1.0)
    f.add_argument("--pos-weight", dest="pos_weight", choices=["none", "sqrt"], default="none")

    p = sub.add_parser("pretrain", parents=[common, arch])
    p.add_argument("--name", required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--mask-prob", dest="mask_prob", type=float, default=0.15)
    p.add_argument("--log-every", dest="log_every", type=int, default=100)
    p.add_argument("--val-every", dest="val_every", type=int, default=500)
    p.add_argument("--val-items", dest="val_items", type=int, default=4096)

    e = sub.add_parser("embed", parents=[common])
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)

    a = ap.parse_args()
    {"finetune": finetune, "pretrain": pretrain, "embed": embed}[a.cmd](a)


if __name__ == "__main__":
    main()
