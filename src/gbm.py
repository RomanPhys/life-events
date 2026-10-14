"""LightGBM на агрегатах истории клиента, по модели на таргет.

Признаки только по событиям до отсечки: статистики всей истории и последних W событий
(тех же, что видит нейросеть), то есть активность за 7/30/90 дней, суммы, тренд, доли частых значений полей.

    python -m src.gbm --data prepared/mbd --name gbm
    python -m src.gbm --data prepared/mbd --name gbm_leak --leak     # проба утечки
    python -m src.gbm --data prepared/mbd --name gbm_plus_emb --emb runs/mlm/embeddings.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGETS
from .data import Prepared, load, window_index
from .metrics import save_run


def build_features(P: Prepared, idx: np.ndarray, width: int = 256, top_k: int = 10, leak: bool = False,
                   chunk: int = 20000) -> pd.DataFrame:
    ev, sm = P.events, P.samples
    day, amt, xn = ev["day"], ev["amount"].astype(np.float64), ev["amount_norm"].astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(xn)])
    cs2 = np.concatenate([[0.0], np.cumsum(xn ** 2)])
    csr = np.concatenate([[0.0], np.cumsum(amt)])
    # топ-значения полей по всем событиям (без меток, утечки нет)
    tops = {f: np.bincount(ev[f]).argsort()[::-1][:top_k] for f in P.fields}
    tops = {f: t[t > 1] for f, t in tops.items()}  # без паддинга и OOV

    cut_all = sm["cut_leak"] if leak else sm["cut"]
    frames = []
    for s in range(0, len(idx), chunk):
        ii = idx[s: s + chunk]
        start, cut = sm["start"][ii], cut_all[ii]
        cut_day = sm["cut_day"][ii] + (31.0 if leak else 0.0)
        n = (cut - start).astype(np.float64)
        f = {"n_events": n, "log_n_events": np.log1p(n)}
        last = np.where(cut > start, day[np.maximum(cut - 1, 0)], np.nan)
        f["days_since_last"] = cut_day - last
        f["days_since_first"] = cut_day - day[start]
        mean = (cs[cut] - cs[start]) / np.maximum(n, 1)
        f["amt_mean_all"] = mean
        f["amt_std_all"] = np.sqrt(np.maximum((cs2[cut] - cs2[start]) / np.maximum(n, 1) - mean ** 2, 0))
        f["amt_sum_raw_all"] = csr[cut] - csr[start]

        w, valid = window_index(start, cut, width)
        dw = np.where(valid, cut_day[:, None] - day[w], np.inf)
        xw = np.where(valid, xn[w], np.nan)
        for h in (7, 30, 90):
            f[f"cnt_{h}d"] = (dw <= h).sum(1)
            f[f"amt_sum_{h}d"] = np.where(dw <= h, xw, 0).sum(1)
        f["trend_30_vs_90"] = (f["cnt_30d"] + 1) / ((f["cnt_90d"] - f["cnt_30d"]) / 2 + 1)
        with np.errstate(all="ignore"):
            f["amt_mean_w"] = np.nanmean(xw, 1)
            f["amt_std_w"] = np.nanstd(xw, 1)
            f["amt_max_w"] = np.nanmax(xw, 1)
            f["amt_min_w"] = np.nanmin(xw, 1)
            f["share_pos_w"] = np.nanmean(np.where(valid, xw > 0, np.nan), 1)
            act_days = np.where(valid & (dw <= 30), np.floor(day[w]), np.nan)
            f["active_days_30d"] = pd.DataFrame(act_days).nunique(axis=1).values
            for fld in P.fields:
                vals = np.where(valid, ev[fld][w], -1)
                denom = np.maximum(valid.sum(1), 1)
                for code in tops[fld]:
                    f[f"{fld}_{code}_share"] = (vals == code).sum(1) / denom
                f[f"{fld}_nunique"] = pd.DataFrame(np.where(valid, vals, np.nan)).nunique(axis=1).values
        frames.append(pd.DataFrame(f))
    return pd.concat(frames, ignore_index=True)


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed: int):
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                               random_state=seed, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.predict_proba(Xte)[:, 1], {"best_iteration": int(m.best_iteration_ or 0), "lib": "lightgbm"}
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                           min_samples_leaf=50, early_stopping=True, random_state=seed)
        m.fit(np.vstack([np.asarray(Xtr), np.asarray(Xva)]), np.concatenate([ytr, yva]))
        return m.predict_proba(np.asarray(Xte))[:, 1], {"n_iter": int(m.n_iter_), "lib": "sklearn-hgb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="gbm")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--leak", action="store_true")
    ap.add_argument("--emb", help="npz с эмбеддингами (поле emb, порядок как в samples), добавляются к признакам")
    ap.add_argument("--emb-only", action="store_true", help="только эмбеддинги, без агрегатов")
    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    P = load(a.data)
    t0 = time.time()
    tr = P.subsample_clients(P.split("train"), a.label_frac, a.seed)
    va, te = P.split("valid"), P.split("test")
    feats = {}
    for name, ii in [("train", tr), ("valid", va), ("test", te)]:
        X = pd.DataFrame(index=range(len(ii))) if a.emb_only else build_features(P, ii, a.width, leak=a.leak)
        if a.emb:
            E = np.load(a.emb)["emb"][ii]
            X = pd.concat([X, pd.DataFrame(E, columns=[f"emb_{j}" for j in range(E.shape[1])])], axis=1)
        feats[name] = X
    print(f"признаков: {feats['train'].shape[1]}, train {len(tr):,} / valid {len(va):,} / test {len(te):,}, "
          f"{time.time() - t0:.0f} с")
    y = P.samples["y"]
    preds, info = np.zeros((len(te), len(TARGETS))), {}
    for i, t in enumerate(TARGETS):
        preds[:, i], info[t] = fit_predict(feats["train"], y[tr, i], feats["valid"], y[va, i], feats["test"], a.seed)
    res = save_run(Path(a.runs) / a.name, a.name, vars(a), y[te], preds, P.samples["client"][te],
                   {"model": "gbm", "fit_info": info, "seconds": round(time.time() - t0)})
    print(f"{a.name}: mean ROC-AUC {res['mean_roc_auc']:.4f} (95% CI {res['mean_roc_auc_ci95'][0]:.4f}–"
          f"{res['mean_roc_auc_ci95'][1]:.4f}), mean PR-AUC {res['mean_pr_auc']:.4f}")
    for t, v in res["per_target"].items():
        print(f"  {t}: ROC-AUC {v['roc_auc']:.4f}  PR-AUC {v['pr_auc']:.4f}  доля позитивов {v['pos_rate']:.3f}")


if __name__ == "__main__":
    main()
