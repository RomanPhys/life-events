"""Сводная таблица по runs/*/metrics.json и парное сравнение двух запусков.
Запуски name_s0, name_s1, ... считаются сидами одной конфигурации.

    python -m src.report
    python -m src.report --compare "runs/tr_scratch_s*" "runs/tr_mlm_s*" --data prepared/mbd
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import TARGETS
from .data import load
from .metrics import paired_bootstrap_diff


def fmt(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return "—"
    return f"{np.mean(vals):.4f}" + (f" ± {np.std(vals):.4f}" if len(vals) > 1 else "")


MIN_POSITIVES = 50  # при меньшем числе позитивов ROC-AUC слишком шумная


def robust_targets(runs_dir: Path) -> list[str]:
    """Таргеты, у которых в тесте хотя бы MIN_POSITIVES позитивов."""
    for f in sorted(runs_dir.glob("*/test_predictions.npz")):
        y = np.load(f)["y"]
        return [t for i, t in enumerate(TARGETS) if y[:, i].sum() >= MIN_POSITIVES]
    return list(TARGETS)


def summarize(runs_dir: Path) -> str:
    groups = defaultdict(list)
    for f in sorted(runs_dir.glob("*/metrics.json")):
        r = json.loads(f.read_text())
        groups[re.sub(r"_s\d+$", "", r["name"])].append(r)
    rows = []
    for g, rs in groups.items():
        frac = rs[0]["config"].get("label_frac", 1.0)
        rows.append((frac, g, rs))
    rows.sort(key=lambda x: (-x[0], np.mean([r["mean_roc_auc"] for r in x[2]])))
    robust = robust_targets(runs_dir)
    rob_name = "Mean ROC (" + ", ".join(t.replace("target_", "t") for t in robust) + ")"
    lines = [f"| Запуск | Модель | Доля разметки | Сидов | Mean ROC-AUC | {rob_name} | Mean PR-AUC | "
             + " | ".join(f"{t} ROC" for t in TARGETS) + " |",
             "|" + "---|" * (7 + len(TARGETS))]
    for frac, g, rs in rows:
        rob = [float(np.mean([r["per_target"][t]["roc_auc"] for t in robust])) for r in rs]
        lines.append(f"| {g} | {rs[0].get('model', '')} | {frac:g} | {len(rs)} | "
                     f"{fmt([r['mean_roc_auc'] for r in rs])} | {fmt(rob)} | {fmt([r['mean_pr_auc'] for r in rs])} | "
                     + " | ".join(fmt([r['per_target'][t]['roc_auc'] for r in rs]) for t in TARGETS) + " |")
    pos = rows[0][2][0]["per_target"] if rows else {}
    note = ("\nДоля позитивов в тесте: " + ", ".join(f"{t} {v['pos_rate']:.4f}" for t, v in pos.items())) if pos else ""
    dropped = [t for t in TARGETS if t not in robust]
    if dropped:
        note += (f"\nВ колонке {rob_name} исключены таргеты с < {MIN_POSITIVES} позитивами в тесте: "
                 + ", ".join(dropped) + ", по ним ROC-AUC слишком шумная.")
    return "\n".join(lines) + "\n" + note + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/results.md")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--data", help="prepared-каталог (нужен для --compare: id клиентов теста)")
    a = ap.parse_args()
    if a.compare:
        def load_preds(pattern):
            # шаблон с * берёт все сиды и усредняет их предсказания
            paths = sorted(glob.glob(pattern)) or [pattern]
            arrs = [np.load(Path(r) / "test_predictions.npz") for r in paths]
            return arrs[0]["y"], np.mean([x["p"] for x in arrs], axis=0), len(arrs)

        (ya, pa, na), (yb, pb, nb) = (load_preds(r) for r in a.compare)
        assert np.array_equal(ya, yb), "разные тестовые наборы"
        P = load(a.data)
        clients = P.samples["client"][P.split("test")]
        print(f"{a.compare[1]} ({nb} запуск.) − {a.compare[0]} ({na} запуск.)")
        robust = [i for i, t in enumerate(TARGETS) if ya[:, i].sum() >= MIN_POSITIVES]
        for label, cols in [("все таргеты", list(range(len(TARGETS)))),
                            ("без шумных таргетов", robust)]:
            res = paired_bootstrap_diff(ya[:, cols], pa[:, cols], pb[:, cols], clients)
            print(f"  {label:20s} Δ mean ROC-AUC = {res['diff']:+.4f}, 95% CI [{res['ci95'][0]:+.4f}, "
                  f"{res['ci95'][1]:+.4f}], доля бутстрэп-выборок без улучшения {res['p_b_not_better']:.3f}")
        return
    text = summarize(Path(a.runs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
