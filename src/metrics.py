"""ROC-AUC, PR-AUC и бутстрэп по клиентам."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import TARGETS


def per_target(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for i, t in enumerate(TARGETS):
        yi = y[:, i]
        if yi.min() == yi.max():
            out[t] = {"roc_auc": float("nan"), "pr_auc": float("nan"), "pos_rate": float(yi.mean())}
            continue
        out[t] = {"roc_auc": float(roc_auc_score(yi, p[:, i])), "pr_auc": float(average_precision_score(yi, p[:, i])),
                  "pos_rate": float(yi.mean())}
    return out


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC через ранги (как в тесте Манна-Уитни), связи учитываются. Быстрее sklearn, нужно для бутстрэпа."""
    pos = y == 1
    n1, n0 = pos.sum(), len(y) - pos.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mean_roc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean([fast_auc(y[:, i], p[:, i]) for i in range(y.shape[1])]))


def client_groups(clients: np.ndarray) -> list[np.ndarray]:
    """Индексы строк каждого клиента за O(n log n)."""
    order = np.argsort(clients, kind="stable")
    _, starts = np.unique(clients[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap_mean_roc(y, p, clients, n_boot=200, seed=0) -> tuple[float, float]:
    """95% CI средней ROC-AUC. Ресэмплим клиентов, а не строки, потому что месяцы одного клиента зависимы."""
    rng = np.random.default_rng(seed)
    groups = client_groups(clients)
    uniq = groups
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))])
        vals.append(mean_roc(y[idx], p[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def save_run(run_dir: Path, name: str, config: dict, y, p, clients, extra: dict | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    pt = per_target(y, p)
    res = {
        "name": name, "config": config, "per_target": pt,
        "mean_roc_auc": float(np.nanmean([v["roc_auc"] for v in pt.values()])),
        "mean_pr_auc": float(np.nanmean([v["pr_auc"] for v in pt.values()])),
        "mean_roc_auc_ci95": bootstrap_mean_roc(y, p, clients),
        **(extra or {}),
    }
    (run_dir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    np.savez_compressed(run_dir / "test_predictions.npz", p=p.astype(np.float32), y=y.astype(np.int8))
    return res


def paired_bootstrap_diff(y, p_a, p_b, clients, n_boot=1000, seed=0) -> dict:
    """Разница средней ROC-AUC (b минус a) на одних и тех же бутстрэп-выборках клиентов."""
    rng = np.random.default_rng(seed)
    groups = client_groups(clients)
    uniq = groups
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(mean_roc(y[idx], p_b[idx]) - mean_roc(y[idx], p_a[idx]))
    diffs = np.array(diffs)
    return {"diff": mean_roc(y, p_b) - mean_roc(y, p_a), "ci95": [float(np.percentile(diffs, 2.5)),
                                                                     float(np.percentile(diffs, 97.5))],
            "p_b_not_better": float((diffs <= 0).mean())}
