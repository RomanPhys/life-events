"""Синтетика в схеме MBD для smoke-тестов и проверки, что модели видят порядок событий.

target_1  зависит от агрегатов (активность за 30 дней, доля типа 3)
target_2  зависит только от порядка: тип 7, а через 0-2 дня тип 8. Сами типы 7 и 8 есть у всех клиентов
target_3  зависит от давности последнего события с dst_type11 = 5
target_4  редкий (1-2%), рост активности и тип 11

    python -m src.synthetic --out data/synth --clients 3000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CAT_FIELDS

START = np.datetime64("2021-07-01")
END = np.datetime64("2022-12-31")
MONTHS = pd.date_range("2022-02-01", "2023-01-01", freq="MS")
CARD = {"event_type": 12, "event_subtype": 30, "currency": 4, "src_type11": 40, "src_type12": 120,
        "dst_type11": 40, "dst_type12": 150, "src_type21": 60, "src_type22": 200, "src_type31": 25,
        "src_type32": 80}


def _client_events(rng, n_days):
    rate = rng.lognormal(-0.3, 0.7)
    n = max(5, rng.poisson(rate * n_days))
    days = np.sort(rng.uniform(0, n_days, n))
    pref = rng.dirichlet(np.full(CARD["event_type"], 0.6))
    pref[[7, 8]] = 0.04  # 7 и 8 есть у всех клиентов примерно с одинаковой частотой
    pref /= pref.sum()
    ev = {"event_type": rng.choice(CARD["event_type"], n, p=pref)}
    for f in CAT_FIELDS[1:]:
        k = CARD[f]
        zipf = 1.0 / np.arange(1, k + 1) ** 1.1
        ev[f] = rng.choice(k, n, p=zipf / zipf.sum())
    ev["currency"] = np.where(rng.random(n) < 0.95, 1, ev["currency"])
    sign = np.where(np.isin(ev["event_type"], [0, 1, 2, 3, 4]), -1.0, 1.0)
    amount = sign * rng.lognormal(7 + 0.2 * ev["event_type"], 1.0, n)
    return days, ev, amount


def generate(n_clients: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_days = int((END - START).astype(int))
    month_days = np.array([(np.datetime64(m.date()) - START).astype(int) for m in MONTHS], dtype=float)
    trx_parts, tgt_rows = [], []
    for c in range(n_clients):
        cid = f"c{c:06x}{rng.integers(1 << 20):05x}"
        fold = c % 5
        days, ev, amount = _client_events(rng, n_days)
        # вставляем упорядоченные пары 7 -> 8 перед случайными месяцами и одиночные 7/8 как шум
        add_days, add_types = [], []
        for md in month_days:
            if rng.random() < 0.12:
                d = md - rng.uniform(3, 28)
                add_days += [d, d + rng.uniform(0.05, 2.0)]
                add_types += [7, 8]
            if rng.random() < 0.25:  # шум: 8 раньше 7 или далеко друг от друга
                d = md - rng.uniform(3, 28)
                add_days += [d, d + rng.uniform(4, 20)] if rng.random() < 0.5 else [d + 1.0, d]
                add_types += [7, 8]
        if add_days:
            k = len(add_days)
            days = np.concatenate([days, np.array(add_days)])
            ev = {f: np.concatenate([v, np.array(add_types) if f == "event_type"
                                     else rng.choice(CARD[f], k)]) for f, v in ev.items()}
            amount = np.concatenate([amount, rng.lognormal(7, 1, k)])
        order = np.argsort(days, kind="stable")
        days, amount = days[order], amount[order]
        ev = {f: v[order] for f, v in ev.items()}
        et, dst = ev["event_type"], ev["dst_type11"]

        for m, md in zip(MONTHS, month_days):
            before = days < md
            if before.sum() == 0:
                continue
            last30 = before & (days >= md - 30)
            prev60 = before & (days >= md - 90) & (days < md - 30)
            act = np.log1p(last30.sum())
            share3 = (et[last30] == 3).mean() if last30.any() else 0.0
            # есть ли 7, за которым следующим из 7/8 идёт 8 не позже чем через 2 дня
            idx = np.flatnonzero(last30 & np.isin(et, [7, 8]))
            pair = any(et[i] == 7 and et[j] == 8 and days[j] - days[i] <= 2.0 for i, j in zip(idx[:-1], idx[1:]))
            d5 = days[before & (dst == 5)]
            recency5 = md - d5[-1] if len(d5) else 400.0
            trend = (last30.sum() + 1) / (prev60.sum() / 2 + 1)
            has11 = (et[before & (days >= md - 60)] == 11).any()
            logits = np.array([
                -3.2 + 1.1 * (act - 2.0) + 4.0 * share3,
                -3.6 + 3.4 * pair,
                -2.0 - 0.08 * recency5,
                -5.5 + 1.6 * np.log(trend) + 2.0 * has11,
            ])
            y = (rng.random(4) < 1 / (1 + np.exp(-logits))).astype(int)
            tgt_rows.append({"mon": (m + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                             **{f"target_{i + 1}": int(y[i]) for i in range(4)},
                             "trans_count": int(before.sum()), "diff_trans_date": 0,
                             "client_id": cid, "fold": fold})

        keep = days < n_days
        ts = START + (days[keep] * 86400).astype("timedelta64[s]")
        part = pd.DataFrame({"client_id": cid, "event_time": ts, "amount": amount[keep].round(2), "fold": fold,
                             **{f: ev[f][keep] for f in CAT_FIELDS}})
        trx_parts.append(part)
    trx = pd.concat(trx_parts, ignore_index=True)
    targets = pd.DataFrame(tgt_rows)
    split = targets[["client_id", "fold"]].drop_duplicates()
    return trx, targets, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--clients", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    trx, targets, split = generate(a.clients, a.seed)
    out = Path(a.out)
    for name, df in [("trx", trx), ("targets", targets), ("client_split", split)]:
        (out / name).mkdir(parents=True, exist_ok=True)
        df.to_csv(out / name / "part-00000.csv", index=False)
    print(f"{out}: {len(trx):,} событий, {targets.client_id.nunique():,} клиентов, {len(targets):,} клиенто-месяцев")
    print("доля позитивов:", targets[[f"target_{i}" for i in range(1, 5)]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
