"""Готовит MBD (или синтетику в той же схеме) к обучению.

Что пишется в --out:
  events.npz   события всех клиентов подряд, по клиенту и времени; offsets это границы клиентов
  samples.npz  клиенто-месяцы. Модель видит события [start, cut), cut это первое событие целевого месяца.
               cut_leak считается по концу месяца и нужен только для пробы утечки
  meta.json    словари, нормировка суммы, статистика

Таблицы находятся по колонкам: транзакции (event_time, amount, event_type) и таргеты (target_1, mon).

    python -m src.prepare --raw data/mbd --out prepared/mbd --max-clients 40000 --last-months 6 --time-shift-hours 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASE_DAY, CAT_FIELDS, TARGETS, TEST_FOLD, VALID_FOLD

BIG = 100_000.0  # ключ client*BIG + day отсортирован глобально



def _columns(path: Path) -> tuple[list[str], dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        return schema.names, {f.name: str(f.type) for f in schema}
    cols = list(pd.read_csv(path, nrows=0).columns)
    return cols, {}


def find_tables(root: Path) -> dict:
    dirs = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and (p.suffix in (".parquet", ".csv")) and not p.name.startswith("."):
            dirs.setdefault(p.parent, []).append(p)
    found = {}
    for d, files in dirs.items():
        cols, types = _columns(files[0])
        is_list = any("list" in t for t in types.values())  # формат ptls (массивы) пропускаем
        # parquet разложен по папкам fold=k, берём родительскую
        base = d.parent if d.name.startswith("fold=") else d
        if {"event_time", "amount", "event_type"} <= set(cols) and not is_list:
            found.setdefault("trx", base)
        elif "target_1" in cols and "mon" in cols:
            found.setdefault("targets", base)
        elif set(cols) == {"client_id", "fold"} or (set(cols) >= {"client_id", "fold"} and len(cols) <= 3):
            found.setdefault("client_split", base)
    return found


def read_table(base: Path, columns: list[str], clients: set | None = None) -> pd.DataFrame:
    files = sorted(p for p in base.rglob("*") if p.suffix in (".parquet", ".csv"))
    if files and files[0].suffix == ".parquet":
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        flt = pc.field("client_id").isin(list(clients)) if clients is not None else None
        last_err = None
        for partitioning in ("hive", None):  # fold может быть и колонкой в файле, и именем каталога
            try:
                dataset = ds.dataset(base, format="parquet", partitioning=partitioning)
                cols = [c for c in columns if c in dataset.schema.names]
                return dataset.to_table(columns=cols, filter=flt).to_pandas()
            except Exception as e:
                last_err = e
        raise last_err
    parts = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns
        df = pd.read_csv(f, usecols=[c for c in columns if c in header])
        parts.append(df[df.client_id.isin(clients)] if clients is not None else df)
    return pd.concat(parts, ignore_index=True)



def month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.start_time


def to_day(ts: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(ts).values - BASE_DAY) / np.timedelta64(1, "D")).astype(np.float64)


def encode_vocab(values: pd.Series, max_vocab: int) -> tuple[np.ndarray, int]:
    top = values.value_counts(dropna=True).index[: max_vocab - 2]
    codes = pd.Categorical(values, categories=top).codes.astype(np.int32)  # -1 для редких и NaN
    return np.where(codes < 0, 1, codes + 2).astype(np.int32), len(top) + 2


def prepare(raw: Path, out: Path, max_clients: int | None, last_months: int | None, max_vocab: int,
            n_bins: int, min_events: int, seed: int, time_shift_hours: float = 0.0) -> dict:
    tables = find_tables(raw)
    print("найденные таблицы:", {k: str(v) for k, v in tables.items()})
    if "trx" not in tables or "targets" not in tables:
        raise SystemExit("не нашёл таблицы транзакций и/или таргетов; запустите src.inspect_mbd и укажите пути")

    tgt = read_table(tables["targets"], ["client_id", "mon", "fold", *TARGETS])
    if "fold" not in tgt.columns:
        split = read_table(tables["client_split"], ["client_id", "fold"])
        tgt = tgt.merge(split, on="client_id", how="left")
    tgt = tgt.dropna(subset=["fold"])
    tgt["fold"] = tgt["fold"].astype(int)

    clients = pd.Series(tgt.client_id.unique())
    if max_clients and len(clients) > max_clients:  # детерминированно, по хешу id
        h = pd.util.hash_pandas_object(clients, index=False).values ^ np.uint64(seed)
        clients = clients.iloc[np.argsort(h)[:max_clients]]
    clients = pd.Series(np.sort(clients.values))
    client_set = set(clients)
    tgt = tgt[tgt.client_id.isin(client_set)]

    trx = read_table(tables["trx"], ["client_id", "event_time", "amount", *CAT_FIELDS], client_set)
    fields = [f for f in CAT_FIELDS if f in trx.columns]
    trx = trx.dropna(subset=["event_time"])
    trx["cidx"] = pd.Categorical(trx.client_id, categories=clients.values).codes
    trx = trx[trx.cidx >= 0]
    # время в MBD в UTC (данные начинаются 2020-12-31 21:00), а месяцы таргетов московские.
    # без сдвига в историю попадают первые 3 часа целевого месяца
    trx["day"] = to_day(trx.event_time) + time_shift_hours / 24.0
    trx = trx.sort_values(["cidx", "day"], kind="stable").reset_index(drop=True)
    print(f"событий: {len(trx):,}; клиентов с событиями: {trx.cidx.nunique():,}; "
          f"период: {trx.event_time.min()} .. {trx.event_time.max()}")

    n_clients = len(clients)
    offsets = np.zeros(n_clients + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(trx.cidx.values, minlength=n_clients))
    fold_of_client = tgt.drop_duplicates("client_id").set_index("client_id").fold.reindex(clients.values).values
    train_mask_ev = ~np.isin(fold_of_client[trx.cidx.values], [TEST_FOLD, VALID_FOLD])

    amount = trx.amount.fillna(0).values.astype(np.float64)
    x = np.sign(amount) * np.log1p(np.abs(amount))
    mu, sd = x[train_mask_ev].mean(), x[train_mask_ev].std() + 1e-6
    edges = np.unique(np.quantile(x[train_mask_ev], np.linspace(0, 1, n_bins + 1)[1:-1]))
    events = {"day": trx.day.values.astype(np.float64), "amount": amount.astype(np.float32),
              "amount_norm": ((x - mu) / sd).astype(np.float32),
              "amount_bin": np.searchsorted(edges, x).astype(np.int32), "offsets": offsets}
    vocab = {}
    for f in fields:
        events[f], vocab[f] = encode_vocab(trx[f], max_vocab)

    # клиенто-месяцы
    tgt = tgt.copy()
    tgt["cidx"] = pd.Categorical(tgt.client_id, categories=clients.values).codes
    tgt["m0"] = month_start(tgt.mon)
    tgt["month"] = tgt.m0.dt.year * 100 + tgt.m0.dt.month
    tgt = tgt.sort_values(["cidx", "m0"]).drop_duplicates(["cidx", "m0"])
    cut_day = to_day(tgt.m0)
    leak_day = to_day(tgt.m0 + pd.offsets.MonthBegin(1))
    key = events["day"] + BIG * trx.cidx.values  # глобально отсортирован
    c = tgt.cidx.values.astype(np.int64)
    start = offsets[c]
    cut = np.searchsorted(key, c * BIG + cut_day, side="left")  # строго до начала месяца
    cut_leak = np.searchsorted(key, c * BIG + leak_day, side="left")
    samples = pd.DataFrame({"client": c, "start": start, "cut": cut, "cut_leak": cut_leak, "cut_day": cut_day,
                            "month": tgt.month.values, "fold": tgt.fold.values})
    for i, t in enumerate(TARGETS):
        samples[f"y{i}"] = tgt[t].fillna(0).astype(int).values
    n_before = len(samples)
    samples = samples[samples.cut - samples.start >= min_events]
    if last_months:
        samples = samples.sort_values(["client", "month"]).groupby("client").tail(last_months)
    samples = samples.reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "events.npz", **events)
    np.savez(out / "samples.npz", **{k: samples[k].values for k in
                                     ["client", "start", "cut", "cut_leak", "cut_day", "month", "fold"]},
             y=samples[[f"y{i}" for i in range(4)]].values.astype(np.int8))
    split_name = np.where(samples.fold == TEST_FOLD, "test", np.where(samples.fold == VALID_FOLD, "valid", "train"))
    hist = (samples.cut - samples.start).values
    meta = {
        "time_shift_hours": time_shift_hours, "fields": fields, "vocab": vocab, "amount_mu": float(mu), "amount_sd": float(sd),
        "amount_edges": edges.tolist(), "n_amount_bins": int(len(edges) + 1),
        "n_clients": int(n_clients), "n_events": int(len(trx)), "n_samples": int(len(samples)),
        "samples_before_filters": int(n_before),
        "samples_per_split": pd.Series(split_name).value_counts().to_dict(),
        "positive_rate": {t: float(samples[f"y{i}"].mean()) for i, t in enumerate(TARGETS)},
        "positive_rate_test": {t: float(samples.loc[split_name == "test", f"y{i}"].mean())
                               for i, t in enumerate(TARGETS)},
        "history_len": {"median": float(np.median(hist)), "p90": float(np.quantile(hist, 0.9))},
        "months": sorted(int(m) for m in samples.month.unique()),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({k: meta[k] for k in ["n_clients", "n_events", "n_samples", "samples_per_split",
                                           "positive_rate", "history_len"]}, ensure_ascii=False, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clients", type=int, default=None)
    ap.add_argument("--last-months", type=int, default=None, help="оставить K последних месяцев каждого клиента")
    ap.add_argument("--max-vocab", type=int, default=1000)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--min-events", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-shift-hours", type=float, default=0.0,
                    help="сдвиг event_time к часовому поясу меток; для MBD: 3 (UTC -> МСК)")
    a = ap.parse_args()
    prepare(Path(a.raw), Path(a.out), a.max_clients, a.last_months, a.max_vocab, a.n_bins, a.min_events, a.seed,
            a.time_shift_hours)


if __name__ == "__main__":
    main()
