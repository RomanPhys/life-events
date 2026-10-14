"""Загрузка подготовленных данных и нарезка окон событий (numpy, без torch)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BASE_DOW, TEST_FOLD, VALID_FOLD


@dataclass
class Prepared:
    events: dict
    samples: dict
    meta: dict

    @property
    def fields(self) -> list[str]:
        return self.meta["fields"]

    def split(self, name: str) -> np.ndarray:
        f = self.samples["fold"]
        if name == "test":
            return np.flatnonzero(f == TEST_FOLD)
        if name == "valid":
            return np.flatnonzero(f == VALID_FOLD)
        return np.flatnonzero((f != TEST_FOLD) & (f != VALID_FOLD))

    def subsample_clients(self, idx: np.ndarray, frac: float, seed: int) -> np.ndarray:
        """Оставляет долю клиентов, а не строк. Нужно для label efficiency."""
        if frac >= 1.0:
            return idx
        clients = np.unique(self.samples["client"][idx])
        rng = np.random.default_rng(seed)
        keep = rng.choice(clients, max(1, int(round(frac * len(clients)))), replace=False)
        return idx[np.isin(self.samples["client"][idx], keep)]


def load(prepared_dir: str | Path) -> Prepared:
    d = Path(prepared_dir)
    ev = dict(np.load(d / "events.npz"))
    sm = dict(np.load(d / "samples.npz"))
    meta = json.loads((d / "meta.json").read_text())
    return Prepared(ev, sm, meta)


def window_index(start: np.ndarray, cut: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Индексы последних width событий до cut. Событие с индексом cut уже не попадает."""
    lo = np.maximum(start, cut - width)
    idx = lo[:, None] + np.arange(width)[None, :]
    valid = idx < cut[:, None]
    return np.where(valid, idx, 0), valid


def slice_events(P: Prepared, lo: int, hi: int, client_start: int, cut_day: float) -> dict:
    """Признаки событий [lo, hi) одного клиента для нейросети, hi это отсечка."""
    ev = P.events
    day = ev["day"][lo:hi]
    prev = ev["day"][lo - 1] if lo > client_start else day[0]
    dt = np.diff(day, prepend=prev)
    out = {
        "amount": ev["amount_norm"][lo:hi].astype(np.float32),
        "log_dt": np.log1p(np.maximum(dt, 0)).astype(np.float32),
        "log_to_cut": np.log1p(np.maximum(cut_day - day, 0)).astype(np.float32),
        "dow": ((np.floor(day).astype(np.int64) + BASE_DOW) % 7 + 1),  # 0 зарезервирован под паддинг
        "amount_bin": ev["amount_bin"][lo:hi].astype(np.int64),
        "last_day": float(day[-1]),
    }
    for f in P.fields:
        out[f] = ev[f][lo:hi].astype(np.int64)
    return out


def sample_window(P: Prepared, i: int, max_len: int) -> dict:
    """Последние max_len событий клиента до начала целевого месяца."""
    sm = P.samples
    s, c = int(sm["start"][i]), int(sm["cut"][i])
    return slice_events(P, max(s, c - max_len), c, s, float(sm["cut_day"][i]))
