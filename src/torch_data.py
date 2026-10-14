"""Датасеты и collate для torch. Сама нарезка событий лежит в data.py."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Prepared, sample_window, slice_events

FLOAT_KEYS = ("amount", "log_dt", "log_to_cut")


class SupervisedDataset(Dataset):
    """Клиенто-месяц: последние max_len событий до начала месяца и 4 метки."""

    def __init__(self, P: Prepared, idx: np.ndarray, max_len: int):
        self.P, self.idx, self.max_len = P, np.asarray(idx), max_len

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        item = sample_window(self.P, i, self.max_len)
        item["y"] = self.P.samples["y"][i].astype(np.float32)
        return item


class PretrainDataset(Dataset):
    """Случайные окна из истории клиентов train-фолдов, без меток.
    Отсечка окна это время следующего события, чтобы log_to_cut значил то же, что при дообучении.
    """

    def __init__(self, P: Prepared, clients: np.ndarray, max_len: int, n_items: int, min_len: int = 8,
                 seed: int = 0):
        off = P.events["offsets"]
        lengths = off[clients + 1] - off[clients]
        self.clients = clients[lengths >= 2]
        self.P, self.max_len, self.n_items, self.min_len = P, max_len, n_items, min_len
        self.seed = seed

    def __len__(self):
        return self.n_items

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k))  # воспроизводимо при любом числе воркеров
        c = int(self.clients[rng.integers(len(self.clients))])
        off = self.P.events["offsets"]
        s, end = int(off[c]), int(off[c + 1])
        e = int(rng.integers(min(s + self.min_len, end), end + 1))
        lo = max(s, e - self.max_len)
        day = self.P.events["day"]
        cut_day = float(day[e]) if e < end else float(day[e - 1]) + 1.0
        return slice_events(self.P, lo, e, s, cut_day)


def collate(batch: list[dict]) -> dict:
    n = len(batch)
    lengths = [len(b["amount"]) for b in batch]
    T = max(lengths)
    out = {"mask": torch.zeros(n, T, dtype=torch.bool), "length": torch.tensor(lengths, dtype=torch.long)}
    keys = [k for k in batch[0] if k not in ("y", "last_day")]
    for k in keys:
        is_float = k in FLOAT_KEYS
        t = torch.zeros(n, T, dtype=torch.float32 if is_float else torch.long)
        for j, b in enumerate(batch):
            t[j, : lengths[j]] = torch.from_numpy(np.asarray(b[k]))
        out[k] = t
    for j, L in enumerate(lengths):
        out["mask"][j, :L] = True
    if "y" in batch[0]:
        out["y"] = torch.from_numpy(np.stack([b["y"] for b in batch]))
    return out


def to_device(b: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if k != "length" else v) for k, v in b.items()}
