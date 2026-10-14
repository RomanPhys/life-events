"""Модели для последовательностей событий.

EventEmbedding    эмбеддинги категориальных полей, день недели, сумма, log dt, log времени до отсечки,
                  дальше Linear и LayerNorm
EventTransformer  pre-LN encoder с CLS и головами для MLM. Позиции считаются от последнего события,
                  поэтому при правом паддинге свежесть события кодируется одинаково для любой длины истории
EventGRU          та же EventEmbedding и GRU, аналог Supervised RNN из статьи MBD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EventEmbedding(nn.Module):
    def __init__(self, meta: dict, d_model: int, emb_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.fields = meta["fields"]
        self.cat = nn.ModuleDict({f: nn.Embedding(meta["vocab"][f], emb_dim, padding_idx=0) for f in self.fields})
        self.dow = nn.Embedding(8, 4, padding_idx=0)
        self.proj = nn.Linear(emb_dim * len(self.fields) + 4 + 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, b: dict) -> torch.Tensor:
        parts = [self.cat[f](b[f]) for f in self.fields]
        parts.append(self.dow(b["dow"]))
        parts.append(torch.stack([b["amount"], b["log_dt"], b["log_to_cut"]], dim=-1))
        return self.drop(self.norm(self.proj(torch.cat(parts, dim=-1))))


class EventTransformer(nn.Module):
    def __init__(self, meta: dict, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                 max_len: int = 256, n_targets: int = 4):
        super().__init__()
        self.fields, self.max_len = meta["fields"], max_len
        self.emb = EventEmbedding(meta, d_model, dropout=dropout)
        self.pos = nn.Embedding(max_len + 1, d_model)  # позиция max_len отдана CLS
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_targets))
        self.mlm_heads = nn.ModuleDict({f: nn.Linear(d_model, meta["vocab"][f]) for f in self.fields})
        self.mlm_amount = nn.Linear(d_model, meta["n_amount_bins"])

    def encode(self, b: dict, mlm_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.emb(b)
        if mlm_mask is not None:
            x = torch.where(mlm_mask.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
        B, T, _ = x.shape
        mask = b["mask"]
        lengths = mask.sum(1, keepdim=True)
        recency = (lengths - 1 - torch.arange(T, device=x.device).unsqueeze(0)).clamp(0, self.max_len - 1)
        x = x + self.pos(recency).to(x.dtype)
        cls = (self.cls + self.pos.weight[self.max_len]).to(x.dtype).expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], dim=1)
        return self.norm(self.encoder(h, src_key_padding_mask=pad))

    def forward(self, b: dict) -> torch.Tensor:
        return self.head(self.encode(b)[:, 0])

    def pooled(self, b: dict) -> torch.Tensor:
        """Эмбеддинг клиента: среднее по событиям (CLS при MLM не обучается)."""
        h = self.encode(b)[:, 1:]
        m = b["mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def mlm_loss(self, b: dict, mask_prob: float = 0.15) -> tuple[torch.Tensor, dict]:
        mask = b["mask"]
        sel = (torch.rand(mask.shape, device=mask.device) < mask_prob) & mask
        rows = (sel.sum(1) == 0).nonzero(as_tuple=True)[0]
        if len(rows):  # в каждой последовательности хотя бы одно маскированное событие
            pos = (torch.rand(len(rows), device=mask.device) * mask[rows].sum(1)).long()
            sel[rows, pos] = True
        h = self.encode(b, mlm_mask=sel)[:, 1:][sel]
        losses = {f: F.cross_entropy(self.mlm_heads[f](h).float(), b[f][sel]) for f in self.fields}
        losses["amount_bin"] = F.cross_entropy(self.mlm_amount(h).float(), b["amount_bin"][sel])
        total = sum(losses.values()) / len(losses)
        return total, {k: float(v.detach()) for k, v in losses.items()}


class EventGRU(nn.Module):
    def __init__(self, meta: dict, d_model: int = 128, hidden: int = 256, dropout: float = 0.1, n_targets: int = 4,
                 **_):
        super().__init__()
        self.emb = EventEmbedding(meta, d_model, dropout=dropout)
        self.rnn = nn.GRU(d_model, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_targets))

    def pooled(self, b: dict) -> torch.Tensor:
        x = self.emb(b)
        packed = pack_padded_sequence(x, b["mask"].sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return h[-1]

    def forward(self, b: dict) -> torch.Tensor:
        return self.head(self.pooled(b))


def build_model(kind: str, meta: dict, cfg: dict) -> nn.Module:
    if kind == "transformer":
        return EventTransformer(meta, d_model=cfg["d_model"], n_layers=cfg["layers"], n_heads=cfg["heads"],
                                dropout=cfg["dropout"], max_len=cfg["max_len"])
    if kind == "gru":
        return EventGRU(meta, d_model=cfg["d_model"], hidden=cfg["hidden"], dropout=cfg["dropout"])
    raise ValueError(kind)
