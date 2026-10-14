"""Тесты моделей и скриптов обучения, без torch пропускаются.
Проверяют то, что чаще всего ломается: зависимость от паддинга, маски, загрузку весов, обучаемость.
"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tests.helpers import tiny_prepared


@unittest.skipIf(torch is None, "torch не установлен")
class TorchModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.data import load
        from src.models import EventGRU, EventTransformer
        from src.torch_data import SupervisedDataset, collate
        torch.manual_seed(0)
        cls.P = load(tiny_prepared())
        cls.collate = staticmethod(collate)
        idx = cls.P.split("train")
        lengths = cls.P.samples["cut"][idx] - cls.P.samples["start"][idx]
        # короткая и длинная история: при батчевании короткая получит паддинг
        cls.short, cls.long = idx[np.argmin(lengths)], idx[np.argmax(lengths)]
        cls.ds = SupervisedDataset(cls.P, np.array([cls.short, cls.long]), max_len=64)
        cls.tr = EventTransformer(cls.P.meta, d_model=32, n_layers=2, n_heads=4, max_len=64).eval()
        cls.gru = EventGRU(cls.P.meta, d_model=32, hidden=48).eval()

    def test_shapes(self):
        b = self.collate([self.ds[0], self.ds[1]])
        self.assertEqual(tuple(self.tr(b).shape), (2, 4))
        self.assertEqual(tuple(self.gru(b).shape), (2, 4))
        self.assertEqual(tuple(self.tr.pooled(b).shape), (2, 32))

    def test_padding_invariance(self):
        alone = self.collate([self.ds[0]])
        batched = self.collate([self.ds[0], self.ds[1]])
        self.assertLess(alone["mask"].shape[1], batched["mask"].shape[1])
        with torch.no_grad():
            for m in (self.tr, self.gru):
                torch.testing.assert_close(m(alone)[0], m(batched)[0], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(self.tr.pooled(alone)[0], self.tr.pooled(batched)[0], atol=1e-4, rtol=1e-4)

    def test_mlm_learns_on_one_batch(self):
        from src.models import EventTransformer
        torch.manual_seed(0)
        m = EventTransformer(self.P.meta, d_model=32, n_layers=2, n_heads=4, max_len=64, dropout=0.0)
        b = self.collate([self.ds[0], self.ds[1]])
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            torch.manual_seed(1)  # одни и те же маски: проверяем способность запомнить батч
            loss, parts = m.mlm_loss(b, 0.3)
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertTrue(np.isfinite(loss.item()))
        self.assertLess(loss.item(), 0.5 * first)
        self.assertIn("amount_bin", parts)

    def test_supervised_loss_decreases(self):
        from src.models import EventTransformer
        import torch.nn.functional as F
        torch.manual_seed(0)
        m = EventTransformer(self.P.meta, d_model=32, n_layers=2, n_heads=4, max_len=64, dropout=0.0)
        b = self.collate([self.ds[0], self.ds[1]])
        b["y"] = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(50):
            loss = F.binary_cross_entropy_with_logits(m(b), b["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.1)


@unittest.skipIf(torch is None, "torch не установлен")
class TrainScriptsTest(unittest.TestCase):
    """pretrain -> finetune -> embed -> report на маленьких данных на CPU."""

    def test_end_to_end(self):
        from src import report, train
        data = tiny_prepared()
        runs = tempfile.mkdtemp(prefix="runs_")
        common = dict(data=data, runs=runs, seed=0, workers=0, eval_batch=64, no_amp=True, d_model=32, layers=2,
                      heads=4, hidden=48, dropout=0.1, max_len=64)
        train.pretrain(argparse.Namespace(**common, name="mlm", batch=16, lr=1e-3, wd=0.01, steps=12,
                                          mask_prob=0.15, log_every=5, val_every=5, val_items=32))
        ckpt = Path(runs) / "mlm" / "encoder.pt"
        self.assertTrue(ckpt.exists())
        ft = dict(batch=32, lr=1e-3, wd=0.01, epochs=2, patience=2, steps_per_epoch=5, label_frac=1.0,
                  pos_weight="sqrt")
        train.finetune(argparse.Namespace(**common, **ft, model="transformer", name="tr_mlm_s0", init=str(ckpt)))
        train.finetune(argparse.Namespace(**common, **ft, model="gru", name="gru_s0", init=None))
        res = json.loads((Path(runs) / "tr_mlm_s0" / "metrics.json").read_text())
        self.assertEqual(res["model"], "transformer+mlm")
        self.assertTrue(0.0 <= res["mean_roc_auc"] <= 1.0)
        out = Path(runs) / "mlm" / "emb.npz"
        train.embed(argparse.Namespace(data=data, ckpt=str(ckpt), out=str(out), workers=0, eval_batch=64,
                                       no_amp=True, runs=runs, seed=0))
        E = np.load(out)["emb"]
        self.assertEqual(E.shape, (len(np.load(Path(data) / "samples.npz")["cut"]), 32))
        self.assertIn("tr_mlm", report.summarize(Path(runs)))


if __name__ == "__main__":
    unittest.main()
