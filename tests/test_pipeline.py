"""Тесты данных, признаков и метрик (без torch)."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import TEST_FOLD, VALID_FOLD
from src.data import load, sample_window, window_index
from src.gbm import build_features
from src.metrics import client_groups, fast_auc, paired_bootstrap_diff, per_target
from tests.helpers import tiny_prepared


class PrepareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = load(tiny_prepared())

    def test_events_sorted_within_clients(self):
        ev = self.P.events
        off = ev["offsets"]
        self.assertTrue(np.all(np.diff(off) >= 0))
        for c in range(0, len(off) - 1, 7):
            d = ev["day"][off[c]:off[c + 1]]
            self.assertTrue(np.all(np.diff(d) >= 0))

    def test_cut_is_strictly_before_target_month(self):
        ev, sm = self.P.events, self.P.samples
        for i in range(len(sm["cut"])):
            s, c, end = sm["start"][i], sm["cut"][i], ev["offsets"][sm["client"][i] + 1]
            self.assertTrue(np.all(ev["day"][s:c] < sm["cut_day"][i]))
            if c < end:  # событие с индексом cut уже в целевом месяце
                self.assertGreaterEqual(ev["day"][c], sm["cut_day"][i])
            self.assertLessEqual(c, sm["cut_leak"][i])

    def test_windows_never_see_target_month(self):
        sm = self.P.samples
        for i in range(0, len(sm["cut"]), 5):
            w = sample_window(self.P, i, 64)
            self.assertLess(w["last_day"], sm["cut_day"][i])
            self.assertEqual(len(w["amount"]), min(64, sm["cut"][i] - sm["start"][i]))
            self.assertTrue(np.all(w["dow"] >= 1) and np.all(w["dow"] <= 7))
            self.assertTrue(np.all(w["log_dt"] >= 0))

    def test_folds_are_client_disjoint(self):
        cl = self.P.samples["client"]
        sets = {n: set(cl[self.P.split(n)]) for n in ("train", "valid", "test")}
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["train"] & sets["valid"])
        self.assertFalse(sets["valid"] & sets["test"])
        self.assertTrue(np.all(self.P.samples["fold"][self.P.split("test")] == TEST_FOLD))
        self.assertTrue(np.all(self.P.samples["fold"][self.P.split("valid")] == VALID_FOLD))

    def test_vocab_codes_in_range(self):
        for f in self.P.fields:
            codes = self.P.events[f]
            self.assertGreaterEqual(codes.min(), 1)  # 0 зарезервирован под паддинг
            self.assertLess(codes.max(), self.P.meta["vocab"][f])
        self.assertLess(self.P.events["amount_bin"].max(), self.P.meta["n_amount_bins"])

    def test_label_fraction_subsamples_clients(self):
        tr = self.P.split("train")
        sub = self.P.subsample_clients(tr, 0.2, seed=0)
        n_all = len(np.unique(self.P.samples["client"][tr]))
        n_sub = len(np.unique(self.P.samples["client"][sub]))
        self.assertAlmostEqual(n_sub / n_all, 0.2, delta=0.05)
        # все месяцы выбранного клиента остаются вместе
        kept = np.unique(self.P.samples["client"][sub])
        self.assertEqual(len(sub), np.isin(self.P.samples["client"][tr], kept).sum())


class FeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = load(tiny_prepared())

    def test_aggregates_match_naive_computation(self):
        P, sm, ev = self.P, self.P.samples, self.P.events
        idx = np.arange(0, len(sm["cut"]), 11)
        X = build_features(P, idx, width=10_000)
        for row, i in enumerate(idx):
            s, c, cd = sm["start"][i], sm["cut"][i], sm["cut_day"][i]
            day, xn = ev["day"][s:c], ev["amount_norm"][s:c].astype(np.float64)
            self.assertEqual(X.n_events[row], c - s)
            self.assertAlmostEqual(X.days_since_last[row], cd - day[-1], places=6)
            self.assertEqual(X.cnt_30d[row], int((cd - day <= 30).sum()))
            self.assertAlmostEqual(X.amt_mean_all[row], xn.mean(), places=4)
            self.assertAlmostEqual(X.amt_sum_30d[row], xn[cd - day <= 30].sum(), places=4)
            et = ev["event_type"][s:c]
            for col in [k for k in X.columns if k.startswith("event_type_") and k.endswith("_share")][:3]:
                code = int(col.split("_")[2])
                self.assertAlmostEqual(X[col][row], (et == code).mean(), places=6)

    def test_window_index(self):
        idx, valid = window_index(np.array([0, 10]), np.array([3, 30]), 5)
        self.assertEqual(idx[0][valid[0]].tolist(), [0, 1, 2])
        self.assertEqual(idx[1][valid[1]].tolist(), [25, 26, 27, 28, 29])


class MetricsTest(unittest.TestCase):
    def test_fast_auc_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(0)
        y = (rng.random(3000) < 0.05).astype(int)
        s = np.round(rng.random(3000) + 0.5 * y, 1)
        self.assertAlmostEqual(fast_auc(y, s), roc_auc_score(y, s), places=10)

    def test_client_groups(self):
        g = client_groups(np.array([3, 1, 3, 2, 1, 3]))
        self.assertEqual([x.tolist() for x in g], [[1, 4], [3], [0, 2, 5]])

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(1)
        n = 4000
        y = (rng.random((n, 4)) < 0.1).astype(int)
        weak, strong = rng.random((n, 4)) + 0.2 * y, rng.random((n, 4)) + 1.0 * y
        clients = rng.integers(0, 1000, n)
        r = paired_bootstrap_diff(y, weak, strong, clients, n_boot=200)
        self.assertGreater(r["ci95"][0], 0)
        r = paired_bootstrap_diff(y, weak, weak, clients, n_boot=50)
        self.assertEqual(r["diff"], 0.0)

    def test_constant_target_is_nan_not_crash(self):
        y = np.zeros((10, 4), dtype=int)
        y[:5, 0] = 1
        out = per_target(y, np.random.default_rng(0).random((10, 4)))
        self.assertTrue(np.isnan(out["target_2"]["roc_auc"]))


if __name__ == "__main__":
    unittest.main()
