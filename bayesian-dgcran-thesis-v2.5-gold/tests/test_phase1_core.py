"""Automated Unit Test Suite for Phase 1 Core Modules."""
import unittest, tempfile, os, json
import numpy as np

from src.data.preprocessing import (
    impute_train_linear_bounded,
    impute_causal_locf,
    build_calendar_features,
    chronological_split,
    fit_scaler,
    create_sliding_windows,
    inject_missing_and_noise,
)
from src.metrics.evaluation import (
    compute_point_metrics,
    compute_interval_metrics,
    compute_crps_empirical_fair,
    compute_crps_gaussian,
    evaluate_all_metrics,
)
from src.metrics.diebold_mariano import diebold_mariano_test
from src.calibration.nemhcc import NEMHCCCalibrator
from src.utils.io_schema import save_experiment_result


class TestPhase1Core(unittest.TestCase):
    def test_leak_free_preprocessing_and_moment_recovery(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(100.0, 20.0, size=(200, 10, 3))
        raw[10, 2, 0] = np.nan
        raw[150, 3, 0] = np.nan  # inside val/test
        tr, va, te, (t1, t2) = chronological_split(raw, (0.6, 0.2, 0.2))
        tr_imp = impute_train_linear_bounded(tr)
        self.assertFalse(np.isnan(tr_imp).any())
        va_imp = impute_causal_locf(va, tr_imp[-1])
        self.assertFalse(np.isnan(va_imp).any())
        # Verify causal LOCF used previous step (149 - 120 = 29), not future step
        self.assertAlmostEqual(va_imp[30, 3, 0], va_imp[29, 3, 0], places=6)

        scaler = fit_scaler(tr_imp)
        tr_norm = scaler.transform(tr_imp)
        self.assertAlmostEqual(float(np.mean(tr_norm[:, :, 0])), 0.0, places=5)
        self.assertAlmostEqual(float(np.std(tr_norm[:, :, 0])), 1.0, places=5)
        # Variance scales by sigma^2
        var_phys = scaler.inverse_variance(np.array([1.0]), target_channel=0)[0]
        self.assertAlmostEqual(var_phys, scaler.std[0] ** 2, places=5)

        cal = build_calendar_features(len(tr_norm))
        win = create_sliding_windows(tr_norm, cal, history_len=12, horizon_len=12)
        self.assertEqual(win["X"].shape[1:], (12, 10, 3))
        self.assertEqual(win["Y"].shape[1:], (12, 10))

        corrupted = inject_missing_and_noise(win["X"], missing_rate=0.3, noise_std=0.1, block_missing=True)
        self.assertEqual(corrupted.shape, win["X"].shape)

    def test_fair_empirical_crps_exact_values(self):
        # Exact analytical test: M=3, Y=(0,1,2), y=0.5 -> CRPS_fair = 1/6
        samples = np.array([[[[0.0]]], [[[1.0]]], [[[2.0]]]])  # (3, 1, 1, 1)
        y_true = np.array([[[0.5]]])
        self.assertAlmostEqual(compute_crps_empirical_fair(y_true, samples), 1.0 / 6.0, places=10)

    def test_nemhcc_coverage_and_online_causality(self):
        rng = np.random.default_rng(7)
        M_cal, M_test, T_p, N = 300, 120, 12, 5
        mu_cal = rng.normal(50.0, 5.0, size=(M_cal, T_p, N))
        sig_cal = np.full((M_cal, T_p, N), 2.0)
        y_cal = mu_cal + rng.normal(0.0, 3.0, size=(M_cal, T_p, N))  # under-dispersed raw sigma

        mu_test = rng.normal(50.0, 5.0, size=(M_test, T_p, N))
        sig_test = np.full((M_test, T_p, N), 2.0)
        y_test = mu_test + rng.normal(0.0, 3.0, size=(M_test, T_p, N))

        cal = NEMHCCCalibrator(alpha=0.05, rho_cal=0.995)
        lo, hi = cal.online_rolling_predict(y_cal, mu_cal, sig_cal, y_test, mu_test, sig_test, n_update=40)
        iv = compute_interval_metrics(y_test, lo, hi, alpha=0.05)
        # Raw Gaussian 95% coverage would be ~81%, NE-MHCC should bring it near >= 93%
        self.assertGreater(iv["PICP"], 0.92)
        self.assertLess(iv["MHPICE"], 0.05)

        full = evaluate_all_metrics(y_test, mu_test, sig_test ** 2, 0.25 * sig_test ** 2, lo, hi, alpha=0.05)
        self.assertIn("overall", full)
        self.assertIn("horizon_60min", full["per_horizon"])
        # Sparsification MAE should be non-empty
        self.assertEqual(len(full["sparsification"]["MAE"]), 7)

    def test_diebold_mariano_and_json_export(self):
        rng = np.random.default_rng(123)
        loss1 = rng.normal(10.0, 1.0, size=400)
        loss2 = rng.normal(11.5, 1.0, size=400)
        dm = diebold_mariano_test(loss1, loss2, horizon=12)
        self.assertLess(dm["DM_stat"], -5.0)
        self.assertLess(dm["HLN_pvalue_one_sided"], 0.001)

        with tempfile.TemporaryDirectory() as tmp:
            p = save_experiment_result(tmp, "PeMS08", "Bayesian-DGCRAN", {"overall": {"MAE": 14.2}}, {"d": 8}, dm)
            self.assertTrue(os.path.exists(p))
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["dataset"], "PeMS08")


if __name__ == "__main__":
    unittest.main()
