import math
import unittest

import numpy as np

import adaptive_contacts as ac


class AdaptiveContactsTests(unittest.TestCase):
    def test_power_threshold_increases_with_separation_and_caps(self):
        seps = np.array([6, 24, 100, 1000, 4640], dtype=np.float32)
        theta = ac.threshold_for_sequence_separation(seps)
        self.assertAlmostEqual(float(theta[0]), ac.DEFAULT_THETA0, places=6)
        self.assertTrue(np.all(np.diff(theta) >= -1e-7))
        self.assertLessEqual(float(theta[-1]), ac.DEFAULT_THETA_MAX)
        self.assertGreaterEqual(float(theta[-1]), 0.93)

    def test_static_binarization_matches_fixed_threshold(self):
        prob = np.array(
            [
                [0.0, 0.86, 0.84],
                [0.86, 0.0, 0.90],
                [0.84, 0.90, 0.0],
            ],
            dtype=np.float32,
        )
        got = ac.binarize_contact_probability(prob, adaptive=False, min_seq_sep=0)
        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(got, expected)

    def test_adaptive_threshold_suppresses_borderline_long_range_contact(self):
        n = 130
        prob = np.zeros((n, n), dtype=np.float32)
        prob[0, 7] = prob[7, 0] = 0.86
        prob[0, 120] = prob[120, 0] = 0.90
        static = ac.binarize_contact_probability(prob, adaptive=False)
        adaptive = ac.binarize_contact_probability(prob, adaptive=True)
        self.assertEqual(static[0, 7], 1.0)
        self.assertEqual(static[0, 120], 1.0)
        self.assertEqual(adaptive[0, 7], 1.0)
        self.assertEqual(adaptive[0, 120], 0.0)

    def test_density_gamma_and_mcc_helpers(self):
        n = 80
        truth = np.zeros((n, n), dtype=np.float32)
        pred = np.zeros((n, n), dtype=np.float32)
        for i in range(n - 8):
            truth[i, i + 8] = truth[i + 8, i] = 1.0
            pred[i, i + 8] = pred[i + 8, i] = 1.0
        for i in range(0, n - 50, 10):
            truth[i, i + 50] = truth[i + 50, i] = 1.0
        density = ac.contact_density_by_separation_bins(pred)
        mcc = ac.matthews_corrcoef_by_separation_bins(pred, truth)
        gamma = ac.fit_contact_decay_gamma(pred, n_bins=8)
        self.assertGreater(density['short'], 0.0)
        self.assertIn('medium', mcc)
        self.assertTrue(math.isnan(gamma) or gamma >= 0.0)


if __name__ == '__main__':
    unittest.main()
