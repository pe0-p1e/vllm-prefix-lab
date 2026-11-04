import unittest
from scripts.plot_cpu_kv_results import affine_fit


class FitTests(unittest.TestCase):
    def test_known_transfer_model(self):
        fit=affine_fit([.1,.5,1,2],[12,52,102,202])
        self.assertAlmostEqual(fit['intercept_ms'],2)
        self.assertAlmostEqual(fit['slope_ms_per_gib'],100)
        self.assertAlmostEqual(fit['r_squared'],1)
        self.assertAlmostEqual(fit['rmse_ms'],0)

    def test_unidentifiable_model(self):
        with self.assertRaises(ValueError):affine_fit([1,1,1],[2,3,4])
