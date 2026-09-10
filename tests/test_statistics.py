"""统计公式单测：scipy/sklearn 可用时对照官方实现，验证手写公式正确性。

运行：python -m unittest discover -s tests -v
"""
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hypothesis_testing import _welch_t, bootstrap_ci, simulate_campaign
from sample_size import n_two_proportion, n_continuous

try:
    import scipy.stats as stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class TestWelchT(unittest.TestCase):
    def test_welch_t_matches_scipy(self):
        """手写 Welch t 应与 scipy.ttest_ind(equal_var=False) 一致。"""
        if not _HAS_SCIPY:
            self.skipTest("scipy 不可用，跳过对照")
        rng = random.Random(1)
        a = [rng.gauss(10, 2) for _ in range(40)]
        b = [rng.gauss(11, 2.5) for _ in range(50)]
        t_ours, p_ours = _welch_t(a, b)
        t_ref, p_ref = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(t_ours, t_ref, places=6)
        self.assertAlmostEqual(p_ours, p_ref, places=6)

    def test_ab_campaign_significant(self):
        """仿真 A/B 数据（对照组 5.2% vs 实验组 7.0%）应显著。"""
        data = simulate_campaign(seed=42)
        t, p = _welch_t(data["control"], data["test"])
        self.assertLess(p, 0.05)


class TestBootstrapCI(unittest.TestCase):
    def test_ci_covers_true_diff(self):
        a = [0.0] * 50 + [1.0] * 50   # 均值 0.5
        b = [0.0] * 30 + [1.0] * 70   # 均值 0.7
        lo, hi = bootstrap_ci(a, b, n_boot=500, seed=3)
        self.assertLessEqual(lo, 0.2)
        self.assertGreaterEqual(hi, 0.2)


class TestSampleSize(unittest.TestCase):
    def test_two_proportion_known_value(self):
        """按 Chow 公式独立手算参考值（p1=0.45, p2=0.55, 双侧 α=0.05, 功效 80%）。"""
        za, zb = 1.959964, 0.841621
        p1, p2 = 0.45, 0.55
        pbar = (p1 + p2) / 2.0
        num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        expected = math.ceil(num ** 2 / (p2 - p1) ** 2)
        self.assertEqual(n_two_proportion(p1, p2), expected)

    def test_n_continuous_known_value(self):
        """连续终点样本量：独立手算参考值。"""
        za, zb = 1.959964, 0.841621
        expected = math.ceil(2 * (za + zb) ** 2 * 1.0 ** 2 / (1.1 - 0.6) ** 2)
        self.assertEqual(n_continuous(0.6, 1.1, sd=1.0), expected)

    def test_monotonicity(self):
        """效应量越大，所需样本量越小。"""
        self.assertGreater(n_two_proportion(0.45, 0.60), n_two_proportion(0.45, 0.65))
        self.assertGreater(n_continuous(0.6, 1.0, sd=1.0), n_continuous(0.6, 1.2, sd=1.0))

    def test_positive_integer(self):
        self.assertGreater(n_two_proportion(0.45, 0.55), 0)
        self.assertIsInstance(n_two_proportion(0.45, 0.55), int)


if __name__ == "__main__":
    unittest.main()
