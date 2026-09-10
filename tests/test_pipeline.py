"""流水线组件冒烟 / 回归测试（重点是"数据泄露"回归 + 关键组件正确性）。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from build_warehouse import _split_statements
from patient_segmentation import _pure_python_kmeans, _standardize


class TestSplitStatements(unittest.TestCase):
    def test_basic(self):
        parts = _split_statements("SELECT 1;\nSELECT 2;\n")
        self.assertEqual(parts, ["SELECT 1;", "SELECT 2;"])

    def test_inline_comment(self):
        # 行内注释里的分号不应被当作语句终止符
        parts = _split_statements("SELECT 1 -- 注释; 分号在注释里\n;\n")
        self.assertEqual(len(parts), 1)
        self.assertTrue(parts[0].startswith("SELECT 1"))

    def test_semicolon_in_string(self):
        # 字符串字面量内的分号不算终止符
        parts = _split_statements("SELECT 'a;b' AS x;\nSELECT 2;\n")
        self.assertEqual(len(parts), 2)
        self.assertIn("'a;b'", parts[0])

    def test_missing_trailing_semicolon(self):
        # 末条语句缺分号也应保留执行
        parts = _split_statements("SELECT 1;\nSELECT 2")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1], "SELECT 2")


class TestStandardize(unittest.TestCase):
    def test_zscore(self):
        mat = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        out = _standardize(mat)
        for c in range(2):
            col = [r[c] for r in out]
            mean = sum(col) / len(col)
            var = sum((x - mean) ** 2 for x in col) / len(col)
            self.assertAlmostEqual(mean, 0.0, places=9)
            self.assertAlmostEqual(var, 1.0, places=9)

    def test_empty(self):
        self.assertEqual(_standardize([]), [])


class TestKMeansFallback(unittest.TestCase):
    def test_clusters_well_separated_data(self):
        """纯 Python KMeans 应对明显分离的 3 簇给出正确聚类。"""
        mat = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
               [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
               [20.0, 20.0], [20.1, 20.0], [20.0, 20.1]]
        labels = _pure_python_kmeans(mat, k=3, iters=50, seed=1)
        self.assertEqual(len(labels), len(mat))
        self.assertEqual(len(set(labels)), 3)
        # 同一簇内的点标签一致
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[3], labels[5])
        self.assertEqual(labels[6], labels[7])
        self.assertEqual(labels[6], labels[8])
        # 不同簇的点标签互异
        self.assertNotEqual(labels[0], labels[3])
        self.assertNotEqual(labels[3], labels[6])


class TestNoLeakageFeatures(unittest.TestCase):
    """数据泄露回归测试：目标变量的派生特征必须从特征集中移除。"""

    BANNED = {"is_active_status", "estimated_completion_year"}

    def test_leakage_features_removed(self):
        from trial_ai_models import engineer_features, load_trials
        trials = load_trials()
        if len(trials) < 30:
            self.skipTest("无试验数据，跳过")
        _, _, cols = engineer_features(trials)
        leaked = self.BANNED & set(cols)
        self.assertFalse(leaked, f"发现目标泄露特征仍在特征集: {leaked}")


class TestDetectMolecule(unittest.TestCase):
    def test_detect_molecule(self):
        from ingest_clinicaltrials import _detect_molecule
        self.assertEqual(_detect_molecule("SURPASS-1: Tirzepatide Study"), "tirzepatide")
        self.assertEqual(_detect_molecule("A Study of Semaglutide in Adults"), "semaglutide")
        self.assertEqual(_detect_molecule("Randomized Trial of Experimental Drug"), "other-glp1")


if __name__ == "__main__":
    unittest.main()
