"""患者分层（Segmentation / Targeting 对应 JD 商业价值链要求）。

基于校准仿真患者池（n=2000，标注 is_simulated=1）做 K-Means 分层。
优先使用 scikit-learn；缺失时回退到纯 Python K-Means（KMeans++ 初始化）。
输出：data/processed/patient_segments.csv + 各层画像汇总
"""
import csv
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

N_PATIENTS = 2000
K = 4
FEATURES = ["age", "bmi", "income_k", "glp1_aware", "cost_sensitive", "tech_savvy"]
SEED = config.SEED

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def generate_patient_pool() -> list[dict]:
    """生成校准仿真患者池（参数基于公开流行病学/消费者调查）。"""
    rng = random.Random(SEED)
    rows = []
    for i in range(N_PATIENTS):
        # 成人 25-75 岁，BMI 依中国+美国肥胖率混合分布（校准仿真）
        age = rng.gauss(48, 12)
        bmi = rng.gauss(30, 5.5)
        income_k = rng.gauss(55, 28)
        glp1_aware = 1 if rng.random() < 0.55 else 0
        cost_sensitive = 1 if rng.random() < 0.5 else 0
        tech_savvy = 1 if rng.random() < 0.4 else 0
        rows.append({
            "patient_id": i + 1,
            "age": round(max(18, min(85, age)), 1),
            "bmi": round(max(18, min(60, bmi)), 1),
            "income_k": round(max(5, income_k), 1),
            "glp1_aware": glp1_aware,
            "cost_sensitive": cost_sensitive,
            "tech_savvy": tech_savvy,
            "is_simulated": 1,
        })
    return rows


def _standardize(mat: list[list[float]]) -> list[list[float]]:
    """按列 z-score 标准化（纯标准库，供无 sklearn 时回退，与 sklearn 路径保持一致）。"""
    if not mat:
        return mat
    n_cols = len(mat[0])
    means = [sum(r[c] for r in mat) / len(mat) for c in range(n_cols)]
    sds = []
    for c in range(n_cols):
        var = sum((r[c] - means[c]) ** 2 for r in mat) / len(mat)
        sds.append(math.sqrt(var) if var > 0 else 1.0)
    return [[(r[c] - means[c]) / sds[c] for c in range(n_cols)] for r in mat]


def _pure_python_kmeans(mat: list[list[float]], k: int, iters: int = 50, seed: int = SEED) -> list[int]:
    """KMeans++ 初始化 + Lloyd 迭代（标准库实现，供无 sklearn 时回退）。"""
    rng = random.Random(seed)
    n, d = len(mat), len(mat[0])

    # KMeans++ 初始化
    centers = [mat[rng.randrange(n)]]
    for _ in range(k - 1):
        d2 = [min(sum((a - c) ** 2 for a, c in zip(p, cen)) for cen in centers) for p in mat]
        total = sum(d2)
        r = rng.random() * total
        acc, chosen = 0.0, n - 1
        for i, dd in enumerate(d2):
            acc += dd
            if acc >= r:
                chosen = i
                break
        centers.append(mat[chosen])

    labels = [0] * n
    for _ in range(iters):
        changed = False
        for i, p in enumerate(mat):
            best = min(range(k), key=lambda c: sum((a - b) ** 2 for a, b in zip(p, centers[c])))
            if best != labels[i]:
                labels[i] = best
                changed = True
        if not changed:
            break
        # 更新中心
        sums = [[0.0] * d for _ in range(k)]
        cnts = [0] * k
        for p, lab in zip(mat, labels):
            cnts[lab] += 1
            for j, v in enumerate(p):
                sums[lab][j] += v
        for c in range(k):
            if cnts[c]:
                centers[c] = [s / cnts[c] for s in sums[c]]
    return labels


def run_segmentation(patients: list[dict]) -> Path:
    mat = [[p[f] for f in FEATURES] for p in patients]
    if _HAS_SKLEARN:
        X = StandardScaler().fit_transform(np.array(mat, dtype=float))
        labels = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X)
        method = "sklearn KMeans (StandardScaler)"
    else:
        labels = _pure_python_kmeans(_standardize(mat), K)
        method = "纯 Python KMeans（回退，已标准化）"
    for p, lab in zip(patients, labels):
        p["segment"] = int(lab)

    out = config.DATA_PROCESSED / "patient_segments.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id"] + FEATURES + ["segment", "is_simulated"])
        writer.writeheader()
        writer.writerows(patients)
    _print_profiles(patients, method)
    return out


def _print_profiles(patients: list[dict], method: str):
    log.info(f"[patient_segmentation] 方法: {method}，K={K}")
    groups = defaultdict(list)
    for p in patients:
        groups[p["segment"]].append(p)
    for seg, rows in sorted(groups.items()):
        avg = lambda key: round(sum(r[key] for r in rows) / len(rows), 1)
        log.info(f"  段{seg} (n={len(rows)}): 平均年龄 {avg('age')}, BMI {avg('bmi')}, "
              f"收入 {avg('income_k')}k, 知晓率 {avg('glp1_aware'):.2f}, "
              f"价格敏感 {avg('cost_sensitive'):.2f}, 数字渠道 {avg('tech_savvy'):.2f}")


if __name__ == "__main__":
    config.setup_logging()
    run_segmentation(generate_patient_pool())
