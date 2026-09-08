"""生物统计：临床试验样本量与功效计算（辉瑞 AI Pilot / CRDC 生物统计分析方向）。

提供两组经典公式（Chow et al., Sample Size Calculations in Clinical Research）：
  1) 二分类终点（应答率）两独立样本优效性检验样本量
  2) 连续终点（如体重变化%）两独立样本优效性检验样本量
纯标准库实现；示例数字取自已发表的 GLP-1 关键试验公开结果（示意用途，正式使用以方案为准）。

输出：data/processed/sample_size_results.csv
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

Z_ALPHA = 1.959964   # 双侧 α=0.05
Z_ALPHA_1S = 1.644854  # 单侧 α=0.05
Z_BETA = 0.841621   # 功效 80%


def n_two_proportion(p1: float, p2: float, alpha: float = 0.05,
                     power: float = 0.80, two_sided: bool = True) -> int:
    """应答率 p1（对照）vs p2（试验）优效性检验：每组样本量。"""
    za = Z_ALPHA if two_sided else Z_ALPHA_1S
    zb = Z_BETA if power == 0.80 else 0.8416
    pbar = (p1 + p2) / 2.0
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(math.ceil(num ** 2 / (p2 - p1) ** 2))


def n_continuous(mu1: float, mu2: float, sd: float, alpha: float = 0.05,
                 power: float = 0.80, two_sided: bool = True) -> int:
    """连续终点（均值差）优效性检验：每组样本量。"""
    za = Z_ALPHA if two_sided else Z_ALPHA_1S
    zb = Z_BETA if power == 0.80 else 0.8416
    return int(math.ceil(2 * (za + zb) ** 2 * sd ** 2 / (mu2 - mu1) ** 2))


def run() -> Path:
    # 示例 1（二分类终点）：应答率 55% vs 45%（10pp 差，接近真实 GLP-1 RCT 的组间差异量级）
    n_resp = n_two_proportion(0.45, 0.55)
    # 示例 2（连续终点）：HbA1c 变化 -1.1% vs -0.6%，SD=1.0（示意，参考 SURPASS 系列量级）
    n_cont = n_continuous(0.6, 1.1, sd=1.0)

    rows = [
        {"scenario": "二分类终点 应答率：试验组 55% vs 对照组 45%（差 10pp）",
         "type": "two_proportion", "n_per_arm": n_resp, "total_n": 2 * n_resp,
         "note": "示意参数，接近真实 GLP-1 RCT 量级"},
        {"scenario": "连续终点 HbA1c 降幅：-1.1% vs -0.6%，SD=1.0",
         "type": "continuous", "n_per_arm": n_cont, "total_n": 2 * n_cont,
         "note": "示意参数，参考 SURPASS 系列量级"},
    ]
    out = config.DATA_PROCESSED / "sample_size_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "type", "n_per_arm", "total_n", "note"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"[sample_size] {r['scenario']} -> 每组 {r['n_per_arm']} 例（总 {r['total_n']}）")
    print(f"[sample_size] 完成 -> {out}")
    return out


if __name__ == "__main__":
    run()
