"""A/B 实验评估（对应 JD：营销效果测量、A/B testing principles）。

场景：数字渠道投放对比实验 —— 对照组（标准内容） vs 实验组（GLP-1 教育内容）
输出：Welch t 检验 + bootstrap 95% CI + 功效分析（全部标准库实现）
"""
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42


def simulate_campaign(n_control=800, n_test=800, seed=SEED) -> dict:
    """生成实验数据（仿真，标注用途：方法演示）。

    对照组转化率 ~5.2%，实验组 ~7.0% —— 模拟"患者教育内容提升线上问诊预约转化"。
    """
    rng = random.Random(seed)
    control = [1 if rng.random() < 0.052 else 0 for _ in range(n_control)]
    test = [1 if rng.random() < 0.070 else 0 for _ in range(n_test)]
    return {"control": control, "test": test, "is_simulated": 1}


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t 检验（双尾）。返回 (t 统计量, 近似 p 值)。

    p 值用正态近似 + 学生 t 自由度校正；无 scipy 时精度足够演示用途。
    """
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else 0.0
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-12)
    # 双侧 p 值（正态近似）
    p = 2.0 * (1.0 - _normal_cdf(abs(t)))
    return t, p


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bootstrap_ci(a: list[float], b: list[float], n_boot=2000, seed=SEED) -> tuple[float, float]:
    """bootstrap 95% CI 作用于 (组B均值 - 组A均值)。"""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        diffs.append(statistics.mean(sb) - statistics.mean(sa))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    return lo, hi


def power_analysis(effect_size: float, alpha=0.05, power=0.80, base_rate=0.052) -> int:
    """粗略功效分析：达到给定功效所需的最小样本量（单侧 z 检验近似）。"""
    z_alpha = 1.6449 if alpha == 0.05 else 1.96
    z_beta = 0.8416 if power == 0.80 else 1.2816
    p = base_rate
    q = base_rate + effect_size
    var = p * (1 - p) + q * (1 - q)
    n = var * (z_alpha + z_beta) ** 2 / (effect_size ** 2)
    return int(math.ceil(n))


def run_ab_test() -> dict:
    data = simulate_campaign()
    c, t = data["control"], data["test"]
    conv_c = statistics.mean(c)
    conv_t = statistics.mean(t)
    lift = (conv_t - conv_c) / conv_c
    t_stat, p_value = _welch_t(c, t)
    lo, hi = bootstrap_ci(c, t)

    result = {
        "n_control": len(c), "n_test": len(t),
        "conv_control_pct": round(conv_c * 100, 2),
        "conv_test_pct": round(conv_t * 100, 2),
        "lift_pct": round(lift * 100, 2),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "boot_ci_diff": (round(lo * 100, 2), round(hi * 100, 2)),
        "needed_n_for_80pct_power": power_analysis(conv_t - conv_c),
        "is_simulated": 1,
    }
    print("[hypothesis_testing] A/B 结果：")
    print(f"  对照组转化率 {result['conv_control_pct']}% vs 实验组 {result['conv_test_pct']}% "
          f"（提升 {result['lift_pct']}%）")
    print(f"  Welch t = {result['t_stat']}, p = {result['p_value']}"
          + ("（显著）" if p_value < 0.05 else "（不显著）"))
    print(f"  bootstrap 95% CI（绝对差，百分点）: {result['boot_ci_diff']}")
    return result


if __name__ == "__main__":
    run_ab_test()
