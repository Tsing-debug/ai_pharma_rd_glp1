"""全局配置：路径与关键参数。

所有路径相对于本文件所在目录解析，保证流水线在任何工作目录下可运行。
"""
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
DB_PATH = DATA_PROCESSED / "analytics.db"

# 全局随机种子（各模块统一引用，保证结果可复现）
SEED = 42

# 核心分析品牌（礼来视角：替尔泊肽 vs 司美格鲁肽）
BRAND_FOCUS = {
    "tirzepatide": "替尔泊肽（Mounjaro/Zepbound，礼来）",
    "semaglutide": "司美格鲁肽（Ozempic/Wegovy，诺和诺德）",
}

# 中国 GLP-1 市场规模测算参数（仿真模块用，基于公开报告校准）
CN_MARKET_PARAMS = {
    "adult_population_mn": 1400,       # 中国成人人口（百万）量级
    "obesity_prevalence": 0.16,        # 中国成人肥胖率（公开研究约 14-16%）
    "overweight_prevalence": 0.34,     # 超重率
    "glp1_treatment_penetration_2030": 0.012,  # 2030 治疗渗透率假设（仿真）
    "annual_treatment_cost_usd": 1800,         # 年治疗费用假设（仿真正规渠道）
}

# 预测参数
FORECAST_YEARS = [2025, 2026, 2027, 2028, 2029, 2030]
BACKTEST_WINDOW = 2  # MAPE 回测：用最后 N 年做 holdout


def setup_logging(level: int = logging.INFO) -> None:
    """统一日志配置。入口脚本（run_pipeline / 各模块 __main__）调用一次即可。"""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
