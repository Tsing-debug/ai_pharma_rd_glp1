"""品牌销量预测（对应 JD：市场预测 / 品牌表现）。

方法：线性趋势最小二乘 + CAGR 阻尼趋势的简单集成；
评估：walk-forward MAPE 回测（最后 BACKTEST_WINDOW 年做 holdout）。
全部标准库实现；statsmodels ARIMA / Prophet 可后续接入做对比（见 requirements）。
"""
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def _load_series() -> dict[str, list[tuple[int, float]]]:
    """读入真实品牌收入序列 {brand_id: [(year, revenue_usd_m), ...]}。"""
    series: dict[str, list[tuple[int, float]]] = {}
    with open(config.DATA_RAW / "financials_lilly_novo.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            series.setdefault(r["brand_id"], []).append(
                (int(r["year"]), float(r["revenue_usd_m"])))
    for v in series.values():
        v.sort()
    return series


def _ols_trend(vals: list[float]) -> tuple[float, float]:
    """最小二乘直线：y = slope * x + intercept（x 为相对索引）。"""
    n = len(vals)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(vals) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, vals))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    return slope, intercept


def _cagr(last: float, first: float, n_years: int) -> float:
    return (last / max(first, 1e-9)) ** (1 / max(n_years, 1)) - 1.0


def _mape(actual: list[float], pred: list[float]) -> float:
    return 100.0 * statistics.mean(
        abs(a - p) / abs(a) for a, p in zip(actual, pred) if a != 0)


def forecast_brand(points: list[tuple[int, float]]) -> dict:
    years = [p[0] for p in points]
    vals = [p[1] for p in points]
    last_year = years[-1]

    # walk-forward 回测：用前 n - k 年拟合并预测 holdout 年
    hold = min(config.BACKTEST_WINDOW, max(len(vals) - 2, 0))
    if hold == 0:  # 数据太少无法回测（仅 1-2 年）
        mape_ols = mape_ensemble = float("nan")
        hist = vals
    else:
        hist, actual = vals[:-hold], vals[-hold:]
        slope, intercept = _ols_trend(hist)
        ols_pred = [slope * (len(hist) + i) + intercept for i in range(hold)]
        mape_ols = _mape(actual, ols_pred)

        # CAGR 阻尼趋势（参照近期增速，逐年衰减 20%）
        cagr_rate = _cagr(hist[-1], hist[0], len(hist) - 1)
        cagr_pred = []
        cur = hist[-1]
        for i in range(hold):
            cagr_pred.append(cur * (1 + cagr_rate * (0.8 ** i)))
            cur = cagr_pred[-1]

        # 集成 = 简单平均
        ensemble = [(o + c) / 2 for o, c in zip(ols_pred, cagr_pred)]
        mape_ensemble = _mape(actual, ensemble)

    # 用全量数据预测未来
    f_slope, f_intercept = _ols_trend(vals)
    f_cagr = _cagr(vals[-1], vals[0], len(vals) - 1)
    forecast = []
    for i, fy in enumerate(config.FORECAST_YEARS):
        ols_fc = f_slope * (len(vals) - 1 + i + 1) + f_intercept
        cur = vals[-1] * (1 + f_cagr * (0.8 ** (i + 1)))
        forecast.append((fy, max(0.0, (ols_fc + cur) / 2)))

    return {
        "years": years, "vals": vals, "last_year": last_year,
        "mape_ols": round(mape_ols, 2), "mape_ensemble": round(mape_ensemble, 2),
        "forecast": forecast,
    }


def run_forecasting() -> Path:
    out = config.DATA_PROCESSED / "forecast_brands.csv"
    series = _load_series()
    rows = []
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["brand_id", "year", "revenue_usd_m", "type", "mape_backtest_pct"])
        for brand, points in series.items():
            res = forecast_brand(points)
            print(f"[forecasting] {brand}: 回测 MAPE={res['mape_ensemble']}%")
            for y, v in zip(res["years"], res["vals"]):
                writer.writerow([brand, y, round(v, 1), "actual", ""])
            for fy, fv in res["forecast"]:
                writer.writerow([brand, fy, round(fv, 1), "forecast", res["mape_ensemble"]])
            rows.append(res)
    print(f"[forecasting] 完成 -> {out}")
    return out


if __name__ == "__main__":
    run_forecasting()
