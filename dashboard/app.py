"""Streamlit 看板：礼来视角 GLP-1 商业分析（4 页）。

运行：streamlit run dashboard/app.py
数据源：data/processed/（由 src/run_pipeline.py 生成）
"""
import csv
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

st.set_page_config(page_title="Lilly GLP-1 Commercial Analytics", layout="wide")


def load_csv(name: str) -> list[dict]:
    p = config.DATA_PROCESSED / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_query(name: str) -> list[dict]:
    return load_csv(f"query_results/{name}")


@st.cache_data
def load_all():
    return {
        "brands": load_query("query_01.csv"),
        "yoy": load_query("query_02.csv"),
        "launch": load_query("query_03.csv"),
        "safety": load_query("query_05.csv"),
        "trials": load_query("query_06.csv"),
        "china": load_query("query_07.csv"),
        "segments": load_csv("patient_segments.csv"),
    }


def to_table(rows: list[dict]) -> dict:
    if not rows:
        return {"列": []}
    cols = list(rows[0].keys())
    return {c: [r[c] for r in rows] for c in cols}


def _pivot(rows: list[dict], x: str, series: str | None, y: str) -> pd.DataFrame:
    """简易透视：返回 DataFrame（x 为行索引、series 为列），兼容新版 st.line_chart。

    旧版直接传 dict 的写法在 streamlit 1.30+ 已弃用。
    """
    def _key(v):
        try:
            return (0, float(v))  # 数值优先，保持时间顺序
        except (TypeError, ValueError):
            return (1, str(v))
    idx = sorted({r[x] for r in rows}, key=_key)
    out: dict[str, list] = {}
    for r in rows:
        sv = r[series] if series else y  # 无 series 时用指标名作列
        out.setdefault(sv, [None] * len(idx))[idx.index(r[x])] = float(r[y])
    df = pd.DataFrame(out, index=idx)
    df.index.name = x
    return df


st.sidebar.title("🎯 Lilly GLP-1 Commercial Analytics")
page = st.sidebar.radio("页面", ["品牌表现", "市场监测", "患者分层", "真实世界信号"])
data = load_all()

if page == "品牌表现":
    st.title("品牌表现 Brand Performance")
    kpi = data["brands"]
    if kpi:
        latest_year = max(int(r["year"]) for r in kpi)
        st.metric("最新数据年", latest_year)
        c1, c2, c3 = st.columns(3)
        latest = [r for r in kpi if int(r["year"]) == latest_year]
        for col, r in zip([c1, c2, c3], sorted(latest, key=lambda x: -float(x["revenue_usd_m"]))[:3]):
            col.metric(r["brand_name"], f"{float(r['revenue_usd_m']):,.0f}M",
                       f"份额 {r['share_pct']}%")
        st.subheader("各品牌收入走势")
        st.line_chart(_pivot(kpi, "year", "brand_name", "revenue_usd_m"))
        st.dataframe(to_table(kpi))

elif page == "市场监测":
    st.title("市场监测 Market Monitoring")
    yoy = data["yoy"]
    st.subheader("分子级同比增速 YoY")
    st.dataframe(to_table(yoy))
    launch = data["launch"]
    if launch:
        st.subheader("Launch Tracker：tirzepatide 上市后曲线")
        st.line_chart(_pivot(launch, "years_since_launch", "brand_name", "revenue_usd_m"))
        st.dataframe(to_table(launch))
    china = data["china"]
    if china:
        st.subheader("中国 GLP-1 市场测算（仿真，is_simulated=1）")
        st.line_chart(_pivot(china, "year", None, "market_usd_bn"))
        st.caption("⚠️ 仿真数据，仅方法演示。正式使用请替换为 CDE/医保局公开数据。")

elif page == "患者分层":
    st.title("患者分层 Patient Segmentation")
    seg = data["segments"]
    if seg:
        by_seg: dict[str, list[dict]] = {}
        for r in seg:
            by_seg.setdefault(r["segment"], []).append(r)
        cols = st.columns(len(by_seg))
        for col, (s, rows) in zip(cols, sorted(by_seg.items())):
            n = len(rows)
            avg = lambda k: sum(float(r[k]) for r in rows) / n
            col.metric(f"段 {s} (n={n})",
                       f"BMI {avg('bmi'):.0f} · {avg('age'):.0f} 岁",
                       f"知晓率 {avg('glp1_aware'):.0%}")
        st.caption("仿真患者池（is_simulated=1），方法演示用途。")
        st.dataframe(to_table(seg))

else:
    st.title("真实世界信号 RWE")
    safety = data["safety"]
    if safety:
        st.subheader("FAERS 不良事件 Top（真实数据）")
        st.dataframe(to_table(safety))
    trials = data["trials"]
    if trials:
        st.subheader("临床管线：公司 × 阶段（真实数据）")
        st.dataframe(to_table(trials))
    st.caption("数据来源：FDA OpenFDA / ClinicalTrials.gov，样例子集；API 全量接入见 src/ingest_*.py。")
