"""主流水线：数据接入 → 数仓 → 患者分层 → A/B 检验 → 预测 → 高管简报。

用法：python src/run_pipeline.py
核心部分仅依赖 Python 标准库；sklearn 等为可选增强。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from ingest_clinicaltrials import main as ingest_trials
from ingest_openfda import main as ingest_faers
from build_warehouse import build as build_warehouse
from trial_ai_models import run as run_trial_ai
from nlp_analysis import run as run_nlp
from sample_size import run as run_sample_size
from patient_segmentation import run_segmentation, generate_patient_pool
from hypothesis_testing import run_ab_test
from forecasting import run_forecasting


def _read_query_csv(name: str) -> list[dict]:
    p = config.DATA_PROCESSED / "query_results" / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_processed_csv(name: str) -> list[dict]:
    p = config.DATA_PROCESSED / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(v: str) -> str:
    """把查询结果里的数值列格式化为千分位显示（供简报使用）。"""
    try:
        return f"{float(v):,.1f}"
    except ValueError:
        return v


def build_executive_brief(ab: dict) -> Path:
    q1 = _read_query_csv("query_01.csv")    # 各品牌历年收入+份额
    q2 = _read_query_csv("query_02.csv")    # YoY 增速
    q4 = _read_query_csv("query_04.csv")    # 2022 vs 最新份额对比
    q5 = _read_query_csv("query_05.csv")    # FAERS Top 事件
    q6 = _read_query_csv("query_06.csv")    # 管线分布
    q7 = _read_query_csv("query_07.csv")    # 中国市场仿真

    latest_year = max(int(r["year"]) for r in q1) if q1 else 0
    top_brand = max(q1, key=lambda r: float(r["revenue_usd_m"])) if q1 else {}
    top_safety = q5[0] if q5 else {}
    lilly_trials = sum(int(r["trial_count"]) for r in q6 if r["company"] == "Eli Lilly") if q6 else 0
    cn_latest = q7[-1] if q7 else {}

    lines = [
        "# Executive Brief：替尔泊肽 × GLP-1 商业分析（礼来视角）",
        "",
        f"> 生成时间：由 `src/run_pipeline.py` 自动产出。仿真数据已按 `is_simulated` 标注。",
        "",
        "## 一、品牌表现（真实公开数据）",
        "",
        f"- 最新数据年：**{latest_year}**；当年收入最高品牌：**{top_brand.get('brand_name', '-')}**"
        f"（{_fmt(top_brand.get('revenue_usd_m', '-'))} 百万美元，份额 {top_brand.get('share_pct', '-')}%）",
        "- 同比增速（最近可得年）：",
    ]
    yoy = [r for r in sorted(q2, key=lambda r: (r["molecule"], r["year"]))
           if r.get("yoy_growth_pct")]
    for r in yoy[-6:]:
        lines.append(f"  - {r['brand_name']} {r['year']}: {r['yoy_growth_pct']}%")
    lines += [
        "",
        "## 二、竞品格局（2022 → 最新）",
        "",
    ]
    for r in q4[:4]:
        lines.append(f"- {r['company']} · {r['brand_name']}: {_fmt(r['rev_2022'])} → {_fmt(r['rev_latest'])}（百万美元）")
    lines += [
        "",
        "## 三、真实世界安全信号（FAERS，真实数据）",
        "",
        f"- Top 不良事件：**{top_safety.get('event_type', '-')}**（{_fmt(top_safety.get('total_reports', '-'))} 例，"
        f"严重率 {top_safety.get('serious_rate_pct', '-')}%）——对应患者教育/说明书重点提示方向。",
        "",
        "## 四、临床管线（ClinicalTrials.gov，真实数据）",
        "",
        f"- 礼来相关试验 {lilly_trials} 项（样例子集；接入 API 后可全量）",
        f"- 明星试验：SURMOUNT-5（替尔泊肽 vs 司美格鲁肽头对头）为品牌差异化叙事核心证据。",
        "",
        "## 五、患者分层与营销实验（仿真，方法演示）",
        "",
        f"- K-Means（K=4）输出 4 类患者画像，可用于精准内容推送优先级（见 `patient_segments.csv`）。",
        f"- A/B 实验：对照组转化率 {ab['conv_control_pct']}% vs 实验组 {ab['conv_test_pct']}%，"
        f"提升 {ab['lift_pct']}%（p={ab['p_value']}，bootstrap 95% CI {ab['boot_ci_diff']}）；"
        f"80% 功效所需每组样本 ≈ {ab['needed_n_for_80pct_power']}。",
        "",
        "## 六、中国市场测算（仿真，明确标注）",
        "",
    ]
    if cn_latest:
        lines.append(f"- 仿真情景：2030 年中国 GLP-1 市场约 **{_fmt(cn_latest.get('market_usd_bn', '-'))} 亿美元**"
                     f"（渗透率 {cn_latest.get('penetration_pct', '-')}%）。仅用于方法演示，"
                     f"正式使用请替换为 CDE/医保局公开数据。")

    # ── AI 临床试验结果预测 ─────────────────────────────
    ai = _read_processed_csv("trial_model_results.csv")
    metrics, importance = {}, []
    for r in ai:
        fold = r.get("fold", "")
        if r.get("model") == "feature_importance":
            importance.append((r["fold"], float(r.get("auc") or 0)))
        elif str(fold).replace(".", "", 1).isdigit():
            metrics.setdefault(r["model"], []).append(float(r.get("auc") or 0))
    lines += ["", "## 七、AI：临床试验结果预测（ClinicalTrials.gov 真实数据）", ""]
    if metrics:
        for m, v in metrics.items():
            lines.append(f"- {m}: CV AUC = {sum(v) / len(v):.3f}")
        best = max(metrics, key=lambda m: sum(metrics[m]) / len(metrics[m]))
        top = sorted(importance, key=lambda kv: -abs(kv[1]))[:3]
        imp_str = "、".join(f"{f}({c:+.2f})" for f, c in top) if top else "（sklearn 不可用，略）"
        lines.append(f"- 最佳模型 **{best}**；主导特征：{imp_str}")
    else:
        lines.append("- 样本不足或依赖缺失，本次未跑（需 ≥30 条且 numpy 可用）。")

    # ── 生物统计样本量 ───────────────────────────────────
    ss = _read_processed_csv("sample_size_results.csv")
    if ss:
        lines += ["", "## 八、生物统计：样本量设计（呼应 CRDC 生物统计分析）", ""]
        for r in ss:
            lines.append(f"- {r['scenario']}：每组 **{r['n_per_arm']}** 例（总 {r['total_n']}）— {r.get('note', '')}")

    # ── NLP 管线关键词 ───────────────────────────────────
    nlp = _read_processed_csv("trial_nlp_keywords.csv")
    all_kw = [r["keyword"] for r in nlp if r.get("scope") == "ALL"]
    if all_kw:
        lines += ["", "## 九、NLP：试验管线关键词（研发情报）", ""]
        lines.append(f"- 全库 Top5：{'、'.join(all_kw[:5])}")

    lines += [
        "",
        "---",
        "*更多方法细节与数据来源见 README「数据来源与真实性标注」。*",
    ]
    out = config.REPORTS_DIR / "executive_brief.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    print("=" * 60)
    print("Lilly GLP-1 Commercial Analytics Pipeline")
    print("=" * 60)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ingest_trials()
    ingest_faers()
    build_warehouse()
    run_trial_ai()
    run_nlp()
    run_sample_size()
    run_segmentation(generate_patient_pool())
    ab = run_ab_test()
    run_forecasting()
    brief = build_executive_brief(ab)
    print(f"[run_pipeline] 高管简报 -> {brief}")
    print("=" * 60)
    print("完成。看板：streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
