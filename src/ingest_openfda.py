"""FDA OpenFDA 不良事件（FAERS）真实数据接入。

从 OpenFDA /drug/event 按活性成分聚合不良事件计数；
多年度（2022-2024）+ 分页拉取，降低单年/单页抽样偏差；
网络不可用时回退到仓库内样例数据。
输出：data/processed/faers.csv（is_simulated=0）
"""
import csv
import json
import logging
import sys
import urllib.request
import urllib.parse
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

API_URL = "https://api.fda.gov/drug/event.json"
SUBSTANCES = ["TIRZEPATIDE", "SEMAGLUTIDE"]
EVENT_GROUPS = {
    "Gastrointestinal": ["gastrointestinal", "abdominal pain", "diarrhea", "constipation"],
    "Nausea & Vomiting": ["nausea", "vomiting"],
    "Weight Decrease": ["weight decreased"],
    "Injection Site": ["injection site"],
}
YEARS = [2022, 2023, 2024]  # 多年度覆盖
MAX_REPORTS = 1000  # OpenFDA 单请求 limit 上限
MAX_PAGES = 3       # 每物质每年最多翻页数（演示用，控制请求量）


def _group_event(reaction: str) -> str:
    r = reaction.lower()
    for group, keywords in EVENT_GROUPS.items():
        if any(k in r for k in keywords):
            return group
    return "Other"


def fetch_from_api(years: list[int] | None = None, out_csv: Path | None = None) -> Path | None:
    """多年度 + 分页拉取 FAERS 并聚合；任何请求/解析失败即回退（保持全量诚实）。"""
    years = years or YEARS
    out_csv = out_csv or config.DATA_PROCESSED / "faers.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in years:
        for substance in SUBSTANCES:
            counts: dict[str, list[int]] = {}
            skip = 0
            while skip < MAX_REPORTS * MAX_PAGES:
                params = {
                    "search": f'patient.drug.openfda.substance_name:"{substance}"'
                              f'+AND+receivedate:[{year}0101+TO+{year}1231]',
                    "limit": str(MAX_REPORTS),
                    "skip": str(skip),
                }
                url = f"{API_URL}?{urllib.parse.urlencode(params)}"
                try:
                    with urllib.request.urlopen(url, timeout=20) as resp:
                        raw = resp.read().decode("utf-8")
                except Exception as exc:  # 网络/超时/HTTP 错误 → 回退
                    log.warning(f"[ingest_openfda] API 请求失败（{type(exc).__name__}: {exc}），回退到样例数据。")
                    return None

                try:
                    payload = json.loads(raw)
                except ValueError as exc:  # 响应非合法 JSON → 回退
                    log.warning(f"[ingest_openfda] API 响应解析失败（{type(exc).__name__}: {exc}），回退到样例数据。")
                    return None

                results = payload.get("results", [])
                for r in results:
                    for reac in r.get("patient", {}).get("reaction", []):
                        g = _group_event(reac.get("reactionmeddrapt", ""))
                        counts.setdefault(g, [0, 0])
                        counts[g][0] += 1
                        if r.get("serious"):
                            counts[g][1] += 1
                if len(results) < MAX_REPORTS:  # 本页未满 → 无更多数据
                    break
                skip += MAX_REPORTS

            for g, (total, serious) in counts.items():
                rows.append({
                    "year": year, "molecule": substance.title(),
                    "event_type": g, "report_count": total,
                    "serious_count": serious, "is_simulated": 0,
                })

    if not rows:
        return None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "molecule", "event_type",
                                               "report_count", "serious_count",
                                               "is_simulated"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[ingest_openfda] 完成：{len(rows)} 条聚合计数（{len(years)} 年 × {len(SUBSTANCES)} 分子）")
    return out_csv


def load_fallback() -> Path:
    src = config.DATA_RAW / "faers_sample.csv"
    dst = config.DATA_PROCESSED / "faers.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = fetch_from_api() or load_fallback()
    log.info(f"[ingest_openfda] 完成 -> {out}")


if __name__ == "__main__":
    config.setup_logging()
    main()
