"""FDA OpenFDA 不良事件（FAERS）真实数据接入。

从 OpenFDA /drug/event 按活性成分聚合不良事件计数；
网络不可用时回退到仓库内样例数据。
输出：data/processed/faers.csv（is_simulated=0）
"""
import csv
import json
import sys
import urllib.request
import urllib.parse
from pathlib import Path

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
MAX_REPORTS = 1000  # 每次拉取条数上限（演示用）


def _group_event(reaction: str) -> str:
    r = reaction.lower()
    for group, keywords in EVENT_GROUPS.items():
        if any(k in r for k in keywords):
            return group
    return "Other"


def fetch_from_api(year=2024, out_csv: Path | None = None) -> Path | None:
    """拉取某年 FAERS 数据并聚合，失败返回 None。"""
    out_csv = out_csv or config.DATA_PROCESSED / "faers.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for substance in SUBSTANCES:
        params = {
            "search": f'patient.drug.openfda.substance_name:"{substance}"'
                      f'+AND+receivedate:[{year}0101+TO+{year}1231]',
            "limit": str(MAX_REPORTS),
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[ingest_openfda] API 不可用（{exc}），回退到样例数据。")
            return None

        counts: dict[str, list[int]] = {}
        for r in payload.get("results", []):
            for reac in r.get("patient", {}).get("reaction", []):
                g = _group_event(reac.get("reactionmeddrapt", ""))
                counts.setdefault(g, [0, 0])
                counts[g][0] += 1
                if r.get("serious"):
                    counts[g][1] += 1
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
    print(f"[ingest_openfda] 完成 -> {out}")


if __name__ == "__main__":
    main()
