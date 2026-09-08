"""ClinicalTrials.gov 真实管线数据接入。

从 ClinicalTrials.gov API v2 拉取 GLP-1 相关试验并输出标准 CSV；
网络不可用时回退到仓库内样例数据（data/raw/clinical_trials_sample.csv）。
输出：data/processed/trials.csv（is_simulated=0）
"""
import csv
import json
import sys
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

API_URL = "https://clinicaltrials.gov/api/v2/studies"
# API v2 查询语法：AREA[字段]过滤 + AND/OR 组合
QUERY_TERM = ('AREA[Condition]("Obesity" OR "Diabetes Mellitus, Type 2") '
              'AND (AREA[InterventionName]("tirzepatide" OR "semaglutide" OR "orforglipron" OR "GLP-1"))')
FIELDS = "protocolSection.identificationModule.nctId," \
         "protocolSection.identificationModule.organization.fullName," \
         "protocolSection.identificationModule.briefTitle," \
         "protocolSection.designModule.phases," \
         "protocolSection.conditionsModule.conditions," \
         "protocolSection.statusModule.overallStatus," \
         "protocolSection.designModule.enrollmentInfo.count," \
         "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"


def fetch_from_api(limit=1000, out_csv: Path | None = None) -> Path | None:
    """拉取真实数据，成功返回输出路径，失败或无结果返回 None。"""
    params = {
        "query.term": QUERY_TERM,
        "fields": FIELDS,
        "pageSize": str(limit),
        "countTotal": "true",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # 网络/解析失败均回退
        print(f"[ingest_clinicaltrials] API 不可用（{exc}），回退到样例数据。")
        return None

    rows = []
    for s in payload.get("studies", []):
        ps = s.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        design = ps.get("designModule", {})
        cond = ps.get("conditionsModule", {})
        status = ps.get("statusModule", {})
        sponsor = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})

        phases = design.get("phases", [])
        rows.append({
            "nct_id": ident.get("nctId", ""),
            "company": sponsor.get("name", ""),
            "molecule": _detect_molecule(ident.get("briefTitle", "")),
            "phase": ",".join(phases),
            "indication": " | ".join(cond.get("conditions", [])),
            "status": status.get("overallStatus", ""),
            "start_year": int((status.get("startDateStruct", {}).get("date") or "0")[:4]) or 0,
            "enrollment": design.get("enrollmentInfo", {}).get("count", 0) or 0,
            "is_simulated": 0,
        })
    if not rows:  # 接口通了但无结果：按失败处理，回退样例
        print("[ingest_clinicaltrials] API 返回 0 条结果，回退到样例数据。")
        return None
    return _write_csv(rows, out_csv)


def _detect_molecule(title: str) -> str:
    t = title.lower()
    for m in ("tirzepatide", "semaglutide", "orforglipron", "dulaglutide"):
        if m in t:
            return m
    return "other-glp1"


def _write_csv(rows, out_csv: Path | None) -> Path:
    out_csv = out_csv or config.DATA_PROCESSED / "trials.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["nct_id", "company", "molecule", "phase", "indication",
                  "status", "start_year", "enrollment", "is_simulated"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def load_fallback() -> Path:
    """回退：直接采用仓库内样例（真实公开试验信息）。"""
    src = config.DATA_RAW / "clinical_trials_sample.csv"
    dst = config.DATA_PROCESSED / "trials.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = fetch_from_api() or load_fallback()
    print(f"[ingest_clinicaltrials] 完成 -> {out}")


if __name__ == "__main__":
    main()
