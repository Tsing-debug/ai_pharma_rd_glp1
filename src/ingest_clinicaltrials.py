"""ClinicalTrials.gov 真实管线数据接入。

从 ClinicalTrials.gov API v2 拉取 GLP-1 相关试验并输出标准 CSV；
网络不可用时回退到仓库内样例数据（data/raw/clinical_trials_sample.csv）。
输出：data/processed/trials.csv（is_simulated=0）
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

MAX_PAGES = 5  # 分页上限（演示用，控制请求量）


def fetch_from_api(limit=1000, out_csv: Path | None = None,
                   max_pages: int = MAX_PAGES) -> Path | None:
    """分页拉取真实数据（API v2 用 pageToken 翻页），成功返回输出路径，失败或无结果返回 None。"""
    rows = []
    page_token = None
    page = 0
    for page in range(max_pages):
        params = {
            "query.term": QUERY_TERM,
            "fields": FIELDS,
            "pageSize": str(limit),
            "countTotal": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:  # 网络/超时/HTTP 错误 → 回退
            log.warning(f"[ingest_clinicaltrials] API 请求失败（{type(exc).__name__}: {exc}），回退到样例数据。")
            return None

        try:
            payload = json.loads(raw)
        except ValueError as exc:  # 响应非合法 JSON → 回退
            log.warning(f"[ingest_clinicaltrials] API 响应解析失败（{type(exc).__name__}: {exc}），回退到样例数据。")
            return None

        for s in payload.get("studies", []):
            rows.append(_extract_study(s))

        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    if not rows:  # 接口通了但无结果：按失败处理，回退样例
        log.warning("[ingest_clinicaltrials] API 返回 0 条结果，回退到样例数据。")
        return None
    log.info(f"[ingest_clinicaltrials] API 拉取 {len(rows)} 条（{page + 1} 页）")
    return _write_csv(rows, out_csv)


def _extract_study(s: dict) -> dict:
    """把一条 API study 记录转成统一 schema 的字典。"""
    ps = s.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    design = ps.get("designModule", {})
    cond = ps.get("conditionsModule", {})
    status = ps.get("statusModule", {})
    sponsor = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})

    phases = design.get("phases", [])

    # 提取设计信息（如果 API 返回）
    study_design = design.get("studyDesignInfo", [])
    design_str = ", ".join([sd.get("name", "") for sd in study_design]) if study_design else ""

    # 提取主要终点（如果 API 返回）
    outcomes = ps.get("outcomesModule", {})
    primary_endpoint = ""
    if outcomes.get("primaryOutcomeMeasures"):
        primary_endpoint = outcomes["primaryOutcomeMeasures"][0].get("title", "")

    # 提取入组信息（如果 API 返回）
    recruitment = design.get("recruitmentInfo", {}).get("conditionList", [])
    recruitment_str = ", ".join(recruitment) if recruitment else ""

    return {
        "nct_id": ident.get("nctId", ""),
        "company": sponsor.get("name", ""),
        "molecule": _detect_molecule(ident.get("briefTitle", "")),
        "phase": ",".join(phases),
        "indication": " | ".join(cond.get("conditions", [])),
        "status": status.get("overallStatus", ""),
        "start_year": int((status.get("startDateStruct", {}).get("date") or "0")[:4]) or 0,
        "enrollment": design.get("enrollmentInfo", {}).get("count", 0) or 0,
        "title": ident.get("briefTitle", ""),
        "design": design_str,
        "endpoint": primary_endpoint,
        "recruitment": recruitment_str,
        "is_simulated": 0,
    }


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
                  "status", "start_year", "enrollment", "title", "design", 
                  "endpoint", "recruitment", "is_simulated"]
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
    log.info(f"[ingest_clinicaltrials] 完成 -> {out}")


if __name__ == "__main__":
    config.setup_logging()
    main()
