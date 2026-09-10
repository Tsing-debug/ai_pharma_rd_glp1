"""SQLite 数仓构建 + 分析 SQL 执行（仅标准库）。

1) 读取 data/processed 下的 CSV（由 ingest 模块产出）
2) 按 sql/schema.sql 建表并灌数
3) 执行 sql/analysis_queries.sql，结果写入 data/processed/query_results/
"""
import csv
import logging
import sqlite3
import sys
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    db = config.DB_PATH
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # 1) 建表
    cur.executescript((config.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))

    # 索引优化（大幅提升查询性能）
    cur.executescript("""
        -- 事实表索引
        CREATE INDEX IF NOT EXISTS idx_fact_brand_sales_year ON fact_brand_sales(year);
        CREATE INDEX IF NOT EXISTS idx_fact_brand_sales_brand_year ON fact_brand_sales(brand_id, year);
        CREATE INDEX IF NOT EXISTS idx_fact_brand_sales_sim ON fact_brand_sales(is_simulated);
        CREATE INDEX IF NOT EXISTS idx_fact_trials_nct ON fact_trials(nct_id);
        CREATE INDEX IF NOT EXISTS idx_fact_trials_phase ON fact_trials(phase);
        CREATE INDEX IF NOT EXISTS idx_fact_faers_molecule ON fact_faers(molecule);
        CREATE INDEX IF NOT EXISTS idx_fact_faers_year ON fact_faers(year);
        CREATE INDEX IF NOT EXISTS idx_fact_faers_event ON fact_faers(event_type);
    """)

    # 2) 维度表
    years = sorted({int(r["year"]) for r in _load_csv(config.DATA_RAW / "financials_lilly_novo.csv")}
                   | {int(r["year"]) for r in _load_csv(config.DATA_RAW / "china_market_sim.csv")})
    cur.executemany("INSERT OR IGNORE INTO dim_time(year) VALUES (?)", [(y,) for y in years])
    cur.executemany("INSERT OR IGNORE INTO dim_region(region_id, region_name, country) VALUES (?,?,?)", [
        ("GLOBAL", "Global", "Multiple"),
        ("CN", "China", "China"),
    ])
    brands = {}
    for r in _load_csv(config.DATA_RAW / "financials_lilly_novo.csv"):
        brands.setdefault(r["brand_id"], r)
    cur.executemany(
        "INSERT OR IGNORE INTO dim_brand(brand_id, brand_name, company, molecule, category, is_simulated) "
        "VALUES (?,?,?,?,?,?)",
        [(b, v["brand_name"], v["company"], v["molecule"], v["category"], int(v["is_simulated"]))
         for b, v in brands.items()],
    )

    # 3) 事实表
    fin = _load_csv(config.DATA_RAW / "financials_lilly_novo.csv")
    cn = _load_csv(config.DATA_RAW / "china_market_sim.csv")
    sales_rows = []
    for r in fin:
        sales_rows.append((r["brand_id"], r["region_id"], int(r["year"]),
                           float(r["revenue_usd_m"]), int(r["scripts_units"] or 0), int(r["is_simulated"])))
    for r in cn:
        # 中国市场规模换算：bn -> 以 "CN-GLP1" 虚拟品牌入库，保持事实表结构统一
        sales_rows.append(("cn_glp1_market", "CN", int(r["year"]),
                           float(r["market_usd_bn"]) * 1000.0, 0, 1))
    cur.executemany(
        "INSERT OR IGNORE INTO fact_brand_sales(brand_id, region_id, year, revenue_usd_m, scripts_units, is_simulated) "
        "VALUES (?,?,?,?,?,?)", sales_rows)

    trials = _load_csv(config.DATA_PROCESSED / "trials.csv")
    cur.executemany(
        "INSERT OR IGNORE INTO fact_trials(nct_id, company, molecule, phase, indication, status, start_year, enrollment, title, design, endpoint, recruitment, is_simulated) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(r["nct_id"], r["company"], r["molecule"], r["phase"], r["indication"],
          r["status"], int(r["start_year"] or 0), int(r["enrollment"] or 0),
          r.get("title", ""), r.get("design", ""), r.get("endpoint", ""), r.get("recruitment", ""),
          int(r["is_simulated"]))
         for r in trials])

    faers = _load_csv(config.DATA_PROCESSED / "faers.csv")
    cur.executemany(
        "INSERT OR IGNORE INTO fact_faers(year, molecule, event_type, report_count, serious_count, is_simulated) "
        "VALUES (?,?,?,?,?,?)",
        [(int(r["year"]), r["molecule"], r["event_type"], int(r["report_count"]),
          int(r["serious_count"]), int(r["is_simulated"])) for r in faers])

    conn.commit()

    # 4) 执行分析 SQL
    out_dir = config.DATA_PROCESSED / "query_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = (config.SQL_DIR / "analysis_queries.sql").read_text(encoding="utf-8")
    for i, stmt in enumerate(_split_statements(script), start=1):
        try:
            rows = cur.execute(stmt).fetchall()
        except sqlite3.Error as exc:
            log.info(f"  [build_warehouse] 查询 Q{i} 失败: {exc}")
            continue
        if not rows:
            continue
        cols = [d[0] for d in cur.description]
        path = out_dir / f"query_{i:02d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        log.info(f"  [build_warehouse] Q{i} -> {path.name} ({len(rows)} 行)")

    conn.close()
    log.info(f"[build_warehouse] 数仓构建完成 -> {db}")


def _split_statements(script: str) -> list[str]:
    """按分号拆分 SQL（忽略 -- 注释，含行内注释；字符串字面量内的分号不算终止符）。

    末条语句缺分号时也保留（SQLite 可执行）。
    """
    stmts, buf = [], []
    in_string = False
    for line in script.splitlines():
        # 不在字符串内时剥掉行内 -- 注释
        stripped = line
        if not in_string:
            cut = stripped.find("--")
            if cut != -1:
                stripped = stripped[:cut]
        stripped = stripped.strip()
        if not stripped:
            continue

        # 扫描该行维护字符串状态（'' 转义会两次翻转，净效果正确）
        for ch in stripped:
            if ch == "'":
                in_string = not in_string

        buf.append(stripped)
        if not in_string and stripped.endswith(";"):
            stmts.append("\n".join(buf))
            buf = []
    if buf:  # 末条语句无分号
        stmts.append("\n".join(buf))
    return stmts


if __name__ == "__main__":
    config.setup_logging()
    build()
