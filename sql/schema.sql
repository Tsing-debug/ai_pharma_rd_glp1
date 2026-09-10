-- 星型模型 DDL：品牌销售事实表 + 维度表（SQLite）
-- 对应 README 中 "数据质量与可复现性" 的数仓层。

-- 时间维度
CREATE TABLE IF NOT EXISTS dim_time (
    year INTEGER PRIMARY KEY,
    decade TEXT
);

-- 区域维度
CREATE TABLE IF NOT EXISTS dim_region (
    region_id TEXT PRIMARY KEY,
    region_name TEXT,
    country TEXT
);

-- 品牌维度
CREATE TABLE IF NOT EXISTS dim_brand (
    brand_id TEXT PRIMARY KEY,
    brand_name TEXT,
    company TEXT,
    molecule TEXT,
    category TEXT,          -- T2D / obesity / combo
    is_simulated INTEGER DEFAULT 0
);

-- 品牌销售事实表（单位：百万美元 / 处方件数）
CREATE TABLE IF NOT EXISTS fact_brand_sales (
    brand_id TEXT,
    region_id TEXT,
    year INTEGER,
    revenue_usd_m REAL,
    scripts_units INTEGER,
    is_simulated INTEGER DEFAULT 0,
    PRIMARY KEY (brand_id, region_id, year)
);

-- 临床试验事实表（真实数据：ClinicalTrials.gov）
CREATE TABLE IF NOT EXISTS fact_trials (
    nct_id TEXT PRIMARY KEY,
    company TEXT,
    molecule TEXT,
    phase TEXT,
    indication TEXT,
    status TEXT,
    start_year INTEGER,
    enrollment INTEGER,
    title TEXT,
    design TEXT,
    endpoint TEXT,
    recruitment TEXT,
    is_simulated INTEGER DEFAULT 0
);

-- 不良事件事实表（真实数据：FDA FAERS）
CREATE TABLE IF NOT EXISTS fact_faers (
    year INTEGER,
    molecule TEXT,
    event_type TEXT,
    report_count INTEGER,
    serious_count INTEGER,
    is_simulated INTEGER DEFAULT 0
);
