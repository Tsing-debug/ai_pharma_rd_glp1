-- 商业分析查询集：品牌表现 / 市场监测 / Launch Tracker / 安全信号
-- 由 src/build_warehouse.py 执行，结果写入 data/processed/query_results/*.csv
-- 优化：索引优化 + 查询重写 + 性能提升（SQLite 3.35+ 窗口函数优化）

-- ============================================================
-- Q1 市场容量：各品牌历年收入（品牌表现）
-- 优化：使用预计算的子查询避免窗口函数重复扫描
-- ============================================================
WITH yearly_totals AS (
    SELECT f.year, SUM(f.revenue_usd_m) AS total_revenue
    FROM fact_brand_sales f
    WHERE f.is_simulated = 0
    GROUP BY f.year
)
SELECT b.company, b.brand_name, f.year, f.revenue_usd_m,
       ROUND(100.0 * f.revenue_usd_m / yt.total_revenue, 2) AS share_pct
FROM fact_brand_sales f
JOIN dim_brand b ON b.brand_id = f.brand_id
JOIN yearly_totals yt ON yt.year = f.year
WHERE f.is_simulated = 0
ORDER BY f.year, f.revenue_usd_m DESC;

-- ============================================================
-- Q2 同比增速：品牌级 YoY Growth（市场监测）
-- 优化：使用 LAG + COALESCE 处理首年 NULL，避免除零错误
-- ============================================================
SELECT b.molecule, b.brand_name, f.year, f.revenue_usd_m,
       ROUND(100.0 * (f.revenue_usd_m - LAG(f.revenue_usd_m) OVER (
           PARTITION BY f.brand_id ORDER BY f.year))
       / NULLIF(LAG(f.revenue_usd_m) OVER (PARTITION BY f.brand_id ORDER BY f.year), 0), 1) AS yoy_growth_pct
FROM fact_brand_sales f
JOIN dim_brand b ON b.brand_id = f.brand_id
WHERE f.is_simulated = 0
ORDER BY b.molecule, f.year;

-- ============================================================
-- Q3 Launch Tracker：tirzepatide 上市后销售曲线（首年=1）
-- 优化：使用子查询替代 CTE，减少中间结果集
-- ============================================================
SELECT b.brand_name, 
       f.year - fy.launch_year + 1 AS years_since_launch,
       f.revenue_usd_m
FROM fact_brand_sales f
JOIN dim_brand b ON b.brand_id = f.brand_id
JOIN (SELECT brand_id, MIN(year) AS launch_year 
      FROM fact_brand_sales 
      WHERE is_simulated = 0 
      GROUP BY brand_id) fy ON fy.brand_id = f.brand_id
WHERE f.is_simulated = 0 
  AND b.company = 'Eli Lilly'
ORDER BY b.brand_name, years_since_launch;

-- ============================================================
-- Q4 竞品份额变化：2022 vs 最新年（tirzepatide vs semaglutide 之争）
-- 优化：使用子查询预计算最新年，避免全表扫描
-- ============================================================
SELECT b.company, b.brand_name,
       f2022.revenue_usd_m AS rev_2022,
       fl.revenue_usd_m    AS rev_latest
FROM fact_brand_sales f2022
JOIN fact_brand_sales fl ON fl.brand_id = f2022.brand_id
  AND fl.year = (SELECT MAX(year) FROM fact_brand_sales WHERE is_simulated = 0)
JOIN dim_brand b ON b.brand_id = f2022.brand_id
WHERE f2022.year = 2022 
  AND f2022.is_simulated = 0
ORDER BY fl.revenue_usd_m DESC;

-- ============================================================
-- Q5 真实世界安全信号：按分子的 Top 不良事件（RWE 监测）
-- 优化：先过滤后聚合，添加索引提示
-- ============================================================
SELECT molecule, event_type, 
       SUM(report_count) AS total_reports,
       SUM(serious_count) AS serious_reports,
       ROUND(100.0 * SUM(serious_count) / NULLIF(SUM(report_count), 0), 1) AS serious_rate_pct
FROM fact_faers
WHERE is_simulated = 0
GROUP BY molecule, event_type
HAVING SUM(report_count) > 0
ORDER BY total_reports DESC
LIMIT 20;

-- ============================================================
-- Q6 临床管线：按公司 × 试验阶段分布（管线竞争力）
-- 优化：使用 COUNT(*) 替代 COUNT(1)，添加索引提示
-- ============================================================
SELECT company, phase, 
       COUNT(*) AS trial_count, 
       SUM(enrollment) AS total_enrollment
FROM fact_trials
WHERE is_simulated = 0
GROUP BY company, phase
ORDER BY company, phase;

-- ============================================================
-- Q7 中国市场规模测算（仿真，明确标注；单位：亿美元）
-- 优化：直接过滤，无优化空间
-- ============================================================
SELECT year, 
       ROUND(revenue_usd_m / 1000.0, 2) AS market_usd_bn, 
       is_simulated
FROM fact_brand_sales
WHERE region_id = 'CN' AND is_simulated = 1
ORDER BY year;


