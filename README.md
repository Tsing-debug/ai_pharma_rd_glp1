# GLP-1 × AI 医药研发分析

> 以 GLP-1 代谢疾病赛道（替尔泊肽 / 司美格鲁肽）为真实场景，完整演示 AI 落地项目从 0 到 1 的全流程：数据接入 → 特征工程 → 建模评估 → 统计设计 → 看板与报告。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## ⚠️ 数据诚实性原则

本仓库所有数据**要么来自真实公开来源**（ClinicalTrials.gov / FDA OpenFDA / 公司年报），**要么是明确标注的“校准仿真”**（文件名含 `_sim`，字段 `is_simulated=1`）。

---

## 📌 项目亮点

- **AI 落地真实研发场景**：临床试验结果预测（LR / RF / 手写 3 层 MLP，numpy 反向传播，5 折 CV AUC）
- **全链路可复现**：一条命令串联数据接入 → 数仓建模 → AI 建模 → 统计设计 → 自动报告
- **零依赖核心流水线**：仅需 Python 标准库 + numpy 即可运行，面试现场可直接跑通
- **数据真实性内建**：真实数据与仿真数据严格区分，标注规范写入代码和文档
- **多职能覆盖**：AI 建模 + 生物统计 + NLP 文本挖掘 + 商业分析看板

---

## 🚀 快速开始

```bash
cd ai_pharma_rd_glp1

# 核心流水线（零依赖，仅需 Python 标准库 + numpy）
python src/run_pipeline.py

# 可选依赖（强烈建议安装）
pip install -r requirements.txt

# 交互看板
streamlit run dashboard/app.py

# 单元测试（统计公式对照 scipy 验证、数据泄露回归）
python -m unittest discover -s tests -v
```

### 流水线输出

| 输出文件 | 说明 |
|---|---|
| `data/processed/analytics.db` | SQLite 数仓（星型模型） |
| `data/processed/query_results/*.csv` | SQL 分析查询结果（份额、增速、launch tracker、安全信号） |
| `data/processed/trial_model_results.csv` | AI 临床试验结果预测（3 模型 × 5 折 CV AUC + 特征重要性） |
| `data/processed/trial_nlp_keywords.csv` | 试验管线文本挖掘关键词 |
| `data/processed/sample_size_results.csv` | 生物统计样本量计算 |
| `data/processed/patient_segments.csv` | 患者分层结果 |
| `data/processed/forecast_*.csv` | 品牌销量预测（含 MAPE 回测） |
| `reports/executive_brief.md` | 自动生成的高管简报 |

---

## 🏗️ 项目结构

```
ai_pharma_rd_glp1/
├── sql/
│   ├── schema.sql                  # 星型模型 DDL（事实表 + 维度表）
│   └── analysis_queries.sql        # 商业分析查询（份额/增速/launch/安全信号）
├── data/
│   ├── raw/
│   │   ├── financials_lilly_novo.csv      # 真实（约数）：礼来/诺和品牌年收入
│   │   ├── china_market_sim.csv           # 仿真（标注）：中国 GLP-1 市场规模测算
│   │   ├── clinical_trials_sample.csv     # 真实：代表性临床试验（含 SURMOUNT/SURPASS 系列）
│   │   └── faers_sample.csv               # 真实（示例）：FAERS 不良事件信号
│   └── processed/                  # 流水线产物
├── src/
│   ├── ingest_clinicaltrials.py    # ClinicalTrials.gov API 接入（失败自动回退样例）
│   ├── ingest_openfda.py           # FDA OpenFDA FAERS 接入（失败自动回退样例）
│   ├── build_warehouse.py          # 建 SQLite 数仓 + 执行分析 SQL
│   ├── trial_ai_models.py          # 〖AI 核心〗LR/RF + 手写 3 层 MLP + CV AUC
│   ├── sample_size.py              # 〖生物统计〗样本量/功效计算
│   ├── nlp_analysis.py             # 〖NLP〗试验管线文本挖掘
│   ├── patient_segmentation.py     # 患者分层（K-Means，缺依赖时纯 Python 回退）
│   ├── hypothesis_testing.py       # A/B 实验：Welch t 检验 + bootstrap CI + 功效分析
│   ├── forecasting.py              # 品牌销量预测：线性趋势 + CAGR 阻尼集成 + MAPE 回测
│   └── run_pipeline.py             # 主入口：数据接入 → 数仓 → AI 建模 → 统计 → 报告
├── dashboard/app.py                # Streamlit 看板（品牌/市场/患者/安全 4 页）
├── docs/
│   ├── resume_project_description.md   # 简历项目描述（bullet + STAR）
│   └── interview_qa.md                 # 面试追问 Q&A
├── config.py
└── requirements.txt
```

---

## 📊 数据来源与真实性标注

| 数据 | 真实性 | 来源 | 刷新方式 |
|---|---|---|---|
| 品牌年收入（Mounjaro/Zepbound/Trulicity/Ozempic/Wegovy） | 真实（约数，已标注） | 公司年报 / 公开报道；**正式使用前请用最新 10-K/年报核对** | `src/ingest_financials.py`（模板） |
| 临床试验管线 | 真实 | ClinicalTrials.gov API v2 | `python src/ingest_clinicaltrials.py` |
| 不良事件信号 | 真实 | FDA OpenFDA `drug/event` | `python src/ingest_openfda.py` |
| 中国 GLP-1 市场测算 | 仿真（标注） | 基于公开报告校准（CDE 获批、医保谈判、流行病学） | 接入 CDE/医保局公开数据后可替换 |
| 患者画像池 | 仿真（标注） | 基于公开患病率/消费者调查参数校准 | — |

---

## 🎯 与辉瑞 AI Pilot（深度学习工程师方向）能力映射

| JD 要求 | 本项目的对应实现 |
|---|---|
| AI 解决医药研发真实问题 | 临床试验结果预测（LR/RF/手写 MLP，5 折 CV AUC）——研发场景直接落地 |
| 深度学习 / 神经网络 | `trial_ai_models.py`：numpy 手写 3 层 MLP 反向传播（不依赖框架，可现场推导） |
| AI 落地项目 0→1 | `run_pipeline.py` 一键串联：数据接入 → 特征工程 → 建模评估 → 报告 |
| 生物统计分析（CRDC 职能） | `sample_size.py`：样本量/功效计算；`hypothesis_testing.py`：t 检验 + bootstrap |
| 安全数据处理与评估（CRDC 职能） | OpenFDA FAERS 接入 + 安全信号 SQL + 严重率指标 |
| I–IV 期临床试验数据 | ClinicalTrials.gov API v2 真实管线（1000 条量级）+ 数仓建模 |
| 数据科学 / 生物信息背景适配 | NLP 文本挖掘 + 统计 + ML 全栈覆盖 |
| 跨国企业工程化意识 | 数据真实性标注、零依赖可复现、CI 友好（`.github/workflows` 可加） |

---

## 🗺️ Roadmap

- [ ] 用最新 ClinicalTrials.gov 数据刷新全量管线（`python src/ingest_clinicaltrials.py`）
- [ ] 扩展 AI 模块：加特征（试验时长、双盲设计、多中心数）与基线模型对比
- [ ] 中国本地化（CDE 获批清单、医保谈判目录）作为加分演示
- [ ] GCP / 数据管理意识写入文档（辉瑞 CRDC 关键词）
- [ ] Power BI 版看板（商业岗位备用）

---
