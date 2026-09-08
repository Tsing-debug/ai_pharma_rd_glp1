# GLP-1 医药研发分析平台

> 一个面向 GLP-1 代谢疾病领域的端到端数据科学项目，覆盖 **AI 临床试验预测**、**真实世界安全监测**、**生物统计设计**与**商业智能看板**。

---

## 📌 项目定位

GLP-1 受体激动剂（如司美格鲁肽、替尔泊肽）是当前代谢疾病领域最重要的药物赛道之一。本项目以此为场景，构建了一套完整的数据分析流水线，将**公开临床试验数据**、**FDA 不良事件报告（FAERS）** 和**上市公司财务数据**进行整合，用于辅助研发决策与商业分析。

简单来说，这套流水线回答了以下几个问题：

- 某项临床试验的成功概率有多高？（AI 预测）
- 当前 GLP-1 领域的研发管线在关注哪些适应症和靶点？（NLP 文本挖掘）
- 某一药物在真实世界中的安全性信号是否值得警惕？（FAERS 信号检测）
- 各品牌的销售趋势如何？下一个增长点在哪里？（销量预测与市场追踪）
- 针对特定终点指标，需要多大的样本量才能检测出显著差异？（统计功效计算）

---

## 🧩 核心功能模块

| 模块 | 技术实现 | 产出 |
| :--- | :--- | :--- |
| **AI 临床试验预测** | 逻辑回归 / 随机森林 / 手写 3 层 MLP（基于 NumPy，不依赖深度学习框架），5 折交叉验证 | 各模型 AUC 对比 + 特征重要性排序 |
| **生物统计工具包** | 样本量 / 功效计算、Welch t 检验、Bootstrap 置信区间 | 统计设计参数与假设检验结果 |
| **NLP 文本挖掘** | 试验标题与适应症字段的关键词提取与频次分析 | 热门适应症 / 研究设计类型分布 |
| **数据仓库建设** | SQLite 星型模型（事实表 + 维度表） + 预置分析 SQL | 市场份额、增速、Launcher Tracker 等衍生指标 |
| **商业销量预测** | 线性趋势 + CAGR 阻尼集成，MAPE 回测验证 | 未来 4 个季度销量预测曲线 |
| **交互式可视化** | Streamlit 构建的多页看板（品牌 / 市场 / 患者 / 安全） | 可交互的图表与筛选器 |

---

## ⚠️ 数据来源与真实性说明

本项目秉持数据透明原则，所有数据来源均在文件中明确标注：

- **真实数据**（来自官方公开渠道）：
  - 品牌年收入：礼来 / 诺和诺德年报及公开财务报道
  - 临床试验管线：ClinicalTrials.gov API v2
  - 不良事件信号：FDA OpenFDA `drug/event` 接口

- **标注仿真数据**（用于填补方法演示缺口）：
  - 中国 GLP-1 市场测算规模
  - 患者画像池与人口统计学分布

> 所有仿真数据文件命名均包含 `_sim` 后缀，且数据表中设有 `is_simulated=1` 标识字段。任何分析结果均可追溯到数据来源属性。

---

## 🚀 快速上手

```bash
# 1. 克隆仓库
git clone https://github.com/Tsing-debug/ai_pharma_rd_glp1.git
cd ai_pharma_rd_glp1

# 2. 运行完整流水线（核心模块仅依赖 Python 标准库 + NumPy）
python src/run_pipeline.py

# 3. 安装可视化依赖（可选，用于启动看板）
pip install -r requirements.txt

# 4. 启动 Streamlit 交互看板
streamlit run dashboard/app.py
```

### 一键运行后产生的输出物

| 输出路径 | 内容说明 |
| :--- | :--- |
| `data/processed/analytics.db` | 星型模型 SQLite 数仓 |
| `data/processed/query_results/*.csv` | 市场增速、份额、安全信号等预置查询结果 |
| `data/processed/trial_model_results.csv` | 三个 AI 模型的 AUC 对比与特征重要性 |
| `data/processed/trial_nlp_keywords.csv` | 临床试验文本关键词统计 |
| `data/processed/sample_size_results.csv` | 不同参数下的样本量计算结果 |
| `data/processed/forecast_*.csv` | 各品牌销量预测数据（含 MAPE 回测指标） |
| `reports/executive_brief.md` | 自动生成的简要分析报告 |

---

## 📁 项目结构

```
ai_pharma_rd_glp1/
├── sql/
│   ├── schema.sql                 # 星型模型 DDL
│   └── analysis_queries.sql       # 商业分析预置查询
├── data/
│   ├── raw/                       # 原始数据（真实 + 仿真标注）
│   └── processed/                 # 流水线输出产物
├── src/
│   ├── ingest_clinicaltrials.py   # ClinicalTrials.gov 数据接入
│   ├── ingest_openfda.py          # FDA FAERS 数据接入
│   ├── build_warehouse.py         # 数仓构建与 SQL 执行
│   ├── trial_ai_models.py         # 临床试验结果预测模型
│   ├── sample_size.py             # 样本量 / 功效计算
│   ├── nlp_analysis.py            # 试验管线文本挖掘
│   ├── patient_segmentation.py    # 患者分层聚类
│   ├── hypothesis_testing.py      # A/B 检验（t 检验 + Bootstrap）
│   ├── forecasting.py             # 销量预测与回测
│   └── run_pipeline.py            # 主流水线入口
├── dashboard/
│   └── app.py                     # Streamlit 看板
├── config.py
└── requirements.txt
```

---

## 🔬 技术细节与设计考量

### AI 模型设计
临床试验结果预测本质上是一个二分类问题（试验是否达到主要终点）。除调用 sklearn 基线模型外，项目从零实现了 3 层 MLP 的反向传播算法（使用 NumPy），目的不在于超越 SOTA，而在于展示对神经网络底层计算图的理解。

### 数仓建模
采用星型模型设计，围绕临床试验事实表构建时间、药物、适应症、机构等维度表。这种设计使得下游的“市场份额变化”、“同靶点试验数量趋势”等分析查询可以用标准的 SQL 聚合高效完成。

### 统计方法的实用性
样本量计算模块支持单样本 / 两样本 / 配对设计的均值和率值检验，可直接用于研究方案撰写阶段的样本量论证。

---

## 📋 后续迭代方向

- [ ] 接入最新 ClinicalTrials.gov 数据刷新全量管线
- [ ] 扩展 AI 特征工程（加入试验时长、盲法设计、多中心数量等结构性特征）
- [ ] 补充中国本地化数据源（CDE 获批清单、医保谈判目录）
- [ ] 增加模型可解释性分析（SHAP / LIME）
- [ ] 将看板部署为在线 Demo

---

## 📄 免责声明

**本项目为个人研究与学习用途，非任何企业官方产出。** 所有真实数据均来自公开 API 与年报，使用请遵守各数据源的相关许可协议。仿真数据仅用于方法学演示，不代表真实市场情况。

---

## 📬 交流与反馈

欢迎通过 GitHub Issues 提出建议或疑问。

---

**License**: MIT © Tsing-debug
