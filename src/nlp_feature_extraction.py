"""Level 3 NLP 特征工程：试验标题、Protocol 复杂度、招募语言分析。

仅使用标准库实现 TF-IDF、复杂度指标、语言特征提取。
输出特征向量用于模型集成。
"""
import csv
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ============================================================
# Task 1: TF-IDF 向量化试验标题
# ============================================================

class SimpleTFIDF:
    """极简 TF-IDF 实现（无 sklearn 依赖）"""
    
    def __init__(self, max_features=50, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}  # term -> idx
        self.idf = {}    # term -> idf
        self.n_docs = 0
    
    def _tokenize(self, text: str) -> list[str]:
        """简单分词：降小写 + 移除标点 + 去停用词"""
        stopwords = {
            "a", "an", "and", "the", "of", "in", "to", "for", "with",
            "is", "are", "be", "was", "were", "study", "trial", "phase",
            "versus", "vs", "randomized", "double", "blind", "placebo",
            "controlled", "patients", "subjects", "subjects", "on"
        }
        text = re.sub(r"[^a-z0-9 ]", "", text.lower())
        return [w for w in text.split() if w and w not in stopwords and len(w) > 2]
    
    def fit(self, docs: list[str]):
        """计算 IDF"""
        doc_freqs = Counter()
        all_tokens = []
        
        for doc in docs:
            tokens = set(self._tokenize(doc))
            doc_freqs.update(tokens)
            all_tokens.extend(tokens)
        
        self.n_docs = len(docs)
        # 保留出现 >= min_df 次的词
        freq_terms = [t for t, c in doc_freqs.most_common() if c >= self.min_df][:self.max_features]
        
        self.vocab = {t: i for i, t in enumerate(freq_terms)}
        for term in freq_terms:
            df = doc_freqs[term]
            self.idf[term] = math.log(self.n_docs / (1 + df))
    
    def transform(self, doc: str) -> np.ndarray | list:
        """返回 TF-IDF 向量"""
        tokens = self._tokenize(doc)
        token_counts = Counter(tokens)
        
        vec = np.zeros(len(self.vocab)) if _HAS_NUMPY else [0.0] * len(self.vocab)
        
        for term, idx in self.vocab.items():
            if term in token_counts:
                tf = token_counts[term] / len(tokens) if tokens else 0
                idf = self.idf[term]
                vec[idx] = tf * idf
        
        return vec


# ============================================================
# Task 2: Protocol 复杂度指标
# ============================================================

def calculate_protocol_complexity(title: str, design: str, endpoint: str) -> dict:
    """计算 Protocol 复杂度指标"""
    
    # 文本长度特征
    title_len = len(title or "")
    design_len = len(design or "")
    endpoint_len = len(endpoint or "")
    total_text_len = title_len + design_len + endpoint_len
    
    # 关键词计数特征
    combined_text = (title or "") + " " + (design or "") + " " + (endpoint or "")
    
    # 复杂度指示词
    complexity_keywords = {
        "stratified": 2,
        "randomized": 1,
        "crossover": 2,
        "adaptive": 3,
        "sequential": 2,
        "factorial": 3,
        "open-label": 0,  # 简单
        "unblinded": 0,   # 简单
        "phase out": -1,  # 结束阶段
    }
    
    complexity_score = 0
    keyword_count = 0
    text_lower = combined_text.lower()
    
    for keyword, score in complexity_keywords.items():
        if keyword in text_lower:
            complexity_score += score
            keyword_count += 1
    
    # 句子复杂度（估算：用句号、冒号、分号数量）
    sentence_count = len(re.split(r'[.!?;]', combined_text)) - 1
    sentence_count = max(1, sentence_count)
    avg_sentence_len = total_text_len / sentence_count
    
    # 医学术语密度
    medical_terms = {
        "cardiovascular", "hypertension", "diabetes", "obesity",
        "efficacy", "safety", "adverse", "event", "biomarker",
        "dosage", "pharmacokinetic", "pharmacodynamic"
    }
    medical_term_count = sum(1 for term in medical_terms if term in text_lower)
    
    return {
        "title_length": title_len,
        "design_length": design_len,
        "endpoint_length": endpoint_len,
        "total_text_length": total_text_len,
        "complexity_score": complexity_score,
        "complexity_keyword_count": keyword_count,
        "avg_sentence_length": avg_sentence_len,
        "medical_term_density": medical_term_count / max(1, len(combined_text) / 100),
    }


# ============================================================
# Task 3: 招募语言分析
# ============================================================

def analyze_recruitment_language(recruitment: str) -> dict:
    """分析招募文本的紧急程度、地理范围、入选难度"""
    
    if not recruitment:
        return {
            "urgent_language": 0,
            "global_scope": 0,
            "selective_criteria": 0,
            "recruitment_simplicity": 0,
        }
    
    text_lower = recruitment.lower()
    
    # 紧急指示词
    urgent_keywords = {"urgent", "emergency", "immediately", "asap", "critical", "now"}
    urgent_language = 1 if any(k in text_lower for k in urgent_keywords) else 0
    
    # 全球范围指示词
    global_keywords = {"global", "international", "worldwide", "multiple countries", "us", "europe", "asia"}
    global_scope = 1 if any(k in text_lower for k in global_keywords) else 0
    
    # 选择性标准指示词（越多说明入选难度越高）
    selective_keywords = {
        "severe": 2, "refractory": 2, "uncontrolled": 1,
        "comorbid": 1, "stable": 1, "no prior": 1
    }
    selective_criteria = sum(score for k, score in selective_keywords.items() if k in text_lower)
    
    # 招募便利度（相反指标）：简单的招募条件
    simple_keywords = {"all", "any", "regardless", "no restrictions"}
    recruitment_simplicity = 1 if any(k in text_lower for k in simple_keywords) else 0
    
    return {
        "urgent_language": urgent_language,
        "global_scope": global_scope,
        "selective_criteria": min(selective_criteria, 5),  # 上限5
        "recruitment_simplicity": recruitment_simplicity,
    }


# ============================================================
# Task 4: 集成所有 Level 3 特征
# ============================================================

def extract_level3_features(trials: list[dict]) -> tuple[list[str], list]:
    """提取 Level 3 NLP 特征"""
    
    # Step 1: 训练 TF-IDF（仅用标题）
    titles = [r.get("title") or "" for r in trials]
    tfidf = SimpleTFIDF(max_features=10, min_df=2)
    tfidf.fit(titles)
    
    log.info(f"[nlp_l3] TF-IDF 词汇表: {list(tfidf.vocab.keys())[:5]}...")
    
    # Step 2: 提取所有特征
    feature_names = []
    feature_vectors = []
    
    for r in trials:
        features = {}
        
        # TF-IDF 特征
        title_vec = tfidf.transform(r.get("title") or "")
        title_vec_list = title_vec.tolist() if _HAS_NUMPY else title_vec
        for i, val in enumerate(title_vec_list):
            features[f"tfidf_{i}"] = val
        
        # Protocol 复杂度特征
        complexity = calculate_protocol_complexity(
            r.get("title", ""), r.get("design", ""), r.get("endpoint", "")
        )
        features.update(complexity)
        
        # 招募语言特征
        recruitment_lang = analyze_recruitment_language(r.get("recruitment", ""))
        features.update(recruitment_lang)
        
        feature_vectors.append(features)
    
    # 收集所有特征名
    if feature_vectors:
        feature_names = list(feature_vectors[0].keys())
    
    return feature_names, feature_vectors


# ============================================================
# 命令行测试
# ============================================================

def main():
    """测试 Level 3 特征提取"""
    trials = []
    csv_path = config.DATA_PROCESSED / "trials.csv"
    
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            trials = list(csv.DictReader(f))
    
    if not trials:
        log.info("[nlp_l3] 无试验数据，跳过。")
        return
    
    log.info(f"[nlp_l3] 处理 {len(trials)} 条试验...")
    
    feature_names, feature_vectors = extract_level3_features(trials)
    
    log.info(f"[nlp_l3] 提取 {len(feature_names)} 个 Level 3 特征:")
    for name in feature_names[:15]:
        log.info(f"  - {name}")
    if len(feature_names) > 15:
        log.info(f"  ... 共 {len(feature_names)} 个特征")
    
    # 示例输出
    if feature_vectors:
        log.info(f"[nlp_l3] 示例特征向量（第一条试验）:")
        for name, val in list(feature_vectors[0].items())[:10]:
            log.info(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")


if __name__ == "__main__":
    config.setup_logging()
    main()
