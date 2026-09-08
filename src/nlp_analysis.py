"""NLP：临床试验管线文本挖掘（辉瑞 AI Pilot / 研发情报方向）。

对 ClinicalTrials.gov 真实试验的标题 + 适应症做词频挖掘：
  - 全库 Top 关键词（研发热点）
  - 按分子（tirzepatide/semaglutide/orforglipron）的差异化关键词（管线定位差异）
纯标准库实现（collections.Counter + 简单 tokenizer + 停用词）。

输出：data/processed/trial_nlp_keywords.csv
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
    "study", "trial", "phase", "open", "label", "randomized", "randomised",
    "double", "blind", "parallel", "arm", "arms", "effect", "effects",
    "versus", "vs", "compared", "comparison", "participants", "patients",
    "evaluate", "evaluating", "assessment", "treatment", "treated",
    # 试验状态词 + 领域高频噪声词
    "completed", "recruiting", "active", "not", "yet", "unknown", "terminated",
    "withdrawn", "enrolling", "invitation", "suspended", "by", "type", "mellitus",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z-]{2,}", (text or "").lower())
            if t not in STOPWORDS]


def load_trials() -> list[dict]:
    p = config.DATA_PROCESSED / "trials.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run() -> Path:
    trials = load_trials()
    if not trials:
        print("[nlp_analysis] 无试验数据，跳过。")
        return config.DATA_PROCESSED / "trial_nlp_keywords.csv"

    all_counter: Counter = Counter()
    per_mol: dict[str, Counter] = {}
    for r in trials:
        text = r.get("indication") or ""  # 只挖掘适应症文本，避免状态字段污染
        toks = _tokens(text)
        all_counter.update(toks)
        mol = r.get("molecule") or "other"
        per_mol.setdefault(mol, Counter()).update(toks)

    rows = []
    for word, cnt in all_counter.most_common(15):
        rows.append({"scope": "ALL", "keyword": word, "count": cnt})
    for mol, c in sorted(per_mol.items()):
        for word, cnt in c.most_common(8):
            rows.append({"scope": mol, "keyword": word, "count": cnt})

    out = config.DATA_PROCESSED / "trial_nlp_keywords.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "keyword", "count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[nlp_analysis] 试验 {len(trials)} 条；全库 Top10: "
          + ", ".join(f"{w}({c})" for w, c in all_counter.most_common(10)))
    print(f"[nlp_analysis] 完成 -> {out}")
    return out


if __name__ == "__main__":
    run()
