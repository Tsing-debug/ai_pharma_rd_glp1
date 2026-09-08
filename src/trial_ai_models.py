"""临床试验结果预测（辉瑞 AI Pilot 核心模块，深度学习工程师定位）。

目标：根据试验设计特征（分期、适应症、入组人数、申办方、启动年份）
预测试验是否"成功完成"（ClinicalTrials.gov status == Completed）。

方法：
  - 特征工程：phase 深度、obesity/diabetes/CV 适应症标志、log(入组数)、申办方标志、启动年份
  - 模型三件套：Logistic Regression、Random Forest（sklearn，可选）
    + 手写 3 层 MLP（numpy 反向传播，无深度学习框架依赖）
  - 评估：5-fold 分层 CV + AUC（Mann-Whitney 秩和，含平局校正）+ Accuracy

输出：data/processed/trial_model_results.csv（每折指标 + 特征重要性）
"""
import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

SEED = 42
FOLD = 5


def load_trials() -> list[dict]:
    p = config.DATA_PROCESSED / "trials.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _phase_depth(phase: str) -> int:
    depths = [int(x) for x in re.findall(r"(\d)", phase or "")]
    return max(depths, default=0)


def engineer_features(trials: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows, cols = [], ["phase_depth", "has_obesity", "has_diabetes", "has_cardio",
                      "log_enrollment", "sponsor_lilly", "sponsor_novo", "start_year"]
    for r in trials:
        ind = (r.get("indication") or "").lower()
        status = (r.get("status") or "").strip().lower()
        if not status:
            continue
        rows.append([
            _phase_depth(r.get("phase", "")),
            1 if "obesity" in ind or "overweight" in ind else 0,
            1 if "diabetes" in ind else 0,
            1 if any(k in ind for k in ("heart", "cardio", "cardiovascular")) else 0,
            math.log1p(float(r.get("enrollment") or 0)),
            1 if "eli lilly" in (r.get("company") or "").lower() else 0,
            1 if "novo" in (r.get("company") or "").lower() else 0,
            int(r.get("start_year") or 0),
            1 if status == "completed" else 0,
        ])
    if not rows:
        return np.zeros((0, len(cols))), np.zeros(0), cols
    arr = np.array(rows, dtype=float)
    return arr[:, :-1], arr[:, -1], cols


# ── 评估：AUC（Mann-Whitney 秩和，含平局校正）+ Accuracy ──────────
def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    s, yy = scores[order], y[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n0 = int(yy.sum()), len(yy) - int(yy.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _stratified_folds(y: np.ndarray, k: int = FOLD) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    idx1 = rng.permutation(np.where(y == 1)[0])
    idx0 = rng.permutation(np.where(y == 0)[0])
    folds = []
    for f in range(k):
        test = np.concatenate([idx1[f::k], idx0[f::k]])
        train = np.setdiff1d(np.arange(len(y)), test)
        folds.append((train, test))
    return folds


# ── 手写 3 层 MLP（numpy 反向传播，无框架依赖）──────────────────────
def manual_mlp(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, hidden=8,
               lr=0.1, epochs=300) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xvs = (Xv - mu) / sd
    rng = np.random.default_rng(SEED)
    d, h = Xs.shape[1], hidden
    W1 = rng.normal(0, 0.5, (d, h)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.5, (h, 1)); b2 = 0.0

    def forward(x):
        hid = np.tanh(x @ W1 + b1)
        return hid, hid @ W2 + b2

    n = len(y)
    for _ in range(epochs):
        hid, o = forward(Xs)
        prob = 1.0 / (1.0 + np.exp(-o))
        dy = (prob - y.reshape(-1, 1)) / n
        W2 -= lr * (hid.T @ dy); b2 -= lr * float(dy.sum())
        dh = dy @ W2.T * (1.0 - hid ** 2)
        W1 -= lr * (Xs.T @ dh); b1 -= lr * dh.sum(0)
    _, o = forward(Xvs)
    return (1.0 / (1.0 + np.exp(-o))).ravel()


# ── 各模型训练：返回验证集概率 ──────────────────────────────────────
def _fit_predict(name: str, Xtr, ytr, Xte) -> np.ndarray:
    if name == "Manual MLP (numpy backprop)":
        return manual_mlp(Xtr, ytr, Xte)
    if name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=SEED)
    else:  # Random Forest
        model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def run() -> Path:
    out = config.DATA_PROCESSED / "trial_model_results.csv"
    trials = load_trials()
    if len(trials) < 30 or not _HAS_NUMPY:
        print(f"[trial_ai_models] 跳过：样本 {len(trials)}，numpy={_HAS_NUMPY}（需 ≥30 且 numpy 可用）")
        return out

    X, y, cols = engineer_features(trials)
    if len(y) < 30:
        print(f"[trial_ai_models] 跳过：有效样本 {len(y)} < 30")
        return out
    print(f"[trial_ai_models] 样本 {len(y)}（完成 {int(y.sum())} / 未完成 {int(len(y) - y.sum())}），"
          f"特征 {len(cols)} 个；sklearn={_HAS_SKLEARN}")

    models = ["Manual MLP (numpy backprop)"]
    if _HAS_SKLEARN:
        models = ["Logistic Regression", "Random Forest"] + models

    rows = []
    summary = {}
    for name in models:
        a, acc = [], []
        for tr, te in _stratified_folds(y):
            p = _fit_predict(name, X[tr], y[tr], X[te])
            a.append(_auc(p, y[te]))
            acc.append(float(((p >= 0.5) == y[te]).mean()))
        mean_auc = float(np.nanmean(a))
        summary[name] = {"auc": mean_auc, "acc": float(np.mean(acc))}
        for f in range(FOLD):
            rows.append({"model": name, "fold": f + 1, "auc": round(a[f], 4),
                         "accuracy": round(acc[f], 4)})
        print(f"  {name:34s} CV AUC={mean_auc:.4f}  Acc={summary[name]['acc']:.3f}")

    best = max(summary, key=lambda k: summary[k]["auc"])
    print(f"[trial_ai_models] 最佳模型: {best} (AUC={summary[best]['auc']:.4f})")

    # 特征重要性（LR 系数 / RF importances）
    imp: dict[str, float] = {}
    if _HAS_SKLEARN:
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xs, y)
        for c, v in zip(cols, lr.coef_[0]):
            imp[c] = float(v)
    for c, v in sorted(imp.items(), key=lambda kv: -abs(kv[1]))[:5]:
        print(f"  特征 {c}: 系数 {v:+.3f}")
        rows.append({"model": "feature_importance", "fold": c, "auc": round(v, 4), "accuracy": ""})

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fold", "auc", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[trial_ai_models] 完成 -> {out}")
    return out


if __name__ == "__main__":
    run()
