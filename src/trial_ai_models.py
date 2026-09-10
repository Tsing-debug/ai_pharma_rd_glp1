"""临床试验结果预测（辉瑞 AI Pilot 核心模块，深度学习工程师定位）。

目标：根据试验设计特征（分期、适应症、入组人数、申办方、启动年份）
预测试验是否"成功完成"（ClinicalTrials.gov status == Completed）。

方法：
  - 特征工程：
    * Level 0: 基础特征（8个）
    * Level 1: 时间与周期特征（5个）
    * Level 2: 设计文本特征（6个）
    * Level 3: NLP 深度挖掘（~20个）- TF-IDF、Protocol 复杂度、招募语言
  - 模型三件套：Logistic Regression、Random Forest（sklearn，可选）
    + 手写 3 层 MLP（numpy 反向传播，无深度学习框架依赖）
  - 评估：训练集内 5-fold 分层 CV 调参 + 独立分层 holdout 最终验证
    （AUC：Mann-Whitney 秩和，含平局校正）+ Accuracy

输出：data/processed/trial_model_results.csv（每折指标 + 特征重要性）
"""
import csv
import logging
import math
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# 导入 Level 3 NLP 特征提取
try:
    from nlp_feature_extraction import extract_level3_features, SimpleTFIDF
    _HAS_NLP_L3 = True
except ImportError:
    _HAS_NLP_L3 = False

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

SEED = config.SEED
FOLD = 5


def load_trials() -> list[dict]:
    p = config.DATA_PROCESSED / "trials.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tune_random_forest(X: np.ndarray, y: np.ndarray) -> dict:
    """网格搜索 Random Forest 超参数"""
    log.info("[tuning] Random Forest 网格搜索...")
    results = []
    
    # 网格搜索参数（小样本下网格不宜过大：108 组 × 5 折既耗时又加剧选择偏差；
    # 收缩为 16 组，选参后仍在独立 holdout 上验证）
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    
    best_auc = 0
    best_params = {}
    
    for n_est in param_grid["n_estimators"]:
        for max_d in param_grid["max_depth"]:
            for min_split in param_grid["min_samples_split"]:
                for min_leaf in param_grid["min_samples_leaf"]:
                    aucs = []
                    for tr, te in _stratified_folds(y):
                        model = RandomForestClassifier(
                            n_estimators=n_est, max_depth=max_d,
                            min_samples_split=min_split, min_samples_leaf=min_leaf,
                            random_state=SEED, n_jobs=-1
                        )
                        model.fit(X[tr], y[tr])
                        p = model.predict_proba(X[te])[:, 1]
                        aucs.append(_auc(p, y[te]))
                    
                    mean_auc = float(np.nanmean(aucs))
                    results.append({
                        "n_estimators": n_est, "max_depth": max_d,
                        "min_samples_split": min_split, "min_samples_leaf": min_leaf,
                        "auc": mean_auc
                    })
                    
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_params = {
                            "n_estimators": n_est, "max_depth": max_d,
                            "min_samples_split": min_split, "min_samples_leaf": min_leaf
                        }
    
    log.info(f"[tuning] 最佳 Random Forest: AUC={best_auc:.4f}, params={best_params}")
    return {"best": best_params, "auc": best_auc}


def tune_logistic_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """网格搜索 Logistic Regression 超参数"""
    log.info("[tuning] Logistic Regression 网格搜索...")
    results = []
    
    # 网格搜索参数（elasticnet 仅 saga 支持，配合 l1_ratio=0.5）
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver_penalty": [("l2", "lbfgs"), ("elasticnet", "saga")]
    }
    
    best_auc = 0
    best_params = {}
    
    for C in param_grid["C"]:
        for penalty, solver in param_grid["solver_penalty"]:
            aucs = []
            for tr, te in _stratified_folds(y):
                try:
                    model = LogisticRegression(
                        C=C, penalty=penalty, solver=solver,
                        max_iter=1000, random_state=SEED,
                        l1_ratio=0.5 if penalty == "elasticnet" else None
                    )
                    model.fit(X[tr], y[tr])
                    p = model.predict_proba(X[te])[:, 1]
                    aucs.append(_auc(p, y[te]))
                except ValueError:
                    continue
            
            if aucs:
                mean_auc = float(np.nanmean(aucs))
                results.append({
                    "C": C, "penalty": penalty, "solver": solver,
                    "auc": mean_auc
                })
                
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"C": C, "penalty": penalty, "solver": solver}
    
    log.info(f"[tuning] 最佳 Logistic Regression: AUC={best_auc:.4f}, params={best_params}")
    return {"best": best_params, "auc": best_auc}


def tune_mlp(X: np.ndarray, y: np.ndarray) -> dict:
    """网格搜索 MLP 超参数"""
    log.info("[tuning] MLP 网格搜索...")
    results = []
    
    # 网格搜索参数（收缩网格：8 组 × 5 折，避免 numpy 手写网络在小样本上耗时过久）
    param_grid = {
        "hidden": [8, 16],
        "lr": [0.05, 0.1],
        "epochs": [200, 300]
    }
    
    best_auc = 0
    best_params = {}
    
    for hidden in param_grid["hidden"]:
        for lr in param_grid["lr"]:
            for epochs in param_grid["epochs"]:
                aucs = []
                for tr, te in _stratified_folds(y):
                    p = manual_mlp(X[tr], y[tr], X[te], hidden=hidden, lr=lr, epochs=epochs)
                    aucs.append(_auc(p, y[te]))
                
                mean_auc = float(np.nanmean(aucs))
                results.append({
                    "hidden": hidden, "lr": lr, "epochs": epochs,
                    "auc": mean_auc
                })
                
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"hidden": hidden, "lr": lr, "epochs": epochs}
    
    log.info(f"[tuning] 最佳 MLP: AUC={best_auc:.4f}, params={best_params}")
    return {"best": best_params, "auc": best_auc}


def _phase_depth(phase: str) -> int:
    """Phase 深度：Phase 1→1, Phase 2→2, Phase 3a/3b/3c→3"""
    depths = [int(x) for x in re.findall(r"(\d)", phase or "")]
    return max(depths, default=0)


def _is_phase3_plus(phase: str) -> int:
    """Phase 3+ 试验（更可能完成）"""
    phase = (phase or "").upper()
    return 1 if "3A" in phase or "3B" in phase or "3C" in phase or "PHASE 3" in phase else 0


def _estimate_duration_days(indication: str, phase: str) -> float:
    """估算试验周期（天）"""
    # 基准时长（月）：Phase 1(1-2), Phase 2(3-6), Phase 3(6-24)
    phase = phase or ""
    if "1" in phase:
        base_months = 1.5
    elif "2" in phase:
        base_months = 4.5
    elif "3" in phase:
        base_months = 12.0
    else:
        base_months = 3.0
    
    # 调整因子：肥胖试验更长，联合疗法更长
    indication = (indication or "").lower()
    if "obesity" in indication or "overweight" in indication:
        base_months *= 1.3  # 肥胖试验随访更长
    if any(k in indication for k in ("combination", "combo")):
        base_months *= 1.2  # 联合疗法更长
    
    return base_months * 30.0  # 转换为天


def engineer_features(trials: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """特征工程：核心特征 + Level 1 + Level 2 + Level 3 增强特征"""
    
    # ── Level 0-2 特征列表 ────────────────────────────────
    cols_base = [
        # 基础特征（不含任何由 status 直接派生的特征，避免目标泄露）
        "phase_depth", "has_obesity", "has_diabetes", "has_cardio",
        "log_enrollment", "sponsor_lilly", "sponsor_novo", "start_year",
        # Level 1 增强特征
        "has_phase3", "log_duration_days", "has_cv_outcomes",
        "has_healthy_volunteers",
        # Level 2 特征
        "design_double_blind", "design_placebo_controlled", "design_global",
        "endpoint_cv", "recruitment_multi_center",
    ]
    
    # ── Level 3 NLP 特征 ────────────────────────────────
    cols_nlp = []
    nlp_features_list = []

    # 注意：不直接改写模块级 _HAS_NLP_L3（赋值会使其成为局部变量，导致
    # 函数内后续读取抛 UnboundLocalError），用局部 nlp_ok 承载。
    nlp_ok = _HAS_NLP_L3
    if nlp_ok:
        try:
            cols_nlp, nlp_features_list = extract_level3_features(trials)
        except Exception as e:
            log.info(f"[engineer_features] NLP L3 提取失败: {e}，跳过")
            nlp_ok = False
            cols_nlp, nlp_features_list = [], []
    else:
        cols_nlp, nlp_features_list = [], []
    
    cols = cols_base + cols_nlp
    
    # ── 提取特征向量 ────────────────────────────────────
    rows = []

    # start_year 缺失（<=1900 视为缺失）用有效年份中位数填充，避免 0 污染特征
    valid_years = [int(r.get("start_year") or 0) for r in trials
                   if 1900 < int(r.get("start_year") or 0) < 2100]
    median_year = int(np.median(valid_years)) if valid_years else 2020
    
    for idx, r in enumerate(trials):
        ind = (r.get("indication") or "").lower()
        title = (r.get("title") or "").lower()
        status = (r.get("status") or "").strip().lower()
        if not status:
            continue
        
        # Level 0-2 特征提取（原有逻辑）
        has_cv = any(k in ind for k in ("cardiovascular", "heart", "cv", "cardio"))
        has_cv_outcomes = any(k in ind for k in ("cvoutcomes", "cv outcomes", "cardiovascular outcomes", 
                                                   "mace", "major adverse cardiac"))
        has_healthy = any(k in ind for k in ("healthy", "healthy volunteer", "healthy volunteers"))
        start_year = int(r.get("start_year") or 0)
        if not 1900 < start_year < 2100:
            start_year = median_year  # 缺失年份回退到中位数，避免 0 污染特征
        phase = r.get("phase", "") or ""
        
        design_text = (r.get("design") or "").lower() + " " + (r.get("title") or "").lower()
        endpoint_text = (r.get("endpoint") or "").lower()
        recruitment_text = (r.get("recruitment") or "").lower()
        
        design_double_blind = 1 if any(k in design_text for k in ("double-blind", "double blind")) else 0
        design_placebo_controlled = 1 if any(k in design_text for k in ("placebo-controlled", "placebo controlled")) else 0
        design_global = 1 if any(k in design_text for k in ("global", "international")) else 0
        
        endpoint_cv = 1 if any(k in endpoint_text for k in ("cv", "cardiovascular", "mace", "heart")) else 0
        
        recruitment_multi_center = 1 if any(k in recruitment_text for k in ("multi-center", "multi-center", 
                                                                              "multicenter", "multiple centers")) else 0
        
        # 注意：不再构造 is_active_status / estimated_completion_year——
        # 二者由 status 或"当前时点"派生，会把目标信息泄漏进特征（数据泄露，面试高频考点）。
        
        row = [
            _phase_depth(r.get("phase", "")),
            1 if "obesity" in ind or "overweight" in ind else 0,
            1 if "diabetes" in ind else 0,
            1 if has_cv else 0,
            math.log1p(float(r.get("enrollment") or 0)),
            1 if "eli lilly" in (r.get("company") or "").lower() else 0,
            1 if "novo" in (r.get("company") or "").lower() else 0,
            start_year,
            _is_phase3_plus(r.get("phase", "")),
            math.log1p(_estimate_duration_days(ind, r.get("phase", ""))),
            1 if has_cv_outcomes else 0,
            1 if has_healthy else 0,
            design_double_blind,
            design_placebo_controlled,
            design_global,
            endpoint_cv,
            recruitment_multi_center,
        ]
        
        # ── 添加 Level 3 NLP 特征 ────────────────────────
        if nlp_ok and idx < len(nlp_features_list):
            nlp_feats = nlp_features_list[idx]
            for col_name in cols_nlp:
                row.append(nlp_feats.get(col_name, 0.0))
        elif nlp_ok:
            row.extend([0.0] * len(cols_nlp))
        
        # 目标变量
        row.append(1 if status == "completed" else 0)
        rows.append(row)
    
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
def _fit_predict(name: str, Xtr, ytr, Xte, params: dict = None) -> np.ndarray:
    if name == "Manual MLP (numpy backprop)":
        hidden = params.get("hidden", 8) if params else 8
        lr = params.get("lr", 0.1) if params else 0.1
        epochs = params.get("epochs", 300) if params else 300
        return manual_mlp(Xtr, ytr, Xte, hidden=hidden, lr=lr, epochs=epochs)
    if name == "Logistic Regression":
        C = params.get("C", 1.0) if params else 1.0
        penalty = params.get("penalty", "l2") if params else "l2"
        solver = params.get("solver", "lbfgs") if params else "lbfgs"
        model = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                   max_iter=1000, random_state=SEED)
    else:  # Random Forest
        n_est = params.get("n_estimators", 200) if params else 200
        max_d = params.get("max_depth", 6) if params else 6
        min_split = params.get("min_samples_split", 2) if params else 2
        min_leaf = params.get("min_samples_leaf", 1) if params else 1
        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                       min_samples_split=min_split, min_samples_leaf=min_leaf,
                                       random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def run() -> Path:
    out = config.DATA_PROCESSED / "trial_model_results.csv"
    trials = load_trials()
    if len(trials) < 30 or not _HAS_NUMPY:
        log.info(f"[trial_ai_models] 跳过：样本 {len(trials)}，numpy={_HAS_NUMPY}（需 ≥30 且 numpy 可用）")
        return out

    X, y, cols = engineer_features(trials)
    if len(y) < 30:
        log.info(f"[trial_ai_models] 跳过：有效样本 {len(y)} < 30")
        return out
    log.info(f"[trial_ai_models] 样本 {len(y)}（完成 {int(y.sum())} / 未完成 {int(len(y) - y.sum())}），"
          f"特征 {len(cols)} 个；sklearn={_HAS_SKLEARN}")

    # ── 分层 holdout（80/20）：最终验证集不参与调参，避免选择偏差 ──
    rng = np.random.default_rng(SEED)
    idx1 = rng.permutation(np.where(y == 1)[0])
    idx0 = rng.permutation(np.where(y == 0)[0])
    n1_tr, n0_tr = int(0.8 * len(idx1)), int(0.8 * len(idx0))
    tr_idx = np.concatenate([idx1[:n1_tr], idx0[:n0_tr]])
    te_idx = np.concatenate([idx1[n1_tr:], idx0[n0_tr:]])
    Xtr_all, ytr_all = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # 标准化统计量只从训练集估计（避免测试信息泄漏）
    mu, sd = Xtr_all.mean(0), Xtr_all.std(0) + 1e-9
    Xtr_std, Xte_std = (Xtr_all - mu) / sd, (Xte - mu) / sd

    # ── 调参：网格搜索在训练集内做 5 折 CV（不再用固定拍脑袋参数）──
    tuned: dict[str, dict] = {}
    if _HAS_SKLEARN:
        tuned["Logistic Regression"] = tune_logistic_regression(Xtr_std, ytr_all)
        tuned["Random Forest"] = tune_random_forest(Xtr_all, ytr_all)
    tuned["Manual MLP (numpy backprop)"] = tune_mlp(Xtr_all, ytr_all)

    # ── 评估：训练集 CV + 独立 holdout ──────────────────────
    log.info("\n--- 模型性能评估（训练集内调参后，再报 holdout 最终指标） ---")
    rows, summary = [], {}
    for name, tune_result in tuned.items():
        params = tune_result["best"]
        a, acc = [], []
        for tr, te in _stratified_folds(ytr_all):
            try:
                if name == "Logistic Regression":
                    p = _fit_predict(name, Xtr_std[tr], ytr_all[tr], Xtr_std[te], params)
                else:
                    p = _fit_predict(name, Xtr_all[tr], ytr_all[tr], Xtr_all[te], params)
                a.append(_auc(p, ytr_all[te]))
                acc.append(float(((p >= 0.5) == ytr_all[te]).mean()))
            except Exception as e:
                log.info(f"  [{name}] 第 {len(a)+1} 折出错: {e}")
                continue

        cv_auc = float(np.nanmean(a)) if a else float("nan")
        if name == "Logistic Regression":
            p_te = _fit_predict(name, Xtr_std, ytr_all, Xte_std, params)
        else:
            p_te = _fit_predict(name, Xtr_all, ytr_all, Xte, params)
        ho_auc = _auc(p_te, yte)
        ho_acc = float(((p_te >= 0.5) == yte).mean())
        summary[name] = {"cv_auc": cv_auc, "holdout_auc": ho_auc, "params": params}

        for f in range(min(FOLD, len(a))):
            rows.append({"model": name, "fold": f + 1, "auc": round(a[f], 4),
                         "accuracy": round(acc[f], 4)})
        rows.append({"model": name, "fold": "holdout", "auc": round(ho_auc, 4),
                     "accuracy": round(ho_acc, 4)})
        log.info(f"  {name:34s} CV AUC={cv_auc:.4f} | holdout AUC={ho_auc:.4f}  Acc={ho_acc:.3f}")

    if summary:
        best = max(summary, key=lambda k: summary[k]["holdout_auc"])
        log.info(f"[trial_ai_models] 最佳模型（按 holdout）: {best} "
              f"(holdout AUC={summary[best]['holdout_auc']:.4f})")

    # ── 特征重要性（只在训练集上拟合：LR 系数 + RF importances）──
    imp: dict[str, tuple[str, float]] = {}
    if _HAS_SKLEARN:
        lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr_std, ytr_all)
        for c, v in zip(cols, lr.coef_[0]):
            imp[f"LR|{c}"] = ("coef", float(v))
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    random_state=SEED, n_jobs=-1).fit(Xtr_all, ytr_all)
        for c, v in zip(cols, rf.feature_importances_):
            imp[f"RF|{c}"] = ("import", float(v))

    log.info(f"[trial_ai_models] 特征数: {len(cols)}")
    log.info(f"[trial_ai_models] Top 10 特征重要性（LR 系数 / RF importances）:")
    for k, (_, v) in sorted(imp.items(), key=lambda kv: -abs(kv[1][1]))[:10]:
        log.info(f"  {k}: {v:+.4f}")
        rows.append({"model": "feature_importance", "fold": k, "auc": round(v, 4), "accuracy": ""})

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fold", "auc", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[trial_ai_models] 完成 -> {out}")
    return out


if __name__ == "__main__":
    config.setup_logging()
    run()
