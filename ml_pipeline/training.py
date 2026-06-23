# ml-pipeline/training.py
"""Model training — Optuna tuning + stacking ensemble."""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from .config import (
    VAR_THRESHOLD, RANDOM_STATE, N_TRIALS, TARGET_RECALL_R,
    cv_strategy, SCORING, soft_objective, report_cv,
)
from .inference import find_best_threshold_inner
from .reporting import (
    format_hyperparams_table,
    format_cv_comparison,
    format_threshold_summary,
    format_final_selection_rationale,
)


# ── Pipeline builders ────────────────────────────────────────────────────────

def build_xgb_pipeline(params: dict, k_features: int, smote_strategy: float) -> ImbPipeline:
    base = XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss")
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=VAR_THRESHOLD)),
        ("rfe", RFE(estimator=base, n_features_to_select=min(k_features, 310), step=20)),
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)),
    ]
    clf_params = {**params, "random_state": RANDOM_STATE, "n_jobs": 1}
    steps.append(("clf", XGBClassifier(**clf_params, eval_metric="logloss")))
    return ImbPipeline(steps)


def build_rf_pipeline(params: dict, k_features: int, smote_strategy: float) -> ImbPipeline:
    base = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=VAR_THRESHOLD)),
        ("rfe", RFE(estimator=base, n_features_to_select=min(k_features, 310), step=20)),
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)),
    ]
    clf_params = {**params, "random_state": RANDOM_STATE, "n_jobs": 1}
    steps.append(("clf", RandomForestClassifier(**clf_params)))
    return ImbPipeline(steps)


def build_lgbm_pipeline(params: dict, k_features: int, smote_strategy: float) -> ImbPipeline:
    base = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=VAR_THRESHOLD)),
        ("rfe", RFE(estimator=base, n_features_to_select=min(k_features, 310), step=20)),
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)),
    ]
    clf_params = {**params, "random_state": RANDOM_STATE, "n_jobs": 1}
    steps.append(("clf", LGBMClassifier(**clf_params)))
    return ImbPipeline(steps)


# ── Optuna objectives ────────────────────────────────────────────────────────
# FIX LỖI 3: nhận X_train, y_train làm parameter (closure), không dùng global

def objective_xgb(trial, X_train, y_train):
    np.random.seed(RANDOM_STATE)
    smote_strategy = trial.suggest_float("smote_strategy", 0.58, 1.0)
    k_features = trial.suggest_int("k_features", 10, min(100, 310))
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    try:
        pipe = build_xgb_pipeline(params, k_features, smote_strategy)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        return soft_objective(scores["test_macro_f1"].mean(), scores["test_recall_R"].mean())
    except Exception:
        return 0.0


def objective_rf(trial, X_train, y_train):
    np.random.seed(RANDOM_STATE)
    smote_strategy = trial.suggest_float("smote_strategy", 0.58, 1.0)
    k_features = trial.suggest_int("k_features", 10, min(100, 310))
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced_subsample", None]),
    }
    try:
        pipe = build_rf_pipeline(params, k_features, smote_strategy)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        return soft_objective(scores["test_macro_f1"].mean(), scores["test_recall_R"].mean())
    except Exception:
        return 0.0


def objective_lgbm(trial, X_train, y_train):
    np.random.seed(RANDOM_STATE)
    smote_strategy = trial.suggest_float("smote_strategy", 0.58, 1.0)
    k_features = trial.suggest_int("k_features", 10, min(100, 310))
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
    }
    try:
        pipe = build_lgbm_pipeline(params, k_features, smote_strategy)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        return soft_objective(scores["test_macro_f1"].mean(), scores["test_recall_R"].mean())
    except Exception:
        return 0.0


def build_stacking_ensemble(pipe_xgb, pipe_rf, pipe_lgbm) -> StackingClassifier:
    """Xây dựng Stacking Classifier từ các model cơ bản."""
    # Stacking dùng cv=3 để tăng tốc
    stacking_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=[
            ('xgb', pipe_xgb),
            ('rf', pipe_rf),
            ('lgbm', pipe_lgbm),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver='lbfgs',
        ),
        cv=stacking_cv,
        n_jobs=1,
        passthrough=False,
    )


def find_optimal_threshold(clf, X_train, y_train) -> float:
    """Tìm threshold tối ưu bằng out-of-fold predictions sử dụng cross_val_predict."""
    from sklearn.model_selection import cross_val_predict
    from .inference import find_best_threshold_inner
    
    # Lấy xác suất dự đoán out-of-fold cho toàn bộ X_train song song
    y_proba_oof = cross_val_predict(
        clf, X_train, y_train, 
        cv=cv_strategy, 
        method="predict_proba", 
        n_jobs=-1
    )[:, 1]
    
    threshold, _ = find_best_threshold_inner(y_train, y_proba_oof, target_recall=TARGET_RECALL_R)
    return threshold


# ── High-level training functions ───────────────────────────────────────────

def train_all_models(X_train, y_train, n_trials: int = N_TRIALS):
    """
    Chạy Optuna cho cả 3 model, fit trên toàn bộ X_train, trả về fitted pipelines.

    Returns
    -------
    dict với keys: xgb, rf, lgbm, stacking
    Mỗi item là (pipeline_fitted, best_params_dict, threshold)
    """
    import optuna
    from sklearn.metrics import recall_score, accuracy_score

    print("=" * 60)
    print("🔍 OPTUNA TUNING — 3 MODELS")
    print("=" * 60)

    best_configs = {}

    for name, objective in [
        ("XGBoost", objective_xgb),
        ("RandomForest", objective_rf),
        ("LightGBM", objective_lgbm),
    ]:
        print(f"\n── {name} ──")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda t: objective(t, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        best_params = {
            k: v for k, v in study.best_params.items()
            if k not in ("smote_strategy", "k_features")
        }
        smote = study.best_params["smote_strategy"]
        k_feats = study.best_params["k_features"]
        print(f" Best trial: {study.best_trial.number} | value: {study.best_value:.4f}")
        # Lưu cả filtered params (dùng để build pipeline) và full params (dùng cho reporting)
        best_configs[name] = {
            'pipeline_params': best_params,
            'full_params': study.best_params.copy(),
            'k_features': k_feats,
            'smote_strategy': smote
        }

    # ── IN FULL HYPERPARAMETER TABLE ─────────────────────────────────────────────
    print(format_hyperparams_table(best_configs))

    # FIX LỖI 1: build pipelines từ config dict
    cfg_xgb = best_configs["XGBoost"]
    cfg_rf = best_configs["RandomForest"]
    cfg_lgbm = best_configs["LightGBM"]

    pipe_xgb = build_xgb_pipeline(cfg_xgb['pipeline_params'], cfg_xgb['k_features'], cfg_xgb['smote_strategy'])
    pipe_rf  = build_rf_pipeline(cfg_rf['pipeline_params'], cfg_rf['k_features'], cfg_rf['smote_strategy'])
    pipe_lgbm = build_lgbm_pipeline(cfg_lgbm['pipeline_params'], cfg_lgbm['k_features'], cfg_lgbm['smote_strategy'])

    # ── THÊM STACKING CONFIG CHO REPORTING ───────────────────────────────────────
    # Stacking không có Optuna params, nhưng có final_estimator config
    stacking_config = {
        'pipeline_params': {},  # Stacking không có classifier params riêng
        'full_params': {
            'ensemble_type': 'StackingClassifier',
            'final_estimator': 'LogisticRegression',
            'final_estimator.max_iter': 2000,
            'final_estimator.class_weight': 'balanced',
            'final_estimator.solver': 'lbfgs',
            'final_estimator.random_state': RANDOM_STATE,
            'cv': 'StratifiedKFold',
            'cv.n_splits': 3,
            'cv.shuffle': True,
            'cv.random_state': RANDOM_STATE,
            'passthrough': False,
            'n_jobs': 1,
            'base_estimators': ['XGBoost', 'RandomForest', 'LightGBM']
        },
        'k_features': 'N/A (ensemble uses all base features)',
        'smote_strategy': 'N/A (each base has own SMOTE)'
    }
    best_configs['Stacking'] = stacking_config

    # ── CROSS-VALIDATION WITH FULL METRICS ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 CROSS-VALIDATION METRICS (5-fold)")
    print("=" * 60)

    cv_metrics = {}
    cv_models = {
        "XGBoost": pipe_xgb,
        "RandomForest": pipe_rf,
        "LightGBM": pipe_lgbm,
    }

    for name, pipe in cv_models.items():
        # Cross-validate with all needed scorers
        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=cv_strategy,
            scoring={
                'macro_f1': SCORING['macro_f1'],
                'recall_R': SCORING['recall_R'],
            },
            n_jobs=-1,
            return_estimator=True  # Keep estimators for OOF predictions
        )

        # Get OOF predictions for additional metrics (recall_S, accuracy)
        from sklearn.model_selection import cross_val_predict
        y_pred_oof = cross_val_predict(pipe, X_train, y_train, cv=cv_strategy, n_jobs=-1)

        # Compute additional metrics from OOF
        recall_S = recall_score(y_train, y_pred_oof, pos_label=0)
        accuracy = accuracy_score(y_train, y_pred_oof)

        cv_metrics[name] = {
            'macro_f1': cv_res['test_macro_f1'].mean(),
            'recall_R': cv_res['test_recall_R'].mean(),
            'recall_S': recall_S,
            'accuracy': accuracy,
        }

        # Print per-model CV results (compact)
        mf1 = cv_metrics[name]['macro_f1']
        rR = cv_metrics[name]['recall_R']
        rS = cv_metrics[name]['recall_S']
        acc = cv_metrics[name]['accuracy']
        print(f"\n📊 {name}:")
        print(f"   Macro F1  : {mf1*100:6.2f}%")
        print(f"   Recall(R) : {rR*100:6.2f}%")
        print(f"   Recall(S) : {rS*100:6.2f}%")
        print(f"   Accuracy  : {acc*100:6.2f}%")

    # ── CV COMPARISON WITH TRADE-OFF ANALYSIS ───────────────────────────────────
    print(format_cv_comparison(cv_metrics))

    # FIX LỖI 2: fit trên toàn bộ data
    print("\n" + "=" * 60)
    print("🎯 FITTING MODELS ON FULL TRAINING SET")
    print("=" * 60)
    pipe_xgb.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)
    pipe_lgbm.fit(X_train, y_train)

    # Stacking
    print("\n📦 Building stacking ensemble...")
    stacking = build_stacking_ensemble(pipe_xgb, pipe_rf, pipe_lgbm)
    stacking.fit(X_train, y_train)

    # ── THRESHOLD TUNING WITH OOF METRICS ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 THRESHOLD TUNING (OOF)")
    print("=" * 60)

    thresholds = {}
    oof_scores = {}  # For threshold summary (macro_f1, recall_R from OOF)

    for name, pipe in {
        "XGBoost": pipe_xgb,
        "RandomForest": pipe_rf,
        "LightGBM": pipe_lgbm,
        "Stacking": stacking
    }.items():
        print(f"\n── {name} ──")
        th = find_optimal_threshold(pipe, X_train, y_train)

        # Get OOF probs for this model to compute metrics at tuned threshold
        from sklearn.model_selection import cross_val_predict
        y_proba_oof = cross_val_predict(pipe, X_train, y_train,
                                        cv=cv_strategy, method='predict_proba', n_jobs=-1)[:, 1]
        y_pred_oof = (y_proba_oof >= th).astype(int)

        from sklearn.metrics import f1_score, recall_score
        f1_oof = f1_score(y_train, y_pred_oof, average='macro')
        rR_oof = recall_score(y_train, y_pred_oof, pos_label=1)

        thresholds[name] = th
        oof_scores[name] = {'macro_f1': f1_oof, 'recall_R': rR_oof}

        print(f"  Optimal threshold: {th:.3f}")
        print(f"  OOF Macro F1: {f1_oof*100:6.2f}%")
        print(f"  OOF Recall(R): {rR_oof*100:6.2f}%")

    # ── THRESHOLD SUMMARY ────────────────────────────────────────────────────────
    print(format_threshold_summary(thresholds, oof_scores))

    # ── FINAL MODEL SELECTION & RATIONALE ───────────────────────────────────────
    # Determine best model based on CV metrics (Macro F1 primary, Recall(R) secondary)
    best_name = max(cv_metrics.keys(), key=lambda n: (
        cv_metrics[n]['macro_f1'],
        cv_metrics[n]['recall_R']  # tie-breaker
    ))

    cv_df = pd.DataFrame(cv_metrics).T.rename(columns={
        'macro_f1': 'Macro F1',
        'recall_R': 'Recall(R)',
        'recall_S': 'Recall(S)',
        'accuracy': 'Accuracy',
        })[['Macro F1', 'Recall(R)', 'Recall(S)', 'Accuracy']]

    print(format_final_selection_rationale(best_name, cv_df, thresholds, TARGET_RECALL_R))

    return {
        "xgb":     (pipe_xgb,   best_configs["XGBoost"]['pipeline_params'],    thresholds["XGBoost"]),
        "rf":      (pipe_rf,    best_configs["RandomForest"]['pipeline_params'], thresholds["RandomForest"]),
        "lgbm":    (pipe_lgbm,  best_configs["LightGBM"]['pipeline_params'],    thresholds["LightGBM"]),
        "stacking": (stacking,   {},                          thresholds["Stacking"]),
    }