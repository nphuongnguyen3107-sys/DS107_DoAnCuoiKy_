"""
Reporting utilities — format và in báo cáo huấn luyện/đánh giá.

Các module:
- Hyperparameter tables (full config cho từng model)
- CV comparison với trade-off analysis
- Threshold tuning summary
- Final model selection rationale
"""

from typing import Dict, Tuple, Any
import pandas as pd


def format_hyperparams_table(best_configs: Dict[str, Dict]) -> str:
    """
    Tạo bảng đầy đủ hyperparameters của tất cả models.

    Parameters
    ----------
    best_configs : dict
        Structure: {
            model_name: {
                'pipeline_params': dict,  # params cho classifier (không có smote/k_features)
                'full_params': dict,      # full params từ Optuna (có smote, k_features)
                'k_features': int,
                'smote_strategy': float
            }
        }

    Returns
    -------
    str : formatted table
    """
    lines = [
        "=" * 80,
        " " * 20 + "FULL HYPERPARAMETER CONFIGURATION",
        "=" * 80,
    ]

    for name, config in best_configs.items():
        lines.append(f"\n{name.upper()}")
        lines.append("-" * 80)
        lines.append(f"  Pipeline Configuration:")
        lines.append(f"    RFE k_features      : {config['k_features']}")
        lines.append(f"    SMOTE strategy      : {config['smote_strategy']:.4f}")

        # Use full_params (includes all Optuna params including smote_strategy, k_features)
        params = config['full_params']
        lines.append(f"\n  All Optuna Hyperparameters:")

        if params:
            max_key_len = max(len(k) for k in params.keys())
            for key, val in sorted(params.items()):
                if isinstance(val, float):
                    lines.append(f"    {key:<{max_key_len}} : {val:.6f}")
                elif isinstance(val, bool):
                    lines.append(f"    {key:<{max_key_len}} : {val}")
                else:
                    lines.append(f"    {key:<{max_key_len}} : {val}")
        else:
            lines.append("    (no additional hyperparameters)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_cv_comparison(cv_metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Tạo bảng so sánh CV results với trade-off analysis.

    Parameters
    ----------
    cv_metrics : dict
        Structure: {
            model_name: {
                'macro_f1': float,
                'recall_R': float,
                'recall_S': float,
                'accuracy': float,
                'roc_auc': float (optional)
            }
        }

    Returns
    -------
    str : formatted comparison table + insights
    """
    df = pd.DataFrame(cv_metrics).T

    expected_cols = ['macro_f1', 'recall_R', 'recall_S', 'accuracy']
    if 'roc_auc' in df.columns:
        expected_cols.append('roc_auc')

    df = df[expected_cols]
    df.columns = ['Macro F1', 'Recall(R)', 'Recall(S)', 'Accuracy',
                  'ROC-AUC'][:len(expected_cols)]

    lines = [
        "\n" + "=" * 90,
        " " * 25 + "CROSS-VALIDATION COMPARISON",
        "=" * 90,
        "\nMetrics (averaged over 5-fold CV):\n",
    ]

    df_pct = df * 100
    lines.append(df_pct.to_string(float_format=lambda x: f"{x:6.2f}%"))

    lines.append("\n" + "-" * 90)
    lines.append("TRADE-OFF ANALYSIS:")
    lines.append("-" * 90)

    best_f1_name = df['Macro F1'].idxmax()
    best_recallR_name = df['Recall(R)'].idxmax()
    best_accuracy_name = df['Accuracy'].idxmax()

    best_f1_val = df.loc[best_f1_name, 'Macro F1']
    best_recallR_val = df.loc[best_recallR_name, 'Recall(R)']
    best_accuracy_val = df.loc[best_accuracy_name, 'Accuracy']

    lines.append(f"\n• Highest Macro F1:  {best_f1_name:20s} ({best_f1_val*100:5.2f}%)")
    lines.append(f"• Highest Recall(R): {best_recallR_name:20s} ({best_recallR_val*100:5.2f}%)")
    lines.append(f"• Highest Accuracy:  {best_accuracy_name:20s} ({best_accuracy_val*100:5.2f}%)")

    lines.append("\nINSIGHTS:")

    if best_f1_name == best_recallR_name and best_f1_name == best_accuracy_name:
        lines.append(
            f"  ✓ {best_f1_name} dominates ALL metrics — clear winner.\n"
            f"    No trade-off needed; optimal across all dimensions."
        )
    else:
        lines.append("  ✓ Trade-offs observed between models:")

        if best_f1_name != best_recallR_name:
            f1_of_recall_model = df.loc[best_recallR_name, 'Macro F1']
            recall_of_f1_model = df.loc[best_f1_name, 'Recall(R)']
            lines.append(f"\n    - {best_recallR_name} has +{(best_recallR_val - recall_of_f1_model)*100:+.1f}% Recall(R)")
            lines.append(f"      but -{(best_f1_val - f1_of_recall_model)*100:+.1f}% Macro F1 vs {best_f1_name}.")
            lines.append(f"      → Higher Recall(R) comes at cost of overall balance (F1).")

        ensemble_names = [n for n in df.index if any(
            keyword in n.lower() for keyword in ['ensemble', 'stacking', 'voting']
        )]
        if ensemble_names:
            ens = ensemble_names[0]
            lines.append(f"\n    - Ensemble ({ens}):")
            for metric in ['Macro F1', 'Recall(R)', 'Accuracy']:
                if ens in df.index:
                    best_single = df.drop(ens)[metric].max()
                    diff = df.loc[ens, metric] - best_single
                    sign = "+" if diff >= 0 else ""
                    lines.append(f"      {metric}: {sign}{diff*100:+.2f}% vs best single model")
            lines.append("      → Ensemble typically improves stability and calibration.")

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


def format_threshold_summary(
    thresholds: Dict[str, float],
    oof_metrics: Dict[str, Dict[str, float]] = None
) -> str:
    """
    Bảng threshold tuning summary.

    Parameters
    ----------
    thresholds : dict {model_name: optimal_threshold}
    oof_metrics : dict {model_name: {'macro_f1': ..., 'recall_R': ...}}

    Returns
    -------
    str : formatted table
    """
    lines = [
        "\n" + "=" * 80,
        " " * 22 + "THRESHOLD TUNING SUMMARY (OOF)",
        "=" * 80,
        f"{'Model':<25} {'Threshold':<12} {'Macro F1':<15} {'Recall(R)':<15}",
        "-" * 80,
    ]

    for name in sorted(thresholds.keys()):
        th = thresholds[name]
        metrics = oof_metrics.get(name, {}) if oof_metrics else {}
        f1 = metrics.get('macro_f1', 0)
        rR = metrics.get('recall_R', 0)
        lines.append(
            f"{name:<25} {th:<12.3f} "
            f"{f1*100:<14.2f}% {rR*100:<14.2f}%"
        )

    lines.append("=" * 80)
    if oof_metrics:
        lines.append("\nNote: OOF = Out-of-Fold predictions from 5-fold CV (no data leakage).")
    return "\n".join(lines)


def format_final_selection_rationale(
    best_name: str,
    cv_df: pd.DataFrame,
    thresholds: Dict[str, float],
    target_recall: float = 0.80
) -> str:
    """
    Giải thích lý do chọn final model dựa trên trade-off analysis.

    Parameters
    ----------
    best_name : str - tên model được chọn
    cv_df : DataFrame với index là model names, columns là metrics (%)
    thresholds : dict {model_name: threshold}
    target_recall : float - target Recall(R) mong đợi

    Returns
    -------
    str : formatted rationale
    """
    lines = [
        "\n" + "=" * 80,
        " " * 20 + "FINAL MODEL SELECTION RATIONALE",
        "=" * 80,
        f"\nSelected Model: {best_name}",
        "-" * 80,
    ]

    best_f1_name = cv_df['Macro F1'].idxmax()
    best_recall_name = cv_df['Recall(R)'].idxmax()
    best_acc_name = cv_df['Accuracy'].idxmax()

    best_f1_val = cv_df.loc[best_name, 'Macro F1']
    best_recall_val = cv_df.loc[best_name, 'Recall(R)']
    th = thresholds.get(best_name, 0.5)

    meets_target = best_recall_val >= target_recall

    lines.append("\nPERFORMANCE PROFILE:")
    lines.append(f"  Macro F1  : {best_f1_val*100:5.2f}% "
                 f"{'(BEST)' if best_name == best_f1_name else f'(best: {cv_df.loc[best_f1_name, 'Macro F1']*100:5.2f}%)'}")
    lines.append(f"  Recall(R) : {best_recall_val*100:5.2f}% "
                 f"{'(BEST)' if best_name == best_recall_name else f'(best: {cv_df.loc[best_recall_name, 'Recall(R)']*100:5.2f}%)'}")
    lines.append(f"  Accuracy  : {cv_df.loc[best_name, 'Accuracy']*100:5.2f}%")
    lines.append(f"  Threshold : {th:.3f}")

    lines.append("\nDECISION LOGIC:")
    if best_name == best_f1_name and best_name == best_recall_name:
        lines.append(
            f"  ✓ {best_name} dominates ALL metrics — undisputed winner.\n"
            f"    No trade-off needed; optimal across all dimensions."
        )
    elif best_name == best_f1_name:
        lines.append(
            f"  ✓ {best_name} has highest Macro F1 (balanced metric).\n"
            f"    Trade-off: {best_recall_name} has +{(cv_df.loc[best_recall_name, 'Recall(R)'] - best_recall_val)*100:+.1f}% Recall(R),\n"
            f"              but -{(best_f1_val - cv_df.loc[best_recall_name, 'Macro F1'])*100:+.1f}% Macro F1.\n"
            f"    Decision: F1 is more comprehensive for imbalanced clinical data.\n"
            f"              Recall trade-off acceptable (threshold tuned to {th:.3f})."
        )
    elif best_name == best_recall_name:
        recall_improvement = best_recall_val - cv_df.loc[best_f1_name, 'Recall(R)']
        f1_drop = best_f1_val - cv_df.loc[best_f1_name, 'Macro F1']
        lines.append(
            f"  ✓ {best_name} selected for clinical priority — Recall(R) >= {target_recall*100:.0f}% is hard constraint.\n"
            f"    {best_name} achieves {best_recall_val*100:.1f}% Recall(R) vs {best_f1_name}'s {cv_df.loc[best_f1_name, 'Recall(R)']*100:.1f}%.\n"
            f"    Cost: -{abs(f1_drop)*100:.1f}% Macro F1, but clinical safety (catching resistant cases) prioritized."
        )
    else:
        lines.append(
            f"  ✓ {best_name} selected despite not having highest F1 or Recall.\n"
            f"    Reason: Balanced trade-off with ensemble stability.\n"
            f"    Consider: Threshold {th:.3f} tuned to optimize clinical metrics."
        )

    if meets_target:
        lines.append(f"\n  ✓ Meets clinical requirement: Recall(R) >= {target_recall*100:.0f}% ✓")
    else:
        lines.append(
            f"\n  ⚠️  Below target Recall(R) {target_recall*100:.0f}% — consider:\n"
            f"     - Lowering threshold further (may increase FP)\n"
            f"     - Adding more resistant samples (data collection)\n"
            f"     - Trying different sampling strategy (SMOTE variant)"
        )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_model_comparison_detailed(
    models_data: Dict[str, Dict[str, Any]]
) -> str:
    """
    Bảng so sánh chi tiết tất cả models với tất cả metrics có thể.

    Parameters
    ----------
    models_data : dict
        {
            model_name: {
                'macro_f1': float,
                'recall_R': float,
                'recall_S': float,
                'accuracy': float,
                'precision_R': float (optional),
                'roc_auc': float (optional),
                'pr_auc': float (optional),
                'threshold': float,
                'train_time': str (optional),
                'n_features': int (optional)
            }
        }

    Returns
    -------
    str : formatted detailed table
    """
    df = pd.DataFrame(models_data).T

    col_order = ['macro_f1', 'recall_R', 'recall_S', 'accuracy',
                 'precision_R', 'roc_auc', 'pr_auc', 'threshold']
    available_cols = [c for c in col_order if c in df.columns]
    df = df[available_cols]

    col_names = {
        'macro_f1': 'Macro F1',
        'recall_R': 'Recall(R)',
        'recall_S': 'Recall(S)',
        'accuracy': 'Accuracy',
        'precision_R': 'Precision(R)',
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'threshold': 'Threshold',
        'train_time': 'Train Time',
        'n_features': 'Features'
    }
    df_display = df.rename(columns={k: v for k, v in col_names.items() if k in df.columns})

    lines = [
        "\n" + "=" * 100,
        " " * 30 + "DETAILED MODEL COMPARISON",
        "=" * 100,
        "\nAll metrics (CV average or Test set):\n",
    ]

    pct_cols = ['Macro F1', 'Recall(R)', 'Recall(S)', 'Accuracy',
                'Precision(R)', 'ROC-AUC', 'PR-AUC']
    df_pct = df_display.copy()
    for col in pct_cols:
        if col in df_pct.columns:
            df_pct[col] = df_pct[col] * 100

    format_dict = {}
    for col in pct_cols:
        if col in df_pct.columns:
            format_dict[col] = lambda x: f"{x:6.2f}%"
    if 'Threshold' in df_pct.columns:
        format_dict['Threshold'] = lambda x: f"{x:.3f}"
    if 'Features' in df_pct.columns:
        format_dict['Features'] = lambda x: f"{int(x):4d}"

    lines.append(df_pct.to_string(formatters=format_dict))

    lines.append("\n" + "=" * 100)
    return "\n".join(lines)
