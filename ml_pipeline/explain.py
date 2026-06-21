# ml-pipeline/explain.py
"""SHAP explanations — interpretable predictions for clinicians."""

import numpy as np
import pandas as pd
import shap


def build_shap_explainer(model, X_background: pd.DataFrame):
    """
    Tạo SHAP explainer cho model.

    Parameters
    ----------
    model : trained ImbPipeline
    X_background : ~100 mẫu để làm baseline (shap cần background distribution)

    Returns
    -------
    shap.Explainer
    """
    # Lấy phần classifier từ pipeline (bỏ imputer + var_thresh + rfe + smote)
    # shap cần pipeline hoàn chỉnh để xử lý input giống predict
    explainer = shap.Explainer(model.predict_proba, X_background)
    return explainer


def explain_prediction(
    explainer,
    feature_vector: pd.Series,
    expected_features: list[str],
    top_k: int = 10,
) -> dict:
    """
    Giải thích 1 dự đoán — trả về top features đẩy prediction lên Resistant.

    Returns
    -------
    dict với keys: top_features (list[dict]), base_value, prediction_value
    """
    X = feature_vector.reindex(expected_features).values.reshape(1, -1)
    num_features = X.shape[1]
    shap_values = explainer(X, max_evals=max(2 * num_features + 1, 1000))
    # shap_values.values shape: (1, 2, n_features) → lấy class 1 (Resistant)
    vals = shap_values.values[0, :, 1]
    cols = np.array(expected_features)

    # Top features đẩy prediction lên (SHAP value dương = tăng risk)
    top_idx = np.argsort(vals)[::-1][:top_k]
    top_features = [
        {"feature": str(cols[i]),
         "shap_value": round(float(vals[i]), 4),
         "feature_value": round(float(X[0, i]), 4)}
        for i in top_idx
    ]
    return {
        "top_features": top_features,
        "base_value": round(float(shap_values.base_values[0, 1]), 4),
        "prediction_value": round(float(vals.sum() + shap_values.base_values[0, 1]), 4),
    }