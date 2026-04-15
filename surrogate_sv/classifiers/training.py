from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_binary_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "lda_shrinkage": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "logreg_l2": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=4000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def evaluate_models(
    models: Dict[str, Any],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    source_eval: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]], pd.DataFrame]:
    results: Dict[str, Dict[str, object]] = {}
    metric_rows = []

    for name, model in models.items():
        y_pred = model.predict(X_eval)
        y_score = model.predict_proba(X_eval)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_eval, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
            "f1": float(f1_score(y_eval, y_pred)),
            "roc_auc": _safe_roc_auc(y_eval, y_score),
            "pr_auc": _safe_pr_auc(y_eval, y_score),
            "n_eval": int(len(y_eval)),
        }

        results[name] = {
            "model": model,
            "y_true": y_eval,
            "y_pred": y_pred,
            "y_score": y_score,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=[0, 1]),
        }

        row = {"model": name}
        row.update(metrics)
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    subtype_rows = []
    if source_eval is not None:
        source_eval = np.asarray(source_eval)
        for name, out in results.items():
            yp = np.asarray(out["y_pred"])
            for subtype in sorted(np.unique(source_eval)):
                mask = source_eval == subtype
                n = int(mask.sum())
                if n == 0:
                    continue

                y_sub = y_eval[mask]
                yp_sub = yp[mask]
                if np.all(y_sub == 1):
                    rec_mal = float(recall_score(y_sub, yp_sub, pos_label=1))
                elif np.all(y_sub == 0):
                    rec_mal = float("nan")
                else:
                    rec_mal = float(recall_score(y_sub, yp_sub, pos_label=1))

                subtype_rows.append(
                    {
                        "model": name,
                        "collection_name": subtype,
                        "n_eval": n,
                        "malicious_frac": float(np.mean(y_sub)),
                        "malicious_recall": rec_mal,
                    }
                )

    subtype_df = pd.DataFrame(subtype_rows)
    return metrics_df, results, subtype_df
