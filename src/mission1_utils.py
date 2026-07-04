"""Mission 1 helper functions for the fairness lab."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.fairness_lab import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_STATE,
)


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numerical and categorical columns."""

    numeric_columns = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in feature_frame.columns
        if column not in numeric_columns
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def build_logistic_regression_model(feature_frame: pd.DataFrame) -> Pipeline:
    """Build the baseline preprocessing plus logistic regression pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_frame)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def base_rate_metric(y_true, y_pred):
    """Observed positive-class rate in the true labels."""

    return np.mean(y_true)


def ppv_metric(y_true, y_pred):
    """Positive predictive value, also called precision."""

    return precision_score(
        y_true,
        y_pred,
        pos_label=POSITIVE_LABEL,
        zero_division=0,
    )


def confusion_counts_by_group(y_true, y_pred, sensitive_features) -> pd.DataFrame:
    """Return TN, FP, FN, TP counts for each sensitive group."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    sensitive_array = pd.Series(sensitive_features).to_numpy()

    rows = []

    for group in sorted(pd.Series(sensitive_array).unique()):
        mask = sensitive_array == group

        tn, fp, fn, tp = confusion_matrix(
            y_true_array[mask],
            y_pred_array[mask],
            labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ).ravel()

        rows.append(
            {
                "group": group,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    return pd.DataFrame(rows)


def audit_model(
    model_name: str,
    y_true,
    y_pred,
    sensitive_features,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute group metrics and overall fairness metrics."""

    metrics = {
        "n": count,
        "base_rate": base_rate_metric,
        "selection_rate": selection_rate,
        "TPR": true_positive_rate,
        "FPR": false_positive_rate,
        "FNR": false_negative_rate,
        "PPV": ppv_metric,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    group_metrics = metric_frame.by_group.copy()
    group_metrics.index.name = "group"
    group_metrics = group_metrics.reset_index()

    confusion_counts = confusion_counts_by_group(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    group_metrics = group_metrics.merge(
        confusion_counts,
        on="group",
        how="left",
    )

    group_metrics.insert(0, "model", model_name)

    summary_metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "selection_rate": selection_rate(y_true, y_pred),
                "DP_difference": demographic_parity_difference(
                    y_true,
                    y_pred,
                    sensitive_features=sensitive_features,
                ),
                "EO_difference": equalized_odds_difference(
                    y_true,
                    y_pred,
                    sensitive_features=sensitive_features,
                ),
            }
        ]
    )

    return group_metrics, summary_metrics
