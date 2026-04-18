import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _encode_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        encoded = pd.to_numeric(target, errors="coerce")
    else:
        encoded = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        )

    if encoded.isna().any():
        raise ValueError("Target column contains unexpected values after mapping.")

    return encoded.astype(int)


def _build_candidate_pipelines(
    numeric_features: list[str], categorical_features: list[str]
) -> dict[str, Pipeline]:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    logistic_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", logistic_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def train_churn_model(df: pd.DataFrame, target_col: str, metrics_path: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    data = df.copy()
    identifier_columns = [
        column
        for column in data.columns
        if column != target_col
        and column.lower().endswith("id")
        and data[column].nunique(dropna=False) == len(data)
    ]

    y = _encode_target(data[target_col].copy())
    X = data.drop(columns=[target_col, *identifier_columns])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_pipelines = _build_candidate_pipelines(numeric_features, categorical_features)

    best_name = None
    best_pipeline = None
    best_cv_scores = None
    best_score = float("-inf")

    for model_name, candidate in candidate_pipelines.items():
        cv_scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
        )
        mean_score = float(cv_scores.mean())
        if mean_score > best_score:
            best_name = model_name
            best_pipeline = candidate
            best_cv_scores = cv_scores
            best_score = mean_score

    pipeline = best_pipeline
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "task": "classification",
        "selected_model": best_name,
        "auc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "cv_auc_mean": float(best_cv_scores.mean()),
        "cv_auc_std": float(best_cv_scores.std()),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dropped_identifier_columns": identifier_columns,
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return pipeline, X_test, y_test, preds, probs, metrics
