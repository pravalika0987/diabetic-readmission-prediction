import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DB_PATH = "diabetic_db.sqlite"
FEATURE_TABLE = "ml_ready_features"
MODEL_PATH = "best_readmission_model.joblib"
REPORT_PATH = "model_report.txt"


def load_features(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_data(df):
    exclude_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted_binary",
        "readmitted_label",
    ]
    drop_cols = [
        "diag_1_id",
        "diag_2_id",
        "diag_3_id",
        "description_id",
    ]
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols + drop_cols
    ]

    X = df[feature_cols].copy()
    X["visit_to_lab_ratio"] = X["total_visits"] / (X["num_lab_procedures"] + 1)
    X["meds_per_visit"] = X["num_medications"] / (X["total_visits"] + 1)
    X["stay_per_visit"] = X["time_in_hospital"] / (X["total_visits"] + 1)

    y = df["readmitted_binary"].astype(int)
    return X, y


def build_models():
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf_grid = {
        "clf__n_estimators": [100, 150],
        "clf__max_depth": [10, 15],
        "clf__min_samples_split": [2, 5],
    }

    return logistic, rf, rf_grid


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    report = classification_report(y_test, y_pred, digits=4)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    return report, roc_auc, cm


def write_report(report_path, results):
    path = Path(report_path)
    with path.open("w", encoding="utf-8") as f:
        for name, metrics in results.items():
            f.write(f"MODEL: {name}\n")
            f.write("=" * (7 + len(name)) + "\n")
            f.write("Classification Report:\n")
            f.write(metrics["report"] + "\n")
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n" if metrics["roc_auc"] is not None else "ROC AUC: N/A\n")
            f.write("Confusion Matrix:\n")
            f.write(str(metrics["confusion_matrix"]) + "\n")
            f.write("\n")

    return path


def main():
    df = load_features(DB_PATH, FEATURE_TABLE)
    if df.empty:
        raise ValueError("No data found in feature table.")

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic, rf, rf_grid = build_models()

    rf_search = GridSearchCV(
        rf,
        rf_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    logistic.fit(X_train, y_train)

    models = {
        "logistic_regression": logistic,
        "random_forest": best_rf,
    }

    results = {}
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        report, roc_auc, cm = evaluate_model(model, X_test, y_test)
        results[name] = {
            "report": report,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }
        if roc_auc is not None and roc_auc > best_score:
            best_score = roc_auc
            best_model = model

    model_path = Path(MODEL_PATH)
    joblib.dump(best_model, model_path)

    report_path = write_report(REPORT_PATH, results)

    print(f"Training complete. Best model saved to {model_path}")
    print(f"Metrics written to {report_path}")


if __name__ == "__main__":
    main()
