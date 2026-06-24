import argparse
import joblib
import os
import pandas as pd
import sqlite3

MODEL_PATH = "best_readmission_model.joblib"
FEATURE_TABLE = "ml_ready_features"
DB_PATH = "diabetic_db.sqlite"


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. Run train_model.py or run_pipeline.py first."
        )
    return joblib.load(path)


def load_sample_data(db_path, table_name, sample_id=None, n=1):
    conn = sqlite3.connect(db_path)
    if sample_id is not None:
        query = f"SELECT * FROM {table_name} WHERE encounter_id = ?"
        df = pd.read_sql_query(query, conn, params=(sample_id,))
    else:
        query = f"SELECT * FROM {table_name} LIMIT {n}"
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_input(df):
    df = df.copy()
    exclude_cols = ["encounter_id", "patient_nbr", "readmitted_binary", "readmitted_label"]
    drop_cols = ["diag_1_id", "diag_2_id", "diag_3_id", "description_id"]
    feature_cols = [col for col in df.columns if col not in exclude_cols + drop_cols]
    X = df[feature_cols].copy()
    X["visit_to_lab_ratio"] = X["total_visits"] / (X["num_lab_procedures"] + 1)
    X["meds_per_visit"] = X["num_medications"] / (X["total_visits"] + 1)
    X["stay_per_visit"] = X["time_in_hospital"] / (X["total_visits"] + 1)
    return X


def main():
    parser = argparse.ArgumentParser(description="Predict readmission from saved model")
    parser.add_argument("--encounter_id", type=int, help="Optional encounter ID to predict")
    parser.add_argument("--limit", type=int, default=1, help="Number of sample rows to predict")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    df = load_sample_data(DB_PATH, FEATURE_TABLE, sample_id=args.encounter_id, n=args.limit)
    if df.empty:
        raise ValueError("No rows found for prediction.")

    X = prepare_input(df)
    predictions = model.predict(X)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]

    for idx, row in df.iterrows():
        encounter_id = row["encounter_id"]
        target = predictions[idx]
        proba = probabilities[idx] if probabilities is not None else None
        print(f"encounter_id={encounter_id}, predicted_readmitted_binary={target}, probability={proba}")


if __name__ == "__main__":
    main()
