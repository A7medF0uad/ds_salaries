"""
train_model.py

Trains the leakage-safe salary prediction pipeline (RandomForest + TargetEncoder,
log-transformed target, ordinal experience_level, split-safe rare-category
bundling) on the DS Salaries dataset, and saves everything the Streamlit
dashboard (app.py) needs:

    model/salary_model.pkl        -> fitted model, ready to call .predict()
    model/feature_options.pkl     -> dropdown choices + metadata for the UI
    model/feature_importances.csv -> for the EDA tab's importance chart

Run once locally before launching the dashboard:
    python train_model.py

Requires: data/ds_salaries.csv with columns:
    work_year, experience_level, employment_type, job_title, salary,
    salary_currency, salary_in_usd, employee_residence, remote_ratio,
    company_location
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

DATA_PATH = Path("data/ds_salaries.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

THRESHOLD = 5
EXP_MAPPING = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
EXP_LABELS = {1: "Entry-level (EN)", 2: "Mid-level (MI)", 3: "Senior (SE)", 4: "Executive (EX)"}

NUMERIC_COLS = ["work_year", "remote_ratio", "experience_level"]
CAT_HIGH_COLS = ["job_title", "company_location", "employee_residence"]
CAT_LOW_COLS = ["employment_type"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["experience_level"] = df["experience_level"].map(EXP_MAPPING)
    return df


def bundle_rare(frame: pd.DataFrame, rare_map: dict) -> pd.DataFrame:
    frame = frame.copy()
    for col, rare_items in rare_map.items():
        frame.loc[frame[col].isin(rare_items), col] = "Other"
    return frame


def learn_rare_map(frame: pd.DataFrame) -> dict:
    rare_map = {}
    for col in CAT_HIGH_COLS:
        counts = frame[col].value_counts()
        rare_map[col] = set(counts[counts < THRESHOLD].index)
    return rare_map


def build_pipeline() -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat_high", TargetEncoder(target_type="continuous", random_state=42), CAT_HIGH_COLS),
        ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_LOW_COLS),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def get_feature_importances(fitted_model: TransformedTargetRegressor) -> pd.DataFrame:
    ohe_names = fitted_model.regressor_.named_steps["preprocessor"] \
        .named_transformers_["cat_low"].get_feature_names_out(CAT_LOW_COLS)
    all_names = NUMERIC_COLS + CAT_HIGH_COLS + list(ohe_names)
    importances = fitted_model.regressor_.named_steps["regressor"].feature_importances_
    return (
        pd.DataFrame({"Feature": all_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def main():
    df = load_data()
    X = df.drop(columns=["salary", "salary_currency", "salary_in_usd"])
    y = df["salary_in_usd"].astype(float)

    # --- Honest evaluation split (report-only; the deployed model below is
    #     retrained on all the data once these metrics are recorded) ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    eval_rare_map = learn_rare_map(X_train)
    X_train_b = bundle_rare(X_train, eval_rare_map)
    X_val_b = bundle_rare(X_val, eval_rare_map)
    X_test_b = bundle_rare(X_test, eval_rare_map)

    eval_model = build_pipeline()
    eval_model.fit(X_train_b, y_train)

    val_pred = eval_model.predict(X_val_b)
    test_pred = eval_model.predict(X_test_b)

    print("--- VALIDATION METRICS ---")
    print(f"RMSE: ${root_mean_squared_error(y_val, val_pred):,.2f}")
    print(f"R2:   {r2_score(y_val, val_pred):.4f}")

    print("\n--- TEST METRICS ---")
    test_rmse = root_mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    print(f"RMSE: ${test_rmse:,.2f}")
    print(f"R2:   {test_r2:.4f}")

    # --- Final production model: retrain on ALL data for the deployed dashboard.
    #     The split above already gave an honest, unbiased performance estimate;
    #     retraining on everything squeezes out the extra ~40% of data for the
    #     model that actually ships. ---
    final_rare_map = learn_rare_map(X)
    X_full = bundle_rare(X, final_rare_map)

    final_model = build_pipeline()
    final_model.fit(X_full, y)

    joblib.dump(final_model, MODEL_DIR / "salary_model.pkl")

    importances_df = get_feature_importances(final_model)
    importances_df.to_csv(MODEL_DIR / "feature_importances.csv", index=False)

    feature_options = {
        "job_title": sorted(X_full["job_title"].unique().tolist()),
        "employment_type": sorted(X_full["employment_type"].unique().tolist()),
        "company_location": sorted(X_full["company_location"].unique().tolist()),
        "employee_residence": sorted(X_full["employee_residence"].unique().tolist()),
        "work_year_min": int(X_full["work_year"].min()),
        "work_year_max": int(X_full["work_year"].max()),
        "exp_labels": EXP_LABELS,
        "rare_map": final_rare_map,
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }
    joblib.dump(feature_options, MODEL_DIR / "feature_options.pkl")

    print(f"\nSaved model to               {MODEL_DIR / 'salary_model.pkl'}")
    print(f"Saved feature options to     {MODEL_DIR / 'feature_options.pkl'}")
    print(f"Saved feature importances to {MODEL_DIR / 'feature_importances.csv'}")


if __name__ == "__main__":
    main()
