from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path("data/ds_salaries.csv")
MODEL_PATH = Path("model/salary_model.pkl")
OPTIONS_PATH = Path("model/feature_options.pkl")
IMPORTANCES_PATH = Path("model/feature_importances.csv")

EXP_MAPPING = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
EXP_LABEL_ORDER = ["Entry-level (EN)", "Mid-level (MI)", "Senior (SE)", "Executive (EX)"]

st.set_page_config(page_title="DS Salaries Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["experience_level_num"] = df["experience_level"].map(EXP_MAPPING)
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not OPTIONS_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    options = joblib.load(OPTIONS_PATH)
    return model, options


@st.cache_data
def load_importances() -> pd.DataFrame | None:
    if not IMPORTANCES_PATH.exists():
        return None
    return pd.read_csv(IMPORTANCES_PATH)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("Data Science Salaries Dashboard")
st.caption("Explore the dataset and predict salaries with a leakage-safe RandomForest pipeline.")

if not DATA_PATH.exists():
    st.error(
        f"Couldn't find `{DATA_PATH}`. Place your CSV there before running the app "
        "(see README.md for the expected columns)."
    )
    st.stop()

df = load_data()
model, options = load_model()
importances_df = load_importances()

tab_eda, tab_predict = st.tabs(["Data Explorer", "Salary Predictor"])

# ---------------------------------------------------------------------------
# TAB 1: Data Explorer
# ---------------------------------------------------------------------------
with tab_eda:
    st.sidebar.header("Filters")
    exp_filter = st.sidebar.multiselect(
        "Experience level",
        options=sorted(df["experience_level"].unique()),
        default=sorted(df["experience_level"].unique()),
    )
    year_filter = st.sidebar.multiselect(
        "Work year",
        options=sorted(df["work_year"].unique()),
        default=sorted(df["work_year"].unique()),
    )
    remote_filter = st.sidebar.multiselect(
        "Remote ratio",
        options=sorted(df["remote_ratio"].unique()),
        default=sorted(df["remote_ratio"].unique()),
    )

    filtered = df[
        df["experience_level"].isin(exp_filter)
        & df["work_year"].isin(year_filter)
        & df["remote_ratio"].isin(remote_filter)
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (filtered)", f"{len(filtered):,}")
    col2.metric("Median salary (USD)", f"${filtered['salary_in_usd'].median():,.0f}")
    col3.metric("Mean salary (USD)", f"${filtered['salary_in_usd'].mean():,.0f}")
    col4.metric("Max salary (USD)", f"${filtered['salary_in_usd'].max():,.0f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            filtered, x="salary_in_usd", nbins=40, marginal="box",
            title="Salary Distribution (USD)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            filtered, x="experience_level", y="salary_in_usd",
            category_orders={"experience_level": ["EN", "MI", "SE", "EX"]},
            title="Salary Range by Experience Level",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        remote_avg = filtered.groupby("remote_ratio", as_index=False)["salary_in_usd"].mean()
        fig_remote = px.bar(
            remote_avg, x="remote_ratio", y="salary_in_usd",
            title="Average Salary by Remote Ratio",
            labels={"remote_ratio": "Remote Ratio (%)", "salary_in_usd": "Avg Salary (USD)"},
        )
        st.plotly_chart(fig_remote, use_container_width=True)

    with c4:
        top_titles = (
            filtered.groupby("job_title")["salary_in_usd"]
            .agg(["mean", "count"])
            .query("count >= 5")
            .sort_values("mean", ascending=False)
            .head(10)
            .reset_index()
        )
        fig_titles = px.bar(
            top_titles, x="mean", y="job_title", orientation="h",
            title="Top 10 Highest-Paying Job Titles (min. 5 records)",
            labels={"mean": "Avg Salary (USD)", "job_title": ""},
        )
        fig_titles.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_titles, use_container_width=True)

    if importances_df is not None:
        st.divider()
        st.subheader("Model Feature Importance")
        fig_imp = px.bar(
            importances_df.head(10), x="Importance", y="Feature", orientation="h",
            title="Top 10 Drivers of Salary (from the trained model)",
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("Show raw filtered data"):
        st.dataframe(filtered, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Salary Predictor
# ---------------------------------------------------------------------------
with tab_predict:
    if model is None:
        st.warning(
            "No trained model found. Run `python train_model.py` first — "
            "this saves `model/salary_model.pkl` and `model/feature_options.pkl`, "
            "which this tab needs."
        )
    else:
        st.subheader("Predict a salary")
        st.caption(
            f"Model held-out test performance: RMSE ${options['test_rmse']:,.0f}, "
            f"R² {options['test_r2']:.3f}. Treat predictions as a rough estimate, "
            "not a precise figure."
        )

        left, right = st.columns(2)
        with left:
            job_title = st.selectbox("Job title", options["job_title"])
            employment_type = st.selectbox("Employment type", options["employment_type"])
            exp_label = st.selectbox("Experience level", EXP_LABEL_ORDER, index=2)
            work_year = st.slider(
                "Work year",
                min_value=options["work_year_min"],
                max_value=options["work_year_max"] + 1,
                value=options["work_year_max"],
            )
        with right:
            company_location = st.selectbox("Company location", options["company_location"])
            employee_residence = st.selectbox("Employee residence", options["employee_residence"])
            remote_ratio = st.select_slider("Remote ratio (%)", options=[0, 50, 100], value=100)

        exp_level_num = EXP_LABEL_ORDER.index(exp_label) + 1

        def apply_rare_map(value: str, col: str) -> str:
            return "Other" if value in options["rare_map"].get(col, set()) else value

        if st.button("Predict salary", type="primary"):
            input_row = pd.DataFrame([{
                "work_year": work_year,
                "experience_level": exp_level_num,
                "employment_type": employment_type,
                "job_title": apply_rare_map(job_title, "job_title"),
                "employee_residence": apply_rare_map(employee_residence, "employee_residence"),
                "remote_ratio": remote_ratio,
                "company_location": apply_rare_map(company_location, "company_location"),
            }])

            prediction = model.predict(input_row)[0]
            st.success(f"### Estimated salary: ${prediction:,.0f} / year (USD)")
            st.caption(
                "Estimate only — based on historical patterns in the training data, "
                "with a held-out test R² around "
                f"{options['test_r2']:.2f}, meaning real salaries can vary substantially "
                "from this number."
            )
