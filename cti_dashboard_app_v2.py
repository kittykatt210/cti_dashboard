# ============================================================
# Financial Industry CTI Dashboard
# Deployment-ready Streamlit app for GitHub / Streamlit Cloud
# ------------------------------------------------------------
# Expected files in the same GitHub repo folder as this app:
#   incidents_master.csv OR incidents_master(2).csv
#   financial_impact.csv OR financial_impact(1).csv
#   market_impact.csv OR market_impact(2).csv
#   known_exploited_vulnerabilities.csv OR known_exploited_vulnerabilities(3).csv
#   regression_model_metrics.csv
#   regression_model_predictions.csv
#   feature_importance.csv
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Financial Industry CTI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent


def find_file(possible_names: Sequence[str], contains: Optional[str] = None) -> Optional[Path]:
    """Find a file in the app directory by exact candidate names or a contains pattern."""
    for name in possible_names:
        candidate = APP_DIR / name
        if candidate.exists():
            return candidate

    if contains:
        matches = sorted(APP_DIR.glob(f"*{contains}*.csv"))
        if matches:
            return matches[0]

    return None


@st.cache_data(show_spinner=False)
def read_csv_file(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_optional_csv(possible_names: Sequence[str], contains: Optional[str] = None) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    path = find_file(possible_names, contains=contains)
    if path is None:
        return None, None
    try:
        return read_csv_file(str(path)), path
    except Exception as exc:
        st.sidebar.error(f"Could not read {path.name}: {exc}")
        return None, path


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.2f}K"
    return f"${value:,.0f}"


def get_time_frequency(label: str) -> str:
    # pandas 2.x safe frequency aliases
    return {
        "Weekly": "W-MON",
        "Monthly": "MS",
        "Quarterly": "QS",
    }.get(label, "MS")


# ------------------------------------------------------------
# Load files automatically from repo
# ------------------------------------------------------------
incidents, incidents_path = load_optional_csv(
    ["incidents_master.csv", "incidents_master(2).csv"], contains="incidents_master"
)
financial, financial_path = load_optional_csv(
    ["financial_impact.csv", "financial_impact(1).csv"], contains="financial_impact"
)
market, market_path = load_optional_csv(
    ["market_impact.csv", "market_impact(2).csv"], contains="market_impact"
)
kev, kev_path = load_optional_csv(
    ["known_exploited_vulnerabilities.csv", "known_exploited_vulnerabilities(3).csv"],
    contains="known_exploited_vulnerabilities",
)
regression_metrics, regression_metrics_path = load_optional_csv(
    ["regression_model_metrics.csv"], contains="regression_model_metrics"
)
regression_predictions, regression_predictions_path = load_optional_csv(
    ["regression_model_predictions.csv"], contains="regression_model_predictions"
)
feature_importance, feature_importance_path = load_optional_csv(
    ["feature_importance.csv"], contains="feature_importance"
)


# ------------------------------------------------------------
# Build dashboard master dataset
# ------------------------------------------------------------
if incidents is None:
    st.error(
        "No incidents_master CSV file was found. Add incidents_master.csv to the same GitHub repo folder as this app."
    )
    st.stop()

master = incidents.copy()

if financial is not None and "incident_id" in master.columns and "incident_id" in financial.columns:
    master = master.merge(financial, on="incident_id", how="left", suffixes=("", "_financial"))

if market is not None and "incident_id" in master.columns and "incident_id" in market.columns:
    master = master.merge(market, on="incident_id", how="left", suffixes=("", "_market"))

# Prefer regression prediction file for prediction-related fields and extra merged columns.
if regression_predictions is not None and "incident_id" in master.columns and "incident_id" in regression_predictions.columns:
    prediction_cols = [
        c
        for c in [
            "incident_id",
            "predicted_total_loss_usd",
            "prediction_error_usd",
            "log_total_loss_usd",
        ]
        if c in regression_predictions.columns
    ]
    if len(prediction_cols) > 1:
        master = master.merge(
            regression_predictions[prediction_cols],
            on="incident_id",
            how="left",
            suffixes=("", "_prediction"),
        )

# Convert date columns safely.
for date_candidate in ["incident_date", "discovery_date", "disclosure_date", "dateAdded", "dueDate"]:
    if date_candidate in master.columns:
        master[date_candidate] = pd.to_datetime(master[date_candidate], errors="coerce")

# Convert likely numeric columns safely.
for col in master.columns:
    if any(token in col.lower() for token in ["usd", "loss", "cost", "fee", "fine", "revenue", "count", "hours", "records", "affected", "return", "volatility", "ratio", "recovery"]):
        if master[col].dtype == "object":
            master[col] = pd.to_numeric(master[col].replace({",": ""}, regex=True), errors="ignore")

# Create a dashboard risk level if one does not already exist.
risk_col = first_existing_col(master, ["risk_level", "Risk Level", "severity", "incident_severity"])
loss_col = first_existing_col(master, ["total_loss_usd", "total_loss", "inflation_adjusted_usd", "direct_loss_usd"])
date_col = first_existing_col(master, ["incident_date", "disclosure_date", "discovery_date"])
attack_col = first_existing_col(master, ["attack_vector_primary", "attack_type", "Attack Type"])
industry_col = first_existing_col(master, ["industry_primary", "industry", "Industry"])
country_col = first_existing_col(master, ["country_hq", "country", "Country"])
company_col = first_existing_col(master, ["company_name", "Company"])

if risk_col is None:
    if loss_col is not None and pd.api.types.is_numeric_dtype(master[loss_col]):
        master["dashboard_risk_level"] = pd.qcut(
            master[loss_col].rank(method="first"),
            q=4,
            labels=["Low", "Medium", "High", "Critical"],
            duplicates="drop",
        )
    else:
        master["dashboard_risk_level"] = "Unknown"
    risk_col = "dashboard_risk_level"

# Ensure risk labels are strings for filters/charts.
master[risk_col] = master[risk_col].astype(str)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("🛡️ CTI Dashboard")
st.sidebar.caption("GitHub / Streamlit Cloud ready")

with st.sidebar.expander("Loaded files", expanded=False):
    loaded_paths = {
        "Incidents": incidents_path,
        "Financial impact": financial_path,
        "Market impact": market_path,
        "Known Exploited Vulnerabilities": kev_path,
        "Regression metrics": regression_metrics_path,
        "Regression predictions": regression_predictions_path,
        "Feature importance": feature_importance_path,
    }
    for label, path in loaded_paths.items():
        st.write(f"**{label}:** {path.name if path else 'Not found'}")

st.sidebar.header("Filters")

time_granularity = st.sidebar.selectbox(
    "Time Granularity",
    ["Monthly", "Weekly", "Quarterly"],
    index=0,
)
time_freq = get_time_frequency(time_granularity)

filtered = master.copy()

if date_col and filtered[date_col].notna().any():
    min_date = filtered[date_col].min().date()
    max_date = filtered[date_col].max().date()
    selected_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        filtered = filtered[
            (filtered[date_col].dt.date >= start_date)
            & (filtered[date_col].dt.date <= end_date)
        ]

if risk_col:
    risk_options = sorted(filtered[risk_col].dropna().astype(str).unique().tolist())
    selected_risks = st.sidebar.multiselect("Risk Level", risk_options, default=risk_options)
    if selected_risks:
        filtered = filtered[filtered[risk_col].astype(str).isin(selected_risks)]

if attack_col:
    attack_options = sorted(master[attack_col].dropna().astype(str).unique().tolist())
    selected_attacks = st.sidebar.multiselect("Attack Vector", attack_options, default=attack_options)
    if selected_attacks:
        filtered = filtered[filtered[attack_col].astype(str).isin(selected_attacks)]

if industry_col:
    industry_options = sorted(master[industry_col].dropna().astype(str).unique().tolist())
    selected_industries = st.sidebar.multiselect("Industry", industry_options, default=industry_options)
    if selected_industries:
        filtered = filtered[filtered[industry_col].astype(str).isin(selected_industries)]

if country_col:
    country_options = sorted(master[country_col].dropna().astype(str).unique().tolist())
    selected_countries = st.sidebar.multiselect("Country", country_options, default=country_options)
    if selected_countries:
        filtered = filtered[filtered[country_col].astype(str).isin(selected_countries)]


# ------------------------------------------------------------
# Dashboard header
# ------------------------------------------------------------
st.title("🛡️ Financial Industry Cyber Threat Intelligence Dashboard")
st.caption(
    "Interactive Streamlit dashboard for cyber incident trends, financial impact, regression results, and model explainability."
)

# KPI section
total_incidents = len(filtered)
total_loss = filtered[loss_col].sum() if loss_col and pd.api.types.is_numeric_dtype(filtered[loss_col]) else None
avg_loss = filtered[loss_col].mean() if loss_col and pd.api.types.is_numeric_dtype(filtered[loss_col]) else None
high_critical_count = filtered[risk_col].str.lower().isin(["high", "critical"]).sum() if risk_col else 0
unique_attacks = filtered[attack_col].nunique() if attack_col else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Incidents", f"{total_incidents:,}")
kpi2.metric("High/Critical", f"{high_critical_count:,}")
kpi3.metric("Attack Types", f"{unique_attacks:,}")
kpi4.metric("Total Loss", format_currency(total_loss))
kpi5.metric("Average Loss", format_currency(avg_loss))


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Threat Trends",
        "🔥 Threat Heatmap",
        "🎯 Risk Analysis",
        "💰 Regression Results",
        "🧠 Feature Importance",
        "🔎 Data Explorer",
    ]
)


# ------------------------------------------------------------
# Tab 1: Threat Trends
# ------------------------------------------------------------
with tab1:
    st.subheader("Threat Volume Over Time")

    if date_col and risk_col and filtered[date_col].notna().any():
        trend = (
            filtered.dropna(subset=[date_col])
            .groupby([pd.Grouper(key=date_col, freq=time_freq), risk_col], observed=False)
            .size()
            .reset_index(name="count")
        )

        if not trend.empty:
            fig_trend = px.line(
                trend,
                x=date_col,
                y="count",
                color=risk_col,
                markers=True,
                title=f"Threat Volume by Risk Level ({time_granularity})",
            )
            fig_trend.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Incidents",
                legend_title="Risk Level",
                hovermode="x unified",
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No trend data is available after filters.")
    else:
        st.warning("A usable date column was not found for trend analysis.")

    c1, c2 = st.columns(2)
    with c1:
        if attack_col:
            top_attacks = filtered[attack_col].astype(str).value_counts().head(10).reset_index()
            top_attacks.columns = ["Attack Vector", "Incidents"]
            fig_attack = px.bar(
                top_attacks,
                x="Incidents",
                y="Attack Vector",
                orientation="h",
                title="Top 10 Attack Vectors",
            )
            fig_attack.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_attack, use_container_width=True)
    with c2:
        if industry_col:
            top_industries = filtered[industry_col].astype(str).value_counts().head(10).reset_index()
            top_industries.columns = ["Industry", "Incidents"]
            fig_industry = px.bar(
                top_industries,
                x="Incidents",
                y="Industry",
                orientation="h",
                title="Top 10 Industries",
            )
            fig_industry.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_industry, use_container_width=True)


# ------------------------------------------------------------
# Tab 2: Threat Heatmap
# ------------------------------------------------------------
with tab2:
    st.subheader("Threat Heatmap")

    heatmap_candidates = [
        col for col in [risk_col, attack_col, industry_col, country_col, "data_type", "attribution_confidence"] if col and col in filtered.columns
    ]

    if len(heatmap_candidates) >= 2:
        heat_col1, heat_col2 = st.columns(2)
        with heat_col1:
            heatmap_y = st.selectbox("Heatmap Rows", heatmap_candidates, index=0)
        with heat_col2:
            default_x_index = 1 if len(heatmap_candidates) > 1 else 0
            heatmap_x = st.selectbox("Heatmap Columns", heatmap_candidates, index=default_x_index)

        if heatmap_x != heatmap_y:
            heatmap_data = (
                filtered.groupby([heatmap_y, heatmap_x], observed=False)
                .size()
                .reset_index(name="count")
            )
            heatmap_pivot = heatmap_data.pivot(index=heatmap_y, columns=heatmap_x, values="count").fillna(0)

            fig_heatmap = px.imshow(
                heatmap_pivot,
                text_auto=True,
                aspect="auto",
                title=f"Threat Concentration: {heatmap_y} by {heatmap_x}",
            )
            fig_heatmap.update_layout(xaxis_title=heatmap_x, yaxis_title=heatmap_y)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Choose two different fields for the heatmap.")
    else:
        st.warning("Not enough categorical columns were found for a heatmap.")


# ------------------------------------------------------------
# Tab 3: Risk Analysis
# ------------------------------------------------------------
with tab3:
    st.subheader("Risk and Financial Impact Analysis")

    c1, c2 = st.columns(2)
    with c1:
        risk_counts = filtered[risk_col].astype(str).value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Incidents"]
        fig_risk = px.pie(
            risk_counts,
            names="Risk Level",
            values="Incidents",
            title="Risk Level Distribution",
            hole=0.45,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        if loss_col and pd.api.types.is_numeric_dtype(filtered[loss_col]):
            fig_loss = px.box(
                filtered,
                x=risk_col,
                y=loss_col,
                title="Financial Loss by Risk Level",
                labels={risk_col: "Risk Level", loss_col: "Financial Loss"},
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("No numeric financial loss column was found.")

    if attack_col and risk_col:
        risk_attack = (
            filtered.groupby([attack_col, risk_col], observed=False)
            .size()
            .reset_index(name="count")
        )
        fig_risk_attack = px.bar(
            risk_attack,
            x=attack_col,
            y="count",
            color=risk_col,
            barmode="stack",
            title="Risk Level by Attack Vector",
        )
        fig_risk_attack.update_layout(xaxis_title="Attack Vector", yaxis_title="Incidents")
        st.plotly_chart(fig_risk_attack, use_container_width=True)


# ------------------------------------------------------------
# Tab 4: Regression Results
# ------------------------------------------------------------
with tab4:
    st.subheader("Regression Model Results")
    st.write(
        "This section displays Linear Regression and Random Forest Regression outputs for predicting total financial loss."
    )

    if regression_metrics is not None:
        st.dataframe(regression_metrics, use_container_width=True)

        metric_cols = [c for c in ["R2", "MAE", "RMSE"] if c in regression_metrics.columns]
        if "Model" in regression_metrics.columns and metric_cols:
            fig_metrics = px.bar(
                regression_metrics,
                x="Model",
                y=metric_cols,
                barmode="group",
                title="Regression Model Evaluation Metrics",
            )
            fig_metrics.update_layout(yaxis_title="Metric Value", legend_title="Metric")
            st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.warning("regression_model_metrics.csv was not found.")

    st.divider()

    if regression_predictions is not None:
        pred_df = regression_predictions.copy()
        if {"total_loss_usd", "predicted_total_loss_usd"}.issubset(pred_df.columns):
            pred_df["total_loss_usd"] = pd.to_numeric(pred_df["total_loss_usd"], errors="coerce")
            pred_df["predicted_total_loss_usd"] = pd.to_numeric(pred_df["predicted_total_loss_usd"], errors="coerce")
            plot_df = pred_df.dropna(subset=["total_loss_usd", "predicted_total_loss_usd"])

            fig_pred = px.scatter(
                plot_df,
                x="total_loss_usd",
                y="predicted_total_loss_usd",
                title="Actual vs Predicted Total Loss",
                labels={
                    "total_loss_usd": "Actual Total Loss USD",
                    "predicted_total_loss_usd": "Predicted Total Loss USD",
                },
                hover_data=[c for c in ["incident_id", "company_name", "attack_vector_primary", "data_type"] if c in plot_df.columns],
            )
            if not plot_df.empty:
                min_val = min(plot_df["total_loss_usd"].min(), plot_df["predicted_total_loss_usd"].min())
                max_val = max(plot_df["total_loss_usd"].max(), plot_df["predicted_total_loss_usd"].max())
                fig_pred.add_shape(
                    type="line",
                    x0=min_val,
                    y0=min_val,
                    x1=max_val,
                    y1=max_val,
                    line=dict(dash="dash"),
                )
            st.plotly_chart(fig_pred, use_container_width=True)

        if "prediction_error_usd" in pred_df.columns:
            pred_df["prediction_error_usd"] = pd.to_numeric(pred_df["prediction_error_usd"], errors="coerce")
            fig_error = px.histogram(
                pred_df.dropna(subset=["prediction_error_usd"]),
                x="prediction_error_usd",
                nbins=30,
                title="Prediction Error Distribution",
            )
            fig_error.update_layout(xaxis_title="Prediction Error USD", yaxis_title="Incident Count")
            st.plotly_chart(fig_error, use_container_width=True)

            st.markdown("#### Largest Absolute Prediction Errors")
            display_cols = [
                c
                for c in [
                    "incident_id",
                    "company_name",
                    "total_loss_usd",
                    "predicted_total_loss_usd",
                    "prediction_error_usd",
                    "attack_vector_primary",
                    "data_type",
                ]
                if c in pred_df.columns
            ]
            error_table = pred_df[display_cols].copy()
            error_table["abs_error"] = error_table["prediction_error_usd"].abs()
            st.dataframe(
                error_table.sort_values("abs_error", ascending=False).drop(columns=["abs_error"]).head(15),
                use_container_width=True,
            )
    else:
        st.warning("regression_model_predictions.csv was not found.")


# ------------------------------------------------------------
# Tab 5: Feature Importance
# ------------------------------------------------------------
with tab5:
    st.subheader("Model Explainability: Feature Importance")

    if feature_importance is not None and {"Feature", "Importance"}.issubset(feature_importance.columns):
        fi = feature_importance.copy()
        fi["Importance"] = pd.to_numeric(fi["Importance"], errors="coerce")
        fi = fi.dropna(subset=["Importance"]).sort_values("Importance", ascending=False)

        top_n = st.slider("Number of features to show", min_value=5, max_value=50, value=20, step=5)
        top_fi = fi.head(top_n).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            top_fi,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Feature Importances",
        )
        fig_fi.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.dataframe(fi.head(100), use_container_width=True)
    else:
        st.warning("feature_importance.csv was not found or does not contain Feature and Importance columns.")


# ------------------------------------------------------------
# Tab 6: Data Explorer
# ------------------------------------------------------------
with tab6:
    st.subheader("Filtered Master Dataset")
    st.write(f"Showing **{len(filtered):,}** records after filters.")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_cti_dashboard_data.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Raw Data Preview")
    source_choice = st.selectbox(
        "Choose source file to preview",
        [
            "Incidents",
            "Financial Impact",
            "Market Impact",
            "Known Exploited Vulnerabilities",
            "Regression Metrics",
            "Regression Predictions",
            "Feature Importance",
        ],
    )

    source_map = {
        "Incidents": incidents,
        "Financial Impact": financial,
        "Market Impact": market,
        "Known Exploited Vulnerabilities": kev,
        "Regression Metrics": regression_metrics,
        "Regression Predictions": regression_predictions,
        "Feature Importance": feature_importance,
    }
    selected_df = source_map.get(source_choice)
    if selected_df is not None:
        st.write(f"Rows: **{selected_df.shape[0]:,}** | Columns: **{selected_df.shape[1]:,}**")
        st.dataframe(selected_df.head(500), use_container_width=True)
    else:
        st.warning("That source file was not found in the app folder.")
