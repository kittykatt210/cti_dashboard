# CTI Risk Intelligence Dashboard
# Save as cti_dashboard_app.py
# Run with: streamlit run cti_dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="CTI Risk Intelligence Dashboard", page_icon="🛡️", layout="wide")


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    if s.max() == s.min():
        return pd.Series(0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def make_risk_level(score):
    try:
        return pd.qcut(score.rank(method="first"), q=4, labels=["Low", "Medium", "High", "Critical"])
    except Exception:
        return pd.cut(score, bins=[-np.inf, .25, .5, .75, np.inf], labels=["Low", "Medium", "High", "Critical"])

def get_time_frequency(label):
    mapping = {
        "Monthly": "MS",
        "Weekly": "W-MON",
        "Quarterly": "QS"
    }
    return mapping.get(label, "MS")


time_granularity = st.sidebar.selectbox(
    "Time Granularity",
    ["Monthly", "Weekly", "Quarterly"],
    index=0
)

time_freq = get_time_frequency(time_granularity)


st.title("🛡️ Financial Industry CTI Risk Intelligence Dashboard")
st.caption("Integrated dashboard for cyber incident classification, financial impact, market impact, and KEV vulnerability intelligence.")

with st.sidebar:
    st.header("Upload CSV Files")
    uploaded_incidents = st.file_uploader("incidents_master.csv", type="csv")
    uploaded_financial = st.file_uploader("financial_impact.csv", type="csv")
    uploaded_market = st.file_uploader("market_impact.csv", type="csv")
    uploaded_kev = st.file_uploader("known_exploited_vulnerabilities.csv", type="csv")
    uploaded_reg_model_metrics = st.file_uploader("regression_model_metrics.csv", type="csv")
    uploaded_reg_model_predictions = st.file_uploader("regression_model_predictions.csv", type="csv")
    uploaded_importance = st.file_uploader("Optional: feature_importance.csv", type="csv")

if uploaded_incidents is None:
    st.warning("Upload incidents_master.csv to begin.")
    st.stop()

incidents = pd.read_csv(uploaded_incidents)
financial = pd.read_csv(uploaded_financial) if uploaded_financial is not None else pd.DataFrame()
market = pd.read_csv(uploaded_market) if uploaded_market is not None else pd.DataFrame()
kev = pd.read_csv(uploaded_kev) if uploaded_kev is not None else pd.DataFrame()

df = incidents.copy()
incident_id = first_existing(df, ["incident_id", "Incident ID", "id", "ID"])

if incident_id and not financial.empty and incident_id in financial.columns:
    df = df.merge(financial, on=incident_id, how="left", suffixes=("", "_financial"))

if incident_id and not market.empty and incident_id in market.columns:
    df = df.merge(market, on=incident_id, how="left", suffixes=("", "_market"))

# KEV enrichment
incident_cve = first_existing(df, ["cve_id", "cveID", "CVE", "cve"])
kev_cve = first_existing(kev, ["cveID", "cve_id", "CVE", "cve"]) if not kev.empty else None

if incident_cve and kev_cve:
    kev_join = kev[[kev_cve]].drop_duplicates().copy()
    kev_join["kev_exposure"] = 1
    df = df.merge(kev_join, left_on=incident_cve, right_on=kev_cve, how="left")
    df["kev_exposure"] = df["kev_exposure"].fillna(0).astype(int)
else:
    df["kev_exposure"] = 0

# Detect key columns
risk_score_col = first_existing(df, ["cti_risk_score", "risk_score"])
risk_level_col = first_existing(df, ["risk_level", "Risk Level"])
records_col = first_existing(df, ["data_compromised_records", "records_compromised", "compromised_records"])
downtime_col = first_existing(df, ["downtime_hours", "downtime", "outage_hours"])
loss_col = first_existing(df, ["total_loss_usd", "total_loss", "inflation_adjusted_usd", "direct_loss_usd"])
attack_col = first_existing(df, ["attack_vector", "attack_type", "incident_type", "vector"])
data_type_col = first_existing(df, ["data_type", "Data Type"])
date_col = first_existing(df, ["incident_date", "date", "disclosure_date", "Date"])

# Create CTI score if not already present
if not risk_score_col:
    incident_score = 0
    if records_col:
        incident_score += 0.5 * normalize(df[records_col])
    if downtime_col:
        incident_score += 0.5 * normalize(df[downtime_col])
    financial_score = normalize(df[loss_col]) if loss_col else 0
    market_metric = first_existing(df, ["abnormal_return", "stock_price_change", "volatility_change"])
    market_score = normalize(df[market_metric]) if market_metric else 0
    df["cti_risk_score"] = 0.30 * incident_score + 0.25 * financial_score + 0.20 * market_score + 0.25 * df["kev_exposure"]
    risk_score_col = "cti_risk_score"

if not risk_level_col:
    df["risk_level"] = make_risk_level(df[risk_score_col])
    risk_level_col = "risk_level"

df[risk_level_col] = df[risk_level_col].astype(str)
df[risk_score_col] = pd.to_numeric(df[risk_score_col], errors="coerce").fillna(0)

for c in [records_col, downtime_col, loss_col]:
    if c:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Filters
with st.sidebar:
    st.header("Filters")
    risk_options = sorted(df[risk_level_col].dropna().unique().tolist())
    selected_risks = st.multiselect("Risk Level", risk_options, default=risk_options)

    selected_attacks = None
    if attack_col:
        attack_options = sorted(df[attack_col].dropna().astype(str).unique().tolist())
        selected_attacks = st.multiselect("Attack Vector / Type", attack_options, default=attack_options)

    selected_data = None
    if data_type_col:
        data_options = sorted(df[data_type_col].dropna().astype(str).unique().tolist())
        selected_data = st.multiselect("Data Type", data_options, default=data_options)

filtered = df[df[risk_level_col].isin(selected_risks)].copy()
if attack_col and selected_attacks:
    filtered = filtered[filtered[attack_col].astype(str).isin(selected_attacks)]
if data_type_col and selected_data:
    filtered = filtered[filtered[data_type_col].astype(str).isin(selected_data)]

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Incidents", f"{len(filtered):,}")
k2.metric("Avg CTI Risk Score", f"{filtered[risk_score_col].mean():.3f}")
k3.metric("Critical Incidents", f"{(filtered[risk_level_col] == 'Critical').sum():,}")
k4.metric("Total Financial Loss", f"${filtered[loss_col].sum():,.0f}" if loss_col else "N/A")
k5.metric("Compromised Records", f"{filtered[records_col].sum():,.0f}" if records_col else "N/A")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Threat Trends",
    "🔥 Threat Heatmap",
    "🎯 Risk Analysis",
    "🤖 Classification Results",
    "💰 Regression Results",
    "🔎 Data Explorer"
])

with tab1:
    st.subheader("Threat Volume Over Time")
    pd.Grouper(key=date_col, freq=time_freq)
    if date_col in filtered.columns and risk_level_col in filtered.columns:
        trend = (
            filtered
            .dropna(subset=[date_col])
            .groupby(
                [
                    pd.Grouper(key=date_col, freq=time_freq),
                    risk_level_col
                ],
                observed=False
            )
            .size()
            .reset_index(name="count")
        )

        if not trend.empty:
            fig_trend = px.line(
                trend,
                x=date_col,
                y="count",
                color=risk_level_col,
                markers=True,
                title=f"Threat Volume Over Time by Risk Level ({time_granularity})"
            )

            fig_trend.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Incidents",
                legend_title="Risk Level",
                hovermode="x unified"
            )

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No trend data available for the selected filters.")
    else:
        st.warning("Date and risk level columns are required for this chart.")

    st.divider()

    st.subheader("Top Attack Types")

    if attack_type_col in filtered.columns:
        attack_counts = (
            filtered[attack_type_col]
            .value_counts()
            .reset_index()
        )
        attack_counts.columns = [attack_type_col, "count"]

        fig_attack = px.bar(
            attack_counts.head(10),
            x="count",
            y=attack_type_col,
            orientation="h",
            title="Top 10 Attack Types"
        )

        fig_attack.update_layout(
            xaxis_title="Number of Incidents",
            yaxis_title="Attack Type",
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig_attack, use_container_width=True)


# ============================================================
# TAB 2: Threat Heatmap
# ============================================================
with tab2:
    st.subheader("Threat Heatmap")

    heatmap_x = st.selectbox(
        "Heatmap X-Axis",
        [col for col in [attack_type_col, industry_col, country_col] if col in filtered.columns],
        index=0
    )

    heatmap_y = st.selectbox(
        "Heatmap Y-Axis",
        [col for col in [risk_level_col, industry_col, country_col] if col in filtered.columns],
        index=0
    )

    if heatmap_x != heatmap_y:
        heatmap_data = (
            filtered
            .groupby([heatmap_y, heatmap_x], observed=False)
            .size()
            .reset_index(name="count")
        )

        heatmap_pivot = heatmap_data.pivot(
            index=heatmap_y,
            columns=heatmap_x,
            values="count"
        ).fillna(0)

        fig_heatmap = px.imshow(
            heatmap_pivot,
            text_auto=True,
            aspect="auto",
            title=f"Threat Concentration: {heatmap_y} by {heatmap_x}"
        )

        fig_heatmap.update_layout(
            xaxis_title=heatmap_x,
            yaxis_title=heatmap_y
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Choose two different variables for the heatmap.")


# ============================================================
# TAB 3: Risk Analysis
# ============================================================
with tab3:
    st.subheader("Risk Distribution")

    if risk_level_col in filtered.columns:
        risk_counts = (
            filtered[risk_level_col]
            .value_counts()
            .reset_index()
        )
        risk_counts.columns = [risk_level_col, "count"]

        fig_risk = px.pie(
            risk_counts,
            names=risk_level_col,
            values="count",
            title="Risk Level Distribution",
            hole=0.45
        )

        st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()

    st.subheader("Risk by Attack Type")

    if risk_level_col in filtered.columns and attack_type_col in filtered.columns:
        risk_attack = (
            filtered
            .groupby([attack_type_col, risk_level_col], observed=False)
            .size()
            .reset_index(name="count")
        )

        fig_risk_attack = px.bar(
            risk_attack,
            x=attack_type_col,
            y="count",
            color=risk_level_col,
            title="Risk Level by Attack Type",
            barmode="stack"
        )

        fig_risk_attack.update_layout(
            xaxis_title="Attack Type",
            yaxis_title="Number of Incidents",
            legend_title="Risk Level"
        )

        st.plotly_chart(fig_risk_attack, use_container_width=True)


# ============================================================
# TAB 4: Model Results
# ============================================================
with tab4:
    st.subheader("Classification Model Summary")

    st.markdown("""
    This section is designed for the classification portion of the artifact.
    The dashboard can be used to present model results such as accuracy, precision,
    recall, F1-score, and confusion matrix values.
    """)

    model_results = pd.DataFrame({
        "Model": ["Decision Tree", "Logistic Regression", "Random Forest"],
        "Accuracy": [0.84, 0.79, 0.87],
        "Precision": [0.82, 0.77, 0.86],
        "Recall": [0.81, 0.75, 0.85],
        "F1-Score": [0.815, 0.760, 0.855]
    })

    st.dataframe(model_results, use_container_width=True)

    fig_model = px.bar(
        model_results,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Classification Model Evaluation"
    )

    fig_model.update_layout(
        yaxis_title="Score",
        xaxis_title="Model",
        legend_title="Metric"
    )

    st.plotly_chart(fig_model, use_container_width=True)

    st.info(
        "Replace the sample model scores above with the actual evaluation results "
        "from your trained classification models."
    )

# ============================================================
# TAB 5: Regression Results
# ============================================================
with tab5:
    st.subheader("Regression Model Results")

    st.markdown("""
    This section compares the performance of the Linear Regression and Random Forest Regression models.
    The regression models predict estimated financial loss from incident, financial, market, and KEV-related features.
    """)

    try:
        regression_metrics = pd.read_csv("regression_model_metrics.csv")

        st.dataframe(regression_metrics, use_container_width=True)

        fig_reg_metrics = px.bar(
            regression_metrics,
            x="Model",
            y=["R2", "MAE", "RMSE"],
            barmode="group",
            title="Regression Model Evaluation Metrics"
        )

        fig_reg_metrics.update_layout(
            xaxis_title="Model",
            yaxis_title="Metric Value",
            legend_title="Metric"
        )

        st.plotly_chart(fig_reg_metrics, use_container_width=True)

    except FileNotFoundError:
        st.warning("regression_model_metrics.csv was not found. Run the regression script first.")

    st.divider()

    st.subheader("Actual vs Predicted Financial Loss")

    try:
        regression_predictions = pd.read_csv("regression_model_predictions.csv")

        if {
            "total_loss_usd",
            "predicted_total_loss_usd"
        }.issubset(regression_predictions.columns):

            fig_actual_pred = px.scatter(
                regression_predictions,
                x="total_loss_usd",
                y="predicted_total_loss_usd",
                title="Actual vs Predicted Total Loss",
                labels={
                    "total_loss_usd": "Actual Total Loss USD",
                    "predicted_total_loss_usd": "Predicted Total Loss USD"
                },
                hover_data=[
                    col for col in [
                        "incident_id",
                        "attack_vector_primary",
                        "data_type",
                        "systems_affected",
                        "downtime_hours"
                    ]
                    if col in regression_predictions.columns
                ]
            )

            fig_actual_pred.add_shape(
                type="line",
                x0=regression_predictions["total_loss_usd"].min(),
                y0=regression_predictions["total_loss_usd"].min(),
                x1=regression_predictions["total_loss_usd"].max(),
                y1=regression_predictions["total_loss_usd"].max(),
                line=dict(dash="dash")
            )

            st.plotly_chart(fig_actual_pred, use_container_width=True)

        else:
            st.warning("Prediction file must contain total_loss_usd and predicted_total_loss_usd.")

    except FileNotFoundError:
        st.warning("regression_model_predictions.csv was not found. Run the regression script first.")

    st.divider()

    st.subheader("Prediction Error Analysis")

    try:
        regression_predictions = pd.read_csv("regression_model_predictions.csv")

        if "prediction_error_usd" in regression_predictions.columns:
            fig_error = px.histogram(
                regression_predictions,
                x="prediction_error_usd",
                nbins=30,
                title="Distribution of Prediction Errors"
            )

            fig_error.update_layout(
                xaxis_title="Prediction Error USD",
                yaxis_title="Number of Incidents"
            )

            st.plotly_chart(fig_error, use_container_width=True)

            st.dataframe(
                regression_predictions[
                    [
                        col for col in [
                            "incident_id",
                            "total_loss_usd",
                            "predicted_total_loss_usd",
                            "prediction_error_usd",
                            "attack_vector_primary",
                            "data_type"
                        ]
                        if col in regression_predictions.columns
                    ]
                ].sort_values(
                    by="prediction_error_usd",
                    key=lambda x: abs(x),
                    ascending=False
                ).head(10),
                use_container_width=True
            )

    except FileNotFoundError:
        st.warning("regression_model_predictions.csv was not found. Run the regression script first.")
        
# ============================================================
# TAB 6: Data Explorer
# ============================================================
with tab6:
    st.subheader("Filtered Dataset")

    st.write(f"Showing **{len(filtered):,}** records after filters.")

    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_cti_data.csv",
        mime="text/csv"
    )
