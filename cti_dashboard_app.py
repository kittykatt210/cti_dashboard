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


st.title("🛡️ Financial Industry CTI Risk Intelligence Dashboard")
st.caption("Integrated dashboard for cyber incident classification, financial impact, market impact, and KEV vulnerability intelligence.")

with st.sidebar:
    st.header("Upload CSV Files")
    uploaded_incidents = st.file_uploader("incidents_master.csv", type="csv")
    uploaded_financial = st.file_uploader("financial_impact.csv", type="csv")
    uploaded_market = st.file_uploader("market_impact.csv", type="csv")
    uploaded_kev = st.file_uploader("known_exploited_vulnerabilities.csv", type="csv")
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

tab1, tab2, tab3, tab4 = st.tabs(["Executive Overview", "Threat Drilldown", "KEV Intelligence", "Model Explainability"])

with tab1:
    c1, c2 = st.columns(2)
    risk_counts = filtered[risk_level_col].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    c1.plotly_chart(px.bar(risk_counts, x="risk_level", y="count", title="Incident Count by Risk Level"), use_container_width=True)

    if loss_col:
        loss_by_risk = filtered.groupby(risk_level_col, as_index=False)[loss_col].sum()
        c2.plotly_chart(px.bar(loss_by_risk, x=risk_level_col, y=loss_col, title="Total Financial Loss by Risk Level"), use_container_width=True)

    c3, c4 = st.columns(2)
    if loss_col:
        hover_cols = [c for c in [incident_id, attack_col, data_type_col, downtime_col] if c]
        c3.plotly_chart(px.scatter(filtered, x=risk_score_col, y=loss_col, color=risk_level_col,
                                   size=records_col if records_col else None,
                                   hover_data=hover_cols,
                                   title="CTI Risk Score vs. Financial Loss"), use_container_width=True)

    if date_col and filtered[date_col].notna().any():
        trend = filtered.dropna(subset=[date_col]).groupby([pd.Grouper(key=date_col, freq="ME"), risk_level_col]).size().reset_index(name="count")
        c4.plotly_chart(px.line(trend, x=date_col, y="count", color=risk_level_col, title="Incident Trend Over Time"), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    if attack_col:
        attack_counts = filtered.groupby([attack_col, risk_level_col]).size().reset_index(name="count")
        c1.plotly_chart(px.bar(attack_counts, y=attack_col, x="count", color=risk_level_col, orientation="h", title="Top Attack Vectors by Risk Level"), use_container_width=True)

    if attack_col and downtime_col:
        downtime = filtered.groupby([attack_col, risk_level_col], as_index=False)[downtime_col].mean()
        c2.plotly_chart(px.bar(downtime, x=attack_col, y=downtime_col, color=risk_level_col, title="Average Downtime by Attack Vector"), use_container_width=True)

    if data_type_col and records_col:
        data_records = filtered.groupby([data_type_col, risk_level_col], as_index=False)[records_col].sum()
        st.plotly_chart(px.treemap(data_records, path=[risk_level_col, data_type_col], values=records_col, title="Compromised Records by Data Type"), use_container_width=True)

    st.subheader("Incident Drilldown")
    cols = [c for c in [incident_id, date_col, risk_level_col, risk_score_col, attack_col, data_type_col, records_col, downtime_col, loss_col, "kev_exposure"] if c and c in filtered.columns]
    st.dataframe(filtered[cols], use_container_width=True)

with tab3:
    if kev.empty:
        st.info("Upload the KEV dataset to populate this page.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KEV Records", f"{len(kev):,}")
        c2.metric("KEV-Linked Incidents", f"{filtered['kev_exposure'].sum():,.0f}")
        c3.metric("Avg Risk for KEV Incidents", f"{filtered.loc[filtered['kev_exposure'] == 1, risk_score_col].mean():.3f}" if filtered["kev_exposure"].sum() > 0 else "N/A")
        c4.metric("Total Loss for KEV Incidents", f"${filtered.loc[filtered['kev_exposure'] == 1, loss_col].sum():,.0f}" if loss_col else "N/A")

        vendor_col = first_existing(kev, ["vendorProject", "vendor_project", "vendor", "Vendor"])
        ransomware_col = first_existing(kev, ["knownRansomwareCampaignUse", "known_ransomware_campaign_use"])
        c1, c2 = st.columns(2)
        if vendor_col:
            vendor_counts = kev[vendor_col].value_counts().head(15).reset_index()
            vendor_counts.columns = ["vendor", "count"]
            c1.plotly_chart(px.bar(vendor_counts, y="vendor", x="count", orientation="h", title="Top KEV Vendors / Projects"), use_container_width=True)
        if ransomware_col:
            rw_counts = kev[ransomware_col].value_counts().reset_index()
            rw_counts.columns = ["known_ransomware_campaign_use", "count"]
            c2.plotly_chart(px.pie(rw_counts, names="known_ransomware_campaign_use", values="count", title="Known Ransomware Campaign Use"), use_container_width=True)
        st.subheader("KEV Detail Table")
        st.dataframe(kev, use_container_width=True)

with tab4:
    st.subheader("Model Performance Comparison")
    perf = pd.DataFrame({
        "Model": ["Decision Tree", "Logistic Regression", "Random Forest"],
        "Accuracy": [0.9718, 0.5493, 0.8404],
        "Precision": [0.9720, 0.5475, 0.8433],
        "Recall": [0.9718, 0.5493, 0.8404],
        "F1 Score": [0.9716, 0.5482, 0.8365]
    })
    perf_long = perf.melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(perf_long, x="Model", y="Score", color="Metric", barmode="group", title="Classification Model Evaluation"), use_container_width=True)
    st.dataframe(perf, use_container_width=True)

    st.info("Recommended model: Random Forest. Although the Decision Tree achieved the highest accuracy, its importance was dominated by a small number of variables, suggesting possible overfitting. Random Forest provides a stronger balance between predictive performance and generalizability.")

    if uploaded_importance is not None:
        importance = pd.read_csv(uploaded_importance)
        feature_col = first_existing(importance, ["feature", "Feature"])
        importance_col = first_existing(importance, ["importance", "Importance"])
        if feature_col and importance_col:
            top_imp = importance.sort_values(importance_col, ascending=False).head(15)
            st.plotly_chart(px.bar(top_imp, y=feature_col, x=importance_col, orientation="h", title="Top Feature Importances"), use_container_width=True)
            st.dataframe(top_imp, use_container_width=True)
        else:
            st.warning("feature_importance.csv must contain columns named feature and importance.")
    else:
        st.write("Upload feature_importance.csv to display the feature importance chart.")

st.sidebar.download_button("Download Filtered Dashboard Data", filtered.to_csv(index=False), "filtered_cti_dashboard_data.csv", "text/csv")
