import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Stock Cluster & Risk Classifier",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem;}

    .app-banner {
        background: linear-gradient(135deg, #3B8F6F 0%, #4E8AA6 35%, #B8863B 70%, #C2604A 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.4rem;
    }
    .app-banner h1 {color: white; margin: 0 0 6px 0; font-size: 2rem;}
    .app-banner p {color: rgba(255,255,255,0.92); margin: 0; font-size: 0.98rem;}

    div[data-testid="stMetric"] {
        background-color: #F7F7F9;
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        border: 1px solid #ECECEF;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        background-color: #F2F2F5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E8E4FB !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #ECECEF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Risk tiers ordered from lowest to highest risk, mapped to their cluster id
RISK_ORDER = [1, 2, 3, 0]

# ---------------------------------------------------------------------------
# Cluster metadata (derived from analysis of the trained model)
# ---------------------------------------------------------------------------
CLUSTER_INFO = {
    0: {
        "name": "Volatile Value",
        "tagline": "Cheaper, high-swing stocks",
        "risk": "High",
        "risk_color": "#C2604A",
        "description": (
            "Lower share prices with the widest price swings in the dataset. "
            "Earnings per share and cash reserves run high relative to price, "
            "which can mean upside, but the volatility means these names move "
            "fast in both directions."
        ),
    },
    1: {
        "name": "Stable Core",
        "tagline": "The steady, diversified majority",
        "risk": "Low",
        "risk_color": "#3B8F6F",
        "description": (
            "The largest and calmest group in the dataset. Low volatility, "
            "modest returns, and spread across almost every sector. This is "
            "the ballast of a diversified portfolio rather than where the "
            "excitement happens."
        ),
    },
    2: {
        "name": "Mega-Cap Earners",
        "tagline": "Big companies, big income, flat price",
        "risk": "Low-Moderate",
        "risk_color": "#4E8AA6",
        "description": (
            "The biggest balance sheets in the dataset by net income and cash "
            "flow, with a large share count and a low P/E ratio. Price has "
            "been roughly flat recently, typical of large, mature companies "
            "that are cheap relative to their earnings."
        ),
    },
    3: {
        "name": "Premium Growth",
        "tagline": "Expensive, high-return, high-growth",
        "risk": "Moderate-High",
        "risk_color": "#B8863B",
        "description": (
            "The highest share prices, the highest return on equity, and the "
            "highest price gains over the last 13 weeks, but investors are "
            "also paying the richest valuations (P/E and P/B) for that growth."
        ),
    },
}

NUM_COLS = [
    "Current_Price",
    "Price_Change",
    "Volatility",
    "ROE",
    "Cash_Ratio",
    "Net_Cash_Flow",
    "Net_Income",
    "Earnings_Per_Share",
    "Estimated_Shares_Outstanding",
    "P/E_Ratio",
    "P/B_Ratio",
]

METRIC_LABELS = {
    "Current_Price": "Current Price ($)",
    "Price_Change": "13-Week Price Change (%)",
    "Volatility": "Volatility",
    "ROE": "ROE (%)",
    "Cash_Ratio": "Cash Ratio",
    "Net_Cash_Flow": "Net Cash Flow ($)",
    "Net_Income": "Net Income ($)",
    "Earnings_Per_Share": "Earnings Per Share ($)",
    "Estimated_Shares_Outstanding": "Estimated Shares Outstanding",
    "P/E_Ratio": "P/E Ratio",
    "P/B_Ratio": "P/B Ratio",
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.joblib")
    kmeans = joblib.load("kmeans_model.joblib")
    with open("outlier_caps.json") as f:
        caps = json.load(f)
    return scaler, kmeans, caps


@st.cache_data
def load_data():
    df = pd.read_csv("stock_data_with_clusters.csv")
    return df


def format_metric(key, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if key == "Current_Price":
        return f"${value:,.2f}"
    if key in ("Price_Change", "ROE"):
        return f"{value:,.1f}%"
    if key in ("Net_Cash_Flow", "Net_Income"):
        if abs(value) >= 1e9:
            return f"${value / 1e9:,.2f}B"
        return f"${value / 1e6:,.1f}M"
    if key == "Estimated_Shares_Outstanding":
        if value >= 1e9:
            return f"{value / 1e9:,.2f}B"
        return f"{value / 1e6:,.1f}M"
    return f"{value:,.2f}"


def apply_outlier_caps(row, caps):
    capped = row.copy()
    for col in NUM_COLS:
        lower = caps[col]["lower"]
        upper = caps[col]["upper"]
        capped[col] = min(max(row[col], lower), upper)
    return capped


def predict_cluster(row, scaler, kmeans, caps):
    capped = apply_outlier_caps(row, caps)
    features = np.array([[capped[c] for c in NUM_COLS]])
    scaled = scaler.transform(features)
    cluster = int(kmeans.predict(scaled)[0])
    return cluster


def build_comparison_chart(company_row, cluster_id, df):
    cluster_median = df[df["Cluster"] == cluster_id][NUM_COLS].median()
    overall_median = df[NUM_COLS].median()

    labels = [METRIC_LABELS[c] for c in NUM_COLS]
    company_vals = []
    cluster_vals = []
    overall_vals = []

    # Normalize each metric to a 0-100 scale so all metrics fit on one chart
    for c in NUM_COLS:
        max_abs = max(
            abs(company_row[c]), abs(cluster_median[c]), abs(overall_median[c]), 1e-9
        )
        company_vals.append(100 * abs(company_row[c]) / max_abs)
        cluster_vals.append(100 * abs(cluster_median[c]) / max_abs)
        overall_vals.append(100 * abs(overall_median[c]) / max_abs)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=company_vals,
            name="Selected Company",
            orientation="h",
            marker_color=CLUSTER_INFO[cluster_id]["risk_color"],
        )
    )
    fig.add_trace(
        go.Scatter(
            y=labels,
            x=cluster_vals,
            name="Cluster Median",
            mode="markers",
            marker=dict(color="#8A97A5", size=10, symbol="line-ns", line=dict(width=2)),
        )
    )
    fig.update_layout(
        barmode="overlay",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Relative scale (normalized per metric)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def render_risk_meter(current_cluster_id):
    """
    Renders a horizontal strip showing all 4 risk tiers (low to high),
    with the tier that matches the given cluster highlighted.
    """
    pills = []
    for cid in RISK_ORDER:
        info = CLUSTER_INFO[cid]
        is_active = cid == current_cluster_id
        if is_active:
            style = (
                f"background-color:{info['risk_color']}; color:white; "
                f"border:2px solid {info['risk_color']}; transform: scale(1.06); "
                f"box-shadow: 0 2px 10px {info['risk_color']}55;"
            )
            marker = "&#9679; "
        else:
            style = (
                f"background-color:transparent; color:{info['risk_color']}; "
                f"border:2px solid {info['risk_color']}55; opacity:0.55;"
            )
            marker = ""
        pills.append(
            f"""<div style='{style} border-radius:10px; padding:10px 6px;
            text-align:center; flex:1; font-weight:600; font-size:13.5px;'>
            {marker}{info['risk']} RISK</div>"""
        )

    st.markdown(
        f"""
        <div style='display:flex; gap:10px; margin: 6px 0 4px 0;'>
        {''.join(pills)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_info = CLUSTER_INFO[current_cluster_id]
    st.markdown(
        f"<div style='text-align:center; margin-top:6px; font-size:13px; color:#666;'>"
        f"This stock falls under <b style='color:{active_info['risk_color']}'>"
        f"{active_info['risk']} risk</b> — {active_info['name']} cluster</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Model:** K-Means Clustering (k = 4)
        **Preprocessing:** IQR outlier capping + StandardScaler
        **Trained on:** 513 NYSE-listed companies
        **Indicators used:** 11 financial and price-based metrics
        """
    )
    st.divider()

    st.subheader("The 4 Clusters")
    for cid, info in CLUSTER_INFO.items():
        st.markdown(
            f"<span style='color:{info['risk_color']}; font-size:18px;'>&#9679;</span> "
            f"**{info['name']}** — {info['risk']} risk",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "This tool classifies stocks using patterns learned from historical "
        "data. It is for educational purposes only and is not investment "
        "advice."
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-banner">
        <h1>Stock Cluster & Risk Classifier</h1>
        <p>Select a company from the trained dataset to see which risk cluster it
        belongs to and how its financials compare to its peer group.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scaler, kmeans, caps = load_model()
df = load_data()

df["display_label"] = df["Ticker_Symbol"] + " - " + df["Security"]

tab_lookup, tab_browse = st.tabs(["Look Up a Company", "Browse All Clusters"])

# ---------------------------------------------------------------------------
# Tab 1: Look up a single company
# ---------------------------------------------------------------------------
with tab_lookup:
    search_col, _ = st.columns([2, 1])
    with search_col:
        selected_label = st.selectbox(
            "Search by ticker or company name",
            options=sorted(df["display_label"].tolist()),
            index=None,
            placeholder="Start typing a ticker or company name...",
        )

    if selected_label:
        row = df[df["display_label"] == selected_label].iloc[0]
        cluster_id = int(row["Cluster"])
        info = CLUSTER_INFO[cluster_id]

        st.divider()

        header_col, badge_col = st.columns([3, 1])
        with header_col:
            st.subheader(f"{row['Ticker_Symbol']} - {row['Security']}")
            st.markdown(f"Sector: {row['GICS_Sector']} | Sub-Industry: {row['GICS_Sub_Industry']}")
        with badge_col:
            st.markdown(
                f"""
                <div style='text-align:right;'>
                    <span style='background-color:{info['risk_color']}22;
                    color:{info['risk_color']}; padding:6px 14px; border-radius:20px;
                    font-weight:600; border:1px solid {info['risk_color']};'>
                    {info['risk']} RISK
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f"### Cluster: {info['name']}")
        st.markdown(f"*{info['tagline']}*")
        st.write(info["description"])

        st.markdown("#### Risk Classification")
        render_risk_meter(cluster_id)

        st.markdown("#### Key Metrics")
        metric_cols = st.columns(4)
        display_metrics = ["Current_Price", "Price_Change", "ROE", "P/E_Ratio"]
        for col, metric in zip(metric_cols, display_metrics):
            col.metric(METRIC_LABELS[metric], format_metric(metric, row[metric]))

        st.markdown("#### Company vs. Cluster Median")
        st.plotly_chart(
            build_comparison_chart(row, cluster_id, df),
            use_container_width=True,
        )

        st.markdown("#### Other Companies in This Cluster")
        peers = df[(df["Cluster"] == cluster_id) & (df["Ticker_Symbol"] != row["Ticker_Symbol"])]
        peers_sample = peers.sample(min(10, len(peers)), random_state=None)
        st.dataframe(
            peers_sample[["Ticker_Symbol", "Security", "GICS_Sector"]].rename(
                columns={
                    "Ticker_Symbol": "Ticker",
                    "Security": "Company",
                    "GICS_Sector": "Sector",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Select a company above to see its cluster classification.")

# ---------------------------------------------------------------------------
# Tab 2: Browse all clusters
# ---------------------------------------------------------------------------
with tab_browse:
    for cid, info in CLUSTER_INFO.items():
        cluster_df = df[df["Cluster"] == cid]
        with st.expander(
            f"Cluster {cid}: {info['name']} ({len(cluster_df)} companies) - {info['risk']} risk"
        ):
            st.write(info["description"])
            st.markdown("**Top sectors in this cluster:**")
            sector_counts = cluster_df["GICS_Sector"].value_counts().head(5)
            st.bar_chart(sector_counts)
            st.markdown("**Companies:**")
            st.dataframe(
                cluster_df[["Ticker_Symbol", "Security", "GICS_Sector"]]
                .sort_values("Security")
                .rename(
                    columns={
                        "Ticker_Symbol": "Ticker",
                        "Security": "Company",
                        "GICS_Sector": "Sector",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                height=300,
            )
