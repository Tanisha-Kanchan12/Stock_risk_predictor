import json
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Stock Cluster & Risk Classifier - Live",
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
    with open("reference_stats.json") as f:
        ref_stats = json.load(f)
    return scaler, kmeans, caps, ref_stats


@st.cache_data
def load_ticker_universe():
    df = pd.read_csv("stock_data_with_clusters.csv")
    df["display_label"] = df["Ticker_Symbol"] + " - " + df["Security"]
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


def find_statement_value(statement_df, keywords):
    """
    Searches a yfinance financial statement dataframe (balance sheet or cash
    flow) for a row whose label contains any of the given keywords, and
    returns the most recent (first) column's value. Returns None if not found.
    """
    if statement_df is None or statement_df.empty:
        return None
    for label in statement_df.index:
        label_lower = str(label).lower()
        if any(k.lower() in label_lower for k in keywords):
            row = statement_df.loc[label]
            row = row.dropna()
            if len(row) > 0:
                return float(row.iloc[0])
    return None


# ---------------------------------------------------------------------------
# Live data fetch
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_usd_conversion_rate(currency_code):
    """
    Fetches the current exchange rate to convert an amount in the given
    currency into USD. Returns 1.0 for USD or if the rate cannot be found
    (in which case a warning should be shown to the user separately).
    """
    if currency_code in (None, "USD"):
        return 1.0, True
    try:
        fx_ticker = yf.Ticker(f"{currency_code}=X")
        fx_hist = fx_ticker.history(period="5d")
        if fx_hist is None or fx_hist.empty:
            return 1.0, False
        native_per_usd = float(fx_hist["Close"].iloc[-1])
        if native_per_usd == 0:
            return 1.0, False
        return 1.0 / native_per_usd, True
    except Exception:
        return 1.0, False


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_features(ticker_symbol):
    """
    Fetches live/recent data for a ticker from Yahoo Finance and computes the
    same 11 features the model was trained on. Returns a dict of feature
    values (some may be NaN if not available) and a list of warnings for any
    feature that could not be fetched.

    All monetary and price-based fields are converted to USD, since the
    model was trained on USD-normalized data. Ratio/percentage fields
    (ROE, Cash_Ratio, P/E_Ratio, P/B_Ratio, Price_Change) are currency-
    neutral and are left as-is.
    """
    warnings = []
    values = {}

    t = yf.Ticker(ticker_symbol)

    try:
        info = t.info
    except Exception:
        info = {}

    currency = info.get("currency", "USD")
    fx_rate, fx_ok = get_usd_conversion_rate(currency)
    if currency != "USD" and not fx_ok:
        warnings.append(
            f"currency conversion ({currency} to USD rate unavailable, "
            f"values may be in {currency}, not USD)"
        )

    # --- Price history based features: Current_Price, Price_Change, Volatility ---
    hist = t.history(period="6mo")
    if hist is None or hist.empty:
        raise ValueError(f"No price history found for ticker '{ticker_symbol}'.")

    hist = hist.tail(200)
    current_price_native = float(hist["Close"].iloc[-1])

    cutoff_date = hist.index[-1] - timedelta(weeks=13)
    window = hist[hist.index >= cutoff_date]
    if len(window) < 2:
        window = hist.tail(65)

    price_13w_ago_native = float(window["Close"].iloc[0])
    price_change = ((current_price_native - price_13w_ago_native) / price_13w_ago_native) * 100
    volatility_native = float(window["Close"].std())

    # Convert price-based fields to USD to match the training data's scale
    values["Current_Price"] = current_price_native * fx_rate
    values["Price_Change"] = price_change  # percentage, currency-neutral
    values["Volatility"] = volatility_native * fx_rate

    roe = info.get("returnOnEquity")
    values["ROE"] = roe * 100 if roe is not None else np.nan
    if roe is None:
        warnings.append("ROE")

    eps_native = info.get("trailingEps")
    values["Earnings_Per_Share"] = eps_native * fx_rate if eps_native is not None else np.nan
    if eps_native is None:
        warnings.append("Earnings Per Share")

    shares_out = info.get("sharesOutstanding")
    values["Estimated_Shares_Outstanding"] = shares_out if shares_out is not None else np.nan
    if shares_out is None:
        warnings.append("Estimated Shares Outstanding")

    pe = info.get("trailingPE")
    values["P/E_Ratio"] = pe if pe is not None else np.nan
    if pe is None:
        warnings.append("P/E Ratio")

    pb = info.get("priceToBook")
    values["P/B_Ratio"] = pb if pb is not None else np.nan
    if pb is None:
        warnings.append("P/B Ratio")

    net_income_native = info.get("netIncomeToCommon")
    values["Net_Income"] = net_income_native * fx_rate if net_income_native is not None else np.nan
    if net_income_native is None:
        warnings.append("Net Income")

    # --- Balance sheet based feature: Cash_Ratio ---
    cash_ratio = np.nan
    try:
        bs = t.quarterly_balance_sheet
        cash_val = find_statement_value(
            bs, ["cash and cash equivalents", "cash financial"]
        )
        current_liab = find_statement_value(bs, ["current liabilities"])
        if cash_val is not None and current_liab not in (None, 0):
            cash_ratio = cash_val / current_liab
    except Exception:
        pass
    values["Cash_Ratio"] = cash_ratio
    if np.isnan(cash_ratio):
        warnings.append("Cash Ratio")

    # --- Cash flow based feature: Net_Cash_Flow ---
    net_cash_flow_native = np.nan
    try:
        cf = t.quarterly_cash_flow
        net_cash_flow_native = find_statement_value(
            cf, ["changes in cash", "change in cash"]
        )
        if net_cash_flow_native is None:
            free_cf = info.get("freeCashflow")
            if free_cf is not None:
                net_cash_flow_native = float(free_cf)
    except Exception:
        pass
    if net_cash_flow_native is not None and not (
        isinstance(net_cash_flow_native, float) and np.isnan(net_cash_flow_native)
    ):
        values["Net_Cash_Flow"] = net_cash_flow_native * fx_rate
    else:
        values["Net_Cash_Flow"] = np.nan
        warnings.append("Net Cash Flow")

    company_name = info.get("shortName", ticker_symbol)
    sector = info.get("sector", "Unknown")

    return values, warnings, company_name, sector, currency


def impute_missing(values, overall_median):
    imputed_fields = []
    clean = dict(values)
    for col in NUM_COLS:
        val = clean.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            clean[col] = overall_median[col]
            imputed_fields.append(col)
    return clean, imputed_fields


def apply_outlier_caps(row, caps):
    capped = dict(row)
    for col in NUM_COLS:
        lower = caps[col]["lower"]
        upper = caps[col]["upper"]
        capped[col] = min(max(row[col], lower), upper)
    return capped


def predict_cluster(row, scaler, kmeans, caps):
    capped = apply_outlier_caps(row, caps)
    features = pd.DataFrame([[capped[c] for c in NUM_COLS]], columns=NUM_COLS)
    scaled = scaler.transform(features)
    cluster = int(kmeans.predict(scaled)[0])
    return cluster


def build_comparison_chart(live_values, cluster_id, ref_stats):
    cluster_median = ref_stats["cluster_medians"][str(cluster_id)]
    overall_median = ref_stats["overall_median"]

    labels = [METRIC_LABELS[c] for c in NUM_COLS]
    company_vals = []
    cluster_vals = []

    for c in NUM_COLS:
        max_abs = max(
            abs(live_values[c]), abs(cluster_median[c]), abs(overall_median[c]), 1e-9
        )
        company_vals.append(100 * abs(live_values[c]) / max_abs)
        cluster_vals.append(100 * abs(cluster_median[c]) / max_abs)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=company_vals,
            name="Live Value",
            orientation="h",
            marker_color=CLUSTER_INFO[cluster_id]["risk_color"],
        )
    )
    fig.add_trace(
        go.Scatter(
            y=labels,
            x=cluster_vals,
            name="Cluster Median (training data)",
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
        **Trained on:** 513 NYSE-listed companies (historical snapshot)
        **Live data source:** Yahoo Finance (via yfinance)
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
        "This tool classifies stocks using a model trained on historical "
        "data, applied to live financial data at the time you check. It is "
        "for educational purposes only and is not investment advice."
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-banner">
        <h1>Stock Cluster & Risk Classifier — Live</h1>
        <p>Select a company below. The app fetches its current price and
        financials from Yahoo Finance right now, and classifies it using the
        trained clustering model — so the result reflects today's numbers,
        not a fixed snapshot.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scaler, kmeans, caps, ref_stats = load_model()
universe = load_ticker_universe()

col1, col2 = st.columns([3, 1])
with col1:
    selected_label = st.selectbox(
        "Search by ticker or company name",
        options=sorted(universe["display_label"].tolist()),
        index=None,
        placeholder="Start typing a ticker or company name...",
    )
with col2:
    st.write("")
    st.write("")
    fetch_clicked = st.button("Fetch Live Data", type="primary", use_container_width=True)

if selected_label and fetch_clicked:
    ticker_symbol = selected_label.split(" - ")[0].strip()

    with st.spinner(f"Fetching live data for {ticker_symbol} from Yahoo Finance..."):
        try:
            live_values, missing_fields, company_name, sector, currency = fetch_live_features(
                ticker_symbol
            )
        except Exception as e:
            st.error(
                f"Could not fetch live data for '{ticker_symbol}'. "
                f"Yahoo Finance may be rate-limiting requests, the ticker may "
                f"be delisted, or your internet connection may have blocked "
                f"the request. Details: {e}"
            )
            st.stop()

    clean_values, imputed_fields = impute_missing(live_values, ref_stats["overall_median"])
    cluster_id = predict_cluster(clean_values, scaler, kmeans, caps)
    info = CLUSTER_INFO[cluster_id]

    st.divider()

    header_col, badge_col = st.columns([3, 1])
    with header_col:
        st.subheader(f"{ticker_symbol} - {company_name}")
        st.markdown(f"Sector: {sector}")
        st.caption(f"Live data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if currency and currency != "USD":
            st.caption(
                f"This company trades in {currency}. All price and monetary "
                f"values below have been converted to USD to match the "
                f"model's training data."
            )
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

    if missing_fields:
        st.warning(
            "Yahoo Finance did not return live values for: "
            + ", ".join(missing_fields)
            + ". These were filled in using the training dataset's overall "
            "median so the model can still run, but the classification may "
            "be less precise for this company."
        )

    st.markdown(f"### Cluster: {info['name']}")
    st.markdown(f"*{info['tagline']}*")
    st.write(info["description"])

    st.markdown("#### Risk Classification")
    render_risk_meter(cluster_id)

    st.markdown("#### Live Key Metrics")
    metric_cols = st.columns(4)
    display_metrics = ["Current_Price", "Price_Change", "ROE", "P/E_Ratio"]
    for col, metric in zip(metric_cols, display_metrics):
        col.metric(METRIC_LABELS[metric], format_metric(metric, clean_values[metric]))

    st.markdown("#### Live Value vs. Cluster Median (from training data)")
    st.plotly_chart(
        build_comparison_chart(clean_values, cluster_id, ref_stats),
        use_container_width=True,
    )

    st.markdown("#### Other Companies in This Cluster (based on training data)")
    peers = universe[
        (universe["Cluster"] == cluster_id) & (universe["Ticker_Symbol"] != ticker_symbol)
    ]
    peers_sample = peers.sample(min(10, len(peers)))
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

elif not selected_label:
    st.info("Select a company above and click 'Fetch Live Data' to classify it.")
else:
    st.info("Click 'Fetch Live Data' to run the classification.")
