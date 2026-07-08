# 📈 Stock Segmentation & Risk Classification | ML + Streamlit Project
# 📊 Project Overview
This project is an **end-to-end Machine Learning project** built using **Python (Scikit-learn) and Streamlit** to segment publicly traded stocks into risk-based clusters.
The goal of this project is to extract **investment insights**, group stocks by **risk, profitability, and valuation**, and let users classify any company using an **interactive web app**.
This project simulates a **real-world data science workflow**:
- Raw financial data cleaning
- Outlier treatment & scaling
- Clustering model building
- Business-focused Streamlit app
# 🔗 Live App
## ➡️ Streamlit App Link: [Add your deployed Streamlit app link here]
# 🛠 Tools & Technologies Used
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn (KMeans, Hierarchical Clustering, PCA, StandardScaler)**
- **Streamlit, Plotly**
- **yfinance (Live Data)**
- **GitHub (Project Hosting)**
# 📂 Dataset Information
- **Source**: Stock Financial Dataset (CSV)
- **Records**: ~513 companies (US + India-listed)
- **Key Columns**:
  - `Ticker_Symbol`
  - `Security`
  - `GICS_Sector`
  - `GICS_Sub_Industry`
  - `Current_Price`
  - `Price_Change`
  - `Volatility`
  - `ROE`
  - `Cash_Ratio`
  - `Net_Cash_Flow`
  - `Net_Income`
  - `Earnings_Per_Share`
  - `Estimated_Shares_Outstanding`
  - `P/E_Ratio`
  - `P/B_Ratio`
# 📈 Key Analysis Steps
The following steps were performed using Python:
1. **Data Cleaning** (sector label standardization, EPS currency fix)
2. **Outlier Treatment** (IQR capping)
3. **Feature Scaling** (StandardScaler)
4. **Clustering** (K-Means + Hierarchical)
5. **Cluster Validation** (Silhouette Score, Cophenetic Correlation)
# 📊 Analytical Insights
 ## 🔹 Risk Analysis
- Clusters by **Volatility** and **Price_Change**
- Distribution of companies across 4 risk tiers
 ## 🔹 Profitability & Valuation
- % of companies by **ROE** and **Net_Income** bands
- % of companies by **P/E_Ratio** and **P/B_Ratio** range
 ## 🔹 Sector Performance
 ### - Cluster composition by:
  - GICS Sector
  - Cash Ratio
  - P/E Ratio
 ### - Outlier companies by:
  - Volatility
  - ROE
  - Net Income
# 🧠 Model Analysis
All clustering logic used in this project is available in the repository.
Key operations performed:
- Aggregations (`groupby`, cluster-wise mean/median)
- Distance-based clustering (`KMeans`, `AgglomerativeClustering`)
- Dimensionality reduction (`PCA`)
- Elbow method & Silhouette scoring
- Cophenetic correlation & dendrograms
📁 **Notebook File**: `Stock_Clustering_TradeAhead.ipynb`
## 📊 Streamlit Dashboard
The Streamlit app provides:
- Risk tier classification for any company
- Static (historical) and live (Yahoo Finance) modes
- Cluster comparison strip across all 4 risk tiers
- Key metrics driving each classification
# 📈 Visuals Included
## Classification Cards
- Predicted Risk Tier
- ROE, P/E Ratio, Volatility of selected company
## Trend Analysis
- 13-week Price Change trend
## Distribution Analysis
- % of Companies by Risk Tier
- % of Companies by GICS Sector
## Cluster Comparison
- All 4 Risk Tiers side-by-side (Low / Low-Moderate / Moderate-High / High)
- Selected company highlighted among all tiers
## Interactive Filters
- Company / Ticker
- Static vs Live Data Mode
# 📁 Dashboard Preview:
- `app.py` (static demo)
- `app_live.py` (live demo)
# 🧠 Key Insights
## 🔹 Cluster Performance
- Both K-Means and Hierarchical clustering converge on **4 clusters**
- Clusters map cleanly to risk, profitability, and valuation
## 🔹 Data Behavior
- Most financial metrics are right-skewed with extreme outliers
- India-listed (`.NS`) tickers had an EPS currency bug, now fixed
## 🔹 Improvement Opportunities
- Small, high-risk clusters should be reviewed company-by-company
- Clustering should be refreshed periodically as markets shift

