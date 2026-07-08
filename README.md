# Stock Cluster & Risk Classifier - Streamlit App

This folder has two versions of the app:

- **`app.py`** - uses the static training-time dataset. Fast, works offline,
  but the numbers (price, ROE, etc.) are frozen at the time the data was
  collected and will not change if you check again later.
- **`app_live.py`** - fetches the company's current price and financials
  from Yahoo Finance (via the `yfinance` library) every time you check, then
  classifies it using the same trained model. This needs an internet
  connection each time you use it.

## Files

- `app.py` / `app_live.py` - the two Streamlit applications
- `scaler.joblib` - the fitted StandardScaler used during training
- `kmeans_model.joblib` - the fitted K-Means model (k = 4)
- `outlier_caps.json` - the IQR lower/upper bounds used for outlier capping
- `reference_stats.json` - cluster medians and overall median from the
  training data, used by `app_live.py` to compare a live-fetched company
  against its cluster
- `stock_data_with_clusters.csv` - the 513-company training dataset with
  cluster labels already assigned
- `requirements.txt` - required Python packages (includes `yfinance` for the
  live version)

All supporting files must stay in the same folder as the app you are
running, since the app loads them by filename.

## How to run locally

1. Open a terminal in this folder.
2. Install the required packages (skip if already installed):
   ```
   pip install -r requirements.txt
   ```
3. Start whichever version you want:
   ```
   streamlit run app.py
   ```
   or
   ```
   streamlit run app_live.py
   ```
4. Your browser should open automatically. If not, go to:
   ```
   http://localhost:8501
   ```

## How app_live.py works

1. Select a company from the dropdown (the same 513-company universe used
   for training) and click "Fetch Live Data".
2. The app pulls the current price, recent price history, and financial
   ratios for that ticker from Yahoo Finance right now.
3. The live values are capped and scaled the same way the training data
   was, then fed into the already-trained K-Means model to get a cluster
   assignment.
4. If Yahoo Finance does not return a value for some field (this happens
   for some companies), the app fills it in with the training dataset's
   overall median and clearly flags which fields were estimated.

Yahoo Finance occasionally rate-limits repeated requests or changes its data
format, which can cause a fetch to fail. If you see an error, wait a minute
and try again, or try a different ticker. If it keeps happening for many
tickers, the field-name matching inside `fetch_live_features()` in
`app_live.py` may need a small update — this is a known limitation of
scraping live financial data.

## Note

Both apps only classify companies that are already part of the 513-company
training dataset (they use this list to populate the dropdown). Only
`app_live.py` fetches current, up-to-date numbers for whichever company you
select; `app.py` always shows the historical snapshot from training time.

## Bug fix (July 2026)

The original training dataset had a currency bug: for the ~120 India-listed
(`.NS`) tickers, `Current_Price`, `Net_Income`, and other dollar fields had
been correctly converted from INR to USD, but `Earnings_Per_Share` had been
left in raw INR. This made those companies' EPS look about 95x too large,
which distorted their position in the clustering (they were pulled toward
the high-EPS cluster regardless of their actual profile).

This has been fixed: EPS for `.NS` tickers is now correctly converted to
USD, and the model has been fully retrained on the corrected data. All
files in this folder (`scaler.joblib`, `kmeans_model.joblib`,
`outlier_caps.json`, `reference_stats.json`,
`stock_data_with_clusters.csv`) reflect the corrected model.

`app_live.py` has also been updated: it now detects a ticker's native
currency via Yahoo Finance and converts all monetary/price fields (price,
EPS, net income, volatility, net cash flow) to USD before classifying, so
a live-fetched Indian (or other non-USD) stock is compared on the same
scale as the USD-trained model. Ratio fields (ROE, Cash Ratio, P/E, P/B,
13-week price change) do not need conversion since they are currency-
neutral.

## UI update

Both apps now use a colorful gradient banner and a "Risk Classification"
strip: after you look up a company, it shows all 4 risk tiers (Low,
Low-Moderate, Moderate-High, High) side by side, with the tier that matches
the selected company highlighted — so it's immediately clear where a stock
falls among all 4 possible outcomes, not just its own cluster name.

