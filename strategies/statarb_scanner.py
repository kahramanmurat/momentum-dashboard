import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import statsmodels.tsa.stattools as ts
from datetime import datetime, timedelta

# -----------------------------
# Universe Fetch with SECTOR
# -----------------------------
def get_universe_df(universe="nasdaq100"):
    """
    Returns a DataFrame with columns: ['ticker', 'sector', 'sub_industry']
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    
    if universe == "nasdaq100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        
        # Find the table with Ticker/Symbol and Sector
        target_df = None
        for tbl in tables:
            # Normalize cols
            cols = [str(c).lower() for c in tbl.columns]
            if (any("ticker" in c for c in cols) or any("symbol" in c for c in cols)) and \
               (any("sector" in c for c in cols) or any("industry" in c for c in cols)):
                target_df = tbl
                break
        
        if target_df is None:
            # Fallback to index 4 if detection fails (common wikiepdia layout)
            if len(tables) > 4: 
                target_df = tables[4]
            else:
                raise ValueError("Could not extract Nasdaq 100 table with Sectors.")

        # Clean Columns
        # Rename relevant columns to standard names
        rename_map = {}
        for c in target_df.columns:
            cl = str(c).lower()
            if "ticker" in cl or "symbol" in cl:
                rename_map[c] = "ticker"
            elif "sector" in cl:
                rename_map[c] = "sector"
            elif "industry" in cl and "sub" in cl:
                rename_map[c] = "sub_industry"
        
        df = target_df.rename(columns=rename_map)
        
        # Ensure 'sector' exists (if not, fill 'Unknown')
        if "sector" not in df.columns:
            df["sector"] = "Unknown"
            
        # Clean tickers
        df["ticker"] = df["ticker"].astype(str).apply(lambda x: x.replace(".", "-").strip())
        return df[["ticker", "sector"]]

    raise ValueError("Only 'nasdaq100' supported for sector grouping in this MVP.")

# -----------------------------
# Stat Arb Logic
# -----------------------------
def check_cointegration(series_a, series_b):
    """
    Perform Engle-Granger Cointegration Test.
    Returns: (is_coint, p_value)
    """
    # statsmodels coint: Null hypothesis is no cointegration.
    # If p < 0.05, we reject Null -> They ARE cointegrated.
    score, pvalue, _ = ts.coint(series_a, series_b)
    return pvalue < 0.05, pvalue

def calculate_zscore_spread(series_a, series_b, window=30):
    """
    Calculate the Z-Score of the spread A - hedge_ratio * B
    Simple implementation: hedge_ratio = 1 (Price Ratio) or OLS.
    For robustness in this MVP, we use Price Ratio Z-Score (Spread = Ratio).
    Alternative: OLS Residual Z-Score. Let's use OLS Residuals for "Pro" quality.
    """
    # Dynamic Hedge Ratio via Rolling OLS is expensive.
    # Static OLS for the whole lookback window:
    # Spread = Y - beta * X
    
    # 1. Calc Hedge Ratio (Beta)
    poly = np.polyfit(series_b, series_a, 1) # A = beta * B + alpha
    beta = poly[0]
    
    # 2. Construct Spread
    spread = series_a - beta * series_b
    
    # 3. Z-Score (rolling)
    # We only care about the MOST RECENT Z-Score
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z_score = (spread - mean) / std
    
    return z_score.iloc[-1], beta

# -----------------------------
# Main Runner
# -----------------------------
def run_statarb_scan(lookback_days=252):
    print("Fetching Universe & Sectors...")
    uni_df = get_universe_df("nasdaq100")
    tickers = uni_df["ticker"].unique().tolist()
    
    print(f"Downloading Data for {len(tickers)} tickers...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)["Close"]
    
    # Filter for liquidity/completeness
    data = data.dropna(axis=1, how="any") # Drop tickers with missing data
    valid_tickers = data.columns.tolist()
    uni_df = uni_df[uni_df["ticker"].isin(valid_tickers)]
    
    print(f"Valid data for {len(valid_tickers)} tickers.")
    
    results = []
    
    # Group by Sector to reduce search space and improve logic (econ link)
    groups = uni_df.groupby("sector")
    
    print("Starting Pairs Scan (Cointegration)...")
    
    for sector, group in groups:
        sector_tickers = group["ticker"].tolist()
        n = len(sector_tickers)
        if n < 2: continue
        
        print(f"Scanning {sector} ({n} tickers)...")
        
        # Pairwise Loop
        for i in range(n):
            for j in range(i + 1, n):
                t1 = sector_tickers[i]
                t2 = sector_tickers[j]
                
                s1 = data[t1]
                s2 = data[t2]
                
                # 1. Correlation Filter (Cheap)
                corr = s1.corr(s2)
                if corr < 0.85:
                    continue
                
                # 2. Cointegration Test (Expensive)
                # Use log prices for cointegration
                is_coint, p_val = check_cointegration(np.log(s1), np.log(s2))
                
                if is_coint:
                    # 3. Calculate Z-Score Signal
                    # Use log prices for spread construction to trade percentages
                    z_score, beta = calculate_zscore_spread(np.log(s1), np.log(s2))
                    
                    # Filter for actionable signals (abs(z) > 2.0)
                    if abs(z_score) > 2.0:
                        results.append({
                            "Ticker_A": t1,
                            "Ticker_B": t2,
                            "Sector": sector,
                            "Correlation": round(corr, 3),
                            "Coint_P_Value": round(p_val, 4),
                            "Hedge_Ratio": round(beta, 3),
                            "Z_Score": round(z_score, 2),
                            "Action": "SELL A / BUY B" if z_score > 0 else "BUY A / SELL B"
                        })

    # Output
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("Coint_P_Value") # Strongest math link first
        
        out_file = "pairs_opportunities.csv"
        results_df.to_csv(out_file, index=False)
        print(f"\nScan Complete! Found {len(results_df)} opportunities.")
        print(f"Saved to: {out_file}")
        print("\nTop Opportunities:")
        print(results_df.head(10).to_string(index=False))
    else:
        print("\nScan Complete. No pairs met the strict cointegration + z-score criteria today.")

if __name__ == "__main__":
    run_statarb_scan()
