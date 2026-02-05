import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta

def get_nasdaq_100_tickers():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        
        target_df = None
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
                target_df = tbl
                break
        
        if target_df is None and len(tables) > 4: target_df = tables[4]
        
        col = None
        for c in target_df.columns:
            if "ticker" in str(c).lower(): col = c; break
        if col is None: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

        tickers = target_df[col].astype(str).tolist()
        return [t.replace(".", "-").strip() for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META", "AMD", "NFLX", "INTC"]

def run_50y_top3():
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("^NDX") 
    
    print(f"Downloading data (50 Years)...")
    start_date = "1975-01-01"
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    monthly_prices = data.resample("M").last()
    
    print("Calculating 3-Month Momentum...")
    mom_3m = monthly_prices.pct_change(3)
    
    equity = [1.0]
    
    print("Running 50-Year Top 3 Simulation...")
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        
        scores = mom_3m.iloc[i-1]
        candidates = scores.drop("^NDX", errors="ignore").dropna().sort_values(ascending=False)
        
        # TOP 3 ONLY
        top_n = candidates.head(3).index.tolist()
        
        if not top_n:
            ret = 0.0
        else:
            p_c = monthly_prices.loc[dt_curr, top_n]
            p_p = monthly_prices.loc[dt_prev, top_n]
            ret = ((p_c - p_p) / p_p).mean()
            
        equity.append(equity[-1] * (1 + ret))

    total = (equity[-1] - 1) * 100
    
    print("\n--- 50-YEAR TOP 3 RESULTS ---")
    print(f"Total Return: {total:,.0f}%")
    print(f"Final Value of $1: ${equity[-1]:,.2f}")

if __name__ == "__main__":
    run_50y_top3()
