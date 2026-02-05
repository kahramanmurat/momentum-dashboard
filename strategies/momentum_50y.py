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

def run_50y_momentum():
    print("Fetching Nasdaq 100 Universe...")
    tickers = get_nasdaq_100_tickers()
    
    # Nasdaq 100 (QQQ) didn't exist in 1976. We use S&P 500 (SPY) as proxy? 
    # Or just use NDX data if available? 
    # YFinance ^NDX (Nasdaq 100 Index) has data back to 1985.
    # ^GSPC (S&P 500) goes back further.
    tickers.append("^NDX") 
    
    print(f"Downloading data for {len(tickers)} tickers (50 Years)...")
    # Start 1975
    start_date = "1975-01-01"
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    
    # We keep NaNs because most stocks didn't exist
    
    monthly_prices = data.resample("M").last()
    
    # 3-Month Momentum (The Winner)
    print("Calculating 3-Month Momentum...")
    mom_3m = monthly_prices.pct_change(3)
    
    equity = [1.0]
    dates = [monthly_prices.index[12]]
    
    bench_equity = [1.0]
    
    holdings_log = []
    
    print("Running 50-Year Deep Simulation...")
    
    # Start loop
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        # Rank T-1
        scores = mom_3m.iloc[i-1]
        # Exclude Index ^NDX
        candidates = scores.drop("^NDX", errors="ignore").dropna().sort_values(ascending=False)
        
        # Top 5
        top_5 = candidates.head(5).index.tolist()
        
        if not top_5:
            ret = 0.0
        else:
            p_c = monthly_prices.loc[dt_curr, top_5]
            p_p = monthly_prices.loc[dt_prev, top_5]
            ret = ((p_c - p_p) / p_p).mean()
            
        equity.append(equity[-1] * (1 + ret))
        
        # Benchmark (^NDX)
        try:
            b_c = monthly_prices.loc[dt_curr, "^NDX"]
            b_p = monthly_prices.loc[dt_prev, "^NDX"]
            # Handle NaN in index start years
            if np.isnan(b_c) or np.isnan(b_p):
                b_r = 0.0
            else:
                b_r = (b_c - b_p) / b_p
        except:
            b_r = 0.0
            
        bench_equity.append(bench_equity[-1] * (1 + b_r))
        
        if i % 60 == 0: # Print every 5 years
            print(f"Date: {dt_curr.date()} | Equity: {equity[-1]:,.2f}")

    # Stats
    total_strat = (equity[-1] - 1) * 100
    total_bench = (bench_equity[-1] - 1) * 100
    
    print("\n--- 50-YEAR HISTORY RESULTS (1976-2026) ---")
    print(f"Momentum Strategy: {total_strat:,.0f}%")
    print(f"Nasdaq 100 Index:  {total_bench:,.0f}%")
    
    print(f"Final Value of $1: ${equity[-1]:,.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, equity, label=f"3-Month Momentum (+{total_strat:,.0f}%)", color="green", linewidth=2)
    plt.plot(dates, bench_equity, label=f"Nasdaq 100 (+{total_bench:,.0f}%)", color="gray", linestyle="--")
    
    plt.yscale("log")
    plt.title("50-Year Deep History: Momentum vs Nasdaq 100 (Log Scale)")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("momentum_50y_results.png")
    print("Chart saved to: momentum_50y_results.png")

if __name__ == "__main__":
    run_50y_momentum()
