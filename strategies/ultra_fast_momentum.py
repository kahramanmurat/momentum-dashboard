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

def run_lookback_comparison():
    print("Fetching Nasdaq 100 Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print(f"Downloading data for {len(tickers)} tickers (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill()
    
    monthly_prices = data.resample("M").last()
    
    # -----------------------------
    # Calculate Signals
    # -----------------------------
    print("Calculating 1M, 3M, 12M Momentum...")
    mom_1m = monthly_prices.pct_change(1)
    mom_3m = monthly_prices.pct_change(3)
    mom_12m = monthly_prices.pct_change(12)
    
    # Simulation Arrays
    eq_1m = [1.0]
    eq_3m = [1.0]
    eq_12m = [1.0]
    dates = [monthly_prices.index[12]]
    
    bench = [1.0]
    
    print("Running Horse Race (1M vs 3M vs 12M)...")
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        
        # --- 1-MONTH (ULTRA FAST) ---
        # Lookback 1 month
        # We need to look at data available at T-1
        # Signal is return from T-2 to T-1
        scores_1 = mom_1m.iloc[i-1] 
        top_1 = scores_1.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
        if len(top_1) > 0:
            ret_1 = ((monthly_prices.loc[dt_curr, top_1] - monthly_prices.loc[dt_prev, top_1]) / monthly_prices.loc[dt_prev, top_1]).mean()
        else: ret_1 = 0.0
        eq_1m.append(eq_1m[-1] * (1 + ret_1))
        
        # --- 3-MONTH (FAST) ---
        scores_3 = mom_3m.iloc[i-1]
        top_3 = scores_3.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
        if len(top_3) > 0:
            ret_3 = ((monthly_prices.loc[dt_curr, top_3] - monthly_prices.loc[dt_prev, top_3]) / monthly_prices.loc[dt_prev, top_3]).mean()
        else: ret_3 = 0.0
        eq_3m.append(eq_3m[-1] * (1 + ret_3))
        
        # --- 12-MONTH (SLOW) ---
        scores_12 = mom_12m.iloc[i-1]
        top_12 = scores_12.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
        if len(top_12) > 0:
            ret_12 = ((monthly_prices.loc[dt_curr, top_12] - monthly_prices.loc[dt_prev, top_12]) / monthly_prices.loc[dt_prev, top_12]).mean()
        else: ret_12 = 0.0
        eq_12m.append(eq_12m[-1] * (1 + ret_12))
        
        # Bench
        dates.append(dt_curr)
        q_c = monthly_prices.loc[dt_curr, "QQQ"]
        q_p = monthly_prices.loc[dt_prev, "QQQ"]
        q_r = (q_c - q_p) / q_p
        bench.append(bench[-1] * (1 + q_r))

    # Stats
    f_1 = (eq_1m[-1] - 1) * 100
    f_3 = (eq_3m[-1] - 1) * 100
    f_12 = (eq_12m[-1] - 1) * 100
    f_b = (bench[-1] - 1) * 100
    
    print("\n--- MOMENTUM LOOKBACK COMPARISON (20 Years) ---")
    print(f"1-Month (Ultra):   {f_1:,.0f}%")
    print(f"3-Month (Fast):    {f_3:,.0f}%")
    print(f"12-Month (Slow):   {f_12:,.0f}%")
    print(f"Benchmark:         {f_b:,.0f}%")
    
    winner = "3-Month"
    if f_1 > f_3 and f_1 > f_12: winner = "1-Month"
    if f_12 > f_1 and f_12 > f_3: winner = "12-Month"
    
    print(f"\nWinner: {winner}")
    if winner == "3-Month":
        print("Conclusion: 3-Month is the 'Sweet Spot'. 1-Month is too noisy (Mean Reversion).")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_1m, label=f"1-Month RS (+{f_1:,.0f}%)", color="orange", linewidth=1.5, alpha=0.9)
    plt.plot(dates, eq_3m, label=f"3-Month RS (+{f_3:,.0f}%)", color="red", linewidth=2.5)
    plt.plot(dates, eq_12m, label=f"12-Month RS (+{f_12:,.0f}%)", color="green", linewidth=2)
    plt.plot(dates, bench, label=f"QQQ (+{f_b:,.0f}%)", color="gray", linestyle="--")
    plt.yscale("log")
    plt.title("Momentum Lookback: 1M vs 3M vs 12M")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("momentum_lookback_comparison.png")
    print("Chart saved to: momentum_lookback_comparison.png")

if __name__ == "__main__":
    run_lookback_comparison()
