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

def run_master_chart():
    print("--- MASTER CHART: THE BATTLE OF STRATEGIES ---")
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    # 20 Years (Scientific Window)
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    print(f"Downloading Data (Since {start_date})...")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill()
    
    monthly_prices = data.resample("M").last()
    dates = [monthly_prices.index[12]]
    
    # -----------------------------
    # CALCULATE SIGNALS
    # -----------------------------
    print("Calculating Momentum (3M and 12M)...")
    mom_3m = monthly_prices.pct_change(3)   # Fast
    mom_12m = monthly_prices.pct_change(12) # Slow
    
    # -----------------------------
    # STRATEGIES
    # -----------------------------
    # 1. Top 3 Fast (The Ceiling)
    # 2. Top 5 Fast (The Winner)
    # 3. Top 5 Slow (The Old Standard)
    # 4. Benchmark
    
    eq_top3_fast = [1.0]
    eq_top5_fast = [1.0]
    eq_top5_slow = [1.0]
    eq_bench = [1.0]
    
    print("Running Simulations...")
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        # --- Top 3 Fast ---
        scores_3 = mom_3m.iloc[i-1]
        cands_3 = scores_3.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False)
        top3 = cands_3.head(3).index
        if len(top3)>0: r3 = ((monthly_prices.loc[dt_curr, top3] - monthly_prices.loc[dt_prev, top3]) / monthly_prices.loc[dt_prev, top3]).mean()
        else: r3 = 0.0
        eq_top3_fast.append(eq_top3_fast[-1] * (1+r3))
        
        # --- Top 5 Fast ---
        top5f = cands_3.head(5).index
        if len(top5f)>0: r5f = ((monthly_prices.loc[dt_curr, top5f] - monthly_prices.loc[dt_prev, top5f]) / monthly_prices.loc[dt_prev, top5f]).mean()
        else: r5f = 0.0
        eq_top5_fast.append(eq_top5_fast[-1] * (1+r5f))
        
        # --- Top 5 Slow ---
        scores_12 = mom_12m.iloc[i-1]
        cands_12 = scores_12.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False)
        top5s = cands_12.head(5).index
        if len(top5s)>0: r5s = ((monthly_prices.loc[dt_curr, top5s] - monthly_prices.loc[dt_prev, top5s]) / monthly_prices.loc[dt_prev, top5s]).mean()
        else: r5s = 0.0
        eq_top5_slow.append(eq_top5_slow[-1] * (1+r5s))
        
        # --- Benchmark ---
        q_c = monthly_prices.loc[dt_curr, "QQQ"]
        q_p = monthly_prices.loc[dt_prev, "QQQ"]
        q_r = (q_c - q_p) / q_p
        eq_bench.append(eq_bench[-1] * (1+q_r))

    # Stats
    f_3f = (eq_top3_fast[-1]-1)*100
    f_5f = (eq_top5_fast[-1]-1)*100
    f_5s = (eq_top5_slow[-1]-1)*100
    f_b = (eq_bench[-1]-1)*100
    
    print("\n--- MASTER SCOREBOARD (20 Years) ---")
    print(f"1. Top 3 Fast (Aggressive): {f_3f:,.0f}%")
    print(f"2. Top 5 Fast (Balanced):   {f_5f:,.0f}%")
    print(f"3. Top 5 Slow (Standard):   {f_5s:,.0f}%")
    print(f"4. Benchmark (QQQ):         {f_b:,.0f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_top3_fast, label=f"Top 3 Fast (+{f_3f:,.0f}%)", color="purple", linewidth=1.5, linestyle="-.")
    plt.plot(dates, eq_top5_fast, label=f"Top 5 Fast (+{f_5f:,.0f}%)", color="green", linewidth=2.5)
    plt.plot(dates, eq_top5_slow, label=f"Top 5 Slow (+{f_5s:,.0f}%)", color="blue", linewidth=1.5, alpha=0.7)
    plt.plot(dates, eq_bench, label=f"QQQ (+{f_b:,.0f}%)", color="gray", linewidth=2, linestyle="--")
    
    plt.yscale("log")
    plt.title("Master Comparison: Speed (3M vs 12M) & Concentration (Top 3 vs 5)")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig("master_strategy_comparison.png")
    print("Chart saved to: master_strategy_comparison.png")

if __name__ == "__main__":
    run_master_chart()
