import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta

# -----------------------------
# Reused Universe Logic
# -----------------------------
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


# -----------------------------
# CONFIGURATION
# -----------------------------
TOP_N = 5 # Set to 3 for "God Mode" (Highest Return), 5 for "Safe Mode"

def run_fast_momentum():
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
    print("Calculating Fast (3M) Momentum...")
    mom_3m = monthly_prices.pct_change(3)
    
    # -----------------------------
    # Simulation Loop
    # -----------------------------
    equity_curve = [1.0]
    dates = [monthly_prices.index[12]] # Start after 12 months for consistency
    
    bench_equity = [1.0]
    
    print(f"Running Fast Momentum (Top {TOP_N})...")
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        
        # --- FAST MOMENTUM (3M) ---
        scores_3 = mom_3m.iloc[i-1]
        top_n = scores_3.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(TOP_N).index
        
        if len(top_n) > 0:
            prices_curr = monthly_prices.loc[dt_curr, top_n]
            prices_prev = monthly_prices.loc[dt_prev, top_n]
            ret = ((prices_curr - prices_prev) / prices_prev).mean()
        else:
            ret = 0.0
            
        equity_curve.append(equity_curve[-1] * (1 + ret))
        dates.append(dt_curr)
        
        # Benchmark
        q_c = monthly_prices.loc[dt_curr, "QQQ"]
        q_p = monthly_prices.loc[dt_prev, "QQQ"]
        q_r = (q_c - q_p) / q_p
        bench_equity.append(bench_equity[-1] * (1 + q_r))
        
    # Stats
    total = (equity_curve[-1] - 1) * 100
    total_bench = (bench_equity[-1] - 1) * 100
    
    print(f"\n--- FAST MOMENTUM RESULTS (Top {TOP_N}) ---")
    print(f"Momentum Strategy: {total:,.2f}%")
    print(f"Benchmark (QQQ):   {total_bench:,.2f}%")
    
    # -----------------------------
    # EXPORT LATEST PICKS
    # -----------------------------
    import os
    
    last_date = monthly_prices.index[-1]
    latest_scores = mom_3m.iloc[-1]
    top_picks = latest_scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(TOP_N)
    
    print(f"\n--- TOP {TOP_N} PICKS FOR {last_date.strftime('%Y-%m-%d')} ---")
    print(top_picks)
    
    # robust path: sibling 'output' folder
    script_dir = os.path.dirname(os.path.abspath(__file__)) # .../strategies
    output_dir = os.path.join(script_dir, "..", "output")   # .../output
    
    picks_path = os.path.join(output_dir, "fast_momentum_picks.csv")
    chart_path = os.path.join(output_dir, "fast_momentum_results.png")
    
    # Save to CSV
    picks_df = pd.DataFrame(top_picks)
    picks_df.columns = ["3M_Score"]
    picks_df.to_csv(picks_path)
    print(f"Picks saved to: {picks_path}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, equity_curve, label=f"Fast Momentum (Top {TOP_N}) (+{total:,.0f}%)", color="green", linewidth=2)
    plt.plot(dates, bench_equity, label="QQQ Benchmark", color="gray", linestyle="--")
    plt.yscale("log")
    plt.title(f"Fast Momentum (Top {TOP_N}) vs QQQ")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig(chart_path)
    print(f"Chart saved to: {chart_path}")

if __name__ == "__main__":
    run_fast_momentum()
