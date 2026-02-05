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

def run_portfolio_size_test():
    print("Fetching Nasdaq 100 Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print(f"Downloading data for {len(tickers)} tickers (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill()
    
    monthly_prices = data.resample("M").last()
    
    # 3-Month Momentum (The Winner)
    print("Calculating 3-Month Momentum...")
    mom_3m = monthly_prices.pct_change(3)
    
    # Portfolios to test
    sizes = [1, 3, 5, 10, 25]
    
    equity_curves = {s: [1.0] for s in sizes}
    dates = [monthly_prices.index[12]]
    
    equity_curves["QQQ"] = [1.0] # Benchmark
    
    print("Running Concentration Test...")
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        # Get Ranking
        scores = mom_3m.iloc[i-1]
        candidates = scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False)
        
        # Calculate Returns for each size
        for s in sizes:
            top_n = candidates.head(s).index
            if len(top_n) > 0:
                p_c = monthly_prices.loc[dt_curr, top_n]
                p_p = monthly_prices.loc[dt_prev, top_n]
                # Equal Weight Return
                ret = ((p_c - p_p) / p_p).mean()
            else:
                ret = 0.0
            
            equity_curves[s].append(equity_curves[s][-1] * (1 + ret))
            
        # Bench
        q_c = monthly_prices.loc[dt_curr, "QQQ"]
        q_p = monthly_prices.loc[dt_prev, "QQQ"]
        q_r = (q_c - q_p) / q_p
        equity_curves["QQQ"].append(equity_curves["QQQ"][-1] * (1 + q_r))
        
    # Stats
    results = {}
    print("\n--- PORFOLIO SIZE RESULTS (20 Years) ---")
    for s in sizes:
        total = (equity_curves[s][-1] - 1) * 100
        results[s] = total
        print(f"Top {s}: {total:,.0f}%")
    
    print(f"QQQ:    {(equity_curves['QQQ'][-1]-1)*100:,.0f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = {1: "orange", 3: "red", 5: "green", 10: "blue", 25: "purple", "QQQ": "gray"}
    
    for s in sizes:
        plt.plot(dates, equity_curves[s], label=f"Top {s} (+{results[s]:,.0f}%)", color=colors[s], linewidth=2 if s==5 else 1.5)
        
    plt.plot(dates, equity_curves["QQQ"], label="QQQ Benchmark", color="gray", linestyle="--")
    
    plt.yscale("log")
    plt.title("Concentration vs Diversification: Top N Analysis")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("portfolio_size_comparison.png")
    print("Chart saved to: portfolio_size_comparison.png")

if __name__ == "__main__":
    run_portfolio_size_test()
