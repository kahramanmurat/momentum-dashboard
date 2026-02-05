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
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    
    target_df = None
    for tbl in tables:
        cols = [str(c).lower() for c in tbl.columns]
        if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
            target_df = tbl
            break
            
    if target_df is None:
        if len(tables) > 4: target_df = tables[4]
        else: return []

    col = None
    for c in target_df.columns:
        if "ticker" in str(c).lower() or "symbol" in str(c).lower():
            col = c
            break
            
    if col is None: col = target_df.columns[0]
    
    tickers = target_df[col].astype(str).tolist()
    tickers = [t.replace(".", "-").strip() for t in tickers]
    return sorted(list(set(tickers)))

# -----------------------------
# Momentum Logic
# -----------------------------
def run_momentum_strategy():
    print("Fetching Nasdaq 100 Universe...")
    tickers = get_nasdaq_100_tickers()
    # Add Benchmark
    tickers.append("QQQ")
    
    print(f"Downloading data for {len(tickers)} tickers (20 Years)...")
    # We need 12 months for signal + trading history
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    
    # Clean
    # For 20 years, we CANNOT drop columns with missing data at the start (IPOs like FB/META)
    # We keep them. The strategy will simply ignore them until they have data.
    data = data.ffill() 
    
    # Resample to Month End
    # Using 'M' for compatibility with installed pandas version
    monthly_prices = data.resample("M").last()
    
    # Calculate 12-Month Momentum (Signal)
    # Signal: Return over last 12 months
    # Shift(1) is NOT needed for pct_change calculation itself, 
    # BUT we must use Lagged Signal for trading (Signal at T-1 determines T holdings).
    momentum_12m = monthly_prices.pct_change(12)
    
    # Backtest Loop
    # We step through each month
    
    equity_curve = [1.0] # Portfolio starts at 1.0
    dates = [monthly_prices.index[12]] # Start after first 12 months
    
    portfolio_log = []
    
    # Loop starts from the 13th month (index 12)
    # For month T, we look at Momentum at T-1 to decide holdings for T.
    # Return at T is based on Holdings(T-1) * (Price(T)/Price(T-1) - 1)
    
    benchmark_equity = [1.0]
    
    for i in range(13, len(monthly_prices)):
        dt_current = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        
        # 1. Selection (At T-1)
        mom_prev = momentum_12m.iloc[i-1]
        
        # Filter: Exclude QQQ from selection, it's just for benchmark
        candidates = mom_prev.drop("QQQ", errors="ignore").dropna()
        
        # Rank
        top_5 = candidates.sort_values(ascending=False).head(5)
        top_tickers = top_5.index.tolist()
        
        # 2. Performance (T-1 to T)
        # We hold these top_tickers during this month
        
        prices_current = monthly_prices.loc[dt_current, top_tickers]
        prices_prev = monthly_prices.loc[dt_prev, top_tickers]
        
        # Calculate individual returns
        returns = (prices_current - prices_prev) / prices_prev
        
        # Portfolio Return (Equal Weight)
        port_ret = returns.mean()
        
        # Update Equity
        prev_eq = equity_curve[-1]
        new_eq = prev_eq * (1 + port_ret)
        equity_curve.append(new_eq)
        dates.append(dt_current)
        
        # Benchmark Logic
        qqq_curr = monthly_prices.loc[dt_current, "QQQ"]
        qqq_prev = monthly_prices.loc[dt_prev, "QQQ"]
        bench_ret = (qqq_curr - qqq_prev) / qqq_prev
        benchmark_equity.append(benchmark_equity[-1] * (1 + bench_ret))
        
        portfolio_log.append({
            "Date": dt_current.date(),
            "Holdings": ", ".join(top_tickers),
            "Monthly_Return": f"{port_ret*100:.2f}%",
            "Equity": new_eq
        })

    # Results to DataFrame
    results_df = pd.DataFrame(portfolio_log)
    results_df.to_csv("momentum_results.csv", index=False)
    
    # Metrics
    total_ret_strat = (equity_curve[-1] - 1) * 100
    total_ret_bench = (benchmark_equity[-1] - 1) * 100
    
    print("\n--- Strategy Results ---")
    print(f"Strategy Total Return: {total_ret_strat:.2f}%")
    print(f"Benchmark (QQQ) Return: {total_ret_bench:.2f}%")
    
    if not results_df.empty:
        print("\nRecent Holdings (The 'Hot' List):")
        print(results_df.tail(3)[["Date", "Holdings", "Monthly_Return"]].to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, equity_curve, label="Top 5 Momentum (Aggressive)", color="green", linewidth=2)
    plt.plot(dates, benchmark_equity, label="QQQ Benchmark", color="gray", linestyle="--")
    plt.yscale("log") # Log scale essential for 20 year compounding
    plt.title("20-Year Momentum Rotation (Top 5 Nasdaq stocks) vs QQQ (Log Scale)")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("momentum_20y_results.png")
    print("Chart saved to: momentum_20y_results.png")

if __name__ == "__main__":
    run_momentum_strategy()
