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

def run_enhanced_momentum():
    print("Fetching Nasdaq 100 Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ") # Benchmark + Regime Indicator
    
    print(f"Downloading data for {len(tickers)} tickers (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill() # Keep IPOs
    
    # 1. Calculate Regime Filter (QQQ SMA 200)
    print("Calculating Technicals...")
    qqq_price = data["QQQ"]
    qqq_sma200 = qqq_price.rolling(200).mean()
    
    # 2. Calculate Monthly Metrcis
    monthly_prices = data.resample("M").last()
    
    # We need Daily Regime status at the Month End Date
    # Reindex daily SMA to monthly timestamps (using ffill to look at last known day)
    monthly_regime = (qqq_price > qqq_sma200).resample("M").last()
    
    # 3. Calculate Sharpe-Based Momentum (Return / Volatility)
    # 12 Month Return
    ret_12m = monthly_prices.pct_change(12)
    
    # 12 Month Volatility (Annualized)
    # We need daily returns to calc realized vol over last 252 days
    daily_rets = data.pct_change()
    vol_12m = daily_rets.rolling(252).std() * np.sqrt(252)
    monthly_vol = vol_12m.resample("M").last()
    
    # Sharpe (Simple: Return / Vol) - Risk Free Rate assumed 0 for relative ranking
    momentum_sharpe = ret_12m / monthly_vol
    
    # Backtest Loop
    equity_curve = [1.0]
    dates = [monthly_prices.index[12]]
    
    holdings_log = []
    
    # Benchmark
    bench_equity = [1.0]
    
    print("Running Enhanced Simulation...")
    
    for i in range(13, len(monthly_prices)):
        dt_current = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        
        # 1. CHECK REGIME (At T-1)
        # Did the previous month end in a Bull Trend?
        is_bull_market = monthly_regime.iloc[i-1]
        
        if not is_bull_market:
            # BEAR REGIME -> CASH
            # Cash return = 0 (or risk free rate)
            step_ret = 0.0
            holdings = ["CASH"]
        else:
            # BULL REGIME -> TOP 5 MOMENTUM
            
            # Ranking by Sharpe at T-1
            scores = momentum_sharpe.iloc[i-1]
            candidates = scores.drop("QQQ", errors="ignore").dropna()
            
            # Select Top 5
            top_5 = candidates.sort_values(ascending=False).head(5)
            holdings = top_5.index.tolist()
            
            if not holdings:
                step_ret = 0.0
            else:
                # Performance T-1 to T
                prices_curr = monthly_prices.loc[dt_current, holdings]
                prices_prev = monthly_prices.loc[dt_prev, holdings]
                
                rets = (prices_curr - prices_prev) / prices_prev
                step_ret = rets.mean()
        
        # Update Equity
        equity_curve.append(equity_curve[-1] * (1 + step_ret))
        dates.append(dt_current)
        
        # Benchmark
        q_c = monthly_prices.loc[dt_current, "QQQ"]
        q_p = monthly_prices.loc[dt_prev, "QQQ"]
        q_r = (q_c - q_p) / q_p
        bench_equity.append(bench_equity[-1] * (1 + q_r))
        
        holdings_log.append({
            "Date": dt_current.date(),
            "Regime": "BULL" if is_bull_market else "BEAR",
            "Holdings": ", ".join(holdings),
            "Return": f"{step_ret*100:.2f}%"
        })
        
    # Stats
    total_strat = (equity_curve[-1] - 1) * 100
    total_bench = (bench_equity[-1] - 1) * 100
    
    print("\n--- ENHANCED MOMENTUM RESULTS (20 Years) ---")
    print(f"Strategy Return: {total_strat:.2f}%")
    print(f"Benchmark (QQQ): {total_bench:.2f}%")
    
    # Save Log
    pd.DataFrame(holdings_log).to_csv("enhanced_log.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, equity_curve, label="Enhanced Momentum (Regime Filter)", color="blue", linewidth=2)
    plt.plot(dates, bench_equity, label="QQQ Benchmark", color="gray", linestyle="--")
    plt.yscale("log")
    plt.title("Enhanced Momentum (Crash Shield) vs QQQ (Log Scale)")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("momentum_enhanced_results.png")
    print("Chart saved to: momentum_enhanced_results.png")

if __name__ == "__main__":
    run_enhanced_momentum()
