import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta

def get_nasdaq_100_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        tables = pd.read_html(StringIO(response.text))
        target_df = tables[4] if len(tables) > 4 else tables[0]
        col = next((c for c in target_df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
        if col: return [t.replace(".", "-").strip() for t in target_df[col].astype(str).tolist()]
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    except: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

def calculate_max_drawdown(equity_curve):
    """Calculates Max Drawdown % and returns the Series"""
    # Convert list to series if needed
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
        
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min() * 100, drawdown

def run_safe_momentum():
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print("Downloading Data (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # -----------------------------
    # REGIME FILTER (DAILY)
    # -----------------------------
    qqq = data["QQQ"]
    qqq_sma200 = qqq.rolling(200).mean()
    # Shift by 1 to avoid lookahead bias (using yesterday's MA for today's decision)
    # Actually, since we rebalance monthly, we check the rule at Month End.
    # Logic: At month end, if QQQ < SMA200 -> Next month is Cash.
    
    monthly_prices = data.resample("M").last()
    
    # Get the Regime State at the end of each month
    # We resample the boolean T/F to month end
    monthly_regime = (qqq > qqq_sma200).resample("M").last()
    
    # -----------------------------
    # MOMENTUM SIGNAL (3-Month)
    # -----------------------------
    print("Calculating Fast Momentum...")
    mom_3m = monthly_prices.pct_change(3)
    
    # Simulations
    eq_raw = [1.0]
    eq_safe = [1.0]
    dates = [monthly_prices.index[12]]
    
    print("Running Safe vs Raw Simulation...")
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        # 1. RAW MOMENTUM (Top 5)
        scores = mom_3m.iloc[i-1]
        top_n = scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
        if len(top_n) > 0:
            ret_raw = ((monthly_prices.loc[dt_curr, top_n] - monthly_prices.loc[dt_prev, top_n]) / monthly_prices.loc[dt_prev, top_n]).mean()
        else: ret_raw = 0.0
        eq_raw.append(eq_raw[-1] * (1 + ret_raw))
        
        # 2. SAFE MOMENTUM (Top 5 + Filter)
        # Check regime at T-1
        is_bull = monthly_regime.iloc[i-1]
        
        if is_bull:
            # Same return as Raw
            ret_safe = ret_raw
        else:
            # Cash return (assume 0 for simplicity, or risk free rate)
            ret_safe = 0.0
            
        eq_safe.append(eq_safe[-1] * (1 + ret_safe))
        
    # Stats
    total_raw = (eq_raw[-1] - 1) * 100
    total_safe = (eq_safe[-1] - 1) * 100
    
    dd_raw_pct, _ = calculate_max_drawdown(eq_raw)
    dd_safe_pct, _ = calculate_max_drawdown(eq_safe)
    
    print("\n--- SAFE VS RAW RESULTS (20 Years) ---")
    print(f"Raw Fast Momentum:  Return: {total_raw:,.0f}% | Max Drawdown: {dd_raw_pct:.2f}%")
    print(f"Safe Fast Momentum: Return: {total_safe:,.0f}% | Max Drawdown: {dd_safe_pct:.2f}%")
    
    print("\nanalysis:")
    if dd_safe_pct > dd_raw_pct: # remember negative numbers
        print("Note: 'Safe' Drawdown is smaller (closer to 0) which is good.")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_raw, label=f"Raw Momentum (Max Profit) (+{total_raw:,.0f}%)", color="green", linewidth=1.5, alpha=0.6)
    plt.plot(dates, eq_safe, label=f"Safe Momentum (Crash Shield) (+{total_safe:,.0f}%)", color="blue", linewidth=2.5)
    
    plt.yscale("log")
    plt.title("The Safety Patch: Fixing the 2008 Crash")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("safe_momentum_comparison.png")
    print("Chart saved to: safe_momentum_comparison.png")

if __name__ == "__main__":
    run_safe_momentum()
