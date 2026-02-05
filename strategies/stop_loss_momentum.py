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

def run_stop_loss_test():
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print("Downloading Data (Daily, 20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # 3-Month Momentum Signal (calculated on daily data for precision, but sampled monthly)
    # We will compute the ranking at month begins
    
    daily_prices = data
    # Resample Monthly for Signal Generation
    monthly_prices = data.resample("M").last()
    mom_3m = monthly_prices.pct_change(3)
    
    # Thresholds
    stops = [None, -0.10, -0.15, -0.20]
    equity_curves = {s: [1.0] for s in stops}
    
    dates = []
    
    # To run this accurately, we must iterate DAILY to check stops
    # But checking stops on Top 5 chosen Monthly
    
    print("Running Hybrid Simulation (Monthly Entry, Daily Exit)...")
    
    # Identify Month Start indices in Daily Data
    # A bit tricky. Let's iterate days and check if month specific logic applies.
    
    # Use rebal dates from monthly_prices
    rebal_dates = monthly_prices.index
    # Map months to their start dates in available data?
    # Simpler: Iterate daily. If today.month != prev.month, we rebalance.
    
    current_holdings = {s: [] for s in stops} # List of tickers
    entry_prices = {s: {} for s in stops} # Ticker -> Entry Price (for Stop calculation)
    # stopped_out: Set of tickers that hit stop this month (to avoid rebuying)
    stopped_out = {s: set() for s in stops}
    
    # Cash portion? We assume 100% invested. If stop hits, that portion goes to Cash.
    # We track total value. 
    # Value = Cash + Sum(Shares * Price)
    # Normalized: We track daily returns of the portfolio.
    
    # Let's track Portfolio Weights: {Ticker: Weight} + Cash Weight
    weights = {s: {"CASH": 1.0} for s in stops}
    
    # Start after 1 year of data
    start_idx = 252 
    dates = []
    
    for i in range(start_idx, len(daily_prices)):
        dt_curr = daily_prices.index[i]
        dt_prev = daily_prices.index[i-1]
        dates.append(dt_curr)
        
        # Is Rebalance Day? (New Month)
        is_rebal = dt_curr.month != dt_prev.month
        
        day_rets = daily_prices.pct_change().iloc[i]
        
        if is_rebal:
            # 1. Generate Signal (using last month's data)
            # Find the closest monthly data point (T-1 month end)
            # Efficient way: Look at mom_3m. We need the row BEFORE this month.
            # actually we can just re-calculate using the daily data at dt_prev (month end)
            # Let's assume we have the signal.
            
            # Use `asof` or similar logic?
            # Re-implementing ranking dynamically here is safer
            
            # 3M Ret at dt_prev
            p_now = daily_prices.iloc[i-1]
            # T-63 days approx
            try:
                p_3m_ago = daily_prices.iloc[i-1-63]
                mom_series = (p_now - p_3m_ago) / p_3m_ago
                top_5 = mom_series.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index.tolist()
            except:
                top_5 = []
                
            # 2. Reset Portfolios for new month
            for s in stops:
                current_holdings[s] = top_5
                stopped_out[s] = set()
                # Equal weight 20% each
                weights[s] = {t: 0.20 for t in top_5}
                weights[s]["CASH"] = 0.0
                if not top_5: weights[s]["CASH"] = 1.0
                
                # Record Entry Prices (Closing price of dt_prev is effectively entry if we buy at open/close of dt_curr)
                # We use dt_prev close as the reference for "Start of Month Price"
                entry_prices[s] = {t: daily_prices.loc[dt_prev, t] for t in top_5}

        # --- DAILY CHECK ---
        for s in stops:
            # 1. Calc Portfolio Return for the day
            port_ret = 0.0
            
            # Active Tickers
            active_tickers = [t for t in weights[s].keys() if t != "CASH"]
            
            for t in active_tickers:
                w = weights[s][t]
                r = day_rets.get(t, 0.0) # Handle missing
                port_ret += w * r
                
                # CHECK STOP LOSS
                if s is not None:
                    curr_price = daily_prices.loc[dt_curr, t]
                    start_price = entry_prices[s].get(t, curr_price)
                    
                    # Performance since Entry (Month Start)
                    perf = (curr_price - start_price) / start_price
                    
                    if perf < s: # e.g. -0.12 < -0.10
                        # STOP HIT
                        # Sell to Cash
                        # We assume we sell AT CLOSE (which incorporates the loss)
                        # So we keep today's return, but tomorrow weight is 0
                        weights[s]["CASH"] += w
                        del weights[s][t]
                        stopped_out[s].add(t)
            
            # Add Cash Return (0)
            # port_ret is complete
            
            equity_curves[s].append(equity_curves[s][-1] * (1 + port_ret))
            
    # Stats
    print("\n--- STOP-LOSS RESULTS (20 Years) ---")
    for s in stops:
        lbl = f"Stop {s*100:.0f}%" if s else "No Stop"
        tot = (equity_curves[s][-1] - 1) * 100
        print(f"{lbl}: {tot:,.0f}%")
        
    # Plot
    plt.figure(figsize=(10, 6))
    colors = {None: "green", -0.10: "red", -0.15: "orange", -0.20: "blue"}
    
    for s in stops:
        lbl = f"Stop {s*100:.0f}%" if s else "No Stop"
        val = (equity_curves[s][-1] - 1) * 100
        plt.plot(dates, equity_curves[s][1:], label=f"{lbl} (+{val:,.0f}%)", color=colors[s], linewidth=2 if s is None else 1.5)
        
    plt.yscale("log")
    plt.title("Stop-Loss Sensitivity: Cutting Losers vs Whipsaws")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("stop_loss_comparison.png")
    print("Chart saved to: stop_loss_comparison.png")

if __name__ == "__main__":
    run_stop_loss_test()
