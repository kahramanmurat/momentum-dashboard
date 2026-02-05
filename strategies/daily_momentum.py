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

def run_daily_rebalance():
    print("Fetching Nasdaq 100 Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print(f"Downloading data for {len(tickers)} tickers (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill()
    
    # Needs 3 Months (approx 63 trading days) history
    mom_lookback = 63
    
    returns_daily = data.pct_change()
    
    # -----------------------------
    # DAILY SIGNAL GENERATION
    # -----------------------------
    # 3-Month Return for every day
    print("Calculating Daily 3-Month Momentum...")
    mom_daily_3m = data.pct_change(mom_lookback)
    
    # Simulation Arrays
    
    # 1. Monthly (Baseline)
    # We already know this logic, can we simulate it using the daily arrays?
    # Yes, we only change holdings at month end.
    
    # 2. Daily (No Cost)
    # 3. Daily (With Cost)
    
    eq_monthly = [1.0]
    eq_daily_raw = [1.0]
    eq_daily_net = [1.0]
    
    dates = []
    
    holdings_daily = [] # Current Tickers
    
    bench = [1.0]
    
    # Skip first 63 days
    start_idx = mom_lookback + 1
    
    print("Running Daily Simulation (This may take a minute)...")
    
    # Pre-compute Month Ends for the Monthly strategy
    month_ends = set(data.resample("M").last().index)
    holdings_monthly = []
    
    # Start loop
    # Optimization: Iterating 5000 days in python is slow.
    # We will do it anyway for accuracy of logic.
    
    for i in range(start_idx, len(data)):
        dt_curr = data.index[i]
        dt_prev = data.index[i-1]
        
        dates.append(dt_curr)
        
        # Returns for this day (for held assets)
        day_rets = returns_daily.iloc[i]
        
        # --- MONTHLY STRATEGY ---
        # Logic: If today is a new month (or first day of month), rebalance. 
        # Actually easier: If prev date was a month end, rebalance.
        # But 'month_ends' are timestamps.
        # Let's check: was there a month boundary between prev and curr?
        # Simplification: Rebalance if dt_prev.month != dt_curr.month
        
        is_rebal_monthly = (dt_prev.month != dt_curr.month) or i == start_idx
        
        if is_rebal_monthly:
            # Rank based on T-1 (dt_prev)
            scores = mom_daily_3m.iloc[i-1]
            top = scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index.tolist()
            holdings_monthly = top
            
        # Calc Return
        if holdings_monthly:
            r_mo = day_rets[holdings_monthly].mean()
        else: r_mo = 0.0
        
        eq_monthly.append(eq_monthly[-1] * (1 + r_mo))
        
        # --- DAILY STRATEGY ---
        # Always Rebalance
        scores_d = mom_daily_3m.iloc[i-1]
        top_d = scores_d.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index.tolist()
        
        # Did we change holdings?
        # We need to know yesterday's holdings to calc cost
        # holdings_daily is from yesterday
        
        # Turnover Calculation
        # How many stocks left the portfolio?
        # Set difference
        if not holdings_daily:
            turnover = 1.0 # First buy (100% turnover)
        else:
            entering = len(set(top_d) - set(holdings_daily))
            # exiting = len(set(holdings_daily) - set(top_d))
            # Turnover ratio = entering / 5 
            # e.g. 1 change = 20% turnover
            turnover = entering / 5.0
            
        cost = turnover * 0.0010 # 10bps on the amount traded
        
        # Return for the day
        if top_d:
            r_da = day_rets[top_d].mean()
        else: r_da = 0.0
        
        # Update Raw (No Cost)
        eq_daily_raw.append(eq_daily_raw[-1] * (1 + r_da))
        
        # Update Net (With Cost)
        eq_daily_net.append(eq_daily_net[-1] * (1 + r_da - cost))
        
        # Update State
        holdings_daily = top_d
        
        # Bench
        bench.append(bench[-1] * (1 + day_rets["QQQ"]))

    # Stats
    f_mo = (eq_monthly[-1] - 1) * 100
    f_dr = (eq_daily_raw[-1] - 1) * 100
    f_dn = (eq_daily_net[-1] - 1) * 100
    
    print("\n--- DAILY REBALANCING RESULTS (20 Years) ---")
    print(f"Monthly Rebal:       {f_mo:,.0f}%")
    print(f"Daily Rebal (Gross): {f_dr:,.0f}%")
    print(f"Daily Rebal (Net):   {f_dn:,.0f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_monthly[1:], label=f"Monthly (+{f_mo:,.0f}%)", color="green", linewidth=2)
    plt.plot(dates, eq_daily_raw[1:], label=f"Daily Gross (+{f_dr:,.0f}%)", color="blue", linewidth=1, alpha=0.5)
    plt.plot(dates, eq_daily_net[1:], label=f"Daily Net (After Fees) (+{f_dn:,.0f}%)", color="red", linewidth=2.5)
    
    plt.yscale("log")
    plt.title("Friction Test: Monthly vs Daily Rebalancing")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("daily_vs_monthly_momentum.png")
    print("Chart saved to: daily_vs_monthly_momentum.png")

if __name__ == "__main__":
    run_daily_rebalance()
