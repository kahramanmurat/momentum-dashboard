import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta
import os

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

def run_short_term_test():
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
    print("Calculating 1M, 2M, 3M Momentum...")
    
    mom_1m = monthly_prices.pct_change(1)
    mom_2m = monthly_prices.pct_change(2)
    mom_3m = monthly_prices.pct_change(3)
    
    # -----------------------------
    # Simulation
    # -----------------------------
    eq_1m = [1.0]
    eq_2m = [1.0]
    eq_3m = [1.0]
    
    # Start after 3 months to be fair to all
    start_idx = 3 
    dates = [monthly_prices.index[3]]
    
    print("Running The Sprint Race (1M vs 2M vs 3M)...")
    
    for i in range(start_idx + 1, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        # --- Helper Function for Strategy ---
        def get_strat_ret(mom_df):
            scores = mom_df.iloc[i-1]
            top_5 = scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
            if len(top_5) > 0:
                p_c = monthly_prices.loc[dt_curr, top_5]
                p_p = monthly_prices.loc[dt_prev, top_5]
                return ((p_c - p_p) / p_p).mean()
            return 0.0

        # 1-Month
        ret_1 = get_strat_ret(mom_1m)
        eq_1m.append(eq_1m[-1] * (1 + ret_1))
        
        # 2-Month
        ret_2 = get_strat_ret(mom_2m)
        eq_2m.append(eq_2m[-1] * (1 + ret_2))
        
        # 3-Month
        ret_3 = get_strat_ret(mom_3m)
        eq_3m.append(eq_3m[-1] * (1 + ret_3))
        
    # Stats
    f_1 = (eq_1m[-1] - 1) * 100
    f_2 = (eq_2m[-1] - 1) * 100
    f_3 = (eq_3m[-1] - 1) * 100
    
    print("\n--- SHORT-TERM LOOKBACK RESULTS (20 Years) ---")
    print(f"1-Month (Ultra):   {f_1:,.0f}%")
    print(f"2-Month (Super):   {f_2:,.0f}%")
    print(f"3-Month (Fast):    {f_3:,.0f}%")
    
    # Plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_1m, label=f"1-Month (+{f_1:,.0f}%)", color="orange", linewidth=1.5, alpha=0.8)
    plt.plot(dates, eq_2m, label=f"2-Month (+{f_2:,.0f}%)", color="blue", linewidth=1.5, alpha=0.8)
    plt.plot(dates, eq_3m, label=f"3-Month (+{f_3:,.0f}%)", color="red", linewidth=2.5) # Highlight Winner (Assuming 3M)
    
    plt.yscale("log")
    plt.title("The Sprint: 1M vs 2M vs 3M Lookback")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    chart_path = os.path.join(output_dir, "short_term_comparison.png")
    plt.savefig(chart_path)
    print(f"Chart saved to: {chart_path}")

if __name__ == "__main__":
    run_short_term_test()
