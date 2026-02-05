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

def run_stress_test():
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print("Downloading Data (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # -----------------------------
    # INJECT POISON (Fake Bankrupt Stocks)
    # -----------------------------
    print("Injecting Synthetic Poison (Bankruptcies)...")
    
    # We create a dataframe for the fake stocks matching the index of real data
    fake_stocks = pd.DataFrame(index=data.index)
    
    # 1. FAKE_LEHMAN (Crashes in 2008)
    # Behave like QQQ until Aug 2008, then go to 0.
    qqq = data["QQQ"]
    
    # Clone QQQ initially
    lehman = qqq.copy()
    
    # Find crash date (Sept 2008)
    crash_locs = lehman.index[(lehman.index.year == 2008) & (lehman.index.month >= 6)]
    if len(crash_locs) > 0:
        crash_start = crash_locs[0]
        # Drop 99% over 3 months
        # We manually overwrite the prices
        start_idx = data.index.get_loc(crash_start)
        
        # Simulate -50% drop, then -50%, then 0
        current_price = lehman.iloc[start_idx]
        
        for i in range(start_idx, min(start_idx+100, len(lehman))):
            current_price = current_price * 0.90 # Drop 10% per day
            lehman.iloc[i] = current_price
            if current_price < 0.01: 
                lehman.iloc[i] = 0.0
                
        # Set to 0 for rest of history
        if start_idx+100 < len(lehman):
            lehman.iloc[start_idx+100:] = 0.0
            
    fake_stocks["FAKE_LEHMAN"] = lehman
    
    # 2. FAKE_ENRON (Crashes in 2022 - simulating a modern fraud)
    enron = qqq.copy() * 1.5 # Outperformer initially! High Momentum!
    
    crash_locs_2 = enron.index[(enron.index.year == 2022) & (enron.index.month >= 1)]
    if len(crash_locs_2) > 0:
        cs2 = crash_locs_2[0]
        s_idx = data.index.get_loc(cs2)
        price = enron.iloc[s_idx]
        for i in range(s_idx, min(s_idx+100, len(enron))):
            price = price * 0.85 # Drop 15% per day
            enron.iloc[i] = price
            if price < 0.01: enron.iloc[i] = 0.0
        if s_idx+100 < len(enron):
            enron.iloc[s_idx+100:] = 0.0
            
    fake_stocks["FAKE_ENRON"] = enron
    
    # Merge Fake stocks into Data
    combined_data = pd.concat([data, fake_stocks], axis=1)
    
    # -----------------------------
    # RUN STRATEGY (Top 5 / 3-Month)
    # -----------------------------
    print("Running Top 5 Momentum on Poisoned Dataset...")
    
    monthly_prices = combined_data.resample("M").last()
    mom_3m = monthly_prices.pct_change(3)
    
    equity_curve = [1.0]
    dates = [monthly_prices.index[12]]
    
    held_fake_lehman = [] # Track if we held it
    held_fake_enron = []
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        scores = mom_3m.iloc[i-1]
        top_n = scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
        
        # Check what we held
        if "FAKE_LEHMAN" in top_n: held_fake_lehman.append(dt_curr)
        if "FAKE_ENRON" in top_n: held_fake_enron.append(dt_curr)
        
        if len(top_n) > 0:
            prices_curr = monthly_prices.loc[dt_curr, top_n]
            prices_prev = monthly_prices.loc[dt_prev, top_n]
            
            # Handle potential zeros (division by zero)
            # If price was 0 at prev, we can't calculate return (it's delisted)
            # If price goes to 0 at curr, return is -1.0 (-100%)
            
            rets = []
            for t in top_n:
                p0 = prices_prev[t]
                p1 = prices_curr[t]
                if p0 <= 0.001: 
                    # Already dead, return 0 (assuming we couldn't buy it)
                    r = 0.0 
                else:
                    r = (p1 - p0) / p0
                rets.append(r)
            
            ret = sum(rets) / len(rets)
        else:
            ret = 0.0
            
        equity_curve.append(equity_curve[-1] * (1 + ret))

    # Stats
    total = (equity_curve[-1] - 1) * 100
    
    print("\n--- SURVIVORSHIP STRESS TEST ---")
    print(f"Total Return: {total:,.0f}%")
    print(f"Months Holding FAKE_LEHMAN: {len(held_fake_lehman)}")
    print(f"Months Holding FAKE_ENRON:  {len(held_fake_enron)}")
    
    if len(held_fake_lehman) < 3 and len(held_fake_enron) < 3:
        print("\nSUCCESS: Strategy auto-sold the losers immediately as Momentum turned negative.")
    else:
        print("\nWARNING: Strategy held the losers for too long.")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, equity_curve, label=f"Momentum w/ Poison (+{total:,.0f}%)", color="green")
    
    # Plot the Fake Stocks normalized
    l_norm = combined_data["FAKE_LEHMAN"] / combined_data["FAKE_LEHMAN"].iloc[0] * 1000
    e_norm = combined_data["FAKE_ENRON"] / combined_data["FAKE_ENRON"].iloc[0] * 1000
    
    # We plot them on secondary axis or just overlay scaled
    # Just schematic
    
    plt.yscale("log")
    plt.title("Survivorship Stress Test: Did IT Survive?")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("survivorship_test_results.png")
    print("Chart saved to: survivorship_test_results.png")

if __name__ == "__main__":
    run_stress_test()
