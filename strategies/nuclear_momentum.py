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

def run_nuclear_momentum():
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    print("Downloading Data (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # -----------------------------
    # REGIME FILTER (Month End)
    # -----------------------------
    qqq = data["QQQ"]
    qqq_sma200 = qqq.rolling(200).mean()
    monthly_regime = (qqq > qqq_sma200).resample("M").last()
    
    # -----------------------------
    # MOMENTUM SIGNAL (3-Month)
    # -----------------------------
    print("Calculating Fast Momentum...")
    monthly_prices = data.resample("M").last()
    mom_3m = monthly_prices.pct_change(3)
    
    # -----------------------------
    # SIMULATION
    # -----------------------------
    eq_raw_1x = [1.0] # Baseline (Win +84k%)
    eq_safe_1x = [1.0] # Safe Baseline
    eq_safe_2x = [1.0] # Nuclear Option
    
    dates = [monthly_prices.index[13]] # Start index alignment
    
    # Margin Interest Rate (Annual)
    margin_rate = 0.06 # 6%
    monthly_margin_cost = margin_rate / 12.0
    
    print("Running Nuclear Simulation (2x Leverage)...")
    
    start_idx = 13
    
    for i in range(start_idx, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        dates.append(dt_curr)
        
        # 1. Base Strategy Return (Top 5)
        # -----------------------------
        scores = mom_3m.iloc[i-1]
        top_n = scores.drop("QQQ", errors="ignore").dropna().sort_values(ascending=False).head(5).index
        
        if len(top_n) > 0:
            ret_base = ((monthly_prices.loc[dt_curr, top_n] - monthly_prices.loc[dt_prev, top_n]) / monthly_prices.loc[dt_prev, top_n]).mean()
        else:
            ret_base = 0.0
            
        # Update Raw 1x
        eq_raw_1x.append(eq_raw_1x[-1] * (1 + ret_base))
        
        # 2. Regime Check
        # -----------------------------
        is_bull = monthly_regime.iloc[i-1]
        
        # 3. Safe 1x & Safe 2x Logic
        # -----------------------------
        if is_bull:
            # Bull Market: Invest
            ret_s1 = ret_base
            
            # Leverage 2x: (Return * 2) - Interest
            # Note: We borrow 1.0x to get 2.0x exposure. We pay interest on the borrowed 1.0x.
            # ret_2x = (ret_base * 2) - monthly_margin_cost
            ret_s2 = (ret_base * 2) - monthly_margin_cost
        else:
            # Bear Market: Cash
            ret_s1 = 0.0
            ret_s2 = 0.0 # No leverage, no interest, just cash
            
        eq_safe_1x.append(eq_safe_1x[-1] * (1 + ret_s1))
        
        # Check for blowup (Equity cant go below 0)
        curr_eq_2x = eq_safe_2x[-1] * (1 + ret_s2)
        if curr_eq_2x < 0: curr_eq_2x = 0.0
        eq_safe_2x.append(curr_eq_2x)
        
    # Stats
    f_r1 = (eq_raw_1x[-1] - 1) * 100
    f_s1 = (eq_safe_1x[-1] - 1) * 100
    f_s2 = (eq_safe_2x[-1] - 1) * 100
    
    print("\n--- NUCLEAR RESULTS (20 Years) ---")
    print(f"Raw 1x (Control):   {f_r1:,.0f}%")
    print(f"Safe 1x (Control):  {f_s1:,.0f}%")
    print(f"Safe 2x (Lev+Int):  {f_s2:,.0f}%")
    
    if f_s2 > f_r1:
        print("\nSUCCESS: Safe Leverage beat the Raw Strategy!")
    else:
        print("\nFAILURE: Leverage or Interest destroyed the alpha.")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_raw_1x, label=f"Raw 1x (+{f_r1:,.0f}%)", color="green", linewidth=1.5, alpha=0.5)
    plt.plot(dates, eq_safe_1x, label=f"Safe 1x (+{f_s1:,.0f}%)", color="blue", linewidth=1.5, alpha=0.7)
    plt.plot(dates, eq_safe_2x, label=f"Safe 2x (Nuclear) (+{f_s2:,.0f}%)", color="red", linewidth=2)
    
    plt.yscale("log")
    plt.title("The Nuclear Option: Safe 2x Leverage vs Raw 1x")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("nuclear_momentum_results.png")
    print("Chart saved to: nuclear_momentum_results.png")

if __name__ == "__main__":
    run_nuclear_momentum()
