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

def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return ["AAPL", "MSFT", "XOM", "JPM", "JNJ"]
        tables = pd.read_html(StringIO(response.text))
        target_df = tables[0]
        
        col = next((c for c in target_df.columns if "symbol" in str(c).lower()), None)
        if col: return [t.replace(".", "-").strip() for t in target_df[col].astype(str).tolist()]
        return ["AAPL", "MSFT", "XOM", "JPM", "JNJ"]
    except: return ["AAPL", "MSFT", "XOM", "JPM", "JNJ"]

def run_universe_comparison():
    print("Fetching Universes...")
    nasdaq_tickers = get_nasdaq_100_tickers()
    sp500_tickers = get_sp500_tickers()
    
    # 20 Years
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    
    print(f"Downloading Nasdaq 100 ({len(nasdaq_tickers)} tickers)...")
    data_ndx = yf.download(nasdaq_tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    print(f"Downloading S&P 500 ({len(sp500_tickers)} tickers)...")
    data_spx = yf.download(sp500_tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # Resample
    prices_ndx = data_ndx.resample("M").last()
    prices_spx = data_spx.resample("M").last()
    
    # 3-Month Momentum
    print("Calculating Momentum...")
    mom_ndx = prices_ndx.pct_change(3)
    mom_spx = prices_spx.pct_change(3)
    
    eq_ndx = [1.0]
    eq_spx = [1.0]
    dates = [prices_ndx.index[12]]
    
    # Align dates (intersection)
    common_idx = prices_ndx.index.intersection(prices_spx.index)
    
    # Start loop (skip first 13 months for signal warmup)
    start_idx = 13
    if len(common_idx) < 14:
        print("Not enough common history.")
        return

    print("Running Universe Race...")
    
    # Convert index to list/array to iterate by position if needed, 
    # but since indices are aligned, we can just iterate the common index.
    # Note: reset eq lists/dates to match common_idx start
    dates = []
    eq_ndx = [1.0]
    eq_spx = [1.0]
    
    for i in range(start_idx, len(common_idx)):
        dt_curr = common_idx[i]
        dt_prev = common_idx[i-1]
        dates.append(dt_curr)
        
        # --- NASDAQ STRATEGY ---
        try:
            scores_n = mom_ndx.loc[dt_prev] # Signal at T-1
            top_n = scores_n.dropna().sort_values(ascending=False).head(5).index
            if len(top_n) > 0:
                ret_n = ((prices_ndx.loc[dt_curr, top_n] - prices_ndx.loc[dt_prev, top_n]) / prices_ndx.loc[dt_prev, top_n]).mean()
            else: ret_n = 0.0
        except KeyError: ret_n = 0.0
        
        eq_ndx.append(eq_ndx[-1] * (1 + ret_n))
        
        # --- S&P 500 STRATEGY ---
        try:
            scores_s = mom_spx.loc[dt_prev]
            top_s = scores_s.dropna().sort_values(ascending=False).head(5).index
            if len(top_s) > 0:
                ret_s = ((prices_spx.loc[dt_curr, top_s] - prices_spx.loc[dt_prev, top_s]) / prices_spx.loc[dt_prev, top_s]).mean()
            else: ret_s = 0.0
        except KeyError: ret_s = 0.0
        
        eq_spx.append(eq_spx[-1] * (1 + ret_s))
        
    # Stats
    f_ndx = (eq_ndx[-1] - 1) * 100
    f_spx = (eq_spx[-1] - 1) * 100
    
    print("\n--- UNIVERSE COMPARISON (20 Years) ---")
    print(f"Nasdaq 100 Universe: {f_ndx:,.0f}%")
    print(f"S&P 500 Universe:    {f_spx:,.0f}%")
    
    winner = "Nasdaq 100" if f_ndx > f_spx else "S&P 500"
    print(f"\nWinner: {winner}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_ndx[1:], label=f"Nasdaq 100 (+{f_ndx:,.0f}%)", color="green", linewidth=2)
    plt.plot(dates, eq_spx[1:], label=f"S&P 500 (+{f_spx:,.0f}%)", color="blue", linewidth=2)
    
    plt.yscale("log")
    plt.title("Universe Expansion: Nasdaq 100 vs S&P 500")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("nasdaq_vs_sp500_momentum.png")
    print("Chart saved to: nasdaq_vs_sp500_momentum.png")

if __name__ == "__main__":
    run_universe_comparison()
