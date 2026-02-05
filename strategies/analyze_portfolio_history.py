import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from io import StringIO
from datetime import datetime, timedelta

def get_nasdaq_100_tickers():
    # Helper to ensure we have the massive universe to rank against
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "MU", "WDC", "LRCX", "STX", "AMAT"]
        tables = pd.read_html(StringIO(response.text))
        target_df = tables[4] if len(tables) > 4 else tables[0]
        col = next((c for c in target_df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
        if col: return [t.replace(".", "-").strip() for t in target_df[col].astype(str).tolist()]
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "MU", "WDC", "LRCX"]
    except: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "MU", "WDC", "LRCX"]

def analyze_portfolio():
    print("--- AUDITING TOP 5 PORTFOLIO HISTORY ---")
    
    # 1. Get Top 5 Tickers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    try:
        picks_df = pd.read_csv(os.path.join(output_dir, "fast_momentum_picks.csv"))
        if "Unnamed: 0" in picks_df.columns:
            picks_df = picks_df.rename(columns={"Unnamed: 0": "Ticker"})
        top_tickers = picks_df["Ticker"].head(5).tolist()
    except:
        top_tickers = ["MU", "WDC", "LRCX", "STX", "AMAT"] # Fallback
        
    print(f"Analyzing: {', '.join(top_tickers)}")
    
    # 2. Download Data (Context)
    # We need the whole universe to calculate ranks correctly in the past
    universe = get_nasdaq_100_tickers()
    # Ensure our top 5 are in the universe
    for t in top_tickers:
        if t not in universe: universe.append(t)
        
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    print("Downloading historical data for rank calculation...")
    data = yf.download(universe, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    m_data = data.resample("M").last()
    mom_3m = m_data.pct_change(3)
    
    results = []
    
    for ticker in top_tickers:
        # Trace back to find entry
        # We look backwards from today. The first month it fell OUT of top 5 (or universe), the month AFTER is the entry.
        
        # --- NEW: Fetch Sector/Industry Metadata & Options ---
        try:
            t_obj = yf.Ticker(ticker)
            sector = t_obj.info.get('sector', 'N/A')
            industry = t_obj.info.get('industry', 'N/A')
            has_options = "YES" if len(t_obj.options) > 0 else "No"
        except:
            sector = 'N/A'
            industry = 'N/A'
            has_options = "No"
            
        # --- NEW: Calculate Pullback Level (20-Day SMA) ---
        sma_20 = 0.0
        try:
            # We already have 'data' with history.
            # rolling(20).mean().iloc[-1]
            if ticker in data.columns:
                series = data[ticker].dropna()
                if len(series) >= 20:
                    sma_20 = series.rolling(window=20).mean().iloc[-1]
        except:
            sma_20 = 0.0
            
        entry_date = None
        entry_price = 0.0
        
        # Scan last 12 months backwards
        # i=0 is current, i=1 is last month...
        
        streak_start_idx = -1
        
        # We need to find the specific month where Rank <= 5 started.
        # Let's iterate from 12 months ago forward.
        
        in_streak = False
        streak_start = None
        
        # Get dates for last 12 months
        dates = mom_3m.index[-13:]
        
        for d in dates:
            try:
                row = mom_3m.loc[d].dropna().sort_values(ascending=False)
                rank = row.index.get_loc(ticker) + 1
                
                if rank <= 5:
                    if not in_streak:
                        # Streak started here!
                        in_streak = True
                        streak_start = d
                else:
                    if in_streak:
                        # Streak broken?
                        # For this specific analysis, let's just find the *current* streak start.
                        # If rank > 5, we reset.
                        in_streak = False
                        streak_start = None
            except:
                in_streak = False
                streak_start = None
                
        # Now we have the streak start date (Month End).
        # We entry at the CLOSE of that month (or Open of next day, simple backtest logic uses Close).
        
        if streak_start:
            entry_date_str = streak_start.strftime("%Y-%m-%d")
            # Use asof to get the price on or before the streak_start date
            # This handles weekends and "future" month-ends (like Jan 31 when it's Jan 28)
            entry_price = data[ticker].asof(streak_start)
            if pd.isna(entry_price):
                 # Fallback if asof fails (shouldn't happen with ffill, but safety first)
                 entry_price = data[ticker].iloc[-1]
            
            # --- NEW: Calculate Max Price (High Watermark) ---
            # We look at the price window from entry (streak_start) to today
            # We need daily data for this to be accurate
            price_window = data[ticker].loc[streak_start:]
            max_price = price_window.max()
            if pd.isna(max_price): max_price = entry_price
            
            curr_price = data[ticker].iloc[-1].item()
            ret = (curr_price - entry_price) / entry_price
            
            results.append({
                "Ticker": ticker,
                "Sector": sector,
                "Industry": industry,
                "Options_Avail": has_options,
                "SMA_20": sma_20,
                "Entry_Date": entry_date_str,
                "Entry_Price": entry_price,
                "Max_Price": max_price,
                "Current_Price": curr_price,
                "Return": ret,
                "Months_Held": (datetime.now() - streak_start).days // 30
            })
        else:
            # Maybe it just entered today?
            pass

    # Print Table
    print("\n--- PORTFOLIO PERFORMANCE AUDIT ---")
    print(f"{'Ticker':<8} {'Entry Date':<12} {'Held (Mos)':<12} {'Return':<10}")
    print("-" * 45)
    
    for r in results:
        print(f"{r['Ticker']:<8} {r['Entry_Date']:<12} {r['Months_Held']:<12} {r['Return']*100:+.1f}%")
        
    # Save to CSV for Dashboard
    hist_df = pd.DataFrame(results)
    hist_path = os.path.join(output_dir, "portfolio_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"\nHistory saved to: {hist_path}")

if __name__ == "__main__":
    analyze_portfolio()
