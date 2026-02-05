import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
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

def run_scanner():
    print("--- HOURLY BREAKOUT SCANNER (THE SQUEEZE) ---")
    
    # 1. Fetch Universe
    tickers = get_nasdaq_100_tickers()
    print(f"Scanning {len(tickers)} Stocks on 1-HOUR Timeframe...")
    
    # 2. Download Hourly Data
    # Max is 730 days for hourly, but yf often limits less. Let's try 59 days to be safe.
    data = yf.download(tickers, period="59d", interval="1h", group_by='ticker', progress=False, auto_adjust=False)
    
    results = []
    
    print("Analyzing Volatility Patterns...")
    
    for t in tickers:
        try:
            df = data[t].copy()
            if df.empty: continue
            
            # Clean
            df = df.dropna()
            
            # 3. Bollinger Bands (20, 2.0) on Hourly
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['STD_20'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['SMA_20'] + (df['STD_20'] * 2.0)
            df['Lower'] = df['SMA_20'] - (df['STD_20'] * 2.0)
            
            # Bandwidth (The Squeeze Metric)
            # (Upper - Lower) / SMA
            df['Bandwidth'] = (df['Upper'] - df['Lower']) / df['SMA_20']
            
            # Validations
            last_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            # --- SIGNAL 1: THE SQUEEZE (Watchlist) ---
            # Bandwidth is very low (e.g., < 3% width)
            # This is relative, let's use percentile rank later or raw threshold
            is_squeezing = last_bar['Bandwidth'] < 0.03 # Less than 3% range on hourly
            
            # --- SIGNAL 2: THE BREAKOUT (Action) ---
            # Price closed above Upper Band
            breakout_up = (last_bar['Close'] > last_bar['Upper']) and (prev_bar['Close'] <= prev_bar['Upper'])
            
            # Price closed below Lower Band
            breakout_down = (last_bar['Close'] < last_bar['Lower']) and (prev_bar['Close'] >= prev_bar['Lower'])
            
            status = "Normal"
            
            # Priority 1: Fresh Breakout (Actionable)
            if breakout_up: status = "BREAKOUT UP (Buy)"
            elif breakout_down: status = "BREAKDOWN (Short)"
            
            # Priority 2: Momentum Run (Already outside bands)
            elif last_bar['Close'] > last_bar['Upper']: status = "MOMENTUM UP (Running)"
            elif last_bar['Close'] < last_bar['Lower']: status = "MOMENTUM DOWN (Falling)"
            
            # Priority 3: Coiling (Only if inside bands)
            elif is_squeezing: status = "COILING (Wait)"
            
            # Volume Spike Check (Current Vol > 1.5x Average)
            avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
            vol_spike = last_bar['Volume'] > (avg_vol * 1.5)
            
            if status != "Normal":
                results.append({
                    "Ticker": t,
                    "Last_Time": last_bar.name,
                    "Price": last_bar['Close'],
                    "Status": status,
                    "Bandwidth": last_bar['Bandwidth'],
                    "Vol_Spike": "YES" if vol_spike else "No"
                })
                
        except Exception:
            continue
            
    # 4. Display & Export
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Sort by Status and then Tightness (Bandwidth)
        res_df = res_df.sort_values(by=["Status", "Bandwidth"])
        
        print("\n--- SCANNER RESULTS ---")
        print(res_df)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        csv_path = os.path.join(output_dir, "breakouts_today.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"\nSaved to: {csv_path}")
    else:
        print("\nNo setups found right now (Market is quiet).")

if __name__ == "__main__":
    run_scanner()
