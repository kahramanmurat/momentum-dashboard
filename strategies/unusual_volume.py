import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from io import StringIO
from datetime import datetime

def get_nasdaq_100_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return ["AAPL", "NVDA", "TSLA", "AMD", "PLTR"]
        tables = pd.read_html(StringIO(response.text))
        target_df = tables[4] if len(tables) > 4 else tables[0]
        col = next((c for c in target_df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
        if col: return [t.replace(".", "-").strip() for t in target_df[col].astype(str).tolist()]
        return ["AAPL", "NVDA", "TSLA", "AMD", 'PLTR']
    except:
        return ["AAPL", "NVDA", "TSLA", "AMD", "PLTR"]

def run_unusual_volume_scan():
    print("--- 🐳 UNUSUAL VOLUME SCAN (WHALE WATCH) ---")
    
    tickers = get_nasdaq_100_tickers()
    print(f"Scanning {len(tickers)} tickers for Institutional Buying...")
    
    # Download 60 days of history to get good Volume SMA
    # Using 'group_by=ticker' to handle multi-index robustly
    try:
        data = yf.download(tickers, period="60d", interval="1d", group_by='ticker', progress=False, auto_adjust=False)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    volume_hits = []
    
    for t in tickers:
        try:
            # Robust Data Extraction
            if len(tickers) == 1:
                df = data
            else:
                if t not in data.columns: continue
                df = data[t].copy()
                
            if df.empty: continue
            
            # Ensure 'Close' and 'Volume' exist
            if 'Close' not in df.columns or 'Volume' not in df.columns: continue
            
            df = df.dropna()
            if len(df) < 20: continue # Need at least 20 days
            
            # 1. Calculate Volume SMA (20)
            df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
            
            # 2. Get Current Data
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 3. Calculate RVOL (Relative Volume)
            # RVOL = Current Vol / SMA_20
            # If Vol_SMA_20 is 0 (weird data), avoid div by zero
            if curr['Vol_SMA_20'] == 0: continue
            
            rvol = curr['Volume'] / curr['Vol_SMA_20']
            
            # 4. Filter for SIGNIFICANT Volume (> 1.2x average)
            if rvol > 1.2:
                
                # Context: Is it buying or selling?
                # Green Day: Close > Open (or Close > Prev Close)
                # Let's use Close > Prev Close for simplicity of "Price Appreciation"
                price_appr = (curr['Close'] - prev['Close']) / prev['Close']
                
                # Signal Types
                signal = "Active"
                
                # A. WHALE BUYING (Huge Vol + Up Move)
                if rvol > 2.5 and price_appr > 0:
                    signal = "🐳 WHALE BUYING (Panic Buy)"
                    
                # B. INSTITUTIONAL ACCUMULATION (Consistent Vol + Up)
                elif rvol > 1.5 and price_appr > 0:
                    signal = "🏦 INSTITUTIONAL BUYING"
                    
                # C. VOLUME BREAKOUT (Breaking 20-day high on vol)
                elif rvol > 2.0 and curr['Close'] > df['Close'].rolling(20).max().iloc[-2]:
                    signal = "🚀 VOL BREAKOUT"
                    
                # D. CHURNING / TOPPING (High Vol but Price stuck or down)
                elif rvol > 2.0 and abs(price_appr) < 0.005: 
                    signal = "🤔 CHURN (Indecision)"
                    
                # E. DUMPING (High Vol + Down)
                elif rvol > 1.5 and price_appr < -0.01:
                    signal = "📉 HEAVY SELLING"
                    
                # Save Data
                if signal != "Active" or rvol > 1.8: # Only exciting stuff
                    volume_hits.append({
                        "Ticker": t,
                        "Price": curr['Close'],
                        "Change %": f"{price_appr:.2%}",
                        "Volume": f"{curr['Volume']/1_000_000:.1f}M",
                        "RVOL": rvol, # Keep float for sorting
                        "Signal": signal
                    })
                    
        except Exception:
            pass
            
    # Export
    if volume_hits:
        res_df = pd.DataFrame(volume_hits)
        
        # Sort by RVOL descending (Biggest outliers first)
        res_df = res_df.sort_values(by="RVOL", ascending=False)
        
        print("\n--- UNUSUAL VOLUME FOUND ---")
        print(res_df.head())
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        csv_path = os.path.join(output_dir, "unusual_volume.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")
    else:
        print("No unusual volume detected.")
        # Empty file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        pd.DataFrame(columns=["Ticker","Price","Change %","Volume","RVOL","Signal"]).to_csv(os.path.join(output_dir, "unusual_volume.csv"), index=False)

if __name__ == "__main__":
    run_unusual_volume_scan()
