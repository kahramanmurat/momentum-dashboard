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
        return ["AAPL", "AMD", "NVDA", "TSLA", "META"] 
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["AAPL", "AMD", "NVDA", "TSLA", "META"] 

def calculate_indicators(df):
    """
    Calculates EMA_9, EMA_21, RSI_14
    """
    df = df.copy()
    
    # EMA 9 and EMA 21
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def run_scanner():
    print("--- 📅 WEEKLY EMA CROSSOVER STRATEGY ---")
    
    tickers = get_nasdaq_100_tickers()
    print(f"Scanning {len(tickers)} Stocks on 1-WEEK Timeframe...")
    
    # Download Weekly Data
    # We need enough data for EMA_21 and RSI_14. 
    # 1 Year (52 weeks) should be plenty for calculation startup.
    data = yf.download(tickers, period="1y", interval="1wk", group_by='ticker', progress=False, auto_adjust=False)
    
    results = []
    
    for t in tickers:
        try:
            df = data[t].copy()
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.droplevel(0)
                except: pass
                
            if df.empty or len(df) < 25: continue # Need enough history
            df = df.dropna()
            
            # Indicators
            df = calculate_indicators(df)
            
            # Current Candle (Last completed week usually, or current partial week)
            # yfinance '1wk' often updates the current week in progress.
            # We will use the latest available bar.
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- STRATEGY LOGIC ---
            # 1. EMA 9 > EMA 21 (Bullish Alignment)
            bullish_ema = curr['EMA_9'] > curr['EMA_21']
            
            # 2. Crossover Check (Signal generated NOW)
            # Was bearish or equal before?
            ema_cross = (prev['EMA_9'] <= prev['EMA_21']) and bullish_ema
            
            # 3. RSI Conditions
            # RSI > 50 AND RSI > Prev RSI
            rsi_bullish = (curr['RSI'] > 50) and (curr['RSI'] > prev['RSI'])
            
            # We want CROSSOVER + RSI CONFIRMATION
            if ema_cross and rsi_bullish:
                results.append({
                    "Ticker": t,
                    "Price": curr['Close'],
                    "EMA_9": f"{curr['EMA_9']:.2f}",
                    "EMA_21": f"{curr['EMA_21']:.2f}",
                    "RSI": f"{curr['RSI']:.1f}",
                    "Status": "BUY SIGNAL"
                })
                
        except Exception:
            continue
            
    # Export
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, "weekly_ema_strategy.csv")
    
    if results:
        res_df = pd.DataFrame(results)
        print("\n--- WEEKLY SIGNALS FOUND ---")
        print(res_df)
        res_df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")
    else:
        print("\nNo Weekly EMA Signals found.")
        # Create empty with correct columns
        pd.DataFrame(columns=["Ticker", "Price", "EMA_9", "EMA_21", "RSI", "Status"]).to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_scanner()
