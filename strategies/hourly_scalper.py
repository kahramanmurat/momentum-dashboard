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
    Calculates EMA_200, RSI_14, MACD
    """
    df = df.copy()
    
    # EMA 200 (Trend)
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # ATR (14) for Volatility/Beta Proxy
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    return df

def run_scanner():
    print("--- ⚡ TURBO 1H SCALPER (DIP & RIP) ---")
    
    tickers = get_nasdaq_100_tickers()
    print(f"Scanning {len(tickers)} Stocks on 1-HOUR Timeframe...")
    
    # Download Hourly Data (Last 59 days is the max for 1h usually)
    data = yf.download(tickers, period="59d", interval="1h", group_by='ticker', progress=False, auto_adjust=False)
    
    scalps = []
    
    for t in tickers:
        try:
            df = data[t].copy()
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten check again
                try: df.columns = df.columns.droplevel(0)
                except: pass
                
            if df.empty: continue
            df = df.dropna()
            
            # Indicators
            df = calculate_indicators(df)
            
            # Current Candle
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- THE GOLDEN SETUP ---
            # 1. Trend: Stock is BULLISH (Price > EMA 200)
            is_uptrend = curr['Close'] > curr['EMA_200']
            
            # 2. The Pullback: RSI is Low (Sold Off)
            # < 45 is a "Deep Pullback" in an Uptrend. < 30 is Oversold.
            is_cheap = curr['RSI'] < 45
            
            # 3. The Trigger: Momentum Turning?
            # MACD Histogram increasing (ticking up) or Crossed positive
            macd_improving = curr['Histogram'] > prev['Histogram']
            
            # 4. HIGH BETA FILTER (Volatility Check)
            # We want stocks moving at least 0.8% - 1.0% per hour on average
            is_high_beta = curr['ATR_Pct'] >= 0.8
            
            if is_uptrend and is_cheap and is_high_beta:
                
                signal_type = "WATCH (Dip)"
                if macd_improving:
                    signal_type = "⚡ READY (Momentum Up)"
                
                # Check for MACD Crossover specifically
                macd_cross = (prev['MACD'] < prev['Signal']) and (curr['MACD'] > curr['Signal'])
                if macd_cross:
                    signal_type = "🚀 BUY TRIGGER (MACD Cross)"
                
                scalps.append({
                    "Ticker": t,
                    "Price": curr['Close'],
                    "Setup": signal_type,
                    "RSI": f"{curr['RSI']:.1f}",
                    "Trend": "Up (Above EMA200)",
                    "Beta_Vol": f"{curr['ATR_Pct']:.2f}%" # Show Volatility
                })
                
        except Exception:
            continue
            
    # Export
    if scalps:
        res_df = pd.DataFrame(scalps)
        # Sort: Triggers first, then Ready
        res_df.sort_values(by="Setup", ascending=True, inplace=True) # Z-A sorta works, BUY > WATCH
        
        print("\n--- TURBO SCALPS FOUND ---")
        print(res_df)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        csv_path = os.path.join(output_dir, "hourly_scalps.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")
    else:
        print("\nNo Turbo Scalps found (Market might be flat).")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        pd.DataFrame(columns=["Ticker", "Price", "Setup", "RSI", "Trend", "Beta_Vol"]).to_csv(os.path.join(output_dir, "hourly_scalps.csv"), index=False)

if __name__ == "__main__":
    run_scanner()
