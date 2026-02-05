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
        return ["AAPL", "AMD", "NVDA", "TSLA", "META"] # Fallback
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["AAPL", "AMD", "NVDA", "TSLA", "META"] # Fallback

def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Calculates SuperTrend Indicator.
    Returns DataFrame with 'SuperTrend' and 'Trend' columns.
    Trend: 1 = Bullish, -1 = Bearish
    """
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()

    # Basic Upper/Lower Bands
    df['Basic_Upper'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
    df['Basic_Lower'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']

    # Final Upper/Lower Bands initialization
    df['Final_Upper'] = df['Basic_Upper']
    df['Final_Lower'] = df['Basic_Lower']
    df['SuperTrend'] = 0.0
    df['Trend'] = 1 # 1 Up, -1 Down
    
    # Iterative calculation (Pandas is tricky for recursive SuperTrend, looping is safer/clearer)
    # Using arrays for speed
    close = df['Close'].values
    basic_upper = df['Basic_Upper'].values
    basic_lower = df['Basic_Lower'].values
    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    trend = np.zeros(len(df)) # 1 or -1
    
    # Initialize first valid index
    # Need 'period' data points for ATR, so start after that
    start_idx = period
    
    for i in range(start_idx, len(df)):
        # Final Upper
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Final Lower
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Trend
        # If prev trend was Down (-1) and Close > Final Upper -> Trend flips to Up (1)
        # If prev trend was Up (1) and Close < Final Lower -> Trend flips to Down (-1)
        prev_trend = trend[i-1] if i > start_idx else 1
        
        if prev_trend == -1 and close[i] > final_upper[i]:
            trend[i] = 1
        elif prev_trend == 1 and close[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = prev_trend
            
        # SuperTrend Value
        if trend[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
            
    df['SuperTrend'] = supertrend
    df['Trend'] = trend
    
    return df

def run_scanner():
    print("--- SUPERTREND REVERSAL SCANNER (THE RED CIRCLE) ---")
    
    # 1. Tickers
    tickers = get_nasdaq_100_tickers()
    print(f"Scanning {len(tickers)} tickers for Trend Reversals...")
    
    # 2. Download Data (Daily)
    # Need enough for ATR calculation
    data = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', progress=False, auto_adjust=False)
    
    reversals = []
    
    for t in tickers:
        try:
            df = data[t].copy()
            if df.empty: continue
            df = df.dropna()
            
            # 3. Calculate SuperTrend (10, 3)
            df = calculate_supertrend(df, period=10, multiplier=3.0)
            
            # 4. Check for Reversal TODAY
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # BULLISH REVERSAL (Red Circle)
            # Trend changed from -1 (Bearish) to 1 (Bullish)
            if prev['Trend'] == -1 and current['Trend'] == 1:
                # Also verify the crossover logic explicitly just in case
                # Close > Upper Band of previous candle mostly
                
                pct_change = (current['Close'] - prev['Close']) / prev['Close'] * 100
                
                reversals.append({
                    "Ticker": t,
                    "Price": current['Close'],
                    "Signal": "BULLISH REVERSAL (Buy)",
                    "Stop_Loss": current['SuperTrend'], # The new lower band
                    "Change": f"{pct_change:+.2f}%",
                    "Date": current.name.strftime('%Y-%m-%d')
                })
                
        except Exception:
            continue
            
    # 5. Export
    if reversals:
        res_df = pd.DataFrame(reversals)
        print("\n--- TREND REVERSALS FOUND ---")
        print(res_df)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        csv_path = os.path.join(output_dir, "trend_reversals.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")
    else:
        print("\nNo Trend Reversals detected today.")
        # Create empty file to prevent errors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        pd.DataFrame(columns=["Ticker", "Price", "Signal", "Stop_Loss", "Change", "Date"]).to_csv(os.path.join(output_dir, "trend_reversals.csv"), index=False)

if __name__ == "__main__":
    run_scanner()
