import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime, timedelta

def get_support_resistance(series, current_price, window=5, threshold=0.02):
    """
    Identifies Support and Resistance levels based on local swing points.
    Returns nearest_support, nearest_resistance.
    """
    # 1. Identify Local Min/Max (Swing Points)
    # We use a rolling window approach roughly simulating argrelextrema
    # A swing high is higher than N candles before and after.
    
    # Simple vectorization for window=5 (2 before, 2 after) doesn't exist easily in pandas rolling without apply.
    # We will iterate or use shift logic. Shift is fast.
    
    high_swing = (series > series.shift(1)) & (series > series.shift(2)) & \
                 (series > series.shift(-1)) & (series > series.shift(-2))
                 
    low_swing = (series < series.shift(1)) & (series < series.shift(2)) & \
                (series < series.shift(-1)) & (series < series.shift(-2))
                
    resistance_points = series[high_swing]
    support_points = series[low_swing]
    
    # 2. Cluster Levels (Sensitivity)
    # If multiple levels are close (within `threshold`), treat as one zone.
    # We focus on levels relevant to current price (+/- 15% range usually, but let's look broadly).
    
    levels = []
    
    # Concatenate all meaningful pivots
    all_pivots = pd.concat([resistance_points, support_points])
    
    # Filter to last 12 months (series is already 12M usually, but ensure recency weight?)
    # Actually, old levels assume importance too.
    
    if all_pivots.empty:
        return 0, 0
        
    # Valid levels must be somewhat distinct.
    # Simple Alg: Sort, then group.
    sorted_pivots = all_pivots.sort_values()
    
    unique_levels = []
    if not sorted_pivots.empty:
        curr_group = [sorted_pivots.iloc[0]]
        
        for p in sorted_pivots.iloc[1:]:
            # If within threshold of the group average
            avg = sum(curr_group) / len(curr_group)
            if abs(p - avg) / avg < threshold:
                curr_group.append(p)
            else:
                # Close group
                unique_levels.append(sum(curr_group) / len(curr_group))
                curr_group = [p]
        unique_levels.append(sum(curr_group) / len(curr_group))
        
    # 3. Find Nearest
    # Supports are < Current Price
    supports = [l for l in unique_levels if l < current_price]
    resistances = [l for l in unique_levels if l > current_price]
    
    nearest_support = supports[-1] if supports else 0 # Highest Support below price
    nearest_resistance = resistances[0] if resistances else 0 # Lowest Resistance above price
    
    return nearest_support, nearest_resistance

def run_etf_scan():
    print("--- ETF SCANNER: MULTI-TIMEFRAME PERFORMANCE ---")
    
    # 1. THE UNIVERSE (Sectors + Indices + Commodities)
    etfs = {
        # Major Indices
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "DIA": "Dow Jones",
        "IWM": "Russell 2000",
        
        # Sectors
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLC": "Comms",
        "XLY": "Discretionary",
        "XLP": "Staples",
        "XLU": "Utilities",
        "XLB": "Materials",
        "IYR": "Real Estate",
        
        # Thematic / Industry
        "SMH": "Semiconductors",
        "XBI": "Biotech",
        "GDX": "Gold Miners",
        "ARKK": "Innovation",
        
        # Commodities & Bonds
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "Oil",
        "TLT": "20+ Yr Treasury",
        
        # Leveraged (Direxion 3x/2x) - High Volatility
        "TQQQ": "Nasdaq 3x Bull",
        "SQQQ": "Nasdaq 3x Bear",
        "SPXL": "S&P 500 3x Bull",
        "SPXS": "S&P 500 3x Bear",
        "SOXL": "Semi 3x Bull",
        "SOXS": "Semi 3x Bear",
        "FAS": "Financials 3x Bull",
        "FAZ": "Financials 3x Bear",
        "TECL": "Tech 3x Bull",
        "TECS": "Tech 3x Bear",
        "LABU": "Biotech 3x Bull",
        "LABD": "Biotech 3x Bear",
        "YINN": "China 3x Bull",
        "YANG": "China 3x Bear",
        "NUGT": "Gold Miners 2x Bull",
        "DUST": "Gold Miners 2x Bear",
        "ERX": "Energy 2x Bull",
        "ERY": "Energy 2x Bear"
    }
    
    tickers = list(etfs.keys())
    
    print(f"Scanning {len(tickers)} ETFs...")
    
    # Download Data (Last 1 Year is enough for 6M calc with buffer)
    # We need roughly 130 trading days for 6 Months. 2 Years is safe.
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d") 
    
    # Use 'Close' for simplicity, or 'Adj Close' if dividends matter significantly for this view. 
    # Usually traders look at Price Action, so 'Close' is fine, but yfinance auto_adjust=True gives Adj Close in Close col.
    # Let's use auto_adjust=False to get raw price, which matches what people see on charts usually unless they use adjusted charts.
    # The existing sector_dashboard uses auto_adjust=False.
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # Check if data download was successful (sometimes single ticker download returns Series)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    results = []
    
    # Timeframe offsets (Trading Days approx)
    periods = {
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126
    }
    
    for ticker in tickers:
        try:
            if ticker not in data.columns:
                print(f"Data missing for {ticker}")
                continue
                
            series = data[ticker].dropna()
            if series.empty:
                continue
                
            current_price = series.iloc[-1]
            
            # Calculate Metrics & Signals
            row = {
                "Ticker": ticker,
                "Name": etfs.get(ticker, ticker),
                "Current Price": current_price
            }
            
            # --- Technical Indicators ---
            # SMA
            sma_20 = series.rolling(window=20).mean().iloc[-1]
            sma_200 = series.rolling(window=200).mean().iloc[-1]
            
            # Bollinger Bands (20, 2)
            std_20 = series.rolling(window=20).std().iloc[-1]
            upper_band = sma_20 + (2 * std_20)
            lower_band = sma_20 - (2 * std_20)
            
            # RSI (14)
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs)) if avg_loss.iloc[-1] != 0 else 50
            
            # Key Levels (Support / Resistance)
            support_level, res_level = get_support_resistance(series, current_price)
            
            # Determine Signal
            signal = "WAIT"
            
            # 1. DOWNTREND Check
            if current_price < sma_200:
                 signal = "DOWNTREND (Avoid)"
            else:
                # UPTREND (Price > SMA 200)
                
                # 2. CRASH / OVERSOLD (Extreme Value)
                if rsi < 30:
                     signal = "OVERSOLD (Snipe?)"
                     
                # 3. BUY STRUCTURE (Test of Key Support)
                # If Price is within 1.5% of Key Support (and > Support)
                elif support_level > 0 and (current_price - support_level)/current_price < 0.015:
                     signal = "BUY STRUCTURE (Support)"
                     
                # 4. BUY ZONE (Dynamic Pullback)
                # Either touching Lower Band, OR RSI < 55 while Up Trending
                elif current_price < lower_band or (current_price < sma_20 and rsi < 55):
                    signal = "BUY ZONE (Pullback)"
                
                # 5. EXTENDED (Dynamic Profit Take)
                elif current_price > upper_band or rsi > 70:
                    signal = "EXTENDED (Take Profit)"
                    
                # 6. BREAKOUT (Just crossed Resistance)
                elif res_level > 0 and (res_level - current_price)/current_price < 0.01:
                     signal = "BREAKOUT WATCH"
                    
                # 7. MOMENTUM UP (Acceleration)
                else:
                    signal = "UPTREND (Hold)"
            
            # Calculate metrics for each period
            ret_1m = 0.0
            ret_3m = 0.0
            
            for label, days in periods.items():
                if len(series) > days:
                    prev_price = series.iloc[-(days+1)] # N days ago
                    
                    price_change = current_price - prev_price
                    pct_change = (price_change / prev_price) 
                    
                    row[f"{label} Price Chg"] = price_change
                    row[f"{label} % Chg"] = pct_change
                    
                    if label == "1M": ret_1m = pct_change
                    if label == "3M": ret_3m = pct_change
                else:
                    row[f"{label} Price Chg"] = 0.0
                    row[f"{label} % Chg"] = 0.0
            
            # Refine Momentum Signal using Returns
            if signal == "UPTREND (Hold)":
                if ret_1m > ret_3m and ret_1m > 0:
                     signal = "MOMENTUM UP (Add)"
            
            row["Signal"] = signal
            row["RSI"] = rsi
            row["SMA_20"] = sma_20
            row["SMA_200"] = sma_200
            row["Support_Level"] = support_level
            row["Res_Level"] = res_level
            
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Export
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, "etf_scan_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    run_etf_scan()
