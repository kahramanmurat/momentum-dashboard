import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sys
from io import StringIO
from datetime import datetime, timedelta

def get_nasdaq_100_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "MU", "WDC", "LRCX"]
        tables = pd.read_html(StringIO(response.text))
        target_df = tables[4] if len(tables) > 4 else tables[0]
        col = next((c for c in target_df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
        if col: return [t.replace(".", "-").strip() for t in target_df[col].astype(str).tolist()]
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "MU"]
    except: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "MU"]

def check_history(target_ticker):
    print(f"--- CHECKING RANK HISTORY FOR {target_ticker} (Last 12 Months) ---")
    
    tickers = get_nasdaq_100_tickers()
    if target_ticker not in tickers: tickers.append(target_ticker)
    
    # Download 2 years of data to be safe for 3M calc
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    print("Downloading data...")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # Resample to Monthly
    m_data = data.resample("M").last()
    
    # Calculate 3-Month Momentum for every month
    mom_3m = m_data.pct_change(3)
    
    # Iterate backwards 12 months
    print(f"\n{'Date':<15} {'Rank':<10} {'3M Return':<15} {'Status'}")
    print("-" * 60)
    
    consecutive_months = 0
    
    # Get last 13 periods (Today + 12 months back)
    for i in range(1, 14):
        idx = -i
        if idx < -len(mom_3m): break
        
        date = mom_3m.index[idx]
        row = mom_3m.iloc[idx].dropna().sort_values(ascending=False)
        
        if target_ticker in row.index:
            rank = row.index.get_loc(target_ticker) + 1
            ret = row[target_ticker]
            
            status = ""
            if rank <= 5: 
                status = "✅ TOP 5 (BUY)"
                consecutive_months += 1
            elif rank <= 10:
                status = "⚠️ Top 10"
            else:
                status = "❌ Weak"
                
            print(f"{date.strftime('%Y-%m-%d'):<15} #{rank:<9} {ret*100:+.1f}%          {status}")
        else:
            print(f"{date.strftime('%Y-%m-%d'):<15} N/A")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MU"
    check_history(ticker)
