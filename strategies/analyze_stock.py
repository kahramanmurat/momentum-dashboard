import pandas as pd
import numpy as np
import yfinance as yf
import sys
from datetime import datetime, timedelta

def analyze_ticker(ticker):
    print(f"--- ANALYZING {ticker} ---")
    
    # 1. MONTHLY MOMENTUM (The Engine)
    print("1. Checking 3-Month Momentum Rank...")
    # diverse set to get a rank
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC", "QCOM", "AVGO", "TXN", "HON", "SBUX"]
    if ticker not in tickers: tickers.append(ticker)
    
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    m_data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].resample("M").last()
    
    mom_3m = m_data.pct_change(3).iloc[-1].sort_values(ascending=False)
    
    try:
        rank = mom_3m.index.get_loc(ticker) + 1
        score = mom_3m[ticker]
        print(f"   - Rank: #{rank} of {len(tickers)}")
        print(f"   - 3-Month Return: {score*100:+.1f}%")
        if rank <= 5: print("   - STATUS: ✅ BUY LIST (Top 5)")
        else: print(f"   - STATUS: ❌ AVOID (Rank {rank} is too low)")
    except:
        print("   - Error determining rank.")

    # 2. HOURLY VOLATILITY (The Swing)
    print("\n2. Checking Hourly Squeeze (60 Days)...")
    h_data = yf.download(ticker, period="59d", interval="1h", progress=False, auto_adjust=False)
    
    if not h_data.empty:
        h_data['SMA'] = h_data['Close'].rolling(20).mean()
        h_data['STD'] = h_data['Close'].rolling(20).std()
        h_data['Upper'] = h_data['SMA'] + (2 * h_data['STD'])
        h_data['Lower'] = h_data['SMA'] - (2 * h_data['STD'])
        h_data['Bandwidth'] = (h_data['Upper'] - h_data['Lower']) / h_data['SMA']
        
        last = h_data.iloc[-1]
        bw = last['Bandwidth']
        price = last['Close']
        
        print(f"   - Price: ${price.item():.2f}")
        print(f"   - Bandwidth: {bw.item()*100:.2f}%")
        
        if bw.item() < 0.03: print("   - STATUS: ⚠️ COILING (Get Ready)")
        elif price.item() > last['Upper'].item(): print("   - STATUS: ✅ BREAKING OUT")
        elif price.item() < last['Lower'].item(): print("   - STATUS: 🛑 BREAKING DOWN")
        else: print("   - STATUS: 💤 NORMAL (No Setup)")
    else:
        print("   - No hourly data available.")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    analyze_ticker(ticker)
