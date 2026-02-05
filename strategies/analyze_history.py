import pandas as pd
import numpy as np
import yfinance as yf
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

def analyze_history():
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    
    # 20 Years Ago (Start Date)
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    print(f"Downloading Data since {start_date}...")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # 1. CUMULATIVE WINNER (2006-2026)
    # Total Return for every stock
    start_prices = data.iloc[0]
    end_prices = data.iloc[-1]
    
    total_returns = (end_prices - start_prices) / start_prices
    lifetime_winners = total_returns.sort_values(ascending=False).head(5)
    
    print("\n--- LIFETIME CHAMPIONS (2006-2026) ---")
    print("(If you bought and held these for 20 years)")
    for t, r in lifetime_winners.items():
        print(f"{t}: +{r*100:,.0f}%")
        
    # 2. SNAPSHOT WINNER (Who was winning in 2006?)
    # We look at 3-Month Momentum in early 2006
    # Let's say April 2006 (after 3 months of data)
    
    monthly_prices = data.resample("M").last()
    mom_3m = monthly_prices.pct_change(3)
    
    # Find the index closest to 2006-05-01 (Start of sim roughly)
    # The simulation usually starts after 1 year (12 months) of warmup if 12m momentum
    # But for fast momentum we can start earlier.
    # Let's check the first available valid rank.
    
    valid_idx = 4 # Month 4
    if len(monthly_prices) > valid_idx:
        date_2006 = monthly_prices.index[valid_idx]
        scores_2006 = mom_3m.iloc[valid_idx]
        winners_2006 = scores_2006.sort_values(ascending=False).head(5)
        
        print(f"\n--- THE HOT STOCKS OF {date_2006.strftime('%Y')} (20 Years Ago) ---")
        print("(These are the stocks the algorithm bought first)")
        for t, r in winners_2006.items():
            print(f"{t}: +{r*100:,.1f}% (3-Month Momentum)")

if __name__ == "__main__":
    analyze_history()
