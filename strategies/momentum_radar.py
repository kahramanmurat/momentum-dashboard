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

def run_momentum_radar():
    print("--- MOMENTUM RADAR: TRACKING THE CLIMBERS ---")
    
    # 1. Fetch Universe
    tickers = get_nasdaq_100_tickers()
    print(f"Scanning {len(tickers)} Stocks for Momentum Acceleration...")
    
    # 2. Download Data (Last 1 Year)
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # 3. Calculate 3-Month Momentum (The Engine)
    # We need "Today's Score" and "Last Month's Score"
    # To do this correctly, we calculate rolling 3M returns
    
    returns_3m = data.pct_change(63) # approx 63 trading days in 3 months
    
    # Snapshot 1: TODAY (Latest Data)
    latest_r = returns_3m.iloc[-1].dropna()
    
    # Snapshot 2: 1 MONTH AGO (Approx 21 trading days ago)
    prev_r = returns_3m.iloc[-22].dropna()
    
    # 4. RANKING
    # Rank 1 = Highest Return
    rank_today = latest_r.rank(ascending=False)
    rank_prev = prev_r.rank(ascending=False)
    
    # Combine
    df = pd.DataFrame({
        "Current_3M_Ret": latest_r,
        "Current_Rank": rank_today
    })
    
    df["Prev_Rank"] = rank_prev
    
    # 5. CLIMB SCORE (Acceleration)
    # Positive = Climbing (Rank 50 -> Rank 20 = +30)
    df["Climb_Score"] = df["Prev_Rank"] - df["Current_Rank"]
    
    # Clean up (stocks that didn't exist in both periods)
    df = df.dropna()
    
    # 6. FILTERS
    relevant = df[df["Current_Rank"] <= 50]
    
    # Must be in top 50 NOW, but was > 80 before (Huge jump) OR simply highest climb score
    rising_stars = relevant.sort_values(by="Climb_Score", ascending=False).head(15)
    
    # --- NEW: Calculate 20-Day SMA for Rising Stars ---
    # We need to calculate SMA_20 per ticker from 'data'
    # 'data' columns are tickers.
    smas = []
    for t in rising_stars.index:
        try:
            if t in data.columns:
                series = data[t].dropna()
                if len(series) >= 20:
                    val = series.rolling(window=20).mean().iloc[-1]
                    smas.append(val)
                else:
                    smas.append(0.0)
            else:
                smas.append(0.0)
        except:
            smas.append(0.0)
            
    rising_stars['SMA_20'] = smas
    
    # falling stars (for comparison)
    falling_stars = relevant.sort_values(by="Climb_Score", ascending=True).head(5)
    
    print("\n--- RISING STARS (Gaining Momentum) ---")
    print("Stocks charging up the leaderboard in the last 30 days:")
    
    # --- NEW: Fetch Metadata ---
    sectors = []
    industries = []
    options_avail = []
    
    print("Fetching Metadata for Rising Stars...")
    for t in rising_stars.index:
        try:
            t_obj = yf.Ticker(t)
            info = t_obj.info
            sectors.append(info.get('sector', 'N/A'))
            industries.append(info.get('industry', 'N/A'))
            options_avail.append("YES" if len(t_obj.options) > 0 else "No")
        except:
            sectors.append('N/A')
            industries.append('N/A')
            options_avail.append("No")
            
    rising_stars['Sector'] = sectors
    rising_stars['Industry'] = industries
    rising_stars['Options_Avail'] = options_avail
    
    print(rising_stars[["Current_Rank", "Prev_Rank", "Climb_Score", "Sector", "Industry", "Options_Avail", "SMA_20"]])
    
    # 7. EXPORT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    csv_path_rise = os.path.join(output_dir, "rising_stars.csv")
    rising_stars.to_csv(csv_path_rise)
    print(f"\nRising Stars saved to: {csv_path_rise}")
    
    # Visual Polish
    # Top 3 Climbers
    climbers = rising_stars.index[:3].tolist()
    print(f"\n[!] WATCHLIST: {', '.join(climbers)} are moving FAST.")

if __name__ == "__main__":
    run_momentum_radar()
