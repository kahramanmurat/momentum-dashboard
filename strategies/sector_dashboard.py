import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def run_sector_radar():
    print("--- SECTORS RADAR: TRACKING THE FLOW OF MONEY ---")
    
    # 1. THE UNIVERSE (S&P 500 Sectors + Key Industries)
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLC": "Comms (Meta/Goog)",
        "XLY": "Discretionary (Amzn/Tsla)",
        "XLP": "Staples (Safe)",
        "XLU": "Utilities (Safe)",
        "XLB": "Materials",
        "IYR": "Real Estate",
        "SMH": "Semiconductors (Leader)",
        "XBI": "Biotech (High Beta)",
        "GDX": "Gold Miners (Inflation)",
        "ARKK": "Innovation (Speculative)",
        "SPY": "S&P 500 (Baseline)"
    }
    
    tickers = list(sectors.keys())
    
    print(f"Scanning {len(tickers)} Sectors/Industries...")
    
    # Download Data (Last 2 Years is enough for radar)
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # 2. CALCULATE MOMENTUM (The "Speed")
    # We want multiple timeframes to see the rotation
    
    # Current Prices
    latest_price = data.iloc[-1]
    
    # 1 Month Ago (approx 21 trading days)
    price_1m = data.iloc[-22]
    
    # 3 Months Ago (approx 63 trading days)
    price_3m = data.iloc[-64]
    
    # 6 Months Ago (approx 126 trading days)
    price_6m = data.iloc[-127]
    
    # Returns
    ret_1m = (latest_price - price_1m) / price_1m
    ret_3m = (latest_price - price_3m) / price_3m
    ret_6m = (latest_price - price_6m) / price_6m
    
    # Create Dashboard
    df = pd.DataFrame({
        "Ticker": tickers,
        "Name": [sectors[t] for t in tickers]
    }).set_index("Ticker")
    
    df["1M_Return"] = ret_1m
    df["3M_Return"] = ret_3m
    df["6M_Return"] = ret_6m
    
    # 3. RANKING (Who is winning?)
    df["1M_Rank"] = df["1M_Return"].rank(ascending=False)
    df["3M_Rank"] = df["3M_Return"].rank(ascending=False)
    df["6M_Rank"] = df["6M_Return"].rank(ascending=False)
    
    # 4. ROTATION ALERTS (The "Alpha")
    # UPDATED: Use 3M Rank as baseline because 3M is our "Strategy Timeframe".
    # Scenario: "Acceleration" (Bad 3M Rank, Good 1M Rank) -> Moving faster than the trend.
    
    df["Rotation_Score"] = df["3M_Rank"] - df["1M_Rank"] 
    # Positive score = Accelerating vs 3M Trend
    # Negative score = Decelerating vs 3M Trend
    
    # Format for display
    display_df = df.sort_values(by="1M_Return", ascending=False).copy()
    
    # Convert to percentage strings
    out_df = display_df.copy()
    for c in ["1M_Return", "3M_Return", "6M_Return"]:
        out_df[c] = out_df[c].apply(lambda x: f"{x*100:+.1f}%")
        
    print("\n--- SECTOR LEADERBOARD (Sorted by 1-Month Strength) ---")
    print(out_df[["Name", "1M_Return", "3M_Return", "Rotation_Score"]])
    
    # Detect specific signals
    print("\n--- ROTATION ALERTS ---")
    
    # Acceleration
    accel = display_df[ (display_df["1M_Rank"] < display_df["3M_Rank"] - 3) ]
    if not accel.empty:
        print("[!] ACCELERATING (Buy): Faster than their 3M Trend:")
        for t, row in accel.iterrows():
            print(f"    - {row['Name']} ({t}): 3M Rank #{int(row['3M_Rank'])} -> 1M Rank #{int(row['1M_Rank'])}")
            
    # Deceleration
    decel = display_df[ (display_df["1M_Rank"] > display_df["3M_Rank"] + 3) ]
    if not decel.empty:
        print("[!] DECELERATING (Sell): Slower than their 3M Trend:")
        for t, row in decel.iterrows():
            print(f"    - {row['Name']} ({t}): 3M Rank #{int(row['3M_Rank'])} -> 1M Rank #{int(row['1M_Rank'])}")
            
    # 5. EXPORT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, "sector_rotation_report.csv")
    out_df.to_csv(csv_path)
    print(f"\nReport saved to: {csv_path}")
    
    # 6. VISUALIZATION (Heatmap / Scatter)
    plt.figure(figsize=(16, 12))
    
    # Scatter: X = 3M Return (Trend), Y = 1M Return (Momentum)
    x = df["3M_Return"] * 100
    y = df["1M_Return"] * 100
    
    # Color by Rotation Score
    colors = df["Rotation_Score"]
    
    # Plot Scatter
    scatter = plt.scatter(x, y, c=colors, cmap="RdYlGn", alpha=0.6, s=1500, edgecolors="black", zorder=2)
    plt.colorbar(scatter, label="Rotation Score vs 3M (Green=Accelerating, Red=Decelerating)")
    
    # Add labels with 3M Rank (The Strategy)
    import matplotlib.patheffects as pe
    
    names = df.index
    ranks_3m = df["3M_Rank"]
    
    for i, txt in enumerate(names):
        rank_val = int(ranks_3m.iloc[i])
        # Multi-line label: "XLE\n#10"
        label_text = f"{txt}\n#{rank_val}"
        
        t = plt.text(x.iloc[i], y.iloc[i], label_text, 
                     ha='center', va='center', 
                     fontsize=11, fontweight='bold', color='black', zorder=10)
        t.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
        
    # Draw quadrants
    plt.axhline(0, color='black', linewidth=1, linestyle="--", zorder=1)
    plt.axvline(0, color='black', linewidth=1, linestyle="--", zorder=1)
    
    plt.title("Sector Rotation Radar (Aligned to 3-Month Strategy)\n(Labels: Ticker & 3-Month Rank)", fontsize=16)
    plt.xlabel("3-Month Trend (%)", fontsize=14)
    plt.ylabel("1-Month Momentum (%)", fontsize=14)
    plt.grid(True, alpha=0.3, zorder=0)
    
    chart_path = os.path.join(output_dir, "sector_heatmap.png")
    plt.savefig(chart_path)
    print(f"Chart saved to: {chart_path}")

if __name__ == "__main__":
    run_sector_radar()
