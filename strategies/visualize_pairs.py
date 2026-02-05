import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def visualize_pairs(csv_file="pairs_opportunities.csv", output_file="pairs_analysis.png", lookback_days=252):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Run statarb_scanner.py first.")
        return

    if df.empty:
        print("No pairs found in CSV to visualize.")
        return

    # Limit to top 3 pairs to keep chart readable
    top_pairs = df.head(3)
    
    n_pairs = len(top_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(15, 5 * n_pairs))
    if n_pairs == 1:
        axes = np.array([axes]) # Ensure 2D array for indexing

    print(f"Visualizing {n_pairs} pairs...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    for i, row in top_pairs.iterrows():
        t1 = row["Ticker_A"]
        t2 = row["Ticker_B"]
        beta = row["Hedge_Ratio"]
        sector = row["Sector"]
        
        # Download Data
        data = yf.download([t1, t2], start=start_date, end=end_date, progress=False, auto_adjust=False)["Close"]
        
        # Handle single-level columns if only 2 tickers downloaded (rare edge case with yfinance)
        # But usually yfinance returns multi-index or single level depending on how many requested.
        # Safest way: access by column name directly
        s1 = data[t1].ffill()
        s2 = data[t2].ffill()
        
        # -----------------------------
        # Plot 1: Normalized Prices
        # -----------------------------
        ax_price = axes[i, 0]
        
        # Normalize to start at 1.0
        s1_norm = s1 / s1.iloc[0]
        s2_norm = s2 / s2.iloc[0]
        
        ax_price.plot(s1_norm.index, s1_norm, label=t1, color="#1f77b4")
        ax_price.plot(s2_norm.index, s2_norm, label=t2, color="#ff7f0e")
        ax_price.set_title(f"{t1} vs {t2} ({sector}) - Normalized Price")
        ax_price.legend()
        ax_price.grid(True, alpha=0.3)
        
        # -----------------------------
        # Plot 2: Z-Score Spread
        # -----------------------------
        ax_z = axes[i, 1]
        
        # Reconstruct Spread (Log Prices usually for stat arb, matching scanner logic)
        # Spread = log(A) - beta * log(B)
        spread = np.log(s1) - beta * np.log(s2)
        
        # Rolling Z-Score (30 day)
        mean = spread.rolling(30).mean()
        std = spread.rolling(30).std()
        z_score = (spread - mean) / std
        
        ax_z.plot(z_score.index, z_score, color="#2ca02c", label="Z-Score")
        ax_z.axhline(2.0, color="red", linestyle="--", alpha=0.7)
        ax_z.axhline(-2.0, color="red", linestyle="--", alpha=0.7)
        ax_z.axhline(0, color="black", linestyle="-", alpha=0.3)
        
        current_z = z_score.iloc[-1]
        ax_z.set_title(f"Spread Z-Score (Beta={beta:.2f}): {current_z:.2f}")
        ax_z.fill_between(z_score.index, z_score, 2.0, where=(z_score >= 2.0), color="red", alpha=0.3)
        ax_z.fill_between(z_score.index, z_score, -2.0, where=(z_score <= -2.0), color="red", alpha=0.3)
        ax_z.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Analysis saved to: {output_file}")

if __name__ == "__main__":
    visualize_pairs()
