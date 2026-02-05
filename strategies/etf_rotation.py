import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_etf_rotation():
    print("--- 3x LEVERAGED ETF ROTATION (SYNTHETIC) ---")
    
    # Map 3x ETFs to their 1x Underlying Proxies for synthetic history
    # TQQQ -> QQQ
    # SOXL -> SOXX (Semis)
    # TECL -> XLK (Tech)
    # FAS  -> XLF (Financials)
    # ERX  -> XLE (Energy)
    # UPRO -> SPY (S&P 500)
    
    # We download the UNDERLYING assets to generate history back to 2006
    tickers_map = {
        "QQQ": "TQQQ",
        "SOXX": "SOXL",
        "XLK": "TECL",
        "XLF": "FAS",
        "XLE": "ERX", 
        "SPY": "UPRO"
    }
    
    underlying = list(tickers_map.keys())
    
    print("Downloading Underlying Data (20 Years)...")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    data = yf.download(underlying, start=start_date, progress=False, auto_adjust=False)["Close"].ffill()
    
    # -----------------------------
    # GENERATE SYNTHETIC 3x DATA
    # -----------------------------
    print("Generating Synthetic 3x History...")
    # Daily Returns of Underlying
    daily_rets = data.pct_change()
    
    # Lev Cost (drag + expense ratio + borrowing cost)
    # Approx 3% drag per year spread over trading days
    daily_drag = 0.03 / 252 
    
    # Synthetic 3x Returns = (DailyRet * 3) - Drag
    # Note: Leveraged ETFs decay due to volatility drag mathematically, 
    # simple multiplication * 3 captures the daily rebalancing effect (and the decay).
    lev_rets = (daily_rets * 3) - daily_drag
    
    # Reconstruct Prices (Normalized to 1.0 start)
    lev_prices = (1 + lev_rets).cumprod()
    
    # Rename columns to ETF names
    lev_prices = lev_prices.rename(columns=tickers_map)
    
    # -----------------------------
    # REGIME FILTER (QQQ based)
    # -----------------------------
    qqq = data["QQQ"]
    qqq_sma200 = qqq.rolling(200).mean()
    monthly_regime = (qqq > qqq_sma200).resample("M").last()
    
    # -----------------------------
    # MOMENTUM (3-Month Relative Strength)
    # -----------------------------
    # We run momentum on the LEVERAGED prices (to capture the actual vol/trend of the ETF)
    # Or underlying? Usually better on the ETF itself as that's what we trade.
    monthly_lev_prices = lev_prices.resample("M").last()
    mom_3m = monthly_lev_prices.pct_change(3)
    
    # -----------------------------
    # SIMULATION
    # -----------------------------
    eq_curve = [1.0]
    dates = [monthly_lev_prices.index[13]]
    
    print("Running ETF Rotation (Top 2)...")
    
    start_idx = 13
    
    for i in range(start_idx, len(monthly_lev_prices)):
        dt_curr = monthly_lev_prices.index[i]
        dt_prev = monthly_lev_prices.index[i-1]
        dates.append(dt_curr)
        
        # 1. Regime Check
        is_bull = monthly_regime.iloc[i-1]
        
        if not is_bull:
            # Bear Market -> Cash
            ret = 0.0
        else:
            # Bull Market -> Top 2 ETFs
            scores = mom_3m.iloc[i-1]
            top_2 = scores.dropna().sort_values(ascending=False).head(2).index
            
            if len(top_2) > 0:
                p_c = monthly_lev_prices.loc[dt_curr, top_2]
                p_p = monthly_lev_prices.loc[dt_prev, top_2]
                ret = ((p_c - p_p) / p_p).mean()
            else:
                ret = 0.0
                
        eq_curve.append(eq_curve[-1] * (1 + ret))
        
    # Stats
    total_ret = (eq_curve[-1] - 1) * 100
    
    print("\n--- ETF ROTATION RESULTS (20 Years) ---")
    print(f"Strategy: Rotate Top 2 (TQQQ/SOXL/TECL/FAS/ERX)")
    print(f"Total Return: {total_ret:,.0f}%")
    print(f"Final Value of $10k: ${10000 * eq_curve[-1]:,.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, eq_curve, label=f"3x ETF Rotation (+{total_ret:,.0f}%)", color="purple", linewidth=2)
    
    plt.yscale("log")
    plt.title("Leveraged ETF Rotation (Synthetic History 2006-2026)")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("etf_rotation_results.png")
    print("Chart saved to: etf_rotation_results.png")

if __name__ == "__main__":
    run_etf_rotation()
