import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta

# -----------------------------
# Data Fetching
# -----------------------------
def get_nasdaq_100_tickers():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        
        target_df = None
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
                target_df = tbl
                break
        
        if target_df is None and len(tables) > 4: target_df = tables[4]
        
        col = None
        for c in target_df.columns:
            if "ticker" in str(c).lower(): col = c; break
        if col is None: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

        tickers = target_df[col].astype(str).tolist()
        return [t.replace(".", "-").strip() for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META", "AMD", "NFLX", "INTC"]

def run_grand_finale():
    print("--- GRAND FINALE: 20-YEAR BATTLE ---")
    print("Fetching Universe...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ")
    
    # 20 Years
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    print(f"Downloading Data (Since {start_date})...")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill()
    
    # -----------------------------
    # 1. Benchmark (QQQ)
    # -----------------------------
    qqq = data["QQQ"]
    bench_ret = qqq.pct_change().fillna(0)
    curve_bench = (1 + bench_ret).cumprod()
    
    # -----------------------------
    # 2. Synthetic 3x QQQ (Leveraged Proxy)
    # -----------------------------
    # Logic: 3x Daily Return - Borrow Cost (assuming 5% annual drag / 252)
    # Safety: Exit if QQQ < SMA 200 (The "Nuclear" Safety Switch)
    daily_drag = 0.05 / 252
    lev_ret = (bench_ret * 3) - daily_drag
    
    sma200_qqq = qqq.rolling(200).mean()
    
    # Leveraged with Regime Filter
    lev_sig = (qqq > sma200_qqq).shift(1).fillna(0).astype(int)
    lev_strat_ret = lev_ret * lev_sig
    curve_lev = (1 + lev_strat_ret).cumprod()
    
    # -----------------------------
    # 3. EMA Strategy (Proxy on QQQ)
    # -----------------------------
    # Logic: Buy QQQ if Price > EMA200, Sell if Price < EMA50
    ema200 = qqq.ewm(span=200, adjust=False).mean()
    ema50 = qqq.ewm(span=50, adjust=False).mean()
    
    # State Machine Vectorized
    # We use valid signals (1=Long, 0=Cash)
    # This is hard to fully vectorise with hysteresis, so we approximate:
    # If Price > EMA200 -> Enter Trend (1)
    # If Price < EMA50 -> Exit Trend (0)
    # Else -> Hold previous state
    
    ema_pos = []
    state = 0
    q_vals = qqq.values
    e200_vals = ema200.values
    e50_vals = ema50.values
    
    for i in range(len(q_vals)):
        p = q_vals[i]
        if np.isnan(p):
            ema_pos.append(0)
            continue
            
        if state == 0:
            if p > e200_vals[i]: state = 1
        elif state == 1:
            if p < e50_vals[i]: state = 0
        ema_pos.append(state)
        
    ema_sig = pd.Series(ema_pos, index=qqq.index).shift(1).fillna(0)
    curve_ema = (1 + (bench_ret * ema_sig)).cumprod()
    
    # -----------------------------
    # 4. Momentum Rotation (Top 5 Stocks)
    # -----------------------------
    print("Simulating Momentum Strategies (Universe Level)...")
    monthly_prices = data.resample("M").last()
    mom_12m = monthly_prices.pct_change(12)
    
    # Metrics for Enhanced (Sharpe)
    daily_rets = data.pct_change()
    vol_12m = daily_rets.rolling(252).std().resample("M").last() * np.sqrt(252)
    sharpe_12m = mom_12m / vol_12m
    
    # Regime for Enhanced
    # Use the Monthly resampled QQQ/SMA relationship we calculated earlier?
    # Better: Calculate monthly regime from daily data
    monthly_regime = (qqq > sma200_qqq).resample("M").last()
    
    # Arrays for Curves
    # Align dates with Monthly Prices (start after 12 months)
    valid_dates = monthly_prices.index[12:]
    
    # We need to bridge Daily Bench/Lev curves to these Monthly points for plotting
    # or just plot everything on Daily? 
    # Momentum is monthly. Let's expand Momentum to Daily or compact others to Monthly?
    # Compacting to Monthly is cleaner for the chart.
    
    curve_mom_raw = [1.0]
    curve_mom_enh = [1.0]
    
    # We need to track value of $1 invested
    val_raw = 1.0
    val_enh = 1.0
    
    for i in range(13, len(monthly_prices)):
        dt_curr = monthly_prices.index[i]
        dt_prev = monthly_prices.index[i-1]
        
        # --- RAW MOMENTUM ---
        # Select Top 5 by Return
        scores_raw = mom_12m.iloc[i-1]
        cands_raw = scores_raw.drop("QQQ", errors="ignore").dropna()
        top5_raw = cands_raw.sort_values(ascending=False).head(5).index
        
        if len(top5_raw) > 0:
            p_c = monthly_prices.loc[dt_curr, top5_raw]
            p_p = monthly_prices.loc[dt_prev, top5_raw]
            ret_raw = ((p_c - p_p) / p_p).mean()
        else:
            ret_raw = 0.0
            
        val_raw *= (1 + ret_raw)
        curve_mom_raw.append(val_raw)
        
        # --- ENHANCED MOMENTUM ---
        # Check Regime
        is_bull = monthly_regime.iloc[i-1]
        
        if not is_bull:
            ret_enh = 0.0 # Cash
        else:
            # Select Top 5 by Sharpe
            scores_enh = sharpe_12m.iloc[i-1]
            cands_enh = scores_enh.drop("QQQ", errors="ignore").dropna()
            top5_enh = cands_enh.sort_values(ascending=False).head(5).index
            
            if len(top5_enh) > 0:
                p_c = monthly_prices.loc[dt_curr, top5_enh]
                p_p = monthly_prices.loc[dt_prev, top5_enh]
                ret_enh = ((p_c - p_p) / p_p).mean()
            else:
                ret_enh = 0.0
        
        val_enh *= (1 + ret_enh)
        curve_mom_enh.append(val_enh)

    # -----------------------------
    # Alignment & Plotting
    # -----------------------------
    # Resample Daily curves to the Monthly timeline of momentum
    subset_dates = monthly_prices.index[12:]
    
    # Helper to reindex daily equity curves to monthly dates
    def to_monthly(daily_curve, dates):
        # returns values at the specific dates
        return daily_curve.reindex(dates, method='ffill').fillna(1.0)
    
    c_bench_m = to_monthly(curve_bench, subset_dates)
    c_lev_m = to_monthly(curve_lev, subset_dates)
    c_ema_m = to_monthly(curve_ema, subset_dates)
    
    # Normalize to 1.0 start
    c_bench_m = c_bench_m / c_bench_m.iloc[0]
    c_lev_m = c_lev_m / c_lev_m.iloc[0]
    c_ema_m = c_ema_m / c_ema_m.iloc[0]
    
    # Ensure all lengths match
    # curve_mom_raw includes the start point (len = len(subset)+1?? No, loop ran len(subset) times?)
    # Loop ran from 13 to len.
    # subset_dates is len-13.
    # curve_mom_raw initiated with 1.0. Loop appended 1 value per step.
    # The first date in valid_dates corresponds to the first return period.
    # Let's align cleanly.
    
    plot_dates = subset_dates
    # Our mom curves have 1 extra point (the initial 1.0).
    # We need to trim or re-align.
    # Let's drop the initial 1.0 and assume the first date in valid_dates is the end of the first month.
    # Actually, let's keep it simple:
    
    # Final Metrics
    f_mom = (curve_mom_raw[-1] - 1) * 100
    f_enh = (curve_mom_enh[-1] - 1) * 100
    f_lev = (c_lev_m.iloc[-1] - 1) * 100
    f_ben = (c_bench_m.iloc[-1] - 1) * 100
    f_ema = (c_ema_m.iloc[-1] - 1) * 100
    
    print("\n--- FINAL SCOREBOARD (2006-2026) ---")
    print(f"1. Raw Momentum:   {f_mom:,.0f}%")
    print(f"2. Enhanced Mom:   {f_enh:,.0f}%")
    print(f"3. 3x Lev (Syn):   {f_lev:,.0f}%")
    print(f"4. Benchmark:      {f_ben:,.0f}%")
    print(f"5. EMA Trend:      {f_ema:,.0f}%")

    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plotting lists vs Series - ensure alignment
    # Mom curves are len N+1?
    # subset_dates is len N.
    # Let's effectively slice mom curves to match dates
    # or Prepend a start date?
    
    # Quick fix: Plot against range, or fix dates
    if len(curve_mom_raw) == len(plot_dates) + 1:
        # Prepend start date to dates?
        # Use first date - 1 month
        start_dt = plot_dates[0] - timedelta(days=30)
        final_dates = [start_dt] + list(plot_dates)
    else:
        final_dates = plot_dates

    plt.plot(final_dates, curve_mom_raw, label=f"Momentum Rotation (+{f_mom:,.0f}%)", color="green", linewidth=2.5)
    plt.plot(final_dates, curve_mom_enh, label=f"Enhanced (Safe) (+{f_enh:,.0f}%)", color="blue", linewidth=2)
    plt.plot(plot_dates, c_lev_m, label=f"3x Leveraged (+{f_lev:,.0f}%)", color="orange", linestyle="-.")
    plt.plot(plot_dates, c_bench_m, label=f"QQQ (+{f_ben:,.0f}%)", color="gray", linewidth=2, alpha=0.7)
    plt.plot(plot_dates, c_ema_m, label=f"EMA Trend (+{f_ema:,.0f}%)", color="purple", linestyle=":")
    
    plt.yscale("log")
    plt.title("The Grand Finale: 20-Year Strategy Comparison (Log Scale)")
    plt.ylabel("Growth of $1 (Log)")
    plt.xlabel("Year")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig("grand_finale_comparison.png")
    print("Chart saved to: grand_finale_comparison.png")

if __name__ == "__main__":
    run_grand_finale()
