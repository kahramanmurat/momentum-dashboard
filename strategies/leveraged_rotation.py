import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_leveraged_strategy():
    # Universe: The "Nuclear Triad" of 3x Bull ETFs
    tickers = ["TQQQ", "SOXL", "TECL", "QQQ"] # QQQ for benchmark
    
    print(f"Fetching data for {tickers} (3 Years)...")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    
    # Download
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill().dropna()
    
    # Indicators
    # 1. Trend Filter: SMA 200 (Daily)
    sma200 = data.rolling(200).mean()
    
    # 2. Momentum Signal: Volatility Adjusted Momentum (20 Days)
    # Why short term? 3x ETFs move FAST. 12-month is too slow.
    # We want to catch the "Gamma Squeeze" moves.
    # Return / StdDev
    # Actually, usually getting the monthly return is enough for rotation, 
    # but let's stick to the plan: Vol Adjusted.
    
    # Let's resample to Monthly for the decision cycle
    # But we need Daily SMA data for the filter check ON the rebalance day.
    
    monthly_data = data.resample("M").last()
    
    # Simulation Loop
    equity = [1.0] # Strategy
    dates = [monthly_data.index[0]]
    
    bench_equity = [1.0]
    
    # Log
    holdings_log = []
    
    # Start after 200 days (for SMA)
    # Find first month-end after 200 days
    start_idx = 0
    for i in range(len(monthly_data)):
        if monthly_data.index[i] > data.index[200]:
            start_idx = i
            break
            
    print("Running Nuclear Simulation...")
    
    for i in range(start_idx, len(monthly_data)):
        curr_date = monthly_data.index[i]
        prev_date = monthly_data.index[i-1]
        
        # 1. DECISION PHASE (At T-1)
        # We look at data available at prev_date to decide what to hold for (prev_date -> curr_date)
        
        # Get momentum (1 Month Return)
        # Simple Return for speed and aggression
        # Lookback 1 month
        if i == 0: continue
        
        # Calculate Returns of the previous month to rank
        # Price(T-1) / Price(T-2)
        if i < 2: continue
        
        date_t_minus_1 = monthly_data.index[i-1]
        date_t_minus_2 = monthly_data.index[i-2]
        
        prices_t1 = monthly_data.loc[date_t_minus_1]
        prices_t2 = monthly_data.loc[date_t_minus_2]
        
        mom_1m = (prices_t1 - prices_t2) / prices_t2
        
        # Candidates (exclude QQQ)
        candidates = ["TQQQ", "SOXL", "TECL"]
        
        best_ticker = "CASH"
        best_score = -999
        
        for t in candidates:
            # SAFETY CHECK 1: Must be above SMA 200
            # Get SMA value at decision date (T-1)
            # Find closest daily index
            sma_val = sma200.loc[:date_t_minus_1, t].iloc[-1]
            price_val = data.loc[:date_t_minus_1, t].iloc[-1]
            
            if price_val > sma_val:
                # SAFETY CHECK 2: Positive Momentum
                if mom_1m[t] > 0:
                    score = mom_1m[t] # Simple Relative Strength
                    if score > best_score:
                        best_score = score
                        best_ticker = t
                        
        # 2. EXECUTION PHASE (T-1 to T)
        # Calculate return of the chosen asset
        if best_ticker == "CASH":
            step_ret = 0.0 # Safety
        else:
            # Return of best_ticker from T-1 to T
            p_start = monthly_data.loc[prev_date, best_ticker]
            p_end = monthly_data.loc[curr_date, best_ticker]
            step_ret = (p_end - p_start) / p_start
            
        # Update Strategy Equity
        equity.append(equity[-1] * (1 + step_ret))
        dates.append(curr_date)
        
        # Update Benchmark (QQQ)
        q_start = monthly_data.loc[prev_date, "QQQ"]
        q_end = monthly_data.loc[curr_date, "QQQ"]
        q_ret = (q_end - q_start) / q_start
        bench_equity.append(bench_equity[-1] * (1 + q_ret))
        
        holdings_log.append({
            "Date": curr_date.date(),
            "Holding": best_ticker,
            "Return": f"{step_ret*100:.2f}%"
        })

    # Results
    final_strat = (equity[-1] - 1) * 100
    final_bench = (bench_equity[-1] - 1) * 100
    
    print("\n--- NUCLEAR OPTION RESULTS ---")
    print(f"Leveraged Strategy: {final_strat:.2f}%")
    print(f"Benchmark (QQQ): {final_bench:.2f}%")
    
    # Holdings
    log_df = pd.DataFrame(holdings_log)
    if not log_df.empty:
        print("\nRecent Month Holdings:")
        print(log_df.tail(5).to_string(index=False))
        log_df.to_csv("leveraged_log.csv", index=False)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, equity, label="Nuclear (3x Rotation)", color="orange", linewidth=2)
    plt.plot(dates, bench_equity, label="QQQ Benchmark", color="gray", linestyle="--")
    plt.yscale("log") # Log scale needed for these returns
    plt.title("The Nuclear Option: 3x Leveraged Rotation (Log Scale)")
    plt.ylabel("Growth of $1 (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig("leveraged_equity.png")
    print("Chart saved to: leveraged_equity.png")

if __name__ == "__main__":
    run_leveraged_strategy()
