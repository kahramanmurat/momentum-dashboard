import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta

# -----------------------------
# Reused Universe Logic
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
        if col is None: return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"] # Fallback

        tickers = target_df[col].astype(str).tolist()
        return [t.replace(".", "-").strip() for t in tickers]
    except:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META", "AMD", "NFLX", "INTC"]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_rsi_strategies():
    print("Fetching Nasdaq 100...")
    tickers = get_nasdaq_100_tickers()
    # Add Benchmark
    tickers.append("QQQ")
    
    print(f"Downloading data for {len(tickers)} tickers (2 Years)...")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill().dropna(axis=1, thresh=len(data)*0.9)
    
    # We will simulate an Equal Weighted Portfolio of ALL tickers
    # Strategy A: Naive (RSI > 50)
    # Strategy B: Improved (RSI > 55 & Price > SMA200)
    
    eq_naive = pd.Series(0.0, index=data.index)
    eq_improved = pd.Series(0.0, index=data.index)
    eq_bench = pd.Series(0.0, index=data.index)
    
    # Initialize separate equity curves for each ticker to sum them up later
    # Actually, simpler: Compute Strategy Returns per ticker, then mean() across portfolio
    
    print("Calculating Technicals...")
    
    # Pre-compute indicators
    rsi_df = data.apply(lambda x: compute_rsi(x, 14))
    sma200_df = data.rolling(200).mean()
    returns_df = data.pct_change()
    
    # -----------------------------
    # Strategy A: Naive (User Request)
    # -----------------------------
    # Signal: 1 if RSI > 50, else 0
    signal_naive = (rsi_df > 50).astype(int)
    # Lag signal by 1 day (Trade at Open based on yesterday's Close RSI)
    pos_naive = signal_naive.shift(1)
    
    # Returns (Equal Weight Portfolio)
    # Mean across all tickers
    ret_naive_tickers = pos_naive * returns_df
    # Subtract Transaction Costs (Churn)
    # Estimate: Every time signal flips (0->1 or 1->0), pay 10bps cost
    trades_naive = pos_naive.diff().abs()
    cost_naive = trades_naive * 0.0010 
    net_ret_naive = ret_naive_tickers - cost_naive
    
    portfolio_ret_naive = net_ret_naive.mean(axis=1).fillna(0)
    cumulative_naive = (1 + portfolio_ret_naive).cumprod()
    
    # -----------------------------
    # Strategy B: Improved (Institutional)
    # -----------------------------
    # Long Condition: RSI > 55 AND Price > SMA200
    # Sell Condition: RSI < 45
    # This requires state (Hysteresis), harder to vectorize purely.
    # Approx Vectorized:
    # We create a "State" mask.
    # But for speed, let's use a simpler "Strong Trend" logic:
    # Buy if RSI > 55 AND Price > SMA200
    # Hold until RSI < 45
    
    # Iterative Approach for Hysteresis Mask
    print("Running Hysteresis Loop (Improved Strategy)...")
    pos_improved = pd.DataFrame(0, index=data.index, columns=data.columns)
    
    # We iterate by row (slow in python but safe) or by column? 
    # By column (Ticker) is faster.
    for col in data.columns:
        rsi = rsi_df[col]
        price = data[col]
        sma = sma200_df[col]
        
        # State
        in_position = False
        states = []
        
        # Convert to numpy for speed
        rsi_arr = rsi.values
        price_arr = price.values
        sma_arr = sma.values
        
        for i in range(len(rsi_arr)):
            r = rsi_arr[i]
            p = price_arr[i]
            s = sma_arr[i]
            
            if np.isnan(r) or np.isnan(s):
                states.append(0)
                continue
            
            if not in_position:
                # Entry Rule: Strong Momentum + Trend
                if r > 55 and p > s:
                    in_position = True
            else:
                # Exit Rule: Breakdown
                if r < 45:
                    in_position = False
            
            states.append(1 if in_position else 0)
            
        pos_improved[col] = states

    pos_improved = pos_improved.shift(1) # Lag
    
    ret_improved_tickers = pos_improved * returns_df
    trades_improved = pos_improved.diff().abs()
    cost_improved = trades_improved * 0.0010
    net_ret_improved = ret_improved_tickers - cost_improved
    
    portfolio_ret_improved = net_ret_improved.mean(axis=1).fillna(0)
    cumulative_improved = (1 + portfolio_ret_improved).cumprod()

    # Benchmark (QQQ Buy & Hold)
    try:
        bench_ret = data["QQQ"].pct_change().fillna(0)
    except:
        bench_ret = returns_df.mean(axis=1).fillna(0) # Fallback to equal weight index
        
    cumulative_bench = (1 + bench_ret).cumprod()

    # Stats
    total_naive = (cumulative_naive.iloc[-1] - 1) * 100
    total_improved = (cumulative_improved.iloc[-1] - 1) * 100
    total_bench = (cumulative_bench.iloc[-1] - 1) * 100
    
    print("\n--- RSI Experiment Results (2 Years) ---")
    print(f"Benchmark (Buy & Hold): {total_bench:.2f}%")
    print(f"Naive Strategy (RSI > 50): {total_naive:.2f}%")
    print(f"Improved Strategy (Filter + Hysteresis): {total_improved:.2f}%")
    
    # Trade Counts (Efficiency)
    avg_trades_naive = trades_naive.sum().mean()
    avg_trades_improved = trades_improved.sum().mean()
    print(f"Avg Trades per Ticker (Naive): {avg_trades_naive:.0f} (High Churn)")
    print(f"Avg Trades per Ticker (Improved): {avg_trades_improved:.0f} (Efficient)")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_bench.index, cumulative_bench, label="Benchmark (QQQ)", color="gray", linestyle="--")
    plt.plot(cumulative_naive.index, cumulative_naive, label="Naive (RSI > 50)", color="red")
    plt.plot(cumulative_improved.index, cumulative_improved, label="Improved (Trend+Hysteresis)", color="green", linewidth=2)
    
    plt.title("RSI Strategy Experiment: Naive vs Institutional")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rsi_comparison.png")
    print("Chart saved to: rsi_comparison.png")

if __name__ == "__main__":
    backtest_rsi_strategies()
