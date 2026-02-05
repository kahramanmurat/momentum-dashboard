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

def run_ema_strategy():
    print("Fetching Nasdaq 100...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ") # Benchmark
    
    print(f"Downloading data for {len(tickers)} tickers (3 Years)...")
    # Need enough Data for 200 EMA to warm up
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill().dropna(axis=1, thresh=len(data)*0.9) # Filter bad data
    
    print("Calculating EMAs...")
    ema200 = data.ewm(span=200, adjust=False).mean()
    ema50 = data.ewm(span=50, adjust=False).mean()
    returns_df = data.pct_change()
    
    # -----------------------------
    # Strategy Logic (Loop)
    # -----------------------------
    # Buy when Price > EMA200
    # Sell when Price < EMA50
    
    print("Running Strategy Loop...")
    positions = pd.DataFrame(0, index=data.index, columns=data.columns)
    
    for col in data.columns:
        if col == "QQQ": continue
        
        price_arr = data[col].values
        e200_arr = ema200[col].values
        e50_arr = ema50[col].values
        
        state = 0 # Out
        pos_arr = []
        
        for i in range(len(price_arr)):
            p = price_arr[i]
            e200 = e200_arr[i]
            e50 = e50_arr[i]
            
            if np.isnan(p) or np.isnan(e200) or np.isnan(e50):
                pos_arr.append(0)
                continue
                
            if state == 0:
                # Entry: Price > EMA200
                if p > e200:
                    state = 1
            elif state == 1:
                # Exit: Price < EMA50
                if p < e50:
                    state = 0
            
            pos_arr.append(state)
            
        positions[col] = pos_arr

    # Shift positions to trade next day
    positions = positions.shift(1)
    
    # Calculate Portfolio Returns (Equal Weight of Active Positions?)
    # Or Equal Weight of ALL tickers (Cash Drag)?
    # Institutional Standard: Allocating capital to active trades.
    # But for a simple backtest comparing to Buy & Hold, let's assume fully invested in whatever signals we have?
    # No, that implies leverage if we have many signals.
    # Let's assume Equal Weight fixed allocation (e.g. 1/100th of portfolio per stock).
    # If signal is OFF, that portion sits in Cash (0 return).
    
    strategy_returns = positions * returns_df
    
    # Costs
    trades = positions.diff().abs()
    costs = trades * 0.0010
    net_ret = strategy_returns - costs
    
    # Portfolio Return (Mean across universe)
    port_ret = net_ret.mean(axis=1).fillna(0)
    cumulative_strategy = (1 + port_ret).cumprod()
    
    # Benchmark
    bench_ret = data["QQQ"].pct_change().fillna(0)
    cumulative_bench = (1 + bench_ret).cumprod()
    
    # Metrics
    total_strat = (cumulative_strategy.iloc[-1] - 1) * 100
    total_bench = (cumulative_bench.iloc[-1] - 1) * 100
    
    # Trade Count
    avg_trades = trades.sum().mean()
    
    print("\n--- EMA Strategy Results (3 Years) ---")
    print(f"Strategy Return: {total_strat:.2f}%")
    print(f"Benchmark (QQQ): {total_bench:.2f}%")
    print(f"Avg Trades per Ticker: {avg_trades:.1f}")
    
    if total_strat > total_bench:
        print("Verdict: OUTPERFORMANCE match!")
    else:
        print("Verdict: Underperformance (Cash Drag?)")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_strategy.index, cumulative_strategy, label="EMA Strategy (>200, <50)", color="green", linewidth=2)
    plt.plot(cumulative_bench.index, cumulative_bench, label="QQQ Benchmark", color="gray", linestyle="--")
    plt.title("EMA Strategy vs QQQ")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ema_strategy_results.png")
    print("Chart saved to: ema_strategy_results.png")

if __name__ == "__main__":
    run_ema_strategy()
