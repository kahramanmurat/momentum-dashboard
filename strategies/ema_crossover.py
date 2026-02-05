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

def run_ema_crossover():
    print("Fetching Nasdaq 100...")
    tickers = get_nasdaq_100_tickers()
    tickers.append("QQQ") # Benchmark
    
    print(f"Downloading data for {len(tickers)} tickers (3 Years)...")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Close"]
    data = data.ffill().dropna(axis=1, thresh=len(data)*0.9)
    
    print("Calculating Technicals (EMA 9/21)...")
    ema9 = data.ewm(span=9, adjust=False).mean()
    ema21 = data.ewm(span=21, adjust=False).mean()
    returns_df = data.pct_change()
    
    # -----------------------------
    # Strategy Logic (Vectorized)
    # -----------------------------
    # Signal: 1 if EMA9 > EMA21, else 0
    signals = (ema9 > ema21).astype(int)
    
    # Position: Lag signal by 1 day (Trade at Open next day)
    positions = signals.shift(1)
    
    # -----------------------------
    # Returns Calculation
    # -----------------------------
    # We simulate equal weight allocation to all stocks in the universe.
    # If Signal is 0 (Cash), return is 0.
    
    strategy_returns = positions * returns_df
    
    # Transaction Costs (Churn Analysis)
    # This strategy flips A LOT. We must account for costs.
    trades = positions.diff().abs()
    costs = trades * 0.0010 # 10bps per trade (slippage + comms)
    
    net_ret = strategy_returns - costs
    
    # Portfolio Return (Mean across universe)
    # Exclude QQQ col from strategy calcs if present in positions (it is)
    if "QQQ" in net_ret.columns:
        strat_cols = [c for c in net_ret.columns if c != "QQQ"]
        port_ret = net_ret[strat_cols].mean(axis=1).fillna(0)
    else:
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
    
    print("\n--- 9/21 EMA Crossover Results (3 Years) ---")
    print(f"Strategy Return: {total_strat:.2f}%")
    print(f"Benchmark (QQQ): {total_bench:.2f}%")
    print(f"Avg Trades per Ticker: {avg_trades:.1f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_strategy.index, cumulative_strategy, label="9/21 EMA Crossover", color="blue", linewidth=2)
    plt.plot(cumulative_bench.index, cumulative_bench, label="QQQ Benchmark", color="gray", linestyle="--")
    plt.title("9/21 EMA Crossover vs QQQ")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ema_crossover_results.png")
    print("Chart saved to: ema_crossover_results.png")

if __name__ == "__main__":
    run_ema_crossover()
