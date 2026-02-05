import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_rolling_beta(s1, s2, window=60):
    """
    Calculate Rolling OLS Beta efficiently using pandas rolling cov/var.
    beta = Cov(s1, s2) / Var(s2) --> Wait, in OLS s1 = beta*s2 + alpha? 
    Usually Spread = Y - beta*X. So Y is dependent, X is independent.
    Let's assume s1 is Y (dependent), s2 is X (independent).
    """
    # Use log prices for returns-like properties / stationarity
    y = np.log(s1)
    x = np.log(s2)
    
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    return beta

def backtest_pair_vectorized(t1, t2, start_date, end_date, lookback=300):
    """
    Vectorized Backtest for a single pair.
    """
    # Download data
    df = yf.download([t1, t2], start=start_date, end=end_date, progress=False, auto_adjust=False)["Close"]
    
    # Clean data
    df = df.dropna()
    if df.empty:
        return None
    
    s1 = df[t1]
    s2 = df[t2]
    
    # 1. Calc Rolling Beta (Hedge Ratio)
    # We use a 60-day rolling window to adapt to changing correlations
    beta = calculate_rolling_beta(s1, s2, window=60)
    
    # 2. Calc Spread
    # Spread = Log(A) - Beta * Log(B)
    # This represents the deviation from the equilibrium relationship
    spread = np.log(s1) - beta * np.log(s2)
    
    # 3. Calc Z-Score
    # We use a 30-day window to z-score the spread
    spread_mean = spread.rolling(30).mean()
    spread_std = spread.rolling(30).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 4. Generate Signals
    # Z > 2: Spread too wide (A high, B low) -> Short Spread -> Short A, Long B
    # Z < -2: Spread too low (A low, B high) -> Long Spread -> Long A, Short B
    # Exit: Z crosses 0
    
    positions = pd.DataFrame(index=z_score.index)
    positions["long_spread"] = 0 # 1 if long spread
    positions["short_spread"] = 0 # 1 if short spread
    
    # Signal Logic
    # We need to maintain state (in a vectorized way is hard for stateful exit).
    # Iterative approach is safer for "Exit on cross 0".
    
    pos = 0 # 0, 1 (Long Spread), -1 (Short Spread)
    pos_history = []
    
    z_vals = z_score.values
    
    for i in range(len(z_vals)):
        z = z_vals[i]
        
        if np.isnan(z):
            pos_history.append(0)
            continue
            
        if pos == 0:
            if z < -2.0:
                pos = 1 # Long Spread (Long A, Short B)
            elif z > 2.0:
                pos = -1 # Short Spread (Short A, Long B)
        elif pos == 1:
            if z >= 0: # Profit Take
                pos = 0
            elif z < -4.0: # Stop Loss (Spread blew out)
                pos = 0 
        elif pos == -1:
            if z <= 0: # Profit Take
                pos = 0
            elif z > 4.0: # Stop Loss
                pos = 0
                
        pos_history.append(pos)
        
    positions["pos"] = pos_history
    positions["pos"] = positions["pos"].shift(1) # Trade executes on NEXT day open (simulate lag)
    
    # 5. Calculate Returns
    # Dollar Neutral Strategy: 50% Capital in A, 50% Capital in B
    # Long Spread (Pos=1)  -> Long A (50%), Short B (50%)
    # Short Spread (Pos=-1) -> Short A (50%), Long B (50%)
    
    ret_a = s1.pct_change()
    ret_b = s2.pct_change()
    
    # Strategy Return
    # If Pos=1:  0.5 * RetA - 0.5 * RetB
    # If Pos=-1: -0.5 * RetA + 0.5 * RetB
    # Simplified: Pos * 0.5 * (RetA - RetB)
    
    # Note: Accurately, Short Ret is -1 * Ret - Cost. We ignore borrow cost for MVP.
    strat_ret = positions["pos"] * 0.5 * (ret_a - ret_b)
    
    # Transaction Costs (Slippage)
    # 5bps per leg = 10bps per trade
    trades = positions["pos"].diff().abs()
    costs = trades * 0.0010
    
    net_ret = strat_ret - costs
    return net_ret.fillna(0)

def run_backtest(csv_file="pairs_opportunities.csv", start_date="2024-01-01"):
    try:
        pairs_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Pairs file not found.")
        return

    if pairs_df.empty:
        print("No pairs to backtest.")
        return
        
    print(f"Backtesting {len(pairs_df)} pairs since {start_date}...")
    
    end_date = datetime.now()
    results = {}
    metrics = []
    
    for _, row in pairs_df.iterrows():
        t1 = row["Ticker_A"]
        t2 = row["Ticker_B"]
        pair_name = f"{t1}-{t2}"
        
        print(f"Processing {pair_name}...")
        try:
            ret = backtest_pair_vectorized(t1, t2, start_date, end_date)
            if ret is not None and not ret.empty:
                results[pair_name] = ret
                
                # Metrics
                cum_ret = (1 + ret).prod() - 1
                sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
                
                metrics.append({
                    "Pair": pair_name,
                    "Total_Return": f"{cum_ret*100:.2f}%",
                    "Sharpe": f"{sharpe:.2f}"
                })
        except Exception as e:
            print(f"Failed {pair_name}: {e}")
            
    # Aggregation (Portfolio of Pairs)
    if results:
        portfolio_ret = pd.DataFrame(results).mean(axis=1).fillna(0)
        cumulative = (1 + portfolio_ret).cumprod()
        
        # Save Results
        pd.DataFrame(metrics).to_csv("pairs_backtest_results.csv", index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        cumulative.plot(title="Stat Arb Strategy: Equal Weighted Portfolio (2024-Now)", color="purple")
        plt.ylabel("Growth of $1")
        plt.grid(True, alpha=0.3)
        plt.savefig("pairs_equity.png")
        print("\nBacktest Complete!")
        print("Results saved to: pairs_backtest_results.csv")
        print("Chart saved to: pairs_equity.png")
        
        print("\nPerformance Summary:")
        print(pd.DataFrame(metrics))
    else:
        print("No valid results generated.")

if __name__ == "__main__":
    # Start backtest from 2 years ago for robust stats
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    run_backtest(start_date=start)
