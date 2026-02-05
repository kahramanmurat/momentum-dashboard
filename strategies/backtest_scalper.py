import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from io import StringIO
from datetime import datetime, timedelta

# --- SETTINGS ---
# Volatile Growth Stocks (Best for this strategy)
TICKERS = ["NVDA", "TSLA", "AMD", "META", "AMZN", "NFLX", "GOOGL", "MSFT", "AAPL"]
TAKE_PROFIT = 0.02 # 2% stock move (~50-80% option)
STOP_LOSS = 0.01   # 1% stock move (Tight leash)
INITIAL_CAPITAL = 10000
TRADE_SIZE = 1000 # fixed per trade

def calculate_indicators(df):
    """
    Calculates EMA_200, RSI_14, MACD
    """
    df = df.copy()
    
    # EMA 200 (Trend)
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    return df

def run_backtest():
    print("--- 🔬 BACKTEST: TURBO 1H SCALPER (60 DAYS) ---")
    print(f"Goal: +{TAKE_PROFIT*100}% | Stop: -{STOP_LOSS*100}%")
    print("-" * 50)
    
    # Download Hourly Data
    data = yf.download(TICKERS, period="59d", interval="1h", group_by='ticker', progress=False, auto_adjust=False)
    
    total_trades = 0
    wins = 0
    losses = 0
    total_pnl = 0
    
    results = []
    
    for t in TICKERS:
        try:
            df = data[t].copy()
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.droplevel(0)
                except: pass
                
            if df.empty: continue
            df = df.dropna()
            
            # Indicators
            df = calculate_indicators(df)
            
            # Simulation
            in_trade = False
            entry_price = 0
            entry_time = None
            
            ticker_trades = 0
            ticker_wins = 0
            ticker_pnl = 0
            
            # Start after 200 bars (for EMA)
            for i in range(200, len(df)):
                curr_bar = df.iloc[i]
                prev_bar = df.iloc[i-1]
                
                price = curr_bar['Close']
                
                if not in_trade:
                    # --- ENTRY LOGIC ---
                    # 1. Uptrend check
                    if price > curr_bar['EMA_200']:
                        # 2. RSI Dip Check (or recently dipped)
                        # Let's say we enter if RSI < 45 OR (RSI was < 45 in last 3 bars)
                        # AND Momentum is turning up
                        
                        # Simplified: Current bar logic from strategy
                        is_cheap = curr_bar['RSI'] < 45
                        macd_up = curr_bar['Histogram'] > prev_bar['Histogram']
                        
                        if is_cheap and macd_up:
                            in_trade = True
                            entry_price = price
                            entry_time = curr_bar.name
                            ticker_trades += 1
                
                else:
                    # --- EXIT LOGIC ---
                    pct_change = (price - entry_price) / entry_price
                    
                    # Take Profit
                    if pct_change >= TAKE_PROFIT:
                        in_trade = False
                        wins += 1
                        ticker_wins += 1
                        profit = TRADE_SIZE * TAKE_PROFIT
                        total_pnl += profit
                        ticker_pnl += profit
                        # print(f"[{t}] WIN: +{profit:.2f} on {curr_bar.name}")
                        
                    # Stop Loss
                    elif pct_change <= -STOP_LOSS:
                        in_trade = False
                        losses += 1
                        loss = TRADE_SIZE * STOP_LOSS
                        total_pnl -= loss
                        ticker_pnl -= loss
                        # print(f"[{t}] LOSS: -{loss:.2f} on {curr_bar.name}")
            
            win_rate = (ticker_wins / ticker_trades * 100) if ticker_trades > 0 else 0
            results.append({
                "Ticker": t,
                "Trades": ticker_trades,
                "Win_Rate": f"{win_rate:.1f}%",
                "Net_PnL": f"${ticker_pnl:.2f}"
            })
            
            print(f"Tested {t}: {ticker_trades} Trades, Win Rate: {win_rate:.1f}%, PnL: ${ticker_pnl:.2f}")

        except Exception as e:
            print(f"Error testing {t}: {e}")
            continue
            
    print("-" * 50)
    print("--- 📊 OVERALL PERFORMANCE ---")
    overall_win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    print(f"Total Trades: {wins + losses}")
    print(f"Win Rate: {overall_win_rate:.1f}%")
    print(f"Total Theoretical PnL (Adjusted for Size): ${total_pnl:.2f}")
    
    if overall_win_rate > 60:
        print("\n✅ VERDICT: STRATEGY IS PROFITABLE. Win Rate > 60%.")
    else:
        print("\n⚠️ VERDICT: CAUTION. Win Rate is below 60%. Tighten stops.")

if __name__ == "__main__":
    run_backtest()
