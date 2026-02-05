import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from datetime import datetime, timedelta

# Reuse logic from backtester for consistency
from statarb_backtester import calculate_rolling_beta

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_training_data(t1, t2, start_date, end_date):
    """
    Simulates the strategy and captures X (Features) and Y (Outcome) at every entry.
    """
    print(f"Generating ML Dataset for {t1}-{t2}...")
    df = yf.download([t1, t2], start=start_date, end=end_date, progress=False, auto_adjust=False)["Close"]
    df = df.dropna()
    
    if df.empty: return pd.DataFrame()
    
    s1 = df[t1]
    s2 = df[t2]
    
    # Core Indicators
    beta = calculate_rolling_beta(s1, s2, window=60)
    spread = np.log(s1) - beta * np.log(s2)
    
    spread_mean = spread.rolling(30).mean()
    spread_std = spread.rolling(30).std()
    z_score = (spread - spread_mean) / spread_std
    
    # Feature Engineering (What conditions look like AT ENTRY)
    # 1. Volatility (Is the spread wild?)
    feat_vol_10 = spread.rolling(10).std()
    
    # 2. Velocity (Momentum of spread)
    feat_vel_5 = spread.diff(5)
    
    # 3. RSI (Is it overextended?)
    feat_rsi = compute_rsi(spread, 14)
    
    # 4. Beta Stability (Is correlation changing?)
    feat_beta_chg = beta.diff(10).abs()

    # Simulation Loop to capture Trades
    trades = []
    
    # State
    pos = 0 # 0, 1, -1
    entry_price = 0
    entry_idx = 0
    df_idx = z_score.index
    z_vals = z_score.values
    
    # We need to look forward to see if trade won, so we iterate
    for i in range(60, len(z_vals)):
        z = z_vals[i]
        if np.isnan(z): continue
        
        current_date = df_idx[i]
        
        # ENTRY LOGIC
        if pos == 0:
            signal = 0
            if z < -2.0: signal = 1  # Long Spread
            elif z > 2.0: signal = -1 # Short Spread
            
            if signal != 0:
                pos = signal
                entry_idx = i
                
                # Capture Features at the moment of entry (i)
                trades.append({
                    "entry_date": current_date,
                    "pair": f"{t1}-{t2}",
                    "side": signal,
                    "z_score": z,
                    "vol_10": feat_vol_10.iloc[i],
                    "vel_5": feat_vel_5.iloc[i],
                    "rsi": feat_rsi.iloc[i],
                    "beta_chg": feat_beta_chg.iloc[i],
                    "entry_idx": i,
                    "outcome": None # To be filled at exit
                })
        
        # EXIT LOGIC
        elif pos != 0:
            exit_signal = False
            # Profit Take
            if pos == 1 and z >= 0: exit_signal = True
            if pos == -1 and z <= 0: exit_signal = True
            
            # Stop Loss (Expansion)
            if pos == 1 and z < -4.0: exit_signal = True
            if pos == -1 and z > 4.0: exit_signal = True
            
            if exit_signal:
                # Calculate if trade was a win approx (Spread reverted = Win, Spread expanded = Loss)
                # Actually, let's look at P&L
                # P&L approx = Side * (Spread_Exit - Spread_Entry)
                # If Long Spread (1): Profit if Spread increases (from -2 back to 0)
                # Wait, spread def is log(A) - beta*log(B).
                # If z < -2, spread is "low". We want it to go "high" (back to 0). So Long Spread wins if spread increases.
                
                # Careful: The beta changes over time. P&L is complex.
                # Simplified "Labeling": Did we exit at Z=0 (Win) or Z=Stop (Loss)?
                # Or did we hold for too long?
                
                # Let's calculate actual approximate return
                # Entry prices
                p1_in = s1.iloc[entry_idx]
                p2_in = s2.iloc[entry_idx]
                # Exit prices
                p1_out = s1.iloc[i]
                p2_out = s2.iloc[i]
                
                # Returns
                r1 = (p1_out - p1_in)/p1_in
                r2 = (p2_out - p2_in)/p2_in
                
                # Long Spread (Long A, Short B)
                if pos == 1:
                    raw_pl = 0.5*r1 - 0.5*r2
                else: 
                    raw_pl = -0.5*r1 + 0.5*r2
                    
                # Store Outcome (1 if > 0, 0 if <= 0)
                # Mark the LAST trade in the list
                trades[-1]["outcome"] = 1 if raw_pl > 0 else 0
                trades[-1]["return"] = raw_pl
                
                pos = 0 # Reset
                
    return pd.DataFrame(trades)

def train_and_evaluate(trades_df):
    if trades_df.empty:
        print("No trades generated to train on.")
        return

    # Clean
    trades_df = trades_df.dropna()
    print(f"Total Trades for Analysis: {len(trades_df)}")
    if len(trades_df) < 10:
        print("Not enough trades to train ML model (need > 10).")
        return

    # Features X and Target y
    feature_cols = ["z_score", "vol_10", "vel_5", "rsi", "beta_chg"]
    X = trades_df[feature_cols]
    y = trades_df["outcome"]
    
    # Train/Test Split (Time Series Split)
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Initialize Model
    # Random Forest: Good at handling non-linear relationships
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    
    print("\n--- ML Model Metrics (Out-of-Sample) ---")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f} (When AI says 'Trade', spread reverted {prec*100:.0f}% of time)")
    
    # Feature Importance
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importance (What matters?):")
    print(feat_imp)
    
    # PLOT: Equity Curve Comparison
    # Simulate equity of Test Set WITH and WITHOUT ML Filter
    
    test_trades = trades_df.iloc[split:].copy()
    test_trades["ml_prediction"] = clf.predict(X_test)
    
    # Equity Curves
    # 1. Base Strategy (Take all trades)
    test_trades["cum_base"] = test_trades["return"].cumsum()
    
    # 2. ML Strategy (Only take if pred == 1)
    test_trades["ret_ml"] = np.where(test_trades["ml_prediction"] == 1, test_trades["return"], 0)
    test_trades["cum_ml"] = test_trades["ret_ml"].cumsum()
    
    plt.figure(figsize=(10, 6))
    # We plot mostly against trade number index, as dates are not continuous
    plt.plot(test_trades["cum_base"].values, label="Original Strategy (Losses)", color="red", linestyle="--")
    plt.plot(test_trades["cum_ml"].values, label="ML-Filtered Strategy", color="green", linewidth=2)
    plt.title("Impact of AI Filter: Equity Curve comparison")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ml_equity_curve.png")
    print("\nSaved chart: ml_equity_curve.png")

def run_ml_pipeline():
    # 1. Load Pairs
    try:
        pairs = pd.read_csv("pairs_opportunities.csv")
    except:
        print("No pairs file found.")
        return

    # 2. Generate Data
    # For ML, we need MORE data than just 2 pairs to generalize well. 
    # But let's stick to the user's workflow. We will use the history of these 2 pairs.
    # To make it robust, you'd usually train on 50+ pairs.
    
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d") # 3 Years history
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    all_trades = []
    
    for _, row in pairs.iterrows():
        t1 = row["Ticker_A"]
        t2 = row["Ticker_B"]
        df_trades = generate_training_data(t1, t2, start_date, end_date)
        if not df_trades.empty:
            all_trades.append(df_trades)
            
    if not all_trades:
        print("No trades generated backtest.")
        return
        
    full_dataset = pd.concat(all_trades).sort_values("entry_date").reset_index(drop=True)
    
    # 3. Train & Evaluate
    train_and_evaluate(full_dataset)

if __name__ == "__main__":
    run_ml_pipeline()
