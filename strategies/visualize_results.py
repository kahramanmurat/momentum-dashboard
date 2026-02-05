import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_tearsheet(equity_csv="equity_curve.csv", benchmark="QQQ", output_file="strategy_tearsheet.png"):
    # 1. Load Strategy Data
    try:
        df = pd.read_csv(equity_csv)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        strategy_col = "portfolio_equity"
        if strategy_col not in df.columns:
            raise ValueError(f"Column '{strategy_col}' not found in {equity_csv}")
    except FileNotFoundError:
        print(f"Error: {equity_csv} not found. Please run the scanner first.")
        return

    # 2. Load Benchmark Data
    start_date = df.index.min()
    end_date = df.index.max() + pd.Timedelta(days=1)
    
    print(f"Fetching {benchmark} data from {start_date.date()} to {end_date.date()}...")
    bench = yf.download(benchmark, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Handle yfinance multi-index if present (pandas/yfinance version dependent)
    if isinstance(bench.columns, pd.MultiIndex):
        bench = bench.xs(benchmark, axis=1, level=1) if benchmark in bench.columns.levels[1] else bench.iloc[:, 0].to_frame(name="Close")
    
    bench["Close"] = bench["Close"].ffill()
    
    # 3. Align Data
    # Reindex benchmark to strategy dates (filling forward for holidays if strategy has data)
    # Normalize to 1.0 at start
    merged = pd.DataFrame(index=df.index)
    merged["Strategy"] = df[strategy_col]
    merged["Benchmark"] = bench["Close"].reindex(df.index).ffill()
    
    # Normalize
    merged = merged / merged.iloc[0]

    # Calculate Daily Returns for Stats
    merged["Ret_Strat"] = merged["Strategy"].pct_change().fillna(0)
    merged["Ret_Bench"] = merged["Benchmark"].pct_change().fillna(0)

    # Setup Plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.5])

    # -----------------------------
    # Plot 1: Cumulative Returns
    # -----------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(merged.index, merged["Strategy"], label="Strategy (Breakout)", linewidth=2, color="#1f77b4")
    ax1.plot(merged.index, merged["Benchmark"], label=f"Benchmark ({benchmark})", linewidth=1.5, color="#7f7f7f", linestyle="--")
    
    # Annotate Final Return
    strat_tot = (merged["Strategy"].iloc[-1] - 1) * 100
    bench_tot = (merged["Benchmark"].iloc[-1] - 1) * 100
    ax1.set_title(f"Cumulative Return: Strategy {strat_tot:.1f}% vs {benchmark} {bench_tot:.1f}%", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # -----------------------------
    # Plot 2: Underwater Plot (Drawdown)
    # -----------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Calculate Drawdown
    roll_max = merged["Strategy"].cummax()
    drawdown = (merged["Strategy"] / roll_max) - 1.0
    
    ax2.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.3)
    ax2.plot(drawdown.index, drawdown, color="#d62728", linewidth=1)
    
    max_dd = drawdown.min() * 100
    ax2.set_title(f"Underwater Plot (Max Drawdown: {max_dd:.2f}%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)

    # -----------------------------
    # Plot 3: Monthly Heatmap
    # -----------------------------
    ax3 = fig.add_subplot(gs[2])
    
    # Prepare Monthly Returns
    monthly_ret = merged["Ret_Strat"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_ret = monthly_ret * 100 # to percentage
    
    # Create Pivot Table (Year x Month)
    monthly_df = pd.DataFrame({
        "Year": monthly_ret.index.year,
        "Month": monthly_ret.index.month,
        "Return": monthly_ret.values
    })
    
    pivot_table = monthly_df.pivot(index="Year", columns="Month", values="Return")
    # Fill missing value
    pivot_table = pivot_table.fillna(0.0)
    
    # Plot Heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=False, ax=ax3, linewidths=0.5, linecolor='white')
    ax3.set_title("Monthly Returns (%)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Year")
    ax3.set_xlabel("Month")

    # -----------------------------
    # Save
    # -----------------------------
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    print(f"Success! Tearsheet saved to: {output_file}")

if __name__ == "__main__":
    generate_tearsheet()
