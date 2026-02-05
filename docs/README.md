# Institutional Breakout Scanner & Backtester

## Overview
This project implements an **Institutional-Style Trend Following Scanner** designed to identify high-quality breakout setups in the **Nasdaq 100** and **S&P 500**. unlike simple retail scanners, this tool incorporates professional risk management filters including **Volatility Squeeze** detection and **Volatility Targeting** for position sizing.

The system performs two primary functions:
1.  **Scanner**: Scans the universe for today's breakout signals.
2.  **Backtester**: Runs a historical simulation (backtest) on every ticker to validate the strategy's edge.

## Key Features

### 1. Signal Logic (The "Edge")
The strategy looks for **Momentum** confirmed by **Volume** and **Trend**:
*   **Price Breakout**: Close > 55-Day High (Donchian Channel).
*   **Trend Filter**: Price > 200-Day SMA (Long-term uptrend).
*   **Volume Confirmation**: Volume > 1.5x of 20-Day Average Volume.
*   **Strong Close**: The stock must close in the top 30% of its daily range (rejecting wick-based reversals).
*   **Volatility Squeeze**: Checks for Bollinger Band consolidation (low bandwidth) prior to the move.

### 2. Risk Management (Institutional)
*   **Volatility Targeting**: Position sizes are dynamic. High-volatility stocks get smaller allocations, and low-volatility stocks get larger allocations, targeting a constant **20% annualized volatility**.
*   **ATR Risk Checks**: Outputs ATR% to help gauge stop-loss distances.

### 3. Survivorship Bias Mitigation
*   **Note**: This tool utilizes current index constituents (fetched live from Wikipedia). Historical backtests on *current* winners inherently suffer from survivorship bias.
*   **Mitigation**: For production use, this logic should be paired with a Point-in-Time (PIT) database (e.g., Norgate, Sharadar) to include delisted companies.

## Project Structure

```bash
.
├── breakout_scanner_backtest.py  # Main script (Fetcher, Scanner, Backtester)
├── breakouts_today.csv           # REPORT: List of active signals for today
├── backtest_results.csv          # METRICS: Sharpe, CAGR, Drawdown for all tickers
├── equity_curve.csv              # CHART: Aggregate portfolio equity curve
└── README.md                     # Documentation
```

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   Pip dependencies:
    ```bash
    pip install yfinance pandas numpy lxml html5lib requests
    ```

### Running the Scanner
Execute the main script:
```bash
python3 breakout_scanner_backtest.py
```

### Configuration
You can modify the parameters for `run()` at the bottom of the script:
*   `universe`: "nasdaq100" or "sp500"
*   `n_breakout`: Lookback days for breakout (default: 55)
*   `vol_target_annual`: Volatility target for sizing (default: 0.20)

## Output Explanation

### `breakouts_today.csv`
Contains the ranked list of actionable trade ideas.
*   **score**: Proprietary ranking metric (Breakout Magnitude + Volume + Squeeze Quality).
*   **squeeze_pct**: Percentile of volatility width. Values < 0.20 indicate a tight "squeeze" structure.
*   **vol_ratio**: Current Volume / 20-Day Avg Volume.

### `backtest_results.csv`
Provides statistical confidence for each ticker.
*   **sharpe_approx**: Annualized Sharpe Ratio.
*   **cagr_approx**: Compound Annual Growth Rate.
*   **max_drawdown**: Worst peak-to-trough decline.

## 🏆 The "Final Answer" Findings (Advanced Strategy Project)

After testing 17 distinct strategies over 20-50 years, we identified the optimal engine.

### The Winner: 3-Month Momentum
*   **Strategy**: Rank Nasdaq 100 stocks by **3-Month Relative Strength**.
*   **Portfolio**: Buy the **Top 5** (or Top 3) every month.
*   **Rebalance**: Monthly.

### The "Top N" Decision (Risk vs Reward)
*   **Top 5 (+84,682%)**: **Recommended**. The professional balance. Survives if 1 stock goes to zero.
*   **Top 3 (+134,737%)**: **Aggressive**. The absolute highest return, but volatile. If 1 stock dies, you lose 33%.
*   **Top 1 (+13,198%)**: **Too Risky**. Binary "Hit or Miss" risk.

To switch to Top 3, simply edit `fast_momentum.py` and change `TOP_N = 5` to `TOP_N = 3`.

### Deployment
Run this script at the end of every month to get your picks:
```bash
python3 fast_momentum.py
```
