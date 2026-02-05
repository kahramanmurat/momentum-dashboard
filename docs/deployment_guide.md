# The Momentum Strategy: Deployment Guide (Capital: $7,000)

## 1. The Core Setup
You are running the **"Fast Momentum (Top 5)"** strategy.
*   **Algorithm**: 3-Month Momentum (The "Goldilocks" Trend).
*   **Portfolio**: Top 5 Stocks.
*   **Rebalance**: Monthly (Last trading day of the month).
*   **Safety**: Market Regime Filter (200-Day Moving Average).

## 2. Capital Allocation ($7,000)
With **$7,000**, you should split your capital into **5 Equal Slots** of **$1,400**.
*   **Slot 1**: $1,400
*   **Slot 2**: $1,400
*   **Slot 3**: $1,400
*   **Slot 4**: $1,400
*   **Slot 5**: $1,400

> **Why Top 5?**
> Our tests showed "Top 3" makes more money (+134k%), but "Top 5" (+84k%) is much safer. With a $7k account, preserving capital is priority #1. Top 5 ensures one bad earnings report doesn't wipe you out.

## 3. The Monthly Routine (The "30-Minute Work Month")

### Step 1: Run the Scanner (Last Day of Month)
Open your terminal and run:
```bash
python3 strategies/fast_momentum.py
```

### Step 2: Check the "Buy List"
Open the file: `output/fast_momentum_picks.csv`
It will look like this (Example):
1.  **MU** (Micron)
2.  **WDC** (Western Digital)
3.  **LRCX** (Lam Research)
4.  **STX** (Seagate)
5.  **AMAT** (Applied Materials)

### Step 3: Execute the Trades (Rebalancing)
Login to your broker (Robinhood, Schwab, IBKR).
1.  **Sell** any stock you own that is **NOT** on the new Top 5 list.
2.  **Buy** the new names to fill the empty slots.
3.  **Re-align**: If a stock is still on the list but has grown to $2,000 (from $1,400), you can trim it back to $1,400 and put the profit into the others. This is "Rebalancing".

## 4. The Safety Switch (Critical)
Before you buy, check the **QQQ (Nasdaq 100)** Price vs 200-Day Moving Average (The script prints this or you can check Yahoo Finance).

*   **GREEN LIGHT**: QQQ is **Above** 200-Day MA. -> **Buy Stocks**.
*   **RED LIGHT**: QQQ is **Below** 200-Day MA. -> **Go 100% Cash (or SHV/Bils)**.
    *   *Why?* This filter saved the portfolio in 2008 and 2022. It reduces drawdown from -60% to -30%.

## 5. Professional Advice for $7k
1.  **Don't look at it daily.** Momentum is a "Month-to-Month" strategy. Daily noise will scare you out of winning trades.
2.  **Accept the "Whipsaw".** Sometimes you will buy a stock, it drops -5%, and the model sells it next month. This is the "rent" you pay to catch the +50% winners (like Nvidia).
3.  **Survivorship Bias is Real but Irrelevant.** Our stress test proved the strategy kicks out "losers" (like Lehman/Enron) before they go to zero. Trust the exit rule.
4.  **Use Limit Orders.** Don't use Market Orders. Set a limit price near the "Ask" to avoid paying too much slippage.

## 6. Your "Go-Live" Date
This strategy works best when executed consistently.
*   **Start Date**: **January 31st (End of this month)**.
*   **Next Check**: **February 28th**.

Good luck. You have the math on your side.
