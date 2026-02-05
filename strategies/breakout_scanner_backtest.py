import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------
# SURVIVORSHIP BIAS WARNING
# -----------------------------
# This scanner uses the CURRENT constituent list for the Nasdaq 100 / S&P 500.
# Historical backtests here suffer from "Survivorship Bias" because companies
# that went bankrupt or were removed from the index in the past are excluded.
# 
# In a real institutional production environment, you must use "Point-in-Time"
# constituent data (e.g., Norgate Data, Sharadar, or Compustat) to ensure
# you are testing the exact universe that existed on each historical date.
#
# For this MVP, the results are optimistic approximations.
# -----------------------------

# -----------------------------
# Universe fetch (Wikipedia)
# -----------------------------
def get_universe(universe: str) -> list[str]:
    import requests
    from io import StringIO

    universe = universe.lower().strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }

    if universe in ["sp500", "s&p500", "s&p 500"]:
        # Wikipedia list of S&P 500 companies
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        tickers = tables[0]["Symbol"].astype(str).tolist()
        # Yahoo uses '-' instead of '.' for tickers like BRK.B
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers

    if universe in ["nasdaq100", "nasdaq 100", "ndx"]:
        # Wikipedia Nasdaq-100 list
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))

        # Different Wikipedia table layouts exist; try common column names
        candidates = []
        for tbl in tables:
            # Handle MultiIndex columns if present
            cols = [str(c).lower() for c in tbl.columns]
            if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
                candidates.append(tbl)

        if not candidates:
            # Fallback for weird table structures - sometimes it's the 4th table
            if len(tables) > 4:
                 candidates.append(tables[4])
            else:
                raise ValueError("Could not find Nasdaq-100 constituents table on Wikipedia.")

        tbl = candidates[0]
        col = None
        for c in tbl.columns:
            cl = str(c).lower()
            if "ticker" in cl or "symbol" in cl:
                col = c
                break
        
        if col is None:
             # Fallback if column not found by name, assume first column
             col = tbl.columns[0]

        tickers = tbl[col].astype(str).tolist()
        tickers = [t.replace(".", "-").strip() for t in tickers if t and t != "nan"]
        # Some entries might have footnotes; clean
        tickers = [t.split()[0] for t in tickers]
        return sorted(list(set(tickers)))

    raise ValueError("Universe must be 'nasdaq100' or 'sp500'.")


# -----------------------------
# Signal: Institutional-style breakout
# -----------------------------
def compute_signals(df: pd.DataFrame, n_breakout=55, n_vol=20, sma_trend=200) -> pd.DataFrame:
    """
    df: OHLCV dataframe with columns: Open, High, Low, Close, Volume
    Produces breakout signals and features for ranking + backtest.
    """
    d = df.copy()

    # Core levels
    d["HH"] = d["High"].rolling(n_breakout).max().shift(1)  # Donchian entry level (prior N-day high)
    d["LL"] = d["Low"].rolling(20).min().shift(1)           # Donchian exit level (prior 20-day low)

    # Trend filter
    d["SMA200"] = d["Close"].rolling(sma_trend).mean()
    d["trend_ok"] = d["Close"] > d["SMA200"]

    # Volume filter
    d["VAVG20"] = d["Volume"].rolling(n_vol).mean()
    d["vol_ratio"] = d["Volume"] / d["VAVG20"]

    # "Strong close" filter: close in top part of daily range
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    d["close_strength"] = (d["Close"] - d["Low"]) / rng  # 0..1, higher is stronger

    # -----------------------------
    # Volatility Squeeze (Bollinger Band Width)
    # -----------------------------
    # BB calc: 20-day, 2 std
    bb_mid = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["bb_upper"] = bb_mid + 2.0 * bb_std
    d["bb_lower"] = bb_mid - 2.0 * bb_std
    
    # Bandwidth %
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / bb_mid
    
    # Squeeze Percentile (vs last 100 days)
    # If bb_width is in the bottom 15% of its 6-month history, it's a "squeeze"
    d["squeeze_pct"] = d["bb_width"].rolling(126).rank(pct=True)  # 126 days ~= 6 months
    
    # Breakout condition (EOD)
    # Optional: You can filter for squeeze_pct < 0.20 to only catch breakouts from consolidation
    d["breakout"] = (d["Close"] > d["HH"]) & d["trend_ok"] & (d["vol_ratio"] >= 1.5) & (d["close_strength"] >= 0.7)

    # Extra ranking features
    d["breakout_pct"] = (d["Close"] / d["HH"] - 1.0)
    d["atr14"] = atr(d, 14)
    d["atr_pct"] = d["atr14"] / d["Close"]
    return d


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# -----------------------------
# Backtest (single-ticker) with VOLATILITY TARGETING
# - Enter on breakout close
# - Exit on 20-day low break OR trend loss (Close < SMA200)
# - Position Sizing: Target Volatility / Realized Volatility
# -----------------------------
def backtest_single(d: pd.DataFrame, slippage_bps=2.0, commission_per_trade=0.0, vol_target_annual=0.20) -> pd.DataFrame:
    """
    Returns dataframe with position, trades, strategy returns and equity curve.
    vol_target_annual: e.g. 0.20 for 20% annualized vol target.
    """
    df = d.copy().dropna(subset=["HH", "LL", "SMA200", "VAVG20", "atr14"])

    # Realized Volatility (20-day lag) for sizing
    # shift(1) so we use YESTERDAY's vol to size TODAY's position
    df["ret_1d"] = df["Close"].pct_change()
    df["roll_vol"] = df["ret_1d"].rolling(20).std().shift(1) * np.sqrt(252)
    
    # Avoid div by zero or extreme leverage
    df["roll_vol"] = df["roll_vol"].replace(0, np.nan).ffill().fillna(0.20) # default to 20% if unknown

    # Position sizing scalar
    # e.g., if target=20% and realized=10%, size = 2.0x
    # Cap leverage at 2.0x to be realistic
    df["pos_size"] = (vol_target_annual / df["roll_vol"]).clip(0.1, 2.0)

    # Entry & exit rules
    entry = df["breakout"].astype(bool)
    exit_ = ((df["Close"] < df["LL"]) | (df["Close"] < df["SMA200"])).astype(bool)

    # Build position state machine (1=long, 0=flat)
    pos = np.zeros(len(df), dtype=float)
    in_pos = 0.0
    current_size = 0.0
    
    for i in range(len(df)):
        # Check exit first (if we are in pos)
        if in_pos != 0 and exit_.iloc[i]:
            in_pos = 0.0
            current_size = 0.0
        
        # Check entry (if flat)
        elif in_pos == 0 and entry.iloc[i]:
            in_pos = 1.0
            # Size position based on TODAY's sizing signal
            current_size = df["pos_size"].iloc[i]
            
        pos[i] = current_size # Store the size (e.g., 0.0, 1.2, etc.)

    df["position"] = pos

    # Daily returns
    # Strategy Return = Position(yesterday) * Return(today)
    df["strategy_ret_gross"] = df["position"].shift(1).fillna(0.0) * df["ret_1d"].fillna(0.0)

    # Transaction costs
    # Trade = change in signed position size
    df["trade"] = df["position"].diff().fillna(0).abs()
    
    # slippage in bps applied on notional traded
    slippage = (slippage_bps / 10000.0) * df["trade"]
    commission = commission_per_trade * df["trade"]

    df["strategy_ret_net"] = df["strategy_ret_gross"] - slippage - commission
    df["equity"] = (1.0 + df["strategy_ret_net"]).cumprod()
    return df


# -----------------------------
# Multi-ticker runner
# -----------------------------
def run(universe="nasdaq100",
        start="2015-01-01",
        end=None,
        n_breakout=55,
        top_n=25,
        out_dir=".",
        slippage_bps=2.0,
        commission_per_trade=0.0):

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    tickers = get_universe(universe)
    print(f"Universe: {universe} | tickers: {len(tickers)}")
    print("Downloading data... (this can take a few minutes)")

    # yfinance bulk download
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # If only one ticker, yfinance returns a different shape
    if isinstance(data.columns, pd.MultiIndex) is False:
        # single ticker case
        data = pd.concat({tickers[0]: data}, axis=1)

    # Compute "today breakouts" ranking
    breakout_rows = []
    bt_summary = []

    for t in tickers:
        if t not in data.columns.get_level_values(0):
            continue

        df = data[t].dropna()
        if df.empty or len(df) < 260:
            continue

        sig = compute_signals(df, n_breakout=n_breakout)
        last = sig.iloc[-1]

        if bool(last.get("breakout", False)):
            # ranking score: breakout strength + volume + (lower ATR% is slightly better)
            # Upgraded: prefer squeezes (low squeeze_pct)
            score = (
                1000.0 * float(last.get("breakout_pct", 0.0))
                + 10.0 * float(last.get("vol_ratio", 0.0))
                + 2.0 * float(last.get("close_strength", 0.0))
                - 10.0 * float(last.get("squeeze_pct", 0.5)) # Penalize high width (loose expansion already)
            )
            breakout_rows.append({
                "date": sig.index[-1].date().isoformat(),
                "ticker": t,
                "close": float(last["Close"]),
                "breakout_level": float(last["HH"]),
                "breakout_pct": float(last["breakout_pct"]),
                "vol_ratio": float(last["vol_ratio"]),
                "close_strength": float(last["close_strength"]),
                "squeeze_pct": float(last.get("squeeze_pct", 1.0)),
                "score": float(score),
            })

        # Backtest summary for each ticker
        bt = backtest_single(sig, slippage_bps=slippage_bps, commission_per_trade=commission_per_trade)
        if bt.empty:
            continue
        equity = bt["equity"]
        total_return = equity.iloc[-1] - 1.0
        cagr = (equity.iloc[-1]) ** (252.0 / max(1, len(bt))) - 1.0
        dd = (equity / equity.cummax() - 1.0).min()
        vol = bt["strategy_ret_net"].std() * np.sqrt(252.0)
        sharpe = np.nan if vol == 0 else (bt["strategy_ret_net"].mean() * 252.0) / vol

        trades = int(bt["trade"].sum())
        bt_summary.append({
            "ticker": t,
            "bars": len(bt),
            "total_return": float(total_return),
            "cagr_approx": float(cagr),
            "max_drawdown": float(dd),
            "vol_annual": float(vol),
            "sharpe_approx": float(sharpe) if pd.notna(sharpe) else np.nan,
            "trades": trades,
        })

    # Export breakouts today
    breakouts_today = pd.DataFrame(breakout_rows).sort_values("score", ascending=False).head(top_n)
    breakouts_path = f"{out_dir}/breakouts_today.csv"
    breakouts_today.to_csv(breakouts_path, index=False)
    print(f"Saved: {breakouts_path} | rows={len(breakouts_today)}")

    # Export backtest summary
    bt_df = pd.DataFrame(bt_summary).sort_values("sharpe_approx", ascending=False)
    bt_path = f"{out_dir}/backtest_results.csv"
    bt_df.to_csv(bt_path, index=False)
    print(f"Saved: {bt_path} | rows={len(bt_df)}")

    # Optional: a "portfolio equity" across all tickers equally weighted by each ticker's strategy returns
    # (Simple, but shows institutional thinking. Extend later with proper portfolio construction.)
    # We'll compute it from the top tickers by sharpe.
    top = bt_df.dropna(subset=["sharpe_approx"]).head(25)["ticker"].tolist()
    if top:
        rets = []
        dates = None
        for t in top:
            df = data[t].dropna()
            if df.empty:
                continue
            sig = compute_signals(df, n_breakout=n_breakout)
            bt = backtest_single(sig, slippage_bps=slippage_bps, commission_per_trade=commission_per_trade)
            if bt.empty:
                continue
            s = bt["strategy_ret_net"].rename(t)
            rets.append(s)
            dates = bt.index if dates is None else dates.union(bt.index)

        if rets:
            all_rets = pd.concat(rets, axis=1).reindex(dates).fillna(0.0)
            port_ret = all_rets.mean(axis=1)  # equal weight
            equity = (1.0 + port_ret).cumprod()
            eq_df = pd.DataFrame({"date": equity.index, "portfolio_equity": equity.values})
            eq_path = f"{out_dir}/equity_curve.csv"
            eq_df.to_csv(eq_path, index=False)
            print(f"Saved: {eq_path} | rows={len(eq_df)}")

    print("\nTop breakouts today:")
    if len(breakouts_today) == 0:
        print("(none found today with current filters)")
    else:
        print(breakouts_today.head(10).to_string(index=False))


if __name__ == "__main__":
    # Change universe to "sp500" if you want S&P 500
    run(
        universe="nasdaq100",
        start="2015-01-01",
        n_breakout=55,
        top_n=30,
        out_dir=".",
        slippage_bps=2.0,          # 2 bps per entry/exit
        commission_per_trade=0.0,  # set e.g. 0.0001 for 1 bp commission in "return space"
    )
