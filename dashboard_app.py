import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import os
import datetime
from datetime import timedelta

# Set Page Config
st.set_page_config(
    page_title="Momentum Master Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PORTFOLIO_FILE = os.path.join(OUTPUT_DIR, "portfolio_history.csv")
RISING_STARS_FILE = os.path.join(OUTPUT_DIR, "rising_stars.csv")
SECTOR_FILE = os.path.join(OUTPUT_DIR, "sector_rotation_report.csv")
BREAKOUT_FILE = os.path.join(OUTPUT_DIR, "breakouts_today.csv")
TREND_REVERSAL_FILE = os.path.join(OUTPUT_DIR, "trend_reversals.csv")
SCALP_FILE = os.path.join(OUTPUT_DIR, "hourly_scalps.csv")
WEEKLY_EMA_FILE = os.path.join(OUTPUT_DIR, "weekly_ema_strategy.csv")
ETF_FILE = os.path.join(OUTPUT_DIR, "etf_scan_results.csv")

# --- Helper Functions ---

def refresh_data():
    """Runs the master strategy script to update all data."""
    with st.spinner('Refreshing Data (Running Strategies)...'):
        try:
            # Run master_dashboard.py which triggers all sub-strategies
            # We use python3 from the current environment
            result = subprocess.run(
                ["python3", "strategies/master_dashboard.py"],
                capture_output=True,
                text=True,
                check=True
            )
            st.success(f"Data Refreshed Successfully at {datetime.datetime.now().strftime('%H:%M:%S')}")
            # st.text(result.stdout) # Optional debug
        except subprocess.CalledProcessError as e:
            st.error(f"Error refreshing data: {e}")
            st.error(e.stderr)

def get_target_expiry_and_delta(opt_avail):
    """Calculates target expiry and delta based on option availability."""
    if opt_avail == 'YES':
        target_date = datetime.datetime.now() + timedelta(days=120)
        target_exp = target_date.strftime("%b '%y")
        target_delta = "0.70 (ITM)"
        return target_exp, target_delta
    return "-", "-"

def get_buy_zone_signal(sma_20, current_price):
    """Determines buy zone signal and color."""
    if sma_20 <= 0:
        return f"${sma_20:.2f}", "grey"
    
    signal_str = f"${sma_20:.2f}"
    color = "grey"
    
    if current_price > 0 and current_price <= sma_20 * 1.02:
         signal_str += " (BUY NOW)"
         color = "green"
         
    if current_price > 0 and current_price <= sma_20 * 1.02:
         signal_str += " (BUY NOW)"
         color = "green"
         
    return signal_str, color

def calculate_supertrend_for_chart(df, period=10, multiplier=3.0):
    """
    Calculates SuperTrend for visualization.
    """
    # ATR
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()

    # Basic Upper/Lower Bands
    df['Basic_Upper'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
    df['Basic_Lower'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']

    # Final Upper/Lower Bands initialization
    df['Final_Upper'] = df['Basic_Upper']
    df['Final_Lower'] = df['Basic_Lower']
    df['SuperTrend'] = 0.0
    df['Trend'] = 1 # 1 Up, -1 Down
    
    # Iterative calculation
    close = df['Close'].values
    basic_upper = df['Basic_Upper'].values
    basic_lower = df['Basic_Lower'].values
    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    trend = np.zeros(len(df)) 
    
    start_idx = period
    
    for i in range(start_idx, len(df)):
        # Final Upper
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Final Lower
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Trend
        prev_trend = trend[i-1] if i > start_idx else 1
        
        if prev_trend == -1 and close[i] > final_upper[i]:
            trend[i] = 1
        elif prev_trend == 1 and close[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = prev_trend
            
        # SuperTrend Value
        if trend[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
            
    df['SuperTrend'] = supertrend
    df['Trend'] = trend
    
    return df

# --- Load Data ---

@st.cache_data(ttl=60) # Cache for 1 min
def load_portfolio_data():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_rising_stars_data():
    if os.path.exists(RISING_STARS_FILE):
        return pd.read_csv(RISING_STARS_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_sector_data():
    # Sector data might come from sector_dashboard.py csv or we might need to run it
    # master_dashboard.py runs sector_dashboard.py? Let's assume it does or creates similar data.
    # The user manual task said sector_dashboard.py runs... checks master_dashboard source
    if os.path.exists(SECTOR_FILE):
        return pd.read_csv(SECTOR_FILE)
    return pd.DataFrame() # Fallback

@st.cache_data(ttl=60)
def load_breakout_data():
    if os.path.exists(BREAKOUT_FILE):
        return pd.read_csv(BREAKOUT_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_trend_reversal_data():
    if os.path.exists(TREND_REVERSAL_FILE):
        return pd.read_csv(TREND_REVERSAL_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_scalp_data():
    if os.path.exists(SCALP_FILE):
        return pd.read_csv(SCALP_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_weekly_ema_data():
    if os.path.exists(WEEKLY_EMA_FILE):
        return pd.read_csv(WEEKLY_EMA_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_etf_data():
    if os.path.exists(ETF_FILE):
        return pd.read_csv(ETF_FILE)
    return pd.DataFrame()

# --- Layout ---

st.title("🚀 Momentum Master Dashboard")
st.markdown(f"**Last Update:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

if st.button("🔄 Refresh Data (Run Strategies)"):
    refresh_data()
    st.cache_data.clear() # Clear cache to reload new data
    st.rerun()

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Active Picks & Stars", "Sector Rotation", "Hourly Signals", "Trend Reversals", "⚡ Turbo Scalps", "📅 Weekly EMA", "📊 ETF Scan"])

with tab1:
    col1, col2 = st.columns(2)
    
    # 1. Active Picks
    with col1:
        st.subheader("Top 5 Active Picks (Hold These)")
        df_picks = load_portfolio_data()
        
        if not df_picks.empty:
            # Enhance Data
            display_picks = df_picks.copy()
            
            # Apply Logic for Columns
            # Assume columns: Ticker, Options_Avail, Sector, Industry, Entry_Date, Entry_Price, Current_Price, Return, Max_Price, SMA_20
            
            # Add calculated columns for display
            exps = []
            deltas = []
            buy_zones = []
            
            for index, row in display_picks.iterrows():
                opt = row.get('Options_Avail', 'No')
                exp, delta = get_target_expiry_and_delta(opt)
                exps.append(exp)
                deltas.append(delta)
                
                sma = float(row.get('SMA_20', 0))
                price = float(row.get('Current_Price', 0))
                sig, col = get_buy_zone_signal(sma, price)
                buy_zones.append(sig) 

            display_picks['Target Exp'] = exps
            display_picks['Target Delta'] = deltas
            display_picks['Smart Buy Zone'] = buy_zones
            
            # Reorder
            cols_to_show = ['Ticker', 'Options_Avail', 'Target Exp', 'Target Delta', 'Smart Buy Zone', 'Sector', 'Return', 'Margin']
            # Adjust to available columns
            final_cols = [c for c in cols_to_show if c in display_picks.columns or c in ['Target Exp', 'Target Delta', 'Smart Buy Zone']]
            
            # If Margin not there, use Cushion logic if present (Max_Price)
            # Master Dashboard uses Max Price to calc cushion.
            
            st.dataframe(display_picks, height=400)
        else:
            st.warning("No Portfolio Data Found. Click Refresh.")

    # 2. Rising Stars
    with col2:
        st.subheader("✨ Rising Stars (Watch These)")
        df_stars = load_rising_stars_data()
        
        if not df_stars.empty:
            display_stars = df_stars.copy()
            
            exps = []
            deltas = []
            buy_zones = []
            
            for index, row in display_stars.iterrows():
                opt = row.get('Options_Avail', 'No')
                exp, delta = get_target_expiry_and_delta(opt)
                exps.append(exp)
                deltas.append(delta)
                
                sma = float(row.get('SMA_20', 0))
                price = 0 # Might be missing in CSV?
                # If missing, we can't fully color code, but show level
                sig, col = get_buy_zone_signal(sma, price) 
                buy_zones.append(f"${sma:.2f}")

            display_stars['Target Exp'] = exps
            display_stars['Target Delta'] = deltas
            display_stars['Smart Buy Zone'] = buy_zones

            st.dataframe(display_stars, height=400)
        else:
            st.warning("No Rising Stars Data Found.")

with tab2:
    st.subheader("🌍 Sector Rotation Radar")
    df_sector = load_sector_data()
    
    if not df_sector.empty:
        # Expected columns from sector_dashboard.py: Name, 1M_Return, 3M_Return, Rotation_Score
        # We need numeric values. The CSV might have strings "%".
        
        # Clean data
        try:
            plot_df = df_sector.copy()
            for c in ['1M_Return', '3M_Return']:
                if plot_df[c].dtype == object:
                   plot_df[c] = plot_df[c].astype(str).str.rstrip('%').astype(float)
            
            fig = px.scatter(
                plot_df,
                x="3M_Return",
                y="1M_Return",
                color="Rotation_Score",
                text="Ticker",
                hover_data=["Name"],
                size_max=60,
                color_continuous_scale="RdYlGn",
                title="Sector Rotation: Risk ON vs Risk OFF"
            )
            fig.update_traces(textposition='top center', marker=dict(size=20, line=dict(width=1, color='DarkSlateGrey')))
            fig.add_hline(y=0, line_dash="dash", line_color="grey")
            fig.add_vline(x=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting data: {e}")
            st.dataframe(df_sector)
    else:
        st.info("Sector Data missing. Run Refresh.")

with tab3:
    st.subheader("⏱️ Hourly Swing Signals")
    df_breakout = load_breakout_data()
    if not df_breakout.empty:
        # Styling Function
        def style_status(val):
            color = ''
            weight = ''
            if "BREAKOUT" in val: color = '#4caf50' # Green
            elif "BREAKDOWN" in val: color = '#f44336' # Red
            elif "MOMENTUM UP" in val: color = '#81c784' # Light Green
            elif "MOMENTUM DOWN" in val: color = '#e57373' # Light Red
            elif "COILING" in val: color = '#ff9800' # Orange
            
            return f'color: {color}; font-weight: bold' if color else ''

        st.dataframe(df_breakout.style.map(style_status, subset=['Status']), height=400)

with tab4:
    st.subheader("🔄 Trend Reversals (SuperTrend Flip)")
    st.markdown("Stocks where Price just crossed **ABOVE** the Support Line (Red Circle Pattern).")
    
    df_reversal = load_trend_reversal_data()
    if not df_reversal.empty:
        # Check if it has data
        if "Ticker" in df_reversal.columns and len(df_reversal) > 0:
             # Styling
            def style_reversal(val):
                color = ''
                if "BULLISH" in str(val): color = '#4caf50' # Green
                return f'color: {color}; font-weight: bold' if color else ''
            
            st.dataframe(df_reversal.style.map(style_reversal, subset=['Signal']), height=200)
            
            # --- SHOW CHART FOR THE REVERSAL ---
            st.markdown("### 📈 Visual Confirmation")
            
            for index, row in df_reversal.iterrows():
                ticker = row['Ticker']
                st.write(f"Analyzing Reversal: **{ticker}**")
                
                # Download Data for Chart
                try:
                    data = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
                    if not data.empty:
                        # Fix for yfinance returning MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            try:
                                data.columns = data.columns.droplevel(1) 
                            except:
                                pass
                                
                        # Or manual check just in case droplevel fails or structure varies
                        if 'Close' not in data.columns and ticker in data.columns:
                            # It's likely Ticker -> OHLC
                            data = data[ticker]

                         # Calculate Indicator
                        data = calculate_supertrend_for_chart(data)
                        
                        # Plot
                        fig = go.Figure()
                        
                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'], high=data['High'],
                            low=data['Low'], close=data['Close'],
                            name="Price"
                        ))
                        
                        # SuperTrend Line
                        # Split into Green (Up) and Red (Down) segments for visual clarity? 
                        # Or just plot the line. Let's color the line based on Trend.
                        
                        # Create segments for color
                        # This is complex in line charts. Simpler: Plot markers or just one line with conditional color? 
                        # Plotting simple line for now.
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['SuperTrend'],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='SuperTrend (Trailing Stop)'
                        ))
                        
                        # Highlight THE Reversal Day
                        # The reversal happend on row['Date']
                        rev_date_str = row['Date'] # String YYYY-MM-DD
                        
                        # Find the bar
                        # Convert column to datetime if needed
                        # data index is DatetimeIndex.
                        
                        # Add Marker/Annotation
                        fig.add_annotation(
                            x=rev_date_str,
                            y=row['Price'],
                            xref="x",
                            yref="y",
                            text="🔴 BUY FLIP",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="#4caf50", # Green arrow for buy
                            ax=0,
                            ay=-40,
                            bgcolor="#4caf50",
                            bordercolor="#4caf50",
                            font=dict(color="white", size=12)
                        )
                         # Add Circle Shape
                        #  fig.add_shape(
                        #      type="circle",
                        #      xref="x", yref="y",
                        #      x0=rev_date_str, y0=row['Price']*0.98,
                        #      x1=rev_date_str, y1=row['Price']*1.02,
                        #      line_color="Red",
                        #  )

                        fig.update_layout(
                            title=f"{ticker} - SuperTrend Reversal (Entry Signal)",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=500,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Could not load chart for {ticker}: {e}")

        else:
             st.info("No Trend Reversals detected today.")
    else:
        st.info("No Trend Reversals detected today.")

with tab5:
    st.subheader("⚡ Turbo 1H Scalps (High Velocity)")
    st.markdown("Strategy: **Price > EMA_200 (Uptrend)** + **RSI < 45 (Dip)**. Target: 300-400% Options.")
    
    df_scalp = load_scalp_data()
    if not df_scalp.empty:
         # Styling
        def style_scalp(val):
            color = ''
            if "BUY TRIGGER" in str(val): color = '#4caf50' # Green
            elif "READY" in str(val): color = '#ff9800' # Orange/Yellow
            elif "WATCH" in str(val): color = '#607d8b' # Grey Blue
            return f'color: {color}; font-weight: bold' if color else ''
        
        st.dataframe(df_scalp.style.map(style_scalp, subset=['Setup']), height=400)
    else:
        st.info("No Scalp Setups found right now. Market might be overextended or flat.")

with tab6:
    st.subheader("📅 Weekly EMA 9/21 Crossover + RSI > 50")
    st.markdown("Strategy: **EMA 9 > EMA 21** (Crossover) + **RSI > 50** & Rising (Confirmed Momentum).")
    
    df_weekly = load_weekly_ema_data()
    if not df_weekly.empty:
         # Styling
        def style_weekly(val):
            color = ''
            if "BUY SIGNAL" in str(val): color = '#4caf50' # Green
            return f'color: {color}; font-weight: bold' if color else ''
        
        if "Ticker" in df_weekly.columns:
            st.dataframe(df_weekly.style.map(style_weekly, subset=['Status']), height=400)
        else:
            st.dataframe(df_weekly)
    else:
        st.info("No Weekly EMA Crossover Signals found this week.")

with tab7:
    st.subheader("📊 ETF Multi-Timeframe Scanner")
    st.markdown("Performance across **1 Week, 1 Month, 3 Months, and 6 Months**.")
    
    df_etf = load_etf_data()
    if not df_etf.empty:
        # Format Columns
        # Data is float, we want to display as Strings for %, or use Styler
        
        # Helper to color percentages
        def color_pct(val):
            if type(val) != float: return ''
            color = '#4caf50' if val > 0 else '#f44336' if val < 0 else ''
            return f'color: {color}'
            
        def style_signal(val):
            if "BUY STRUCTURE" in str(val): return 'color: #00e676; font-weight: bold; background-color: rgba(0, 230, 118, 0.1); border-left: 5px solid #00e676;'
            if "BUY ZONE" in str(val): return 'color: #4caf50; font-weight: bold; background-color: rgba(76, 175, 80, 0.1);'
            if "BREAKOUT" in str(val): return 'color: #29b6f6; font-weight: bold;'
            if "MOMENTUM UP" in str(val): return 'color: #81c784; font-weight: bold;'
            if "OVERSOLD" in str(val): return 'color: #00e676; font-weight: bold; border: 1px solid #00e676;'
            if "DOWNTREND" in str(val): return 'color: #f44336;'
            if "EXTENDED" in str(val): return 'color: #ff9800;'
            return ''
            
        def color_rsi(val):
            if type(val) != float: return ''
            if val < 30: return 'color: #00e676; font-weight: bold;' # Super Oversold
            if val < 45: return 'color: #4caf50;' # Oversold
            if val > 70: return 'color: #f44336;' # Overbought
            if val > 80: return 'color: #d50000; font-weight: bold;' # Super Overbought
            return 'color: #bbb;'

        # Calculate Display Columns for Support/Resistance
        if 'Support_Level' in df_etf.columns and 'Current Price' in df_etf.columns:
            df_etf['Support'] = df_etf.apply(lambda x: f"${x['Support_Level']:.2f}" if x['Support_Level'] > 0 else "-", axis=1)
            # Add distance if close
            # Actually, let's just make it simpler: Value
            
        if 'Res_Level' in df_etf.columns:
             df_etf['Resistance'] = df_etf.apply(lambda x: f"${x['Res_Level']:.2f}" if x['Res_Level'] > 0 else "-", axis=1)

        # Select columns to display (reorder if needed)
        cols = ['Ticker', 'Signal', 'RSI', 'Name', 'Current Price', 'Support', 'Resistance',
                '1W % Chg', '1M % Chg', '3M % Chg', '6M % Chg',
                '1W Price Chg', '1M Price Chg', '3M Price Chg', '6M Price Chg']
        
        # Filter for existing columns just in case
        valid_cols = [c for c in cols if c in df_etf.columns]
        display_df = df_etf[valid_cols].copy()
        
        # Sort logic: Put BUY ZONES and MOMENTUM UP at the top, then Gainers
        if 'Signal' in display_df.columns:
            # Custom sort priority
            def signal_rank(s):
                if "BUY STRUCTURE" in str(s): return 0
                if "BREAKOUT" in str(s): return 1
                if "OVERSOLD" in str(s): return 2
                if "BUY ZONE" in str(s): return 3
                if "MOMENTUM UP" in str(s): return 4
                if "UPTREND" in str(s): return 5
                if "WAIT" in str(s): return 6
                if "EXTENDED" in str(s): return 7
                return 8 # Downtrend
            
            display_df['Signal_Rank'] = display_df['Signal'].apply(signal_rank)
            display_df = display_df.sort_values(by=['Signal_Rank', '1M % Chg'], ascending=[True, False])
            display_df = display_df.drop(columns=['Signal_Rank'])
        
        # Apply Styler
        st.dataframe(
            display_df.style.format({
                'RSI': '{:.0f}',
                '1W % Chg': '{:+.2%}', '1M % Chg': '{:+.2%}',
                '3M % Chg': '{:+.2%}', '6M % Chg': '{:+.2%}',
                '1W Price Chg': '${:+.2f}', '1M Price Chg': '${:+.2f}',
                '3M Price Chg': '${:+.2f}', '6M Price Chg': '${:+.2f}',
                'Current Price': '${:.2f}'
            }).map(color_pct, subset=['1W % Chg', '1M % Chg', '3M % Chg', '6M % Chg'])
              .map(style_signal, subset=['Signal'])
              .map(color_rsi, subset=['RSI']),
            height=800
        )
    else:
        st.info("No ETF Scan Data found. Please Run Refresh.")
