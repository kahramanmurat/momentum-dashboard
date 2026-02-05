import pandas as pd
import os
import datetime
from datetime import timedelta
import base64
import subprocess
import sys
import json

def run_strategies():
    """Runs the sub-strategies to generate fresh data."""
    print("--- UPDATING DATA (Running Sub-Strategies) ---")
    
    scripts = [
        "fast_momentum.py",
        "analyze_portfolio_history.py",
        "momentum_radar.py",
        "sector_dashboard.py",
        "breakout_scanner.py",
        "supertrend_reversal.py",
        "hourly_scalper.py",
        "weekly_ema_strategy.py",
        "etf_scanner.py"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for s in scripts:
        path = os.path.join(script_dir, s)
        print(f" > Executing {s}...")
        try:
            # Use same python executable as current process
            subprocess.run([sys.executable, path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {s}: {e}")
        except Exception as e:
            print(f"Failed to run {s}: {e}")

def generate_dashboard():
    # 0. UPDATE DATA
    run_strategies()

    print("\n--- BUILDING MASTER DASHBOARD (HTML) ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    
    # 1. READ DATA
    try:
        picks_df = pd.read_csv(os.path.join(output_dir, "fast_momentum_picks.csv"))
        if "Unnamed: 0" in picks_df.columns: 
            picks_df = picks_df.rename(columns={"Unnamed: 0": "Ticker"})
        if "3M_Score" in picks_df.columns:
            picks_df = picks_df.rename(columns={"3M_Score": "Momentum"})
            
        # Read History
        try:
            history_df = pd.read_csv(os.path.join(output_dir, "portfolio_history.csv"))
            # Merge history into picks
            picks_df = pd.merge(picks_df, history_df, on="Ticker", how="left")
        except FileNotFoundError:
            print("Warning: portfolio_history.csv not found.")
        
        stars_df = pd.read_csv(os.path.join(output_dir, "rising_stars.csv"))
        if "Unnamed: 0" in stars_df.columns:
            stars_df = stars_df.rename(columns={"Unnamed: 0": "Ticker"})
            
        sector_df = pd.read_csv(os.path.join(output_dir, "sector_rotation_report.csv"))
        
        breakout_df = pd.read_csv(os.path.join(output_dir, "breakouts_today.csv"))
        
    except FileNotFoundError as e:
        print(f"Error: Missing data files. {e}")
        return

    # 2. PREPARE PLOTLY DATA (Sector Radar)
    plot_data = []
    
    for _, row in sector_df.iterrows():
        try:
            r3m = float(str(row['3M_Return']).replace('%', '').replace('+', ''))
            r1m = float(str(row['1M_Return']).replace('%', '').replace('+', ''))
            score = float(row['Rotation_Score'])
            color = "#4caf50" if score > 0 else "#f44336"
            if score == 0: color = "#ffa000"
            plot_data.append({}) 
        except: continue
            
    tickers = [row['Ticker'] for _, row in sector_df.iterrows()]
    x_vals = [float(str(row['3M_Return']).replace('%','').replace('+','')) for _, row in sector_df.iterrows()]
    y_vals = [float(str(row['1M_Return']).replace('%','').replace('+','')) for _, row in sector_df.iterrows()]
    names = [row['Name'] for _, row in sector_df.iterrows()]
    colors = []
    for _, row in sector_df.iterrows():
        sc = float(row['Rotation_Score'])
        if sc > 0: colors.append("#4caf50")
        elif sc < 0: colors.append("#f44336")
        else: colors.append("#ffa000")
    
    text_labels = tickers
    hover_text = [
        f"<b>{t}</b><br>{n}<br>3M Trend: {x}%<br>1M Mom: {y}%<br>Score: {s:.0f}" 
        for t, n, x, y, s in zip(tickers, names, x_vals, y_vals, sector_df['Rotation_Score'])
    ]

    # 3. GENERATE HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Momentum Master Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #ffffff; }}
            .container {{ max_width: 1200px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .full-width {{ margin-top: 20px; }}
            .card {{ background-color: #2d2d2d; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ text-align: left; border-bottom: 1px solid #444; padding: 8px; color: #888; font-size: 12px; }}
            td {{ padding: 8px; border-bottom: 1px solid #333; font-size: 14px; }}
            tr:last-child td {{ border-bottom: none; }}
            
            .positive {{ color: #4caf50; font-weight: bold; }}
            .negative {{ color: #f44336; font-weight: bold; }}
            .alert-text {{ color: #ff9800; font-weight: bold; }}
            .rank {{ font-weight: bold; color: #ffa000; }}
            
            #plotly-div {{ width: 100%; height: 600px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Momentum Master Dashboard</h1>
                <h3>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</h3>
            </div>
            
            <div class="grid">
                <!-- TOP PICKS -->
                <div class="card">
                    <h2>🚀 Top 5 Active Picks (Hold These)</h2>
                    <p>3-Month Momentum Leaders (Nasdaq 100)</p>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th>Opt</th>
                            <th>Target Exp</th>
                            <th>Target Delta</th>
                            <th>Sector</th>
                            <th>Industry</th>
                            <th>3M Score</th>
                            <th>Entry Date</th>
                            <th>Entry Price</th>
                            <th>Current Price</th>
                            <th>Total Return</th>
                            <th>Stop Loss (15%)</th>
                            <th>Cushion</th>
                        </tr>
    """
    
    # Add Picks Rows
    for _, row in picks_df.head(5).iterrows():
        entry = row.get('Entry_Date', 'N/A')
        total_ret = row.get('Return', 0)
        
        # Handle formatting if missing
        ret_str = "N/A"
        if pd.notnull(total_ret):
            ret_str = f"{float(total_ret)*100:+.1f}%"
            
        # Calculate Stop Loss
        max_price = row.get('Max_Price', 0)
        curr_price = row.get('Current_Price', 0)
        
        stop_price_str = "N/A"
        cushion_str = "N/A"
        cushion_style = ""
        
        if pd.notnull(max_price) and float(max_price) > 0 and pd.notnull(curr_price) and float(curr_price) > 0:
            mx = float(max_price)
            curr = float(curr_price)
            stop_price = mx * 0.85
            stop_price_str = f"${stop_price:.2f}"
            
            cushion = (curr - stop_price) / curr
            cushion_str = f"{cushion*100:.1f}%"
            
            if cushion < 0.05:
                cushion_style = "color: #ff9800; font-weight: bold;" # Orange for warning
            if cushion < 0:
                cushion_style = "color: #f44336; font-weight: bold;" # Red for breach
            
        # Options Color
        opt_val = row.get('Options_Avail', 'No')
        opt_style = "color: #4caf50; font-weight: bold;" if opt_val == 'YES' else "color: #666;"
        
        # Option Strategy
        target_date = datetime.datetime.now() + timedelta(days=120)
        target_exp_str = target_date.strftime("%b '%y") if opt_val == 'YES' else "-"
        target_delta_str = "0.70 (ITM)" if opt_val == 'YES' else "-"

        # Pullback Zone Logic
        sma_20 = float(row.get('SMA_20', 0))
        curr_price = float(row.get('Current_Price', 0))
        buy_zone_str = f"${sma_20:.2f}" if sma_20 > 0 else "N/A"
        buy_style = "color: #888;" # Default grey
        
        # If Current Price is within 2% of SMA_20 (and above it slightly or below it)
        # We define "Buy Zone" as anything between SMA_20 * 0.98 and SMA_20 * 1.02
        if sma_20 > 0 and curr_price > 0:
            if curr_price <= sma_20 * 1.02:
                buy_style = "color: #4caf50; font-weight: bold; background-color: rgba(76, 175, 80, 0.1);"
                buy_zone_str += " (BUY NOW)"

        html += f"<tr><td style='font-weight:bold; font-size:16px;'>{row['Ticker']}</td>"
        html += f"<td style='{opt_style}; font-size:11px;'>{opt_val}</td>"
        html += f"<td style='font-size:12px;'>{target_exp_str}</td>"
        html += f"<td style='font-size:12px;'>{target_delta_str}</td>"
        html += f"<td style='{buy_style}; font-size:13px;'>{buy_zone_str}</td>"
        html += f"<td style='font-size:12px; color:#aaa;'>{row.get('Sector', 'N/A')}</td>"
        html += f"<td style='font-size:12px; color:#aaa;'>{row.get('Industry', 'N/A')}</td>"
        html += f"<td class='positive'>{float(row['Momentum'])*100:+.1f}%</td>"
        html += f"<td>{entry}</td>"
        html += f"<td>${float(row.get('Entry_Price', 0)):.2f}</td>"
        html += f"<td>${float(row.get('Current_Price', 0)):.2f}</td>"
        html += f"<td class='positive'>{ret_str}</td>"
        html += f"<td>{stop_price_str}</td>"
        html += f"<td style='{cushion_style}'>{cushion_str}</td></tr>"
    
    html += """
                    </table>
                </div>
                
                <!-- RISING STARS -->
                <div class="card">
                    <h2>✨ Rising Stars (Watch These)</h2>
                    <p>Fastest Rank Acceleration (Last 30 Days)</p>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th>Opt</th>
                            <th>Target Exp</th>
                            <th>Target Delta</th>
                            <th>Smart Buy Zone</th>
                            <th>Sector</th>
                            <th>Industry</th>
                            <th>Climb Score</th>
                            <th>Rank</th>
                        </tr>
    """
    
    # Add Stars Rows
    for _, row in stars_df.head(5).iterrows():
        # Options Color
        opt_val = row.get('Options_Avail', 'No')
        opt_style = "color: #4caf50; font-weight: bold;" if opt_val == 'YES' else "color: #666;"
        
        # Option Strategy
        target_date = datetime.datetime.now() + timedelta(days=120)
        target_exp_str = target_date.strftime("%b '%y") if opt_val == 'YES' else "-"
        target_delta_str = "0.70 (ITM)" if opt_val == 'YES' else "-"
        
        # Pullback Zone Logic (Current Price is not always in Rising Stars csv, we might need to fetch it or use Last Close)
        # Assuming we fetched it or can fetch it. Actually momentum_radar.py uses 'Close' from yfinance.
        # But rising_stars.csv doesn't have 'Current_Price' column explicitly saved in the print statement but maybe in the file?
        # Let's check momentum_radar.py export. It exports `rising_stars.to_csv`.
        # `rising_stars` is a subset of `df`. `df` has `latest_r`.
        # We need to ensure 'Current_Price' is in the CSV.
        # momentum_radar.py: data.iloc[-1] is current price.
        
        # For now, let's assume we need to calculate it or it might be missing.
        # Actually, let's look at momentum_radar.py export again. It exports `rising_stars`.
        # `rising_stars` dataframe colums: `Current_3M_Ret`, `Current_Rank`, `Prev_Rank`, `Climb_Score`, `Sector`...
        # It DOES NOT have price. I need to add Price to momentum_radar.py first? 
        # Wait, I can probably infer it or fetch it? No, better to add it to momentum_radar.py.
        # For this step, I will add the logic assuming the column exists, and knowing I might need to fix momentum_radar.py
        # Pullback Zone Logic
        sma_20 = float(row.get('SMA_20', 0))
        # Note: rising_stars.csv might not have Current_Price if we didn't add it.
        # But for now we just show the SMA level.
        buy_zone_str = f"${sma_20:.2f}" if sma_20 > 0 else "N/A"
        buy_style = "color: #888;"
        
        html += f"<tr><td style='font-weight:bold;'>{row['Ticker']}</td>"
        html += f"<td style='{opt_style}; font-size:11px;'>{opt_val}</td>"
        html += f"<td style='font-size:12px;'>{target_exp_str}</td>"
        html += f"<td style='font-size:12px;'>{target_delta_str}</td>"
        html += f"<td style='{buy_style}; font-size:13px;'>{buy_zone_str}</td>"
        html += f"<td style='font-size:12px; color:#aaa;'>{row.get('Sector', 'N/A')}</td>"
        html += f"<td style='font-size:12px; color:#aaa;'>{row.get('Industry', 'N/A')}</td>"
        html += f"<td class='positive'>+{int(row['Climb_Score'])} Spots</td>"
        html += f"<td>#{int(row.get('Current_Rank', 0))}</td></tr>"
        
    html += """
                    </table>
                </div>
            </div>

            <!-- HOURLY SIGNALS -->
            <div class="full-width">
                <div class="card">
                    <h2>⏱️ Hourly Swing Signals (The Squeeze)</h2>
                    <p>Volatility Compression on 1H Chart</p>
                    <table>
                        <tr>
                            <th>Ticker</th>
                            <th>Status</th>
                            <th>Bandwidth</th>
                            <th>Volume Spike</th>
                        </tr>
    """
    
    def status_priority(s):
        if "BREAK" in s: return 0
        if "COILING" in s: return 1
        return 2
        
    breakout_df['Priority'] = breakout_df['Status'].apply(status_priority)
    breakout_df = breakout_df.sort_values(by=['Priority', 'Bandwidth'])
    
    for _, row in breakout_df.head(5).iterrows():
        status_style = "color: #e0e0e0;"
        if "BREAKOUT" in row['Status']: status_style = "color: #4caf50; font-weight: bold;"
        elif "BREAKDOWN" in row['Status']: status_style = "color: #f44336; font-weight: bold;"
        elif "COILING" in row['Status']: status_style = "color: #ff9800; font-weight: bold;"
        
        html += f"<tr><td style='font-weight:bold;'>{row['Ticker']}</td>"
        html += f"<td style='{status_style}'>{row['Status']}</td>"
        html += f"<td>{float(row['Bandwidth'])*100:.2f}%</td>"
        html += f"<td>{row['Vol_Spike']}</td></tr>"
        
    html += """
                    </table>
                </div>
            </div>
            
            <br>
            
            <!-- SECTOR HEATMAP (PLOTLY) -->
            <div class="card">
                <h2>🌍 Sector Rotation Radar (Interactive)</h2>
                <div id="plotly-div"></div>
                
                <script>
                    var trace1 = {
                        x: """ + json.dumps(x_vals) + """,
                        y: """ + json.dumps(y_vals) + """,
                        mode: 'markers+text',
                        text: """ + json.dumps(text_labels) + """,
                        hovertext: """ + json.dumps(hover_text) + """,
                        hoverinfo: 'text',
                        marker: {
                            size: 35,
                            color: """ + json.dumps(colors) + """,
                            opacity: 0.8,
                            line: {color: 'white', width: 1}
                        },
                        textfont: {
                            family: 'Segoe UI, sans-serif',
                            size: 11,
                            color: 'white',
                            weight: 'bold'
                        },
                        type: 'scatter'
                    };

                    var data = [trace1];

                    var layout = {
                        title: 'Sector Rotation: Risk ON vs Risk OFF',
                        xaxis: {
                            title: '3-Month Trend (Strategy Baseline)',
                            gridcolor: '#444',
                            zerolinecolor: '#888'
                        },
                        yaxis: {
                            title: '1-Month Momentum (Acceleration)',
                            gridcolor: '#444',
                            zerolinecolor: '#888'
                        },
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: {
                            color: '#e0e0e0'
                        },
                        hovermode: 'closest',
                        shapes: [
                            {
                                type: 'line',
                                x0: -50, y0: 0,
                                x1: 50, y1: 0,
                                line: { color: '#666', width: 1, dash: 'dash' }
                            },
                            {
                                type: 'line',
                                x0: 0, y0: -50,
                                x1: 0, y1: 50,
                                line: { color: '#666', width: 1, dash: 'dash' }
                            }
                        ]
                    };

                    Plotly.newPlot('plotly-div', data, layout);
                </script>
            </div>
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, "dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
        
    print(f"Dashboard generated: {html_path}")

if __name__ == "__main__":
    generate_dashboard()
