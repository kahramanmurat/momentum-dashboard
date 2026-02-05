import yfinance as yf

print("--- CALCULATING MU RETURN ---")
# Download entries
data = yf.download('MU', start='2025-10-31', progress=False, auto_adjust=False)['Close']

entry_price = data.iloc[0].item()  # Oct 31 Open/Close
current_price = data.iloc[-1].item() # Today

ret_pct = (current_price - entry_price) / entry_price

print(f"Entry Date:   2025-10-31")
print(f"Entry Price:  ${entry_price:.2f}")
print(f"Current Price: ${current_price:.2f}")
print(f"TOTAL RETURN: {ret_pct*100:+.2f}%")
