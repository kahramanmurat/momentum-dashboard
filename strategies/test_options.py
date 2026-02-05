import yfinance as yf

def check_options(ticker):
    try:
        t = yf.Ticker(ticker)
        opts = t.options
        if opts:
            print(f"{ticker}: YES ({len(opts)} expirations)")
            # Check volume of nearest expiry to guess liquidity?
            # For now just existence.
        else:
            print(f"{ticker}: NO")
    except Exception as e:
        print(f"{ticker}: Error {e}")

check_options("AAPL")
check_options("BF-B") # Example of weird ticker
check_options("BRK-B")
