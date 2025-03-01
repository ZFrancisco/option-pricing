from backend_logic.outside_calls import get_stock_data
import yfinance as yf

def calculate_beta(ticker):
    beta = yf.Ticker(ticker).info['beta']
    return beta

def current_asset_price(ticker):
    ticker_yahoo = yf.Ticker(ticker)
    data = ticker_yahoo.history()
    last_quote = data['Close'].iloc[-1]
    return last_quote

