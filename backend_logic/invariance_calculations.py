import yfinance as yf

def calculate_beta(yfinance_instance):
    stock = yfinance_instance
    options = stock.options
    opt = stock.option_chain(options[0])
    avg_iv = (opt.calls['impliedVolatility'].mean() + opt.puts['impliedVolatility'].mean()) / 2
    return avg_iv


def current_asset_price(yfinance_instance):
    instance = yfinance_instance
    data = instance.history()
    last_quote = data['Close'].iloc[-1]
    return last_quote


