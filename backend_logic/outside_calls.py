
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import time
import datetime
import pandas as pd
from backend_logic.database import check_if_exists
from backend_logic.database import add_stock_price 
from backend_logic.database import get_rows

columns = ['datestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']

def date_to_unix(date):
    return int(time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple()))   

def get_stock_data(stock, start_date, end_date):
    # maybe move this API 
    # inside an else clause
    # if its not already in db
    symbol_instace = yf.Ticker(stock)
    stock_price = symbol_instace.history(start=start_date, end=end_date)
    stock_price.index = stock_price.index.strftime('%Y-%m-%d')
    already_exists = check_if_exists(stock, start_date, end_date)
    repeated_indices = []
    for row in already_exists:
        if row[0] in stock_price.index:
            repeated_indices.append(row[0])
    stock_price = stock_price.drop(repeated_indices)
    for index, row in stock_price.iterrows():
        add_stock_price(index, stock, row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
    return pd.DataFrame(get_rows(stock, start_date, end_date), columns=columns)

#test = finnhub_client.stock_candles('AAPL', 'D', date_to_unix('2021-01-01'), date_to_unix('2021-01-05'))

table = get_stock_data('AAPL', '2021-01-01', '2021-01-05')
print(table)
