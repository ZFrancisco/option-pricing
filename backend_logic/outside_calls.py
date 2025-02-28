
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import finnhub
import time
import datetime
import pandas as pd
from backend_logic.database import check_if_exists
from backend_logic.database import add_stock_price 

api_key = 'cv0gov1r01qo8ssgn730cv0gov1r01qo8ssgn73g'
finnhub_client = finnhub.Client(api_key=api_key)

def date_to_unix(date):
    return int(time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple()))   

def get_stock_price(stock, start_date, end_date):
    if not check_if_exists(stock, start_date):
        stock_price = finnhub_client.stock_candles(stock, 'D', date_to_unix(start_date), date_to_unix(end_date))

    #FIX THIS
    else:
        return
    return stock_price

#test = finnhub_client.stock_candles('AAPL', 'D', date_to_unix('2021-01-01'), date_to_unix('2021-01-05'))




