import requests
import finnhub
import time
import datetime
import pandas as pd

api_key = 'cv0eoc9r01qo8ssgcjp0cv0eoc9r01qo8ssgcjpg'
finnhub_client = finnhub.Client(api_key=api_key)

def date_to_unix(date):
    return int(time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple()))   

def get_stock_price(stock, start_date, end_date):
    stock_price = finnhub_client.stock_candles(stock, 'D', date_to_unix(start_date), date_to_unix(end_date))
    return stock_price
