import sqlite3
from sqlite3 import Error

## FIX THIS
conn = sqlite3.connect(r"/Users/zackaryfrancisco/option_pricing/yf_db.db")
## FIX THIS
cursor = conn.cursor()

def add_stock_price(date, symbol, open, high, low, close, volume):
    try:
        cursor.execute("INSERT INTO stocks (datestamp, symbol, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)", (date, symbol, open, high, low, close, volume))
        conn.commit()
    except Error as e:
        print(e)

def check_if_exists(stock, start, end):
    cursor.execute("SELECT * FROM stocks WHERE symbol = ? AND datestamp >= ? AND datestamp <= ?", (stock, start, end))
    return cursor.fetchall()

def get_rows(stock, start, end):
    cursor.execute("SELECT * FROM stocks WHERE symbol = ? AND datestamp >= ? AND datestamp <= ?", (stock, start, end))
    return cursor.fetchall()