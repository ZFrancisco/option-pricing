import sqlite3
from sqlite3 import Error


conn = sqlite3.connect(r"C:\Users\Zack\finnhub_data.db")
cursor = conn.cursor()

def add_stock_price(stock, date, open_price, high_price, low_price, close_price, volume):
    try:
        cursor.execute("INSERT INTO stocks (symbol, last_price, timestamp, volume) VALUES (?, ?, ?, ?, ?, ?, ?)", (stock, date, open_price, high_price, low_price, close_price, volume))
        conn.commit()
    except Error as e:
        print(e)

def check_if_exists(stock, date):
    cursor.execute("SELECT * FROM stocks WHERE stock = ? AND date = ?", (stock, date))
    return cursor.fetchall()