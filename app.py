from flask import Flask, render_template, request, jsonify
from backend_logic.pricing_logic import option_pricing, simulate_stock_price
from backend_logic.invariance_calculations import calculate_beta, current_asset_price

app = Flask(__name__, static_folder="static", static_url_path="/")

@app.route('/')
def home_page():
    return render_template('home.html') 

@app.route('/program')
def main_page():
    return render_template('program.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input'}), 400   
    try:    
        stock_ticker = str(data.get('stockTicker'))
        strike_price = int(data.get('strikePrice'))
        days_until_expiration = int(data.get('daysUntilExpiration)'))
    except:
        return jsonify({'error': 'Invalid input'}), 400
    beta = calculate_beta(stock_ticker)
    current_price = current_asset_price(stock_ticker)
    option_price = option_pricing(days_until_expiration, 0.042, 0, beta, 1, current_price, strike_price)
    return jsonify({'optionPrice': option_price})

if __name__ == '__main__':
    app.run(debug=True)