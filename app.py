from flask import Flask, render_template, request, jsonify
from backend_logic.pricing_logic import option_pricing, simulate_stock_price
from backend_logic.invariance_calculations import calculate_beta, current_asset_price
import io
import base64
from matplotlib.figure import Figure
import yfinance as yf

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
        strike_price = float(data.get('strikePrice'))
        days_until_expiration = int(data.get('daysUntilExpiration'))
    except Exception as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    
    instance = yf.Ticker(stock_ticker)
    beta = calculate_beta(instance) 
    current_price = current_asset_price(instance)
    print(beta, current_price)
    option_price = option_pricing(days_until_expiration, 0.042, 0, beta, 1, current_price, strike_price)
        
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    simulation_data = simulate_stock_price(current_price, 0.042, beta, 1, days_until_expiration)
    ax.plot(simulation_data.T)
    ax.set_title(f'Monte Carlo Simulation for {stock_ticker}')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Price ($)')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    formatted_price = round(float(option_price), 2)
    
    return jsonify({
        'optionPrice': formatted_price,
        'simulationImage': img_data
    })

if __name__ == '__main__':
    app.run(debug=True)