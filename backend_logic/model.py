import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_logic.pricing_logic import simulate_stock_price
from backend_logic.pricing_logic import run_simulation