import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
 
# S~LN(mu, sigma)
#S(t + deltat) = S(t) * exp((r - sigma^2/2) * deltat + sigma * sqrt(deltat) * Z), Z = N(0, 1)

#invariances
initial_price_asset = 100      
r = 0.0412 # 1 year treasury bond rate 
beta = 0.2   
t = 1         
N = 10000     
M = 252 

#run monte carlo simulation to generate stock price paths
def simulate_stock_price(S_t, r, sigma, T, M):
    paths = np.zeros((N, M+1))
    paths[:, 0] = S_t
    for i in range(N):
        path =  S_t * np.exp(np.cumsum((r - sigma ** 2 / 2) * T / M + sigma * np.sqrt(T / M) * npr.standard_normal(M)))
        paths[i, 1:] = path
    return pd.DataFrame(paths)

# Y = exp(-rdt) * F(S^k_n+1, t_n+1), what we expect S^k to be at time n + 1
def calculate_expected_value(paths_matrix, r, t, M):
    average_next_step = np.mean(paths_matrix[:, t + 1])
    return math.exp(-r * t/M) * average_next_step

#create polynomial regression model for each time step
def create_model(design_matrix, target_Y):
    model = LinearRegression()
    model.fit(design_matrix, target_Y)
    return model

#design matrix for each time step
def create_design_matrix(all_paths, t, M):
    X_curr_step = all_paths[:, t]
    data = {'bias': 1, 'X': X_curr_step, 'X^2': X_curr_step ** 2, 'X^3': X_curr_step ** 3}
    target_Y = calculate_expected_value(all_paths, r, t, M)
    return pd.DataFrame(data), target_Y

#profit from exercising the option
def exercise_decision(option_exercise, strike_price):
    return max(option_exercise - strike_price, 0)

#backtracking algorithm to return best timestamp and discounted best exercise value
def backtracking(M, r, beta, t, initial_price_asset, strike_price):
    paths = simulate_stock_price(initial_price_asset, r, beta, t, M)
    max_exercise_value = exercise_decision(np.mean(paths[:, M]), strike_price)
    max_exercise_timestamp = M
    for i in range(M - 1, 0, -t):
        max_exercise_value = math.exp(-r * t/M) * max_exercise_value
        design_matrix, target_Y = create_design_matrix(paths, i, M)
        curr_model = create_model(design_matrix, target_Y)
        curr_exercise_value = curr_model.predict(design_matrix)
        if curr_exercise_value > max_exercise_value:
            max_exercise_value = curr_exercise_value
            max_exercise_timestamp = i
    return max_exercise_timestamp, max_exercise_value

# F^k_n+1 = F(S^k_n+1, t_n+1)
# Y = a0X + a1X^2 ... + aN-1X^N + aN
# X = [1, S, S^2, ... , S^N]


