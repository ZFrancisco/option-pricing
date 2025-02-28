import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.integrate
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
import scipy
 
# S~LN(mu, sigma)
#S(t + deltat) = S(t) * exp((r - sigma^2/2) * deltat + sigma * sqrt(deltat) * Z), Z = N(0, 1)

#invariances
initial_price_asset = 100      
r = 0.0412 # 1 year treasury bond rate 
beta = 0.2   
t = 1         
N = 10000     
M = 252
strike_price = 105 

#run monte carlo simulation to generate stock price paths
def simulate_stock_price(S_t, r, sigma, T, M):
    paths = np.zeros((N, M+1))
    paths[:, 0] = S_t
    for i in range(N):
        path =  S_t * np.exp(np.cumsum((r - sigma ** 2 / 2) * T / M + sigma * np.sqrt(T / M) * npr.standard_normal(M)))
        paths[i, 1:] = path
    return pd.DataFrame(paths)

# F(w; tk) = E_Q[k SIGMA j = k+1 exp(- t_j INTEGRAL t_k (r)) * C(omega, tj; tk, T) | F_t_k], where F(omega, k) is the value of continuation from timestep tk to T
def calculate_expected_vector(paths_df, k, M):
    copy_paths =  paths_df.copy()
    for j in range(k+1, M + 1, t):
        discount_factor = np.exp(-(scipy.integrate.quad(lambda x: r, k, j)[0]))
        discounted_column = discount_factor * copy_paths.iloc[:, j]
        copy_paths.iloc[:, j] = discounted_column
    copy_paths = copy_paths.iloc[:, k+1:]
    return copy_paths.mean(axis=1)


#create polynomial regression model for each time step
def create_model(design_matrix, target_Y):
    model = LinearRegression()
    model.fit(design_matrix, target_Y)
    return model

#design matrix for each time step
def create_design_matrix(paths_df, k, t, M):
    unaltered_design = paths_df.iloc[:, k+1:]
    data = {'bias': 1, 'X': unaltered_design, 'X^2': unaltered_design ** 2, 'X^3': unaltered_design ** 3}
    target_Y = calculate_expected_vector(paths_df, k, M)
    return pd.DataFrame(data), target_Y

#profit from exercising the option
def exercise_decision(option_exercise, strike_price):
    return np.maximum(option_exercise - strike_price, 0)

#backtracking algorithm to return best timestamp and discounted best exercise value
def backtracking(M, r, k, beta, t, initial_price_asset, strike_price):
    paths = simulate_stock_price(initial_price_asset, r, beta, t, M)
    exercise_df = pd.DataFrame(np.zeros((N, M + 1)))
    exercise_df.iloc[:, M+1] = exercise_decision(paths.iloc[:, M+1], strike_price)
    recent_non_zero = np.zeros(N)
    for i in range(M, -1, -t):
        in_the_money = paths[exercise_decision(paths.iloc[:, i]) > 0]
        design_matrix, target_Y = create_design_matrix(in_the_money, k, t, M)
        model = create_model(design_matrix, target_Y)
        predicted_vector = model.predict(design_matrix)
        predicted_exercise_value = exercise_decision(predicted_vector, strike_price)
        current_exercise_value = exercise_decision(in_the_money, strike_price)
        best_choice = np.where(predicted_exercise_value > current_exercise_value, predicted_exercise_value, current_exercise_value)
        exercise_df.iloc[in_the_money.index, i] = best_choice
        recent_non_zero[in_the_money.index] = i
    expected_option_price = np.mean(recent_non_zero)
    return exercise_df, expected_option_price

def option_pricing(M, r, k, beta, t, initial_price_asset, strike_price):
    df, option_price = backtracking(M, r, k, beta, t, initial_price_asset, strike_price)
    return option_price

# F^k_n+1 = F(S^k_n+1, t_n+1)
# Y = a0X + a1X^2 ... + aN-1X^N + aN
# X = [1, S, S^2, ... , S^N]


