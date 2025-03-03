import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.integrate
import sklearn as sk
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import math
import scipy
import yfinance as yf
from numpy.polynomial.hermite import hermval
 
# S~LN(mu, sigma)
#S(t + deltat) = S(t) * exp((r - sigma^2/2) * deltat + sigma * sqrt(deltat) * Z), Z = N(0, 1)

#invariances
#run monte carlo simulation to generate stock price paths
def simulate_stock_price(S_t, r, sigma, t, M, N=10000):
    paths = np.zeros((N, M+1))
    paths[:, 0] = S_t
    for i in range(N):
        path = S_t * np.exp(np.cumsum((r - sigma ** 2 / 2) * (t / M) + sigma * np.sqrt(t / M) * npr.standard_normal(M)))
        paths[i, 1:] = path
    return pd.DataFrame(paths)

# F(w; tk) = E_Q[k SIGMA j = k+1 exp(- t_j INTEGRAL t_k (r)) * C(omega, tj; tk, T) | F_t_k], where F(omega, k) is the value of continuation from timestep tk to T
def calculate_expected_vector(paths_df, k, M, t, r, strike_price):
    #design matrix
    copy_paths =  paths_df.copy()
    for j in range(0, copy_paths.shape[1], t):
        discount_factor = math.exp(-(r * (j/M)))
        cash_flow = exercise_decision(copy_paths.iloc[:, j], strike_price)
        discounted_column = discount_factor * cash_flow
        copy_paths.iloc[:, j] = discounted_column
    return np.sum(copy_paths, axis=1)

#create polynomial regression model for each time step
def create_model(design_matrix, target_Y):
    model = Ridge(alpha=0.1)
    model.fit(design_matrix, target_Y)
    return model

#design matrix for each time step
def create_design_matrix_hermite_polynomial(paths_df, k, t, M, r, strike_price, max_degree=2):
    unaltered_design = paths_df.iloc[:, k+1:]
    design_features = pd.DataFrame(index=unaltered_design.index)
    
    for col in unaltered_design.columns:
        for degree in range(max_degree + 1):
            coeffs = [0] * degree + [1]
            col_name = f'{col}_H{degree}' 
            design_features[col_name] = unaltered_design[col].apply(lambda x: hermval(x, coeffs))
    target_Y = calculate_expected_vector(unaltered_design, k, M, t, r, strike_price)
    return design_features, target_Y

#profit from exercising the option
def exercise_decision(option_exercise, strike_price):
    return np.maximum(option_exercise - strike_price, 0)

#backtracking algorithm to return best timestamp and discounted best exercise value

def backtracking(M, r, k, beta, t, initial_price_asset, strike_price, N=10000):
    paths = simulate_stock_price(initial_price_asset, r, beta, t, M, N)
    recent_non_zero = np.zeros(N)
    timestamps = np.zeros(N)
    for i in range(M - 1, -1, -t):
        in_the_money = paths[exercise_decision(paths.iloc[:, i], strike_price) > 0]
        if in_the_money.empty:
            predicted_vector = np.exp(-r * (1 / M)) * in_the_money.iloc[:, i + 1]
            predicted_exercise_value = exercise_decision(predicted_vector, strike_price)
            current_exercise_value = exercise_decision(in_the_money.iloc[:, i], strike_price)
            best_choice = np.where(current_exercise_value > predicted_exercise_value, current_exercise_value, 0)
            zeroes = np.nonzero(best_choice)[0]
            recent_non_zero[zeroes] = best_choice[zeroes]
            timestamps[zeroes] = i
        else:
            design_matrix, target_Y = create_design_matrix_hermite_polynomial(in_the_money, i, t, M, r, strike_price)
            model = create_model(design_matrix, target_Y)
            predicted_vector = model.predict(design_matrix)
            current_exercise_value = exercise_decision(in_the_money.iloc[:, i], strike_price)
            predicted_exercise_value = exercise_decision(predicted_vector, strike_price)
            best_choice = np.where(current_exercise_value > predicted_exercise_value, current_exercise_value, 0)
            zeroes = np.nonzero(best_choice)[0]
            recent_non_zero[zeroes] = best_choice[zeroes]
            timestamps[zeroes] = i      
    for i in range(len(recent_non_zero)):
        timestamp = timestamps[i]
        recent_non_zero[i] = np.exp(-(r * (timestamp/M))) * recent_non_zero[i]
    return np.mean(recent_non_zero)

def option_pricing(M, r, k, beta, t, initial_price_asset, strike_price):
    option_price = backtracking(M, r, k, beta, t, initial_price_asset, strike_price)
    return option_price


# simulate = simulate_stock_price(initial_price_asset, r, beta, t, M)
# print(simulate)
# plt.plot(simulate.T)
# plt.show()

# F^k_n+1 = F(S^k_n+1, t_n+1)
# Y = a0X + a1X^2 ... + aN-1X^N + aN
# X = [1, S, S^2, ... , S^N]


