import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 
# S~LN(mu, sigma)
#S(t + deltat) = S(t) * exp((r - sigma^2/2) * deltat + sigma * sqrt(deltat) * Z), Z = N(0, 1)

def simulate_stock_price(S_t, r, sigma, T, M):
    return S_t * np.exp(np.cumsum((r - sigma ** 2 / 2) * T / M + sigma * np.sqrt(T / M) * npr.standard_normal(M)))

initial_price_asset = 100      
r = 0.0412 # 1 year treasury bond rate 
beta = 0.2   
t = 1         
N = 10000     
M = 252 

#returns a 2D array of N paths with M time steps
def run_simulation(S_t, r, sigma, T, M):
    paths = np.zeros((N, M+1))
    paths[:, 0] = S_t
    for i in range(N):
        path = simulate_stock_price(paths[i, 0], r, sigma, T, M)
        paths[i, 1:] = path
    return paths

monte_paths = run_simulation(initial_price_asset, r, beta, t, M)
plt.plot(monte_paths.T)
plt.show()


# Y = exp(-rdt) * F(S^k_n+1, t_n+1)
# F^k_n+1 = F(S^k_n+1, t_n+1)
# Y = a0X + a1X^2 ... + aN-1X^N + aN
# X = [1, S, S^2, ... , S^N]



