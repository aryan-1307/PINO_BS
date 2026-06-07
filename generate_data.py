import os
import numpy as np
import pandas as pd
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    eps = 1e-7
    denom = sigma * np.sqrt(T) + eps
    d1 = (np.log(S / (K + eps)) + (r + 0.5 * sigma ** 2) * T) / denom
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def make_dataset(num_labeled=40000, num_collocation=20000):
    np.random.seed(42)
    os.makedirs('data/generated', exist_ok=True)
    
    S_lbl = np.random.uniform(50.0, 1000.0, num_labeled)
    K_lbl = np.random.uniform(50.0, 1000.0, num_labeled)
    T_lbl = np.random.uniform(0.1, 3.0, num_labeled)
    r_lbl = np.random.uniform(0.0, 0.10, num_labeled)
    sigma_lbl = np.random.uniform(0.5, 1.0, num_labeled)
    
    prices = black_scholes_call(S_lbl, K_lbl, T_lbl, r_lbl, sigma_lbl)
    
    labeled_df = pd.DataFrame({
        'S': S_lbl, 'K': K_lbl, 'T': T_lbl, 'r': r_lbl, 'sigma': sigma_lbl, 'price': prices
    })
    labeled_df.to_csv('data/generated/bs_dataset.csv', index=False)
    print(f"Generated {num_labeled} labeled samples in data/generated/bs_dataset.csv")
    
    S_col = np.random.uniform(50.0, 1000.0, num_collocation)
    K_col = np.random.uniform(50.0, 1000.0, num_collocation)
    T_col = np.random.uniform(0.1, 3.0, num_collocation)
    r_col = np.random.uniform(0.00, 0.10, num_collocation)
    sigma_col = np.random.uniform(0.5, 1.0, num_collocation)
    
    collocation_df = pd.DataFrame({
        'S': S_col, 'K': K_col, 'T': T_col, 'r': r_col, 'sigma': sigma_col
    })
    collocation_df.to_csv('data/generated/collocation.csv', index=False)
    print(f"Generated {num_collocation} collocation samples in data/generated/collocation.csv")

if __name__ == '__main__':
    make_dataset()