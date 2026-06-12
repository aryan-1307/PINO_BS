import os
import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    eps = 1e-7
    T = np.maximum(T, eps)
    denom = sigma * np.sqrt(T) + eps
    d1 = (np.log(S / (K + eps)) + (r + 0.5 * sigma ** 2) * T) / denom
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def make_dataset(num_surfaces=10000, grid_size=64):
    np.random.seed(42)
    os.makedirs('data/generated', exist_ok=True)

    K_min,     K_max     = 50.0,  1000.0
    r_min,     r_max     = 0.0,   0.10
    sigma_min, sigma_max = 0.05,  1.0
    S_min,     S_max     = 1.0,   1000.0
    T_min,     T_max     = 0.0,   3.0

    S_grid = np.linspace(S_min, S_max, grid_size)
    T_grid = np.linspace(T_min, T_max, grid_size)
    S_mesh, T_mesh = np.meshgrid(S_grid, T_grid, indexing='ij')

    inputs   = []
    surfaces = []

    for i in range(num_surfaces):
        K     = np.random.uniform(K_min, K_max)
        r     = np.random.uniform(r_min, r_max)
        sigma = np.random.uniform(sigma_min, sigma_max)
        V     = black_scholes_call(S_mesh, K, T_mesh, r, sigma)
        inputs.append([K, r, sigma])
        surfaces.append(V.astype(np.float32))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{num_surfaces} surfaces generated...")

    inputs   = np.array(inputs,   dtype=np.float32)
    surfaces = np.array(surfaces, dtype=np.float32)

    param_min = np.array([K_min,  r_min,  sigma_min], dtype=np.float32)
    param_max = np.array([K_max,  r_max,  sigma_max], dtype=np.float32)
    V_scale   = np.float32(S_max)

    np.savez_compressed(
        'data/generated/dataset.npz',
        inputs    = inputs,
        surfaces  = surfaces,
        S_grid    = S_grid.astype(np.float32),
        T_grid    = T_grid.astype(np.float32),
        param_min = param_min,
        param_max = param_max,
        V_scale   = np.array([V_scale]),
    )
    print(f"Saved {num_surfaces} surfaces -> data/generated/dataset.npz")
    print(f"  param_min = {param_min}")
    print(f"  param_max = {param_max}")
    print(f"  V_scale   = {V_scale}")


if __name__ == '__main__':
    make_dataset()
