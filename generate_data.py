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


def make_dataset(num_surfaces=1000, grid_size=64):
    np.random.seed(42)
    os.makedirs('data/generated', exist_ok=True)

    # T starts at 0.0 so that index 0 is the true terminal condition V(S,0)=max(S-K,0)
    S_grid = np.linspace(1.0, 1000.0, grid_size)
    T_grid = np.linspace(0.0, 3.0, grid_size)
    S_mesh, T_mesh = np.meshgrid(S_grid, T_grid, indexing='ij')

    inputs = []
    surfaces = []

    for _ in range(num_surfaces):
        K = np.random.uniform(50.0, 1000.0)
        r = np.random.uniform(0.0, 0.10)
        sigma = np.random.uniform(0.05, 1.0)
        price_surface = black_scholes_call(S_mesh, K, T_mesh, r, sigma)
        inputs.append([K, r, sigma])
        surfaces.append((price_surface / 1000.0).astype(np.float32))

    inputs = np.array(inputs, dtype=np.float32)
    surfaces = np.array(surfaces, dtype=np.float32)

    np.save('data/generated/surface_inputs.npy', inputs)
    np.save('data/generated/surface_outputs.npy', surfaces)
    print(f"Generated {num_surfaces} option-price surfaces.")


if __name__ == '__main__':
    make_dataset()