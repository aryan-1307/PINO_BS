import os
import numpy as np


def exact_solution(x_mesh, t_mesh, lam):
    """
    Exact solution: u(x,t) = exp(-lambda * pi^2 * t) * sin(pi * x)
    Satisfies:
      du/dt = lambda * d2u/dx2  on (0,1) x (0,T)
      u(0,t) = u(1,t) = 0      (Dirichlet BC)
      u(x,0) = sin(pi*x)       (IC)
    """
    return np.exp(-lam * np.pi ** 2 * t_mesh) * np.sin(np.pi * x_mesh)


def make_dataset(num_samples=10000, grid_size=64):
    np.random.seed(42)
    os.makedirs('data/generated', exist_ok=True)

    # Parameter and domain ranges
    lam_min, lam_max = 0.01, 2.0
    x_min,   x_max   = 0.0,  1.0
    t_min,   t_max   = 0.0,  1.0

    x_grid = np.linspace(x_min, x_max, grid_size)
    t_grid = np.linspace(t_min, t_max, grid_size)
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid, indexing='ij')

    inputs   = []
    surfaces = []

    for i in range(num_samples):
        lam = np.random.uniform(lam_min, lam_max)
        u   = exact_solution(x_mesh, t_mesh, lam)
        inputs.append([lam])
        surfaces.append(u.astype(np.float32))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{num_samples} samples generated...")

    inputs   = np.array(inputs,   dtype=np.float32)   # (N, 1)
    surfaces = np.array(surfaces, dtype=np.float32)   # (N, 64, 64)

    # Normalisation stats — same structure as Black-Scholes project
    # Input lambda: min-max to [0, 1]
    param_min = np.array([lam_min], dtype=np.float32)
    param_max = np.array([lam_max], dtype=np.float32)

    # Output u: already in [-1, 1] due to sin — V_scale=1.0 for consistency
    V_scale = np.float32(1.0)

    np.savez_compressed(
        'data/generated/dataset.npz',
        inputs    = inputs,
        surfaces  = surfaces,
        S_grid    = x_grid.astype(np.float32),   # named S_grid for consistency
        T_grid    = t_grid.astype(np.float32),   # named T_grid for consistency
        param_min = param_min,
        param_max = param_max,
        V_scale   = np.array([V_scale]),
    )
    print(f"Saved {num_samples} heat equation solutions -> data/generated/dataset.npz")
    print(f"  param_min = {param_min}  (lambda_min)")
    print(f"  param_max = {param_max}  (lambda_max)")
    print(f"  V_scale   = {V_scale}")


if __name__ == '__main__':
    make_dataset()
