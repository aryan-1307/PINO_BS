import numpy as np


def recover_lambda(x, t, u_val, eps=1e-8):
    """
    Analytically recover lambda from a single observation of u(x,t).

    From exact solution: u(x,t) = exp(-lambda * pi^2 * t) * sin(pi*x)
    Inverting:           lambda  = -log(u / sin(pi*x)) / (pi^2 * t)

    Parameters:
        x     : float, spatial coordinate in (0,1)
        t     : float, time > 0
        u_val : float, observed/predicted u(x,t)

    Returns:
        lambda value or np.nan if inversion is unstable
    """
    if t <= 0:
        return np.nan

    sin_pix = np.sin(np.pi * x)
    if abs(sin_pix) < eps:
        # Near x=0 or x=1 boundary — sin(pi*x) ≈ 0, inversion unstable
        return np.nan

    ratio = u_val / (sin_pix + eps)
    if ratio <= 0:
        # log undefined for non-positive ratio
        return np.nan

    lam = -np.log(ratio) / (np.pi ** 2 * t)

    if lam <= 0:
        # Lambda must be positive
        return np.nan

    return float(lam)
