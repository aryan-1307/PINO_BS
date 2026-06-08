from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np


def black_scholes_call(S, K, T, r, sigma):
    eps = 1e-12

    if T <= 0:
        return max(S - K, 0.0)

    sigma = max(float(sigma), eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)

    return (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def implied_volatility(
    S,
    K,
    T,
    r,
    price,
    sigma_lower=1e-6,
    sigma_upper=5.0
):
    if T <= 0:
        return np.nan

    intrinsic = max(S - K * np.exp(-r * T), 0.0)

    if price <= intrinsic:
        return np.nan

    def objective(sigma):
        return (
            black_scholes_call(
                S,
                K,
                T,
                r,
                sigma
            )
            - price
        )

    try:
        iv = brentq(
            objective,
            sigma_lower,
            sigma_upper,
            maxiter=500
        )
        return float(iv)

    except ValueError:
        return np.nan

