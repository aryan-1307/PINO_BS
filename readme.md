Overview
This project implements a Physics-Informed Neural Operator (PINO) based on a Fourier Neural Operator (FNO) architecture for solving the Black-Scholes option pricing problem.
The project consists of two parts:
1. Forward Problem
Learn the mapping:
(S, K, T, r, sigma) → Option Price
The model uses a Fourier Neural Operator together with a physics-informed loss based on the Black-Scholes partial differential equation.
2. Inverse Problem
Recover implied volatility from PINO-predicted option prices using numerical inversion of the Black-Scholes equation.
(S, K, T, r, Option Price) → Implied Volatility
Implied volatility is computed using Brent’s root-finding method applied to the Black-Scholes pricing formula.

Synthetic Black-Scholes data is generated automatically for training. Real market option-chain data can also be downloaded using yfinance for testing and demonstration.

Execution Order
python generate_data.py
Generates synthetic Black-Scholes training data and collocation points.
python market_data.py
Downloads option-chain data from Yahoo Finance for testing and demonstration.
python train_fwd.py
Trains the PINO model for Black-Scholes option pricing.
python evaluation.py
Evaluates the trained PINO model, recovers implied volatility from PINO-predicted prices, and generates metrics and visualizations.

Summary
The forward PINO model learns the option pricing operator while enforcing the Black-Scholes PDE through a physics-informed loss.
Implied volatility is recovered from the PINO-predicted option prices using numerical inversion of the Black-Scholes equation.