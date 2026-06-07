Overview
This project implements a Physics-Informed Neural Operator (PINO) based on a Fourier Neural Operator (FNO) architecture for solving the Black-Scholes option pricing problem.
The project consists of two parts:
1. Forward Problem
Learn the mapping:
(S, K, T, r, sigma) → Option Price
The model uses a Fourier Neural Operator together with a physics-informed loss based on the Black-Scholes partial differential equation.
2. Inverse Problem
Learn the mapping:
(S, K, T, r, Option Price) → Implied Volatility
A neural network regressor is used to estimate implied volatility from option prices.
Synthetic Black-Scholes data is generated automatically for training. Real market option-chain data can also be downloaded using yfinance for testing and demonstration.
Execution Order
python generate_data.py
python market_data.py
python train_fwd.py
python train_imv.py
python evaluation.py