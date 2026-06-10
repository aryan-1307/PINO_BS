import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def fetch_market_snapshot(ticker_symbol='SPY'):
    ticker = yf.Ticker(ticker_symbol)
    try:
        expirations = ticker.options
        if not expirations:
            print(f"No option chains found for ticker {ticker_symbol}")
            return None
    except Exception as e:
        print(f"Error connecting to yfinance: {e}")
        return None

    underlying = ticker.history(period='1d')['Close'].iloc[-1]
    history    = ticker.history(period='1mo')
    log_returns = np.log(history['Close'] / history['Close'].shift(1))
    hist_vol    = log_returns.std() * np.sqrt(252)
    if np.isnan(hist_vol) or hist_vol == 0:
        hist_vol = 0.20

    # Clamp to training sigma range [0.05, 1.0]
    hist_vol = float(np.clip(hist_vol, 0.05, 1.0))

    today = datetime.now().date()
    records = []
    selected = expirations[:3]
    print(f"Fetching {len(selected)} expirations for {ticker_symbol}...")

    for date_str in selected:
        try:
            calls = ticker.option_chain(date_str).calls
            expiry = datetime.strptime(date_str, "%Y-%m-%d").date()
            days   = (expiry - today).days
            if days <= 0:
                continue
            T = days / 365.25
            if T > 3.0:
                continue

            for _, row in calls.iterrows():
                strike = row['strike']
                bid, ask = row['bid'], row['ask']
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else row['lastPrice']
                if mid <= 0 or strike <= 0:
                    continue
                if not (1.0 <= float(underlying) <= 1000.0):
                    continue
                if not (50.0 <= float(strike) <= 1000.0):
                    continue

                records.append({
                    'S': float(underlying),
                    'K': float(strike),
                    'T': float(T),
                    'r': 0.045,
                    'proxy_sigma': hist_vol,
                    'price': float(mid)
                })
        except Exception:
            continue

    if len(records) == 0:
        print("No valid option contracts extracted.")
        return None

    df = pd.DataFrame(records)
    os.makedirs('data/market', exist_ok=True)
    df.to_csv('data/market/market_data.csv', index=False)
    print(f"Saved {len(df)} records to data/market/market_data.csv")
    return df


if __name__ == '__main__':
    fetch_market_snapshot()
