import os
import yfinance as yf
import pandas as pd
import numpy as np
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
    history = ticker.history(period='1mo')
    log_returns = np.log(history['Close'] / history['Close'].shift(1))
    hist_vol = log_returns.std() * np.sqrt(252)
    if np.isnan(hist_vol) or hist_vol == 0:
        hist_vol = 0.20
        
    today = datetime.now().date()
    records = []
    
    selected_expirations = expirations[:3]
    print(f"Fetching first {len(selected_expirations)} expirations for {ticker_symbol}...")
    
    for chosen_date_str in selected_expirations:
        try:
            opt_chain = ticker.option_chain(chosen_date_str)
            calls = opt_chain.calls
            
            expiry_date = datetime.strptime(chosen_date_str, "%Y-%m-%d").date()
            days_to_maturity = (expiry_date - today).days
            
            if days_to_maturity <= 0:
                continue
                
            T = days_to_maturity / 365.25
            
            for _, row in calls.iterrows():
                strike = row['strike']
                bid = row['bid']
                ask = row['ask']
                mid_price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else row['lastPrice']
                
                if mid_price <= 0 or strike <= 0:
                    continue
                    
                records.append({
                    'S': float(underlying),
                    'K': float(strike),
                    'T': float(T),
                    'r': 0.045,
                    'proxy_sigma': float(hist_vol),
                    'price': float(mid_price)
                })
        except Exception:
            continue
            
    if len(records) == 0:
        print("No valid option contracts extracted.")
        return None
        
    df = pd.DataFrame(records)
    os.makedirs('data/market', exist_ok=True)
    df.to_csv('data/market/market_data.csv', index=False)
    print(f"Successfully saved {len(df)} market option records to data/market/market_data.csv")
    return df

if __name__ == '__main__':
    fetch_market_snapshot()