import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from inverse_nn import ImpliedVolRegressor

def train():
    if not os.path.exists('data/generated/bs_dataset.csv'):
        print("Training data missing. Please run generate_data.py first.")
        return
        
    df = pd.read_csv('data/generated/bs_dataset.csv')
    X = df[['S', 'K', 'T', 'r', 'price']].values.astype(np.float32)
    Y = df['sigma'].values.astype(np.float32)
    
    X_min = np.array([50.0, 50.0, 0.01, 0.00, 0.0], dtype=np.float32)
    X_max = np.array([1000.0, 1000.0, 3.0, 0.10, 1000.0], dtype=np.float32)
    
    X_norm = (X - X_min) / (X_max - X_min)
    
    X_tensor = torch.tensor(X_norm)
    Y_tensor = torch.tensor(Y)
    
    split = int(0.8 * len(X))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    Y_train, Y_val = Y_tensor[:split], Y_tensor[split:]
    
    model = ImpliedVolRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    batch_size = 256
    epochs = 60
    history = []
    print("Starting Inverse Implied Volatility Regressor Training...")
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            b_x, b_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            preds = model(b_x).squeeze(-1)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * b_x.size(0)
            
        epoch_loss /= X_train.size(0)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).squeeze(-1)
            val_loss = criterion(val_preds, Y_val).item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {epoch_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        history.append({
            'epoch': epoch + 1, 'train_loss': epoch_loss, 'val_loss': val_loss
        })
        
    os.makedirs('outputs', exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv('outputs/inv_history.csv', index=False)
    torch.save(model.state_dict(), 'outputs/inv_model.pth')
    print("Inverse model checkpoint saved successfully.")

if __name__ == '__main__':
    train()