import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from fno_pino import PINO1d, compute_pino_loss

def train():
    if not os.path.exists('data/generated/bs_dataset.csv') or not os.path.exists('data/generated/collocation.csv'):
        print("Required training datasets are missing. Please run generate_data.py first.")
        return
        
    # Define device first so it is available for tensor conversion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    df_lbl = pd.read_csv('data/generated/bs_dataset.csv')
    X_lbl = df_lbl[['S', 'K', 'T', 'r', 'sigma']].values.astype(np.float32)
    Y_lbl = df_lbl['price'].values.astype(np.float32)
    
    df_col = pd.read_csv('data/generated/collocation.csv')
    X_col = df_col[['S', 'K', 'T', 'r', 'sigma']].values.astype(np.float32)
    
    X_min = np.array([50.0, 50.0, 0.01, 0.00, 0.05], dtype=np.float32)
    X_max = np.array([1000.0, 1000.0, 3.00, 0.10, 1.00], dtype=np.float32)
    
    X_lbl_norm = (X_lbl - X_min) / (X_max - X_min)
    X_col_norm = (X_col - X_min) / (X_max - X_min)
    
    X_lbl_tensor = torch.tensor(X_lbl_norm, dtype=torch.float32).to(device)
    Y_lbl_tensor = torch.tensor(Y_lbl, dtype=torch.float32).to(device)
    X_col_tensor = torch.tensor(X_col_norm, dtype=torch.float32).to(device)
    
    # Send normalization bounds to device to prevent runtime device mismatch errors
    X_min_t = torch.tensor(X_min, dtype=torch.float32).to(device)
    X_max_t = torch.tensor(X_max, dtype=torch.float32).to(device)
    
    split = int(0.8 * len(X_lbl_tensor))
    X_train, X_val = X_lbl_tensor[:split], X_lbl_tensor[split:]
    Y_train, Y_val = Y_lbl_tensor[:split], Y_lbl_tensor[split:]

    model = PINO1d(
        width=128,
        modes=8,
        num_layers=6
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    batch_size = 256
    epochs = 500
    physics_weight = 0.1

    # Updated: T_max is now dynamically set to epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    
    history = []
    print("Starting Corrected Forward PINO Training Matrix...")
    
    for epoch in range(epochs):
        model.train()
        perm_lbl = torch.randperm(X_train.size(0))
        perm_col = torch.randperm(X_col_tensor.size(0))
        
        epoch_loss, epoch_data, epoch_pde = 0.0, 0.0, 0.0
        steps = X_train.size(0) // batch_size
        
        for step in range(steps):
            idx_lbl = perm_lbl[step * batch_size : (step + 1) * batch_size]
            b_x_lbl, b_y_lbl = X_train[idx_lbl], Y_train[idx_lbl]
            
            idx_col = perm_col[(step * batch_size) % X_col_tensor.size(0) : ((step + 1) * batch_size) % X_col_tensor.size(0)]
            if len(idx_col) < batch_size:
                idx_col = perm_col[:batch_size]
            b_x_col = X_col_tensor[idx_col]
            
            optimizer.zero_grad()
            
            b_x_lbl_expanded = b_x_lbl.unsqueeze(1)
            preds = model(b_x_lbl_expanded)[:, 0, 0]
            
            loss_data = torch.mean((preds - b_y_lbl) ** 2)
            
            # Using the pre-defined device tensors to avoid CPU-GPU conflicts
            b_x_col_raw = b_x_col * (X_max_t - X_min_t) + X_min_t
            loss_pde = compute_pino_loss(model, b_x_col_raw)
            
            loss = loss_data + physics_weight * loss_pde
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
            epoch_data += loss_data.item() * batch_size
            epoch_pde += loss_pde.item() * batch_size
            
        epoch_loss /= (steps * batch_size)
        epoch_data /= (steps * batch_size)
        epoch_pde /= (steps * batch_size)
        
        model.eval()
        with torch.no_grad():
            val_expanded = X_val.unsqueeze(1)
            val_preds = model(val_expanded)[:, 0, 0]
            val_loss = torch.mean((val_preds - Y_val) ** 2).item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.6f} | Data MSE: {epoch_data:.6f} | Physics Loss: {epoch_pde:.6f} | Val MSE: {val_loss:.6f}")
        
        # Advance the learning rate scheduler every epoch
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1, 'train_loss': epoch_loss,
            'data_loss': epoch_data, 'pde_loss': epoch_pde, 'val_loss': val_loss
        })
        
    os.makedirs('outputs', exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv('outputs/fwd_history.csv', index=False)
    torch.save(model.state_dict(), 'outputs/fwd_model.pth')
    print("Forward PINO engine trained and saved successfully.")

if __name__ == '__main__':
    train()