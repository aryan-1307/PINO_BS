import os
import numpy as np
import pandas as pd
import torch
from fno_pino import PINO2d, compute_pino_loss


def train():
    if not os.path.exists('data/generated/surface_inputs.npy') or \
       not os.path.exists('data/generated/surface_outputs.npy'):
        print("Required surface training datasets are missing. Please run generate_data.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    X_data = np.load('data/generated/surface_inputs.npy')
    Y_data = np.load('data/generated/surface_outputs.npy')

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

    grid_size = 64
    # T starts at 0.0 — must match generate_data.py exactly
    s_space = np.linspace(1.0, 1000.0, grid_size)
    t_space = np.linspace(0.0, 3.0, grid_size)
    S_mesh, T_mesh = np.meshgrid(s_space, t_space, indexing='ij')

    S_grid = torch.tensor(S_mesh, dtype=torch.float32).to(device)
    T_grid = torch.tensor(T_mesh, dtype=torch.float32).to(device)

    split = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    Y_train, Y_val = Y_tensor[:split], Y_tensor[split:]

    model = PINO2d(modes1=16, modes2=16, width=64, num_layers=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    batch_size = 32
    epochs = 500
    physics_weight = 0.1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    print("Starting Forward PINO Training...")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        num_samples = X_train.size(0)
        steps = int(np.ceil(num_samples / batch_size))

        epoch_loss = epoch_data = epoch_pde = 0.0

        for step in range(steps):
            start = step * batch_size
            end = min(start + batch_size, num_samples)
            idx = perm[start:end]

            batch_params = X_train[idx].to(device)
            batch_y = Y_train[idx].to(device)

            optimizer.zero_grad()

            preds = model(batch_params, S_grid, T_grid)
            loss_data = torch.mean((preds - batch_y) ** 2)
            loss_pde = compute_pino_loss(model, batch_params, S_grid, T_grid)
            loss = loss_data + physics_weight * loss_pde

            loss.backward()
            optimizer.step()

            n = end - start
            epoch_loss += loss.item() * n
            epoch_data += loss_data.item() * n
            epoch_pde += loss_pde.item() * n

        epoch_loss /= num_samples
        epoch_data /= num_samples
        epoch_pde /= num_samples

        model.eval()
        val_accum = 0.0
        num_val = X_val.size(0)
        val_steps = int(np.ceil(num_val / batch_size))

        with torch.no_grad():
            for v in range(val_steps):
                vs = v * batch_size
                ve = min(vs + batch_size, num_val)
                bx = X_val[vs:ve].to(device)
                by = Y_val[vs:ve].to(device)
                val_preds = model(bx, S_grid, T_grid)
                val_accum += torch.mean((val_preds - by) ** 2).item() * (ve - vs)

        val_loss = val_accum / num_val if num_val > 0 else 0.0

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Data: {epoch_data:.6f} | "
            f"PDE: {epoch_pde:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        scheduler.step()
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'data_loss': epoch_data,
            'pde_loss': epoch_pde,
            'val_loss': val_loss
        })

    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame(history).to_csv('outputs/fwd_history.csv', index=False)
    torch.save(model.state_dict(), 'outputs/fwd_model.pth')
    print("Forward PINO training complete. Model saved to outputs/fwd_model.pth")


if __name__ == '__main__':
    train()