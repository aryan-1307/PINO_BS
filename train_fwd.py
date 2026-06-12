import os
import numpy as np
import pandas as pd
import torch
from fno_pino import PINO2d, compute_pino_loss, relative_l2_loss


def train():
    dataset_path = 'data/generated/dataset.npz'
    if not os.path.exists(dataset_path):
        print("Dataset missing. Run generate_data.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data      = np.load(dataset_path)
    X_raw     = data['inputs']
    Y_raw     = data['surfaces']
    S_grid    = data['S_grid']
    T_grid    = data['T_grid']
    param_min = data['param_min']
    param_max = data['param_max']
    V_scale   = float(data['V_scale'][0])

    X_norm = (X_raw - param_min) / (param_max - param_min + 1e-8)
    Y_norm = Y_raw / V_scale

    X_tensor     = torch.tensor(X_norm, dtype=torch.float32)
    Y_tensor     = torch.tensor(Y_norm, dtype=torch.float32)
    X_raw_tensor = torch.tensor(X_raw,  dtype=torch.float32)

    S_norm = (S_grid - S_grid.min()) / (S_grid.max() - S_grid.min())
    T_norm = (T_grid - T_grid.min()) / (T_grid.max() - T_grid.min())
    S_mesh_n, T_mesh_n = np.meshgrid(S_norm, T_norm, indexing='ij')
    S_grid_2d_n = torch.tensor(S_mesh_n, dtype=torch.float32).to(device)
    T_grid_2d_n = torch.tensor(T_mesh_n, dtype=torch.float32).to(device)

    S_grid_1d = torch.tensor(S_grid, dtype=torch.float32).to(device)
    T_grid_1d = torch.tensor(T_grid, dtype=torch.float32).to(device)
    V_scale_t = torch.tensor(V_scale, dtype=torch.float32).to(device)

    split = int(0.8 * len(X_tensor))
    X_train_n,   X_val_n   = X_tensor[:split],     X_tensor[split:]
    Y_train_n,   Y_val_n   = Y_tensor[:split],     Y_tensor[split:]
    X_train_raw, X_val_raw = X_raw_tensor[:split], X_raw_tensor[split:]

    # Updated: width=128 for more capacity, num_layers=4 unchanged
    model = PINO2d(modes1=16, modes2=16, width=128, num_layers=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch_size  = 64     # increased from 32
    epochs      = 500
    grad_clip   = 1.0
    lambda_pde  = 1.0
    lambda_ic   = 10.0
    lambda_bc   = 10.0
    pde_warmup  = 50
    pde_ramp    = 50

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    os.makedirs('outputs', exist_ok=True)
    best_val_loss = float('inf')
    history = []

    print("Starting PINO Training (normalised inputs/outputs)...")
    print(f"  Train: {len(X_train_n)} | Val: {len(X_val_n)}")
    print(f"  V_scale={V_scale:.1f} | param ranges: {param_min} -> {param_max}")
    print(f"  PDE warmup={pde_warmup} | ramp={pde_ramp}")
    print(f"  width=128 | batch_size=64 | two Fourier weights | MLP layers")

    for epoch in range(1, epochs + 1):
        if epoch <= pde_warmup:
            curr_lambda_pde = 0.0
        elif epoch <= pde_warmup + pde_ramp:
            curr_lambda_pde = lambda_pde * (epoch - pde_warmup) / pde_ramp
        else:
            curr_lambda_pde = lambda_pde

        model.train()
        perm        = torch.randperm(len(X_train_n))
        num_samples = len(X_train_n)
        steps       = int(np.ceil(num_samples / batch_size))

        ep_total = ep_data = ep_pde = ep_ic = ep_bc = 0.0

        for step in range(steps):
            i0  = step * batch_size
            i1  = min(i0 + batch_size, num_samples)
            idx = perm[i0:i1]

            bp_n   = X_train_n[idx].to(device)
            bp_raw = X_train_raw[idx].to(device)
            by_n   = Y_train_n[idx].to(device)

            optimizer.zero_grad()

            V_pred_n = model(bp_n, S_grid_2d_n, T_grid_2d_n)

            l_data = relative_l2_loss(V_pred_n, by_n)

            l_phys, l_pde, l_ic, l_bc = compute_pino_loss(
                V_pred_n, bp_raw, S_grid_1d, T_grid_1d, V_scale_t,
                lambda_pde=curr_lambda_pde,
                lambda_ic=lambda_ic,
                lambda_bc=lambda_bc
            )

            loss = l_data + l_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            n = i1 - i0
            ep_total += loss.item()   * n
            ep_data  += l_data.item() * n
            ep_pde   += l_pde.item()  * n
            ep_ic    += l_ic.item()   * n
            ep_bc    += l_bc.item()   * n

        ep_total /= num_samples
        ep_data  /= num_samples
        ep_pde   /= num_samples
        ep_ic    /= num_samples
        ep_bc    /= num_samples

        model.eval()
        val_accum = 0.0
        num_val   = len(X_val_n)

        with torch.no_grad():
            for v in range(int(np.ceil(num_val / batch_size))):
                vs = v * batch_size
                ve = min(vs + batch_size, num_val)
                vp = model(
                    X_val_n[vs:ve].to(device),
                    S_grid_2d_n, T_grid_2d_n
                )
                val_accum += relative_l2_loss(vp, Y_val_n[vs:ve].to(device)).item() * (ve - vs)

        val_loss = val_accum / num_val

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'outputs/fwd_model_best.pth')

        scheduler.step()

        print(
            f"Ep {epoch:03d}/{epochs} | "
            f"total={ep_total:.4f} data={ep_data:.4f} "
            f"pde={ep_pde:.5f} ic={ep_ic:.5f} bc={ep_bc:.5f} | "
            f"val={val_loss:.4f} best={best_val_loss:.4f} lam_pde={curr_lambda_pde:.2f}"
        )

        history.append({
            'epoch': epoch, 'train_loss': ep_total, 'data_loss': ep_data,
            'pde_loss': ep_pde, 'ic_loss': ep_ic, 'bc_loss': ep_bc,
            'val_loss': val_loss, 'lambda_pde': curr_lambda_pde
        })

    pd.DataFrame(history).to_csv('outputs/fwd_history.csv', index=False)
    torch.save(model.state_dict(), 'outputs/fwd_model_final.pth')
    print(f"Done. Best val rel-L2: {best_val_loss:.6f}")
    print("outputs/fwd_model_best.pth  |  outputs/fwd_model_final.pth")


if __name__ == '__main__':
    train()
