import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from heat_pino import HeatPINO, relative_l2_loss
from inverse_lambda import recover_lambda


def compute_metrics(actual, predicted):
    a = actual.flatten()
    p = predicted.flatten()
    mse     = float(np.mean((a - p) ** 2))
    rmse    = float(np.sqrt(mse))
    mae     = float(np.mean(np.abs(a - p)))
    ss_res  = float(np.sum((a - p) ** 2))
    ss_tot  = float(np.sum((a - np.mean(a)) ** 2))
    r2      = 1.0 - ss_res / (ss_tot + 1e-8)
    rel_l2  = float(np.linalg.norm(a - p) / (np.linalg.norm(a) + 1e-8))
    max_err = float(np.max(np.abs(a - p)))
    mape    = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-8)))) * 100.0
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'MAPE(%)': mape, 'MaxAbsError': max_err,
        'R2': r2, 'RelL2': rel_l2
    }


def run_evaluation():
    dataset_path = 'data/generated/dataset.npz'
    if not os.path.exists(dataset_path):
        print("Evaluation canceled: dataset.npz missing.")
        return

    weights_path = 'outputs/fwd_model_best.pth'
    if not os.path.exists(weights_path):
        weights_path = 'outputs/fwd_model_final.pth'
    if not os.path.exists(weights_path):
        print("Evaluation canceled: no trained weights found. Run train.py first.")
        return

    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/metrics',     exist_ok=True)
    os.makedirs('outputs/figures',     exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on device: {device}  |  weights: {weights_path}")

    data      = np.load(dataset_path)
    X_raw     = data['inputs']
    Y_raw     = data['surfaces']
    x_grid    = data['S_grid']
    t_grid    = data['T_grid']
    param_min = data['param_min']
    param_max = data['param_max']
    V_scale   = float(data['V_scale'][0])

    X_norm = (X_raw - param_min) / (param_max - param_min + 1e-8)
    Y_norm = Y_raw / V_scale

    x_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())
    t_norm = (t_grid - t_grid.min()) / (t_grid.max() - t_grid.min())
    x_mesh_n, t_mesh_n = np.meshgrid(x_norm, t_norm, indexing='ij')
    S_grid_2d_n = torch.tensor(x_mesh_n, dtype=torch.float32).to(device)
    T_grid_2d_n = torch.tensor(t_mesh_n, dtype=torch.float32).to(device)

    split      = int(0.8 * len(X_raw))
    X_test_n   = X_norm[split:]
    X_test_raw = X_raw[split:]
    Y_test_raw = Y_raw[split:]
    Y_test_n   = Y_norm[split:]
    grid_size  = len(x_grid)

    model = HeatPINO().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    batch_size = 64
    preds_norm = []
    X_t = torch.tensor(X_test_n, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            b = X_t[i:i+batch_size].to(device)
            preds_norm.append(model(b, S_grid_2d_n, T_grid_2d_n).cpu().numpy())

    preds_norm = np.concatenate(preds_norm, axis=0)
    preds_raw  = preds_norm * V_scale   # denormalise (V_scale=1.0 here)

    # ------------------------------------------------------------------
    # 1. Forward metrics
    # ------------------------------------------------------------------
    fwd_metrics = compute_metrics(Y_test_raw, preds_raw)
    fwd_metrics['RelL2_norm'] = float(
        np.linalg.norm(preds_norm - Y_test_n) / (np.linalg.norm(Y_test_n) + 1e-8)
    )
    pd.DataFrame(list(fwd_metrics.items()), columns=['Metric', 'Value']).to_csv(
        'outputs/metrics/forward_metrics.csv', index=False
    )
    print(f"Forward metrics: {fwd_metrics}")

    # ------------------------------------------------------------------
    # 2. Inverse lambda recovery — analytical inversion
    # Equivalent to implied_vol.py in Black-Scholes project
    # Filters applied:
    #   t >= 0.1  : near t=0 inversion is unstable (sin term well-defined but exp→1)
    #   x in [0.1, 0.9] : avoid near-boundary where sin(pi*x)→0
    #   |u_val| >= 0.01  : near-zero u causes log instability
    # ------------------------------------------------------------------
    print("Evaluating lambda recovery (analytical inversion)...")
    actual_lams, recovered_lams = [], []
    skipped = 0
    np.random.seed(42)

    for i in range(len(X_test_raw)):
        true_lam = float(X_test_raw[i, 0])
        collected = 0
        attempts  = 0
        while collected < 5 and attempts < 50:
            attempts += 1
            x_idx = np.random.randint(0, grid_size)
            t_idx = np.random.randint(1, grid_size)
            x_val      = float(x_grid[x_idx])
            t_val      = float(t_grid[t_idx])
            pred_u_val = float(preds_raw[i, x_idx, t_idx])

            # Filter 1: skip near t=0
            if t_val < 0.1:
                skipped += 1
                continue
            # Filter 2: skip near boundaries where sin(pi*x)→0
            if x_val < 0.1 or x_val > 0.9:
                skipped += 1
                continue
            # Filter 3: skip near-zero u values
            if abs(pred_u_val) < 0.01:
                skipped += 1
                continue

            lam_rec = recover_lambda(x_val, t_val, pred_u_val)
            if not np.isnan(lam_rec) and 0.01 <= lam_rec <= 2.0:
                actual_lams.append(true_lam)
                recovered_lams.append(lam_rec)
                collected += 1

    print(f"  Lambda samples collected: {len(actual_lams)} | skipped: {skipped}")

    inv_metrics = {}
    if len(actual_lams) > 0:
        inv_metrics = compute_metrics(np.array(actual_lams), np.array(recovered_lams))
        pd.DataFrame(list(inv_metrics.items()), columns=['Metric', 'Value']).to_csv(
            'outputs/metrics/inverse_metrics.csv', index=False
        )
        print(f"Inverse (lambda recovery) metrics: {inv_metrics}")
    else:
        print("Warning: no valid lambda recovery samples found.")

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    # Plot 1: Training loss curves
    if os.path.exists('outputs/fwd_history.csv'):
        hist = pd.read_csv('outputs/fwd_history.csv')
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].semilogy(hist['epoch'], hist['data_loss'], label='Train Rel-L2')
        axes[0].semilogy(hist['epoch'], hist['val_loss'],  label='Val Rel-L2', ls='--')
        axes[0].set_xlabel('Epoch')
        axes[0].set_title('Supervised Loss (Relative L2)')
        axes[0].legend()
        axes[1].semilogy(hist['epoch'], hist['pde_loss'], label='PDE residual')
        axes[1].semilogy(hist['epoch'], hist['ic_loss'],  label='IC u(x,0)=sin(πx)', ls='--')
        axes[1].semilogy(hist['epoch'], hist['bc_loss'],  label='BC u(0,t)=u(1,t)=0', ls=':')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Physics Loss Components (unweighted)')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig('outputs/figures/loss_curves.png', dpi=150)
        plt.close(fig)

    # Plot 2: Forward scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test_raw.flatten(), preds_raw.flatten(), alpha=0.05, s=1, c='steelblue')
    vmin = min(float(Y_test_raw.min()), float(preds_raw.min()))
    vmax = max(float(Y_test_raw.max()), float(preds_raw.max()))
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5)
    plt.xlabel('Actual u(x,t)')
    plt.ylabel('Predicted u(x,t)')
    plt.title(f"Forward PINO  R²={fwd_metrics['R2']:.4f}  RelL2={fwd_metrics['RelL2']:.4f}")
    plt.tight_layout()
    plt.savefig('outputs/figures/forward_scatter.png', dpi=150)
    plt.close()

    # Plot 3: Error histogram
    errors = (preds_raw - Y_test_raw).flatten()
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    plt.axvline(0, color='r', ls='--', lw=1.5)
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.title(f"Error Distribution  MAE={fwd_metrics['MAE']:.4f}  RMSE={fwd_metrics['RMSE']:.4f}")
    plt.tight_layout()
    plt.savefig('outputs/figures/error_histogram.png', dpi=150)
    plt.close()

    # Plot 4: Per-sample RelL2 CDF
    per_sample_rl2 = np.array([
        np.linalg.norm(preds_raw[i] - Y_test_raw[i]) / (np.linalg.norm(Y_test_raw[i]) + 1e-8)
        for i in range(len(Y_test_raw))
    ])
    sorted_rl2 = np.sort(per_sample_rl2)
    cdf = np.arange(1, len(sorted_rl2) + 1) / len(sorted_rl2)
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_rl2, cdf, color='steelblue', lw=2)
    plt.axvline(np.median(per_sample_rl2), color='r', ls='--', lw=1.5,
                label=f'Median={np.median(per_sample_rl2):.4f}')
    plt.xlabel('Per-sample Relative L2 Error')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution of Per-sample Rel-L2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/rel_l2_cdf.png', dpi=150)
    plt.close()

    # Plot 5: Lambda recovery scatter and error histogram
    if len(actual_lams) > 0:
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_lams, recovered_lams, alpha=0.35, s=8, c='darkorange')
        lmin = min(min(actual_lams), min(recovered_lams))
        lmax = max(max(actual_lams), max(recovered_lams))
        plt.plot([lmin, lmax], [lmin, lmax], 'r--', lw=1.5)
        plt.xlabel('True λ')
        plt.ylabel('Recovered λ')
        plt.title(
            f"Lambda Recovery  R²={inv_metrics['R2']:.4f}  "
            f"MAE={inv_metrics['MAE']:.4f}  RelL2={inv_metrics['RelL2']:.4f}"
        )
        plt.tight_layout()
        plt.savefig('outputs/figures/lambda_scatter.png', dpi=150)
        plt.close()

        lam_errors = np.array(recovered_lams) - np.array(actual_lams)
        plt.figure(figsize=(7, 4))
        plt.hist(lam_errors, bins=80, color='darkorange', edgecolor='none', alpha=0.8)
        plt.axvline(0, color='r', ls='--', lw=1.5)
        plt.xlabel('Lambda Error (Recovered - True)')
        plt.ylabel('Count')
        plt.title(f"Lambda Recovery Error  MAE={inv_metrics['MAE']:.4f}")
        plt.tight_layout()
        plt.savefig('outputs/figures/lambda_error_histogram.png', dpi=150)
        plt.close()

    # Plot 6: Best and worst surface heatmaps
    for label, idxs in [('best',  np.argsort(per_sample_rl2)[:3]),
                         ('worst', np.argsort(per_sample_rl2)[::-1][:3])]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        for row_i, i in enumerate(idxs):
            lam_v = float(X_test_raw[i, 0])
            rel   = per_sample_rl2[i]
            for col, (arr, ttl, cmap) in enumerate([
                (Y_test_raw[i],                          f"GT  λ={lam_v:.3f}", 'viridis'),
                (preds_raw[i],                           f"Pred  relL2={rel:.4f}", 'viridis'),
                (np.abs(Y_test_raw[i] - preds_raw[i]),  "Abs Error", 'magma'),
            ]):
                im = axes[row_i, col].imshow(
                    arr, aspect='auto', origin='lower', cmap=cmap,
                    extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]]
                )
                axes[row_i, col].set_title(ttl, fontsize=9)
                axes[row_i, col].set_xlabel('t')
                axes[row_i, col].set_ylabel('x')
                plt.colorbar(im, ax=axes[row_i, col])
        fig.suptitle(f"Heat Equation Surfaces — {label.capitalize()} 3 by Rel-L2", fontsize=13)
        fig.tight_layout()
        fig.savefig(f'outputs/figures/surfaces_{label}.png', dpi=120)
        plt.close(fig)

    # Plot 7: Time slice line plots at t=0.1, 0.5, 1.0
    slice_targets = [0.1, 0.5, 1.0]
    slice_indices = [int(np.argmin(np.abs(t_grid - t))) for t in slice_targets]
    for label, idxs in [('best',  np.argsort(per_sample_rl2)[:3]),
                         ('worst', np.argsort(per_sample_rl2)[::-1][:3])]:
        fig, axes = plt.subplots(3, len(slice_indices), figsize=(14, 10))
        for row_i, i in enumerate(idxs):
            lam_v = float(X_test_raw[i, 0])
            for col, (t_idx, t_val) in enumerate(zip(slice_indices, slice_targets)):
                ax = axes[row_i, col]
                ax.plot(x_grid, Y_test_raw[i, :, t_idx], 'k-', lw=2, label='GT')
                ax.plot(x_grid, preds_raw[i, :, t_idx],  '--', lw=2,
                        color='#DD8452', label='Pred')
                ax.fill_between(x_grid, Y_test_raw[i, :, t_idx],
                                preds_raw[i, :, t_idx], alpha=0.2, color='red')
                if row_i == 0:
                    ax.set_title(f"t = {t_val:.2f}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"λ={lam_v:.3f}", fontsize=8)
                ax.set_xlabel('x')
                if row_i == 0 and col == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"u(x,t) Slices — {label.capitalize()} 3", fontsize=13)
        fig.tight_layout()
        fig.savefig(f'outputs/figures/time_slices_{label}.png', dpi=120)
        plt.close(fig)

    print("\nEvaluation complete. Outputs:")
    print("  outputs/metrics/forward_metrics.csv")
    print("  outputs/metrics/inverse_metrics.csv")
    print("  outputs/figures/loss_curves.png")
    print("  outputs/figures/forward_scatter.png")
    print("  outputs/figures/error_histogram.png")
    print("  outputs/figures/rel_l2_cdf.png")
    print("  outputs/figures/lambda_scatter.png")
    print("  outputs/figures/lambda_error_histogram.png")
    print("  outputs/figures/surfaces_best.png / surfaces_worst.png")
    print("  outputs/figures/time_slices_best.png / time_slices_worst.png")


if __name__ == '__main__':
    run_evaluation()
