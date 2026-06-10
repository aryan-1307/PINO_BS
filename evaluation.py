import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

from fno_pino import PINO2d
from implied_vol import implied_volatility


def compute_metrics(actual, predicted):
    a = actual.flatten()
    p = predicted.flatten()
    mse    = float(np.mean((a - p) ** 2))
    rmse   = float(np.sqrt(mse))
    mae    = float(np.mean(np.abs(a - p)))
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    r2     = 1.0 - ss_res / (ss_tot + 1e-8)
    rel_l2 = float(np.linalg.norm(a - p) / (np.linalg.norm(a) + 1e-8))
    mape   = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-8)))) * 100.0
    max_err = float(np.max(np.abs(a - p)))
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
        print("Evaluation canceled: no trained weights found. Run train_fwd.py first.")
        return

    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/metrics',     exist_ok=True)
    os.makedirs('outputs/figures',     exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on device: {device}  |  weights: {weights_path}")

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

    S_norm = (S_grid - S_grid.min()) / (S_grid.max() - S_grid.min())
    T_norm = (T_grid - T_grid.min()) / (T_grid.max() - T_grid.min())
    S_mesh_n, T_mesh_n = np.meshgrid(S_norm, T_norm, indexing='ij')
    S_grid_2d_n = torch.tensor(S_mesh_n, dtype=torch.float32).to(device)
    T_grid_2d_n = torch.tensor(T_mesh_n, dtype=torch.float32).to(device)

    split      = int(0.8 * len(X_raw))
    X_test_n   = X_norm[split:]
    X_test_raw = X_raw[split:]
    Y_test_raw = Y_raw[split:]
    Y_test_n   = Y_norm[split:]
    grid_size  = len(S_grid)

    fwd_model = PINO2d().to(device)
    fwd_model.load_state_dict(torch.load(weights_path, map_location=device))
    fwd_model.eval()

    batch_size = 32
    preds_norm = []
    X_t = torch.tensor(X_test_n, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            b = X_t[i:i+batch_size].to(device)
            preds_norm.append(fwd_model(b, S_grid_2d_n, T_grid_2d_n).cpu().numpy())
    preds_norm = np.concatenate(preds_norm, axis=0)
    preds_raw  = preds_norm * V_scale

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
    # 2. Implied volatility recovery
    # ------------------------------------------------------------------
    print("Evaluating implied volatility recovery...")
    actual_vols, recovered_vols = [], []
    np.random.seed(42)
    for i in range(len(X_test_raw)):
        K_val, r_val, true_sigma = X_test_raw[i]
        for _ in range(5):
            s_idx = np.random.randint(0, grid_size)
            t_idx = np.random.randint(1, grid_size)
            iv = implied_volatility(
                float(S_grid[s_idx]), K_val,
                float(T_grid[t_idx]), r_val,
                float(preds_raw[i, s_idx, t_idx])
            )
            if not np.isnan(iv):
                actual_vols.append(float(true_sigma))
                recovered_vols.append(iv)

    inv_metrics = {}
    if len(actual_vols) > 0:
        inv_metrics = compute_metrics(np.array(actual_vols), np.array(recovered_vols))
        pd.DataFrame(list(inv_metrics.items()), columns=['Metric', 'Value']).to_csv(
            'outputs/metrics/inverse_metrics.csv', index=False
        )
        print(f"IV recovery metrics: {inv_metrics}")
    else:
        print("Warning: no valid IV recovery samples found.")

    # ------------------------------------------------------------------
    # 3. Market snapshot evaluation
    # ------------------------------------------------------------------
    if os.path.exists('data/market/market_data.csv'):
        print("Evaluating on market data...")
        m_df = pd.read_csv('data/market/market_data.csv')
        market_preds, market_ivs = [], []
        for _, row in m_df.iterrows():
            raw_p   = np.array([[row['K'], row['r'], row['proxy_sigma']]], dtype=np.float32)
            norm_p  = (raw_p - param_min) / (param_max - param_min + 1e-8)
            m_input = torch.tensor(norm_p, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_norm_s = fwd_model(m_input, S_grid_2d_n, T_grid_2d_n).squeeze(0).cpu().numpy()
            pred_surface = pred_norm_s * V_scale
            interp = RegularGridInterpolator(
                (S_grid, T_grid), pred_surface,
                method='linear', bounds_error=False, fill_value=None
            )
            pred_price = float(interp([[row['S'], row['T']]])[0])
            if np.isnan(pred_price) or pred_price <= 0:
                market_preds.append(np.nan)
                market_ivs.append(np.nan)
                continue
            market_preds.append(pred_price)
            market_ivs.append(
                implied_volatility(row['S'], row['K'], row['T'], row['r'], pred_price)
            )
        m_df['pred_price'] = market_preds
        m_df['pred_sigma'] = market_ivs
        m_df.to_csv('outputs/predictions/market_predictions.csv', index=False)
        print("Market predictions saved.")

    # ==================================================================
    # PLOTS
    # ==================================================================

    # ------------------------------------------------------------------
    # Plot 1: Training loss curves — supervised + all physics components
    # ------------------------------------------------------------------
    if os.path.exists('outputs/fwd_history.csv'):
        hist = pd.read_csv('outputs/fwd_history.csv')
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].semilogy(hist['epoch'], hist['data_loss'], label='Train Rel-L2')
        axes[0].semilogy(hist['epoch'], hist['val_loss'],  label='Val Rel-L2', ls='--')
        axes[0].set_xlabel('Epoch')
        axes[0].set_title('Supervised Loss (Relative L2)')
        axes[0].legend()

        axes[1].semilogy(hist['epoch'], hist['pde_loss'], label='PDE residual')
        axes[1].semilogy(hist['epoch'], hist['ic_loss'],  label='IC / Terminal BC', ls='--')
        axes[1].semilogy(hist['epoch'], hist['bc_loss'],  label='Spatial BC', ls=':')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Physics Loss Components (unweighted)')
        axes[1].legend()

        fig.tight_layout()
        fig.savefig('outputs/figures/loss_curves.png', dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Price scatter — actual vs predicted (physical prices)
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test_raw.flatten(), preds_raw.flatten(), alpha=0.05, s=1, c='steelblue')
    vmin = min(float(Y_test_raw.min()), float(preds_raw.min()))
    vmax = max(float(Y_test_raw.max()), float(preds_raw.max()))
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f"Forward PINO  R²={fwd_metrics['R2']:.4f}  RelL2={fwd_metrics['RelL2']:.4f}")
    plt.tight_layout()
    plt.savefig('outputs/figures/price_scatter.png', dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Residual / error distribution histogram
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Plot 4: Per-sample relative L2 distribution (CDF)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Plot 5: IV scatter — true sigma vs recovered IV
    # ------------------------------------------------------------------
    if len(actual_vols) > 0:
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_vols, recovered_vols, alpha=0.35, s=8, c='darkorange')
        plt.plot([0, 1], [0, 1], 'r--', lw=1.5)
        plt.xlabel('True σ')
        plt.ylabel('Recovered IV')
        plt.title(
            f"IV Recovery  R²={inv_metrics['R2']:.4f}  "
            f"MAE={inv_metrics['MAE']:.4f}  RelL2={inv_metrics['RelL2']:.4f}"
        )
        plt.tight_layout()
        plt.savefig('outputs/figures/volatility_scatter.png', dpi=150)
        plt.close()

        # IV error histogram
        iv_errors = np.array(recovered_vols) - np.array(actual_vols)
        plt.figure(figsize=(7, 4))
        plt.hist(iv_errors, bins=80, color='darkorange', edgecolor='none', alpha=0.8)
        plt.axvline(0, color='r', ls='--', lw=1.5)
        plt.xlabel('IV Error (Recovered - True)')
        plt.ylabel('Count')
        plt.title(f"IV Recovery Error Distribution  MAE={inv_metrics['MAE']:.4f}")
        plt.tight_layout()
        plt.savefig('outputs/figures/iv_error_histogram.png', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Plot 6: Best and worst surface heatmaps (GT | Pred | Abs Error)
    # ------------------------------------------------------------------
    for label, idxs in [('best',  np.argsort(per_sample_rl2)[:3]),
                         ('worst', np.argsort(per_sample_rl2)[::-1][:3])]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        for row_i, i in enumerate(idxs):
            K_v, r_v, sig_v = X_test_raw[i]
            rel = per_sample_rl2[i]
            for col, (arr, ttl, cmap) in enumerate([
                (Y_test_raw[i],                          f"GT  K={K_v:.0f} σ={sig_v:.2f} r={r_v:.3f}", 'viridis'),
                (preds_raw[i],                           f"Pred  relL2={rel:.4f}",                       'viridis'),
                (np.abs(Y_test_raw[i] - preds_raw[i]),  "Abs Error",                                     'magma'),
            ]):
                im = axes[row_i, col].imshow(
                    arr, aspect='auto', origin='lower', cmap=cmap,
                    extent=[T_grid[0], T_grid[-1], S_grid[0], S_grid[-1]]
                )
                axes[row_i, col].set_title(ttl, fontsize=9)
                axes[row_i, col].set_xlabel('T (years)')
                axes[row_i, col].set_ylabel('S')
                plt.colorbar(im, ax=axes[row_i, col])
        fig.suptitle(f"Surface Predictions — {label.capitalize()} 3 by Rel-L2", fontsize=13)
        fig.tight_layout()
        fig.savefig(f'outputs/figures/surfaces_{label}.png', dpi=120)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 7: Time-slice line plots — GT vs Pred at T=0.25, 1.0, 2.0 yrs
    # ------------------------------------------------------------------
    slice_targets = [0.25, 1.0, 2.0]
    slice_indices = [int(np.argmin(np.abs(T_grid - t))) for t in slice_targets]

    for label, idxs in [('best',  np.argsort(per_sample_rl2)[:3]),
                         ('worst', np.argsort(per_sample_rl2)[::-1][:3])]:
        fig, axes = plt.subplots(3, len(slice_indices), figsize=(14, 10))
        for row_i, i in enumerate(idxs):
            K_v, r_v, sig_v = X_test_raw[i]
            for col, (t_idx, t_val) in enumerate(zip(slice_indices, slice_targets)):
                ax = axes[row_i, col]
                ax.plot(S_grid, Y_test_raw[i, :, t_idx], 'k-',  lw=2,   label='GT')
                ax.plot(S_grid, preds_raw[i, :, t_idx],  '--',  lw=2,
                        color='#DD8452', label='Pred')
                ax.fill_between(
                    S_grid,
                    Y_test_raw[i, :, t_idx],
                    preds_raw[i, :, t_idx],
                    alpha=0.2, color='red'
                )
                if row_i == 0:
                    ax.set_title(f"T = {t_val:.2f} yr", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"K={K_v:.0f}\nσ={sig_v:.2f}", fontsize=8)
                ax.set_xlabel('S')
                if row_i == 0 and col == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"Price Slices at Fixed T — {label.capitalize()} 3", fontsize=13)
        fig.tight_layout()
        fig.savefig(f'outputs/figures/time_slices_{label}.png', dpi=120)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 8: Market IV smile — predicted sigma across strikes (if available)
    # ------------------------------------------------------------------
    if os.path.exists('outputs/predictions/market_predictions.csv'):
        mp = pd.read_csv('outputs/predictions/market_predictions.csv')
        mp_clean = mp.dropna(subset=['pred_sigma'])
        if len(mp_clean) > 0:
            unique_T = sorted(mp_clean['T'].unique())[:3]
            fig, axes = plt.subplots(1, len(unique_T), figsize=(5 * len(unique_T), 4),
                                     sharey=True)
            if len(unique_T) == 1:
                axes = [axes]
            for ax, t_val in zip(axes, unique_T):
                slice_df = mp_clean[np.abs(mp_clean['T'] - t_val) < 0.02]
                slice_df = slice_df.sort_values('K')
                ax.plot(slice_df['K'], slice_df['pred_sigma'], 'o-',
                        color='steelblue', lw=1.5, ms=4, label='PINO IV')
                ax.set_xlabel('Strike K')
                ax.set_ylabel('Implied Volatility')
                ax.set_title(f"IV Smile  T≈{t_val:.2f} yr")
                ax.legend()
            fig.suptitle("Market IV Smile (PINO Predictions)", fontsize=12)
            fig.tight_layout()
            fig.savefig('outputs/figures/market_iv_smile.png', dpi=150)
            plt.close(fig)

    print("\nEvaluation complete. Outputs:")
    print("  outputs/metrics/forward_metrics.csv")
    print("  outputs/metrics/inverse_metrics.csv")
    print("  outputs/figures/loss_curves.png")
    print("  outputs/figures/price_scatter.png")
    print("  outputs/figures/error_histogram.png")
    print("  outputs/figures/rel_l2_cdf.png")
    print("  outputs/figures/volatility_scatter.png")
    print("  outputs/figures/iv_error_histogram.png")
    print("  outputs/figures/surfaces_best.png  /  surfaces_worst.png")
    print("  outputs/figures/time_slices_best.png  /  time_slices_worst.png")
    print("  outputs/figures/market_iv_smile.png  (if market data present)")


if __name__ == '__main__':
    run_evaluation()
