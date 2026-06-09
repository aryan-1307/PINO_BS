import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from fno_pino import PINO2d
from implied_vol import implied_volatility


def compute_extended_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def run_evaluation():
    if not os.path.exists('data/generated/surface_inputs.npy') or \
       not os.path.exists('data/generated/surface_outputs.npy'):
        print("Evaluation canceled: surface dataset arrays are missing.")
        return

    if not os.path.exists('outputs/fwd_model.pth'):
        print("Evaluation canceled: trained model weights not found at outputs/fwd_model.pth. Run train_fwd.py first.")
        return

    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    X_data = np.load('data/generated/surface_inputs.npy')
    Y_data = np.load('data/generated/surface_outputs.npy')

    split = int(0.8 * len(X_data))
    X_test, Y_test = X_data[split:], Y_data[split:]

    grid_size = 64
    # Must match generate_data.py and train_fwd.py exactly
    s_space = np.linspace(1.0, 1000.0, grid_size)
    t_space = np.linspace(0.0, 3.0, grid_size)
    S_mesh, T_mesh = np.meshgrid(s_space, t_space, indexing='ij')

    S_grid = torch.tensor(S_mesh, dtype=torch.float32).to(device)
    T_grid = torch.tensor(T_mesh, dtype=torch.float32).to(device)

    fwd_model = PINO2d().to(device)
    fwd_model.load_state_dict(torch.load('outputs/fwd_model.pth', map_location=device))
    fwd_model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        fwd_preds = fwd_model(X_tensor, S_grid, T_grid).cpu().numpy()

    # 1. Surface-level forward operator metrics
    fwd_metrics = compute_extended_metrics(Y_test, fwd_preds)
    pd.DataFrame(
        list(fwd_metrics.items()), columns=['Metric', 'Value']
    ).to_csv('outputs/metrics/forward_metrics.csv', index=False)
    print(f"Forward metrics: {fwd_metrics}")

    # 2. Implied volatility recovery on synthetic test surfaces
    print("Evaluating implied volatility recovery...")
    actual_vols = []
    recovered_vols = []
    np.random.seed(42)

    for i in range(len(X_test)):
        K_val, r_val, true_sigma = X_test[i]
        for _ in range(5):
            s_idx = np.random.randint(0, grid_size)
            t_idx = np.random.randint(1, grid_size)   # skip T=0 (payoff, not a tradeable price)
            S_val = s_space[s_idx]
            T_val = t_space[t_idx]
            pred_price = fwd_preds[i, s_idx, t_idx]
            iv = implied_volatility(S_val, K_val, T_val, r_val, pred_price)
            if not np.isnan(iv):
                actual_vols.append(true_sigma)
                recovered_vols.append(iv)

    inv_metrics = compute_extended_metrics(np.array(actual_vols), np.array(recovered_vols))
    pd.DataFrame(
        list(inv_metrics.items()), columns=['Metric', 'Value']
    ).to_csv('outputs/metrics/inverse_metrics.csv', index=False)
    print(f"Inverse (IV recovery) metrics: {inv_metrics}")

    # 3. Market snapshot evaluation
    if os.path.exists('data/market/market_data.csv'):
        print("Mapping market snapshots onto trained operator...")
        m_df = pd.read_csv('data/market/market_data.csv')
        market_preds = []
        market_ivs = []

        for _, row in m_df.iterrows():
            # Forward model input: [K, r, proxy_sigma]
            # proxy_sigma here acts as a volatility seed/context for the operator;
            # the model was trained on true sigma so this is an approximation in practice.
            m_input = torch.tensor(
                [[row['K'], row['r'], row['proxy_sigma']]],
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                pred_surface = fwd_model(m_input, S_grid, T_grid).squeeze(0).cpu().numpy()

            # Bilinear interpolation to read off the specific (S, T) contract coordinate
            # fill_value=None returns NaN for any out-of-bounds query point safely
            interp = RegularGridInterpolator(
                (s_space, t_space), pred_surface,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            pred_price = float(interp([[row['S'], row['T']]])[0])

            if np.isnan(pred_price) or pred_price <= 0:
                market_preds.append(np.nan)
                market_ivs.append(np.nan)
                continue

            market_preds.append(pred_price)
            iv = implied_volatility(row['S'], row['K'], row['T'], row['r'], pred_price)
            market_ivs.append(iv)

        m_df['pred_price'] = market_preds
        m_df['pred_sigma'] = market_ivs
        m_df.to_csv('outputs/predictions/market_predictions.csv', index=False)
        print("Market predictions saved to outputs/predictions/market_predictions.csv")

    # --- Plots ---
    if os.path.exists('outputs/fwd_history.csv'):
        fwd_hist = pd.read_csv('outputs/fwd_history.csv')
        plt.figure()
        plt.plot(fwd_hist['epoch'], fwd_hist['train_loss'], label='Train Loss')
        plt.plot(fwd_hist['epoch'], fwd_hist['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Forward PINO Loss Curves')
        plt.legend()
        plt.savefig('outputs/figures/loss_curve_forward.png')
        plt.close()

    plt.figure()
    plt.scatter(Y_test.flatten(), fwd_preds.flatten(), alpha=0.1, s=1)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f"Forward PINO Surface Validation (R²: {fwd_metrics['R2']:.4f})")
    plt.savefig('outputs/figures/price_scatter.png')
    plt.close()

    if len(actual_vols) > 0:
        plt.figure()
        plt.scatter(actual_vols, recovered_vols, alpha=0.5, s=10)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Vol')
        plt.ylabel('Recovered IV')
        plt.title(f"Implied Volatility Recovery (R²: {inv_metrics['R2']:.4f})")
        plt.savefig('outputs/figures/volatility_scatter.png')
        plt.close()

    print("Evaluation complete.")


if __name__ == '__main__':
    run_evaluation()