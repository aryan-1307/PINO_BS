import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from fno_pino import PINO1d
from implied_vol import implied_volatility


def compute_extended_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def run_evaluation():

    if not os.path.exists('data/generated/bs_dataset.csv'):
        print("Evaluation canceled: data/generated/bs_dataset.csv not found.")
        return

    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    df = pd.read_csv('data/generated/bs_dataset.csv')

    split = int(0.8 * len(df))
    test_df = df.iloc[split:].copy()

    fwd_model = PINO1d()

    if os.path.exists('outputs/fwd_model.pth'):
        fwd_model.load_state_dict(torch.load('outputs/fwd_model.pth'))

    fwd_model.eval()

    X_min_fwd = np.array(
        [50.0, 50.0, 0.01, 0.00, 0.05],
        dtype=np.float32
    )

    X_max_fwd = np.array(
        [1000.0, 1000.0, 3.00, 0.10, 1.00],
        dtype=np.float32
    )

    X_fwd_raw = test_df[
        ['S', 'K', 'T', 'r', 'sigma']
    ].values.astype(np.float32)

    X_fwd_norm = (
        X_fwd_raw - X_min_fwd
    ) / (
        X_max_fwd - X_min_fwd
    )

    fwd_inputs = torch.tensor(
        X_fwd_norm
    ).unsqueeze(1)

    with torch.no_grad():
        fwd_preds = fwd_model(
            fwd_inputs
        )[:, 0, 0].numpy()

    inv_preds = []

    for _, row in test_df.iterrows():

        iv = implied_volatility(
            row['S'],
            row['K'],
            row['T'],
            row['r'],
            row['pred_price'] if 'pred_price' in row else fwd_preds[len(inv_preds)]
        )

        inv_preds.append(iv)

    inv_preds = np.array(inv_preds)

    test_df['pred_price'] = fwd_preds
    test_df['pred_sigma'] = inv_preds

    test_df[
        ['S', 'K', 'T', 'r', 'sigma', 'price', 'pred_price']
    ].to_csv(
        'outputs/predictions/forward_predictions.csv',
        index=False
    )

    test_df[
        ['S', 'K', 'T', 'r', 'price', 'sigma', 'pred_sigma']
    ].to_csv(
        'outputs/predictions/inverse_predictions.csv',
        index=False
    )

    fwd_metrics = compute_extended_metrics(
        test_df['price'].values,
        fwd_preds
    )

    valid_mask = ~np.isnan(inv_preds)

    inv_metrics = compute_extended_metrics(
        test_df['sigma'].values[valid_mask],
        inv_preds[valid_mask]
    )

    pd.DataFrame(
        list(fwd_metrics.items()),
        columns=['Metric', 'Value']
    ).to_csv(
        'outputs/metrics/forward_metrics.csv',
        index=False
    )

    pd.DataFrame(
        list(inv_metrics.items()),
        columns=['Metric', 'Value']
    ).to_csv(
        'outputs/metrics/inverse_metrics.csv',
        index=False
    )

    if os.path.exists('outputs/fwd_history.csv'):

        fwd_hist = pd.read_csv(
            'outputs/fwd_history.csv'
        )

        plt.figure()
        plt.plot(
            fwd_hist['epoch'],
            fwd_hist['train_loss'],
            label='Train Loss'
        )

        plt.plot(
            fwd_hist['epoch'],
            fwd_hist['val_loss'],
            label='Val Loss'
        )

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Forward PINO Loss Curves')
        plt.legend()

        plt.savefig(
            'outputs/figures/loss_curve_forward.png'
        )

        plt.close()

    plt.figure()

    plt.scatter(
        test_df['price'],
        test_df['pred_price'],
        alpha=0.4,
        s=8
    )

    plt.plot(
        [test_df['price'].min(), test_df['price'].max()],
        [test_df['price'].min(), test_df['price'].max()],
        'r--'
    )

    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    plt.title(
        f"Forward PINO Model Validation (R²: {fwd_metrics['R2']:.4f})"
    )

    plt.savefig(
        'outputs/figures/price_scatter.png'
    )

    plt.close()

    plt.figure()

    plt.scatter(
        test_df['sigma'],
        test_df['pred_sigma'],
        alpha=0.4,
        s=8
    )

    plt.plot(
        [test_df['sigma'].min(), test_df['sigma'].max()],
        [test_df['sigma'].min(), test_df['sigma'].max()],
        'r--'
    )

    plt.xlabel('Actual Implied Vol')
    plt.ylabel('Recovered Implied Vol')

    plt.title(
        f"Implied Volatility Recovery (R²: {inv_metrics['R2']:.4f})"
    )

    plt.savefig(
        'outputs/figures/volatility_scatter.png'
    )

    plt.close()

    if os.path.exists('data/market/market_data.csv'):

        m_df = pd.read_csv(
            'data/market/market_data.csv'
        )

        m_fwd_raw = m_df[
            ['S', 'K', 'T', 'r', 'proxy_sigma']
        ].values.astype(np.float32)

        m_fwd_norm = (
            m_fwd_raw - X_min_fwd
        ) / (
            X_max_fwd - X_min_fwd
        )

        m_fwd_in = torch.tensor(
            m_fwd_norm
        ).unsqueeze(1)

        with torch.no_grad():
            m_df['pred_price'] = fwd_model(
                m_fwd_in
            )[:, 0, 0].numpy()

        market_iv = []

        for _, row in m_df.iterrows():

            iv = implied_volatility(
                row['S'],
                row['K'],
                row['T'],
                row['r'],
                row['pred_price']
            )

            market_iv.append(iv)

        m_df['pred_sigma'] = market_iv

        m_df.to_csv(
            'outputs/predictions/market_predictions.csv',
            index=False
        )

        print(
            "Market snapshot successfully evaluated and saved to outputs/predictions/."
        )

    print(
        "Project evaluation complete. Metrics and figures exported to subdirectories."
    )


if __name__ == '__main__':
    run_evaluation()