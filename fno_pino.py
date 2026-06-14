import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

# ---------------------------------------------------------------------------
# PINO2d — uses neuralop FNO backbone for:
#   - Layer normalization between spectral blocks
#   - Both low and high frequency Fourier weights
#   - MLP projection layers inside each block
#   - Well-tested spectral convolution implementation
#
# Input channels fed to FNO: [K_n, r_n, sigma_n, S_n, T_n] — all in [0,1]
# Output: V_norm = V / V_scale, approximately in [0,1]
# ---------------------------------------------------------------------------

class PINO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=128, num_layers=4):
        super().__init__()
        self.width = width

        # neuralop FNO with 5 input channels (K, r, sigma, S, T)
        # all normalised to [0,1]
        self.fno = FNO(
            n_modes=(modes1, modes2),
            in_channels=5,
            out_channels=1,
            hidden_channels=width,
            n_layers=num_layers,
            positional_embedding=None,
        )

    def forward(self, params_n, S_grid_n, T_grid_n):
        # params_n:  (B, 3) normalised [K_n, r_n, sigma_n]
        # S_grid_n:  (Nx, Nt) or (B, Nx, Nt) normalised S in [0,1]
        # T_grid_n:  same shape, normalised T in [0,1]
        B = params_n.shape[0]
        K_n     = params_n[:, 0].view(B, 1, 1)
        r_n     = params_n[:, 1].view(B, 1, 1)
        sigma_n = params_n[:, 2].view(B, 1, 1)

        S_n = S_grid_n.unsqueeze(0).expand(B, -1, -1) if S_grid_n.dim() == 2 else S_grid_n
        T_n = T_grid_n.unsqueeze(0).expand(B, -1, -1) if T_grid_n.dim() == 2 else T_grid_n

        K_e     = K_n.expand(-1, S_n.shape[1], S_n.shape[2])
        r_e     = r_n.expand_as(K_e)
        sigma_e = sigma_n.expand_as(K_e)

        # Stack into (B, Nx, Nt, 5) then permute to (B, 5, Nx, Nt) for FNO
        x = torch.stack([K_e, r_e, sigma_e, S_n, T_n], dim=1)  # (B, 5, Nx, Nt)

        # FNO forward: (B, 5, Nx, Nt) -> (B, 1, Nx, Nt)
        out = self.fno(x)           # (B, 1, Nx, Nt)
        return out.squeeze(1)       # (B, Nx, Nt)


# ---------------------------------------------------------------------------
# 4th-order FD stencils — unchanged, operate on normalised V
# ---------------------------------------------------------------------------

def _fd4_d2_dx(u, dx):
    B, Nx, Nt = u.shape
    u_p = u.permute(0, 2, 1).reshape(B * Nt, 1, Nx)
    kernel = torch.tensor(
        [-1/12, 4/3, -5/2, 4/3, -1/12], dtype=u.dtype, device=u.device
    ).view(1, 1, 5) / (dx ** 2)
    out = F.conv1d(u_p, kernel).squeeze(1)
    return out.view(B, Nt, Nx - 4).permute(0, 2, 1)


def _fd4_d1_dx(u, dx):
    B, Nx, Nt = u.shape
    u_p = u.permute(0, 2, 1).reshape(B * Nt, 1, Nx)
    kernel = torch.tensor(
        [1/12, -2/3, 0.0, 2/3, -1/12], dtype=u.dtype, device=u.device
    ).view(1, 1, 5) / dx
    out = F.conv1d(u_p, kernel).squeeze(1)
    return out.view(B, Nt, Nx - 4).permute(0, 2, 1)


def _fd2_d1_dt(u, dt):
    B, Nx, Nt = u.shape
    u_p = u.reshape(B * Nx, 1, Nt)
    kernel = torch.tensor(
        [-0.5, 0.0, 0.5], dtype=u.dtype, device=u.device
    ).view(1, 1, 3) / dt
    out = F.conv1d(u_p, kernel).squeeze(1)
    return out.view(B, Nx, Nt - 2)


def _pde_residual(V_norm, params_raw, S_grid_1d_raw, T_grid_1d_raw):
    dx = (S_grid_1d_raw[1] - S_grid_1d_raw[0]).item()
    dt = (T_grid_1d_raw[1] - T_grid_1d_raw[0]).item()

    V_SS = _fd4_d2_dx(V_norm, dx)
    V_S  = _fd4_d1_dx(V_norm, dx)
    V_T  = _fd2_d1_dt(V_norm, dt)

    V_SS_v = V_SS[:, :, 1:-1]
    V_S_v  = V_S[:, :,  1:-1]
    V_T_v  = V_T[:, 2:-2, :]
    V_v    = V_norm[:, 2:-2, 1:-1]

    sigma = params_raw[:, 2].view(-1, 1, 1)
    r     = params_raw[:, 1].view(-1, 1, 1)
    S_v   = S_grid_1d_raw[2:-2].view(1, -1, 1)

    # BS PDE: dV/dT - 0.5*sigma^2*S^2*d2V/dS2 - r*S*dV/dS + r*V = 0
    residual = V_T_v - 0.5 * sigma**2 * S_v**2 * V_SS_v - r * S_v * V_S_v + r * V_v
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Full physics loss — all terms on normalised scale, fully unchanged
# ---------------------------------------------------------------------------

def compute_pino_loss(V_norm, params_raw, S_grid_1d_raw, T_grid_1d_raw, V_scale,
                      lambda_pde=1.0, lambda_ic=10.0, lambda_bc=10.0):
    B = params_raw.shape[0]

    l_pde = _pde_residual(V_norm, params_raw, S_grid_1d_raw, T_grid_1d_raw)

    # IC: V_norm(S, T=0) = max(S - K, 0) / V_scale
    V_ic  = V_norm[:, :, 0]
    S_ic  = S_grid_1d_raw.unsqueeze(0).expand(B, -1)
    K_ic  = params_raw[:, 0].view(B, 1).expand_as(V_ic)
    payoff_norm = torch.clamp(S_ic - K_ic, min=0.0) / V_scale
    l_ic  = F.mse_loss(V_ic, payoff_norm)

    # BC lower: V_norm(S=0, T) = 0
    l_bc_low = torch.mean(V_norm[:, 0, :] ** 2)

    # BC upper: V_norm(S_max, T) = (S_max - K*exp(-r*T)) / V_scale
    V_high  = V_norm[:, -1, :]
    S_max   = S_grid_1d_raw[-1].item()
    T_bc    = T_grid_1d_raw.unsqueeze(0).expand(B, -1)
    K_bc    = params_raw[:, 0].view(B, 1).expand_as(V_high)
    r_bc    = params_raw[:, 1].view(B, 1).expand_as(V_high)
    target_high_norm = (S_max - K_bc * torch.exp(-r_bc * T_bc)) / V_scale
    l_bc_high = F.mse_loss(V_high, target_high_norm)

    l_bc  = l_bc_low + l_bc_high
    total = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
    return total, l_pde, l_ic, l_bc


# ---------------------------------------------------------------------------
# Relative L2 loss — scale-invariant, standard in operator learning
# ---------------------------------------------------------------------------

def relative_l2_loss(pred, target):
    B = pred.shape[0]
    diff  = torch.norm(pred.view(B, -1) - target.view(B, -1), p=2, dim=1)
    denom = torch.norm(target.view(B, -1), p=2, dim=1) + 1e-8
    return (diff / denom).mean()
