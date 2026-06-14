import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO


# ---------------------------------------------------------------------------
# PINO for Heat Equation — uses neuralop FNO backbone
# Identical structure to Black-Scholes PINO2d
#
# Forward problem: lambda -> u(x,t)
# Input channels: [lambda_n, x_n, t_n] — all normalised to [0,1]
# Output: u(x,t) — already in [-1,1], V_scale=1.0
# ---------------------------------------------------------------------------

class HeatPINO(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, num_layers=4):
        super().__init__()
        self.width = width
        # 3 input channels: normalised lambda, x, t
        self.fno = FNO(
            n_modes=(modes1, modes2),
            in_channels=3,
            out_channels=1,
            hidden_channels=width,
            n_layers=num_layers,
            positional_embedding=None,
        )

    def forward(self, params_n, S_grid_n, T_grid_n):
        # params_n:  (B, 1) normalised lambda
        # S_grid_n:  (Nx, Nt) or (B, Nx, Nt) normalised x in [0,1]
        # T_grid_n:  same shape, normalised t in [0,1]
        B = params_n.shape[0]

        x_n = S_grid_n.unsqueeze(0).expand(B, -1, -1) if S_grid_n.dim() == 2 else S_grid_n
        t_n = T_grid_n.unsqueeze(0).expand(B, -1, -1) if T_grid_n.dim() == 2 else T_grid_n

        lam_e = params_n.view(B, 1, 1).expand(-1, x_n.shape[1], x_n.shape[2])

        # Stack into (B, 3, Nx, Nt) for FNO
        inp = torch.stack([lam_e, x_n, t_n], dim=1)   # (B, 3, Nx, Nt)

        out = self.fno(inp)        # (B, 1, Nx, Nt)
        return out.squeeze(1)      # (B, Nx, Nt)


# ---------------------------------------------------------------------------
# 4th-order FD stencils — identical to Black-Scholes implementation
# ---------------------------------------------------------------------------

def _fd4_d2_dx(u, dx):
    B, Nx, Nt = u.shape
    u_p = u.permute(0, 2, 1).reshape(B * Nt, 1, Nx)
    kernel = torch.tensor(
        [-1/12, 4/3, -5/2, 4/3, -1/12], dtype=u.dtype, device=u.device
    ).view(1, 1, 5) / (dx ** 2)
    out = F.conv1d(u_p, kernel).squeeze(1)
    return out.view(B, Nt, Nx - 4).permute(0, 2, 1)   # (B, Nx-4, Nt)


def _fd2_d1_dt(u, dt):
    B, Nx, Nt = u.shape
    u_p = u.reshape(B * Nx, 1, Nt)
    kernel = torch.tensor(
        [-0.5, 0.0, 0.5], dtype=u.dtype, device=u.device
    ).view(1, 1, 3) / dt
    out = F.conv1d(u_p, kernel).squeeze(1)
    return out.view(B, Nx, Nt - 2)                     # (B, Nx, Nt-2)


def _pde_residual(u_pred, params_raw, x_grid_1d, t_grid_1d):
    """
    Heat equation PDE residual: du/dt - lambda * d2u/dx2 = 0
    u_pred:    (B, Nx, Nt)
    params_raw: (B, 1) physical lambda values
    """
    dx = (x_grid_1d[1] - x_grid_1d[0]).item()
    dt = (t_grid_1d[1] - t_grid_1d[0]).item()

    u_xx = _fd4_d2_dx(u_pred, dx)   # (B, Nx-4, Nt)
    u_t  = _fd2_d1_dt(u_pred, dt)   # (B, Nx, Nt-2)

    # Trim to common valid interior
    u_xx_v = u_xx[:, :, 1:-1]       # (B, Nx-4, Nt-2)
    u_t_v  = u_t[:, 2:-2, :]        # (B, Nx-4, Nt-2)

    lam = params_raw.view(-1, 1, 1)  # (B, 1, 1)

    # Heat PDE: du/dt - lambda * d2u/dx2 = 0
    residual = u_t_v - lam * u_xx_v
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Full physics loss — same structure as Black-Scholes compute_pino_loss
# ---------------------------------------------------------------------------

def compute_pino_loss(u_pred, params_raw, x_grid_1d, t_grid_1d, V_scale,
                      lambda_pde=1.0, lambda_ic=10.0, lambda_bc=10.0):
    B = params_raw.shape[0]

    # PDE residual
    l_pde = _pde_residual(u_pred, params_raw, x_grid_1d, t_grid_1d)

    # IC: u(x, 0) = sin(pi*x)  [t index 0 = t=0.0]
    u_ic      = u_pred[:, :, 0]                              # (B, Nx)
    x_ic      = x_grid_1d.unsqueeze(0).expand(B, -1)        # (B, Nx)
    ic_target = torch.sin(torch.pi * x_ic)                  # sin(pi*x)
    l_ic      = F.mse_loss(u_ic, ic_target)

    # BC: u(0, t) = 0  and  u(1, t) = 0
    l_bc_left  = torch.mean(u_pred[:, 0,  :] ** 2)
    l_bc_right = torch.mean(u_pred[:, -1, :] ** 2)
    l_bc = l_bc_left + l_bc_right

    total = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
    return total, l_pde, l_ic, l_bc


# ---------------------------------------------------------------------------
# Relative L2 loss — identical to Black-Scholes version
# ---------------------------------------------------------------------------

def relative_l2_loss(pred, target):
    B = pred.shape[0]
    diff  = torch.norm(pred.view(B, -1) - target.view(B, -1), p=2, dim=1)
    denom = torch.norm(target.view(B, -1), p=2, dim=1) + 1e-8
    return (diff / denom).mean()
