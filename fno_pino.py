import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral Convolution layer
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B, self.out_channels, x.size(-2), x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# ---------------------------------------------------------------------------
# PINO2d — receives normalised inputs, produces normalised output
#
# Input channels fed to the network:
#   [K_n, r_n, sigma_n, S_n, T_n]   all in [0, 1]
#
# Output: V_norm = V / V_scale, also approximately in [0, 1]
# ---------------------------------------------------------------------------

class PINO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, num_layers=4):
        super().__init__()
        self.width = width
        self.fc0   = nn.Linear(5, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)]
        )
        self.ws    = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_layers)]
        )
        self.fc1   = nn.Linear(width, 128)
        self.fc2   = nn.Linear(128, 1)

    def forward(self, params_n, S_grid_n, T_grid_n):
        # params_n:  (B, 3)  normalised [K_n, r_n, sigma_n]
        # S_grid_n:  (Nx, Nt) or (B, Nx, Nt)  normalised S in [0, 1]
        # T_grid_n:  same shape, normalised T in [0, 1]
        B = params_n.shape[0]
        K_n     = params_n[:, 0].view(B, 1, 1)
        r_n     = params_n[:, 1].view(B, 1, 1)
        sigma_n = params_n[:, 2].view(B, 1, 1)

        S_n = S_grid_n.unsqueeze(0).expand(B, -1, -1) if S_grid_n.dim() == 2 else S_grid_n
        T_n = T_grid_n.unsqueeze(0).expand(B, -1, -1) if T_grid_n.dim() == 2 else T_grid_n

        K_e     = K_n.expand(-1,     S_n.shape[1], S_n.shape[2])
        r_e     = r_n.expand_as(K_e)
        sigma_e = sigma_n.expand_as(K_e)

        x = torch.stack([K_e, r_e, sigma_e, S_n, T_n], dim=-1)   # (B, Nx, Nt, 5)
        x = self.fc0(x)                                            # (B, Nx, Nt, W)
        x = x.permute(0, 3, 1, 2)                                 # (B, W, Nx, Nt)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 1)                                 # (B, Nx, Nt, W)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)                                       # (B, Nx, Nt)


# ---------------------------------------------------------------------------
# 4th-order FD stencils
# Operate on V_norm but use PHYSICAL S/T grids for derivative scaling.
# This is correct: d(V_norm)/dS = (1/V_scale) * dV/dS.
# The BS PDE in normalised form is identical — linearity means V_scale cancels.
# ---------------------------------------------------------------------------

def _fd4_d2_dx(u, dx):
    B, Nx, Nt = u.shape
    u_p = u.permute(0, 2, 1).reshape(B * Nt, 1, Nx)
    kernel = torch.tensor(
        [-1/12, 4/3, -5/2, 4/3, -1/12], dtype=u.dtype, device=u.device
    ).view(1, 1, 5) / (dx ** 2)
    out = F.conv1d(u_p, kernel).squeeze(1)
    return out.view(B, Nt, Nx - 4).permute(0, 2, 1)   # (B, Nx-4, Nt)


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
    # V_norm:          (B, Nx, Nt)  — normalised output from model
    # params_raw:      (B, 3)       — UN-normalised [K, r, sigma] for PDE coefficients
    # S_grid_1d_raw:   (Nx,)        — physical S values for d/dS scaling
    # T_grid_1d_raw:   (Nt,)        — physical T values for d/dT scaling
    dx = (S_grid_1d_raw[1] - S_grid_1d_raw[0]).item()
    dt = (T_grid_1d_raw[1] - T_grid_1d_raw[0]).item()

    V_SS = _fd4_d2_dx(V_norm, dx)   # (B, Nx-4, Nt)
    V_S  = _fd4_d1_dx(V_norm, dx)   # (B, Nx-4, Nt)
    V_T  = _fd2_d1_dt(V_norm, dt)   # (B, Nx, Nt-2)

    # Valid interior: trim to common region
    V_SS_v = V_SS[:, :, 1:-1]    # (B, Nx-4, Nt-2)
    V_S_v  = V_S[:, :,  1:-1]
    V_T_v  = V_T[:, 2:-2, :]     # (B, Nx-4, Nt-2)
    V_v    = V_norm[:, 2:-2, 1:-1]

    sigma = params_raw[:, 2].view(-1, 1, 1)
    r     = params_raw[:, 1].view(-1, 1, 1)
    S_v   = S_grid_1d_raw[2:-2].view(1, -1, 1)   # physical S, (1, Nx-4, 1)

    # BS PDE (V_norm satisfies same equation as V because the PDE is linear)
    residual = V_T_v - 0.5 * sigma**2 * S_v**2 * V_SS_v - r * S_v * V_S_v + r * V_v
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Full physics loss — all terms operate on normalised V so magnitudes ~O(1)
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
# Supervised loss: relative L2 on normalised outputs (both in [0,1])
# ---------------------------------------------------------------------------

def relative_l2_loss(pred, target):
    B = pred.shape[0]
    diff  = torch.norm(pred.view(B, -1) - target.view(B, -1), p=2, dim=1)
    denom = torch.norm(target.view(B, -1), p=2, dim=1) + 1e-8
    return (diff / denom).mean()
