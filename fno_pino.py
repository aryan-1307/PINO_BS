import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(
                in_channels, out_channels, modes1, modes2,
                dtype=torch.cfloat
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-2), x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class PINO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, num_layers=4):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(5, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, params, S_grid, T_grid):
        batch = params.shape[0]
        K = params[:, 0].view(batch, 1, 1)
        r = params[:, 1].view(batch, 1, 1)
        sigma = params[:, 2].view(batch, 1, 1)

        S = S_grid.unsqueeze(0).expand(batch, -1, -1) if S_grid.dim() == 2 else S_grid
        T = T_grid.unsqueeze(0).expand(batch, -1, -1) if T_grid.dim() == 2 else T_grid

        K = K.expand(-1, S.shape[1], S.shape[2])
        r = r.expand_as(K)
        sigma = sigma.expand_as(K)

        x = torch.stack([K, r, sigma, S, T], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


def compute_pino_loss(model, params, S_grid, T_grid):
    batch = params.shape[0]

    # Detach and re-attach so autograd tracks these fresh leaf tensors
    S = S_grid.unsqueeze(0).repeat(batch, 1, 1).detach().requires_grad_(True)
    T = T_grid.unsqueeze(0).repeat(batch, 1, 1).detach().requires_grad_(True)

    V = model(params, S, T)

    dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
    dV_dT = torch.autograd.grad(V.sum(), T, create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True)[0]

    # Expand scalar params to match (B, grid, grid) for broadcast
    K = params[:, 0].view(batch, 1, 1).expand_as(V)
    r = params[:, 1].view(batch, 1, 1).expand_as(V)
    sigma = params[:, 2].view(batch, 1, 1).expand_as(V)

    # Black-Scholes PDE residual: dV/dT - 0.5*sigma^2*S^2*d2V/dS2 - r*S*dV/dS + r*V = 0
    residual = dV_dT - 0.5 * sigma ** 2 * S ** 2 * d2V_dS2 - r * S * dV_dS + r * V
    pde_loss = torch.mean(residual ** 2)

    # Terminal condition loss: V(S, T=0) = max(S - K, 0)
    # Index 0 on axis-2 is T=0.0 (guaranteed by generate_data.py linspace start)
    V_terminal = V[:, :, 0]
    S_terminal = S[:, :, 0]
    K_terminal = params[:, 0].view(batch, 1).expand_as(V_terminal)
    payoff = torch.clamp(S_terminal - K_terminal, min=0.0)
    terminal_loss = torch.mean((V_terminal - payoff) ** 2)

    # Lower boundary: V(S=0, T) = 0
    V_low = V[:, 0, :]
    boundary_loss_low = torch.mean(V_low ** 2)

    # Upper boundary: V(S_max, T) ≈ S_max - K * exp(-r*T)
    # All slices are (B, grid_size) — shapes must match before arithmetic
    V_high = V[:, -1, :]                              # (B, 64)
    S_high = S[:, -1, :]                              # (B, 64)
    T_high = T[:, -1, :]                              # (B, 64)
    K_high = params[:, 0].view(batch, 1).expand_as(V_high)   # (B, 64)
    r_high = params[:, 1].view(batch, 1).expand_as(V_high)   # (B, 64)
    discounted_strike = K_high * torch.exp(-r_high * T_high)
    boundary_loss_high = torch.mean((V_high - (S_high - discounted_strike)) ** 2)

    return pde_loss + terminal_loss + boundary_loss_low + boundary_loss_high