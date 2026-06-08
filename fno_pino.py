import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class PINO1d(nn.Module):
    def __init__(
        self,
        in_dim=5,
        out_dim=1,
        width=128,
        modes=8,
        num_layers=6
    ):
        super(PINO1d, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(in_dim, self.width)
        
        self.convs = nn.ModuleList([SpectralConv1d(self.width, self.width, modes) for _ in range(num_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(num_layers)])
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)
            
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_pino_loss(
    model,
    x_raw,
    num_points=32
):
    device = x_raw.device
    batch_size = x_raw.shape[0]
    
    K = x_raw[:, 1:2]
    r = x_raw[:, 3:4]
    sigma = x_raw[:, 4:5]
    
    s_space = torch.linspace(
        0.0,
        1000.0,
        num_points,
        device=device
    )
    tau_space = torch.linspace(0.001, 2.0, num_points, device=device)
    S_mesh, tau_mesh = torch.meshgrid(s_space, tau_space, indexing='ij')
    
    S_flat = S_mesh.flatten().view(1, -1, 1).repeat(batch_size, 1, 1)
    tau_flat = tau_mesh.flatten().view(1, -1, 1).repeat(batch_size, 1, 1)
    
    S_flat.requires_grad_(True)
    tau_flat.requires_grad_(True)
    
    K_ext = K.unsqueeze(1).repeat(1, S_flat.shape[1], 1)
    r_ext = r.unsqueeze(1).repeat(1, S_flat.shape[1], 1)
    sigma_ext = sigma.unsqueeze(1).repeat(1, S_flat.shape[1], 1)
    
    inp_pde = torch.cat([S_flat, K_ext, tau_flat, r_ext, sigma_ext], dim=-1)
    V_pde = model(inp_pde)
    
    dV_dS = torch.autograd.grad(V_pde.sum(), S_flat, create_graph=True)[0]
    dV_dtau = torch.autograd.grad(V_pde.sum(), tau_flat, create_graph=True)[0]
    dV_dS2 = torch.autograd.grad(dV_dS.sum(), S_flat, create_graph=True)[0]
    
    pde_res = dV_dtau - 0.5 * (sigma_ext ** 2) * (S_flat ** 2) * dV_dS2 - r_ext * S_flat * dV_dS + r_ext * V_pde
    pde_loss = torch.mean(pde_res ** 2)
    
    s_term = torch.linspace(0.0, 200.0, num_points, device=device).view(1, -1, 1).repeat(batch_size, 1, 1)
    tau_term = torch.zeros_like(s_term)
    K_term = K.unsqueeze(1).repeat(1, num_points, 1)
    r_term = r.unsqueeze(1).repeat(1, num_points, 1)
    sigma_term = sigma.unsqueeze(1).repeat(1, num_points, 1)
    
    inp_terminal = torch.cat([s_term, K_term, tau_term, r_term, sigma_term], dim=-1)
    V_terminal = model(inp_terminal)
    payoff = torch.clamp(s_term - K_term, min=0.0)
    terminal_loss = torch.mean((V_terminal - payoff) ** 2)
    
    tau_bound = torch.linspace(0.0, 2.0, num_points, device=device).view(1, -1, 1).repeat(batch_size, 1, 1)
    S_low = torch.zeros_like(tau_bound)
    K_bound = K.unsqueeze(1).repeat(1, num_points, 1)
    r_bound = r.unsqueeze(1).repeat(1, num_points, 1)
    sigma_bound = sigma.unsqueeze(1).repeat(1, num_points, 1)
    
    inp_low = torch.cat([S_low, K_bound, tau_bound, r_bound, sigma_bound], dim=-1)
    V_low = model(inp_low)
    boundary_loss_low = torch.mean(V_low ** 2)
    
    S_high = torch.full_like(
        tau_bound,
        1000.0
    )
    inp_high = torch.cat([S_high, K_bound, tau_bound, r_bound, sigma_bound], dim=-1)
    V_high = model(inp_high)
    discounted_strike = K_bound * torch.exp(-r_bound * tau_bound)
    boundary_loss_high = torch.mean((V_high - (S_high - discounted_strike)) ** 2)
    
    total_physics_loss = pde_loss + terminal_loss + boundary_loss_low + boundary_loss_high
    return total_physics_loss