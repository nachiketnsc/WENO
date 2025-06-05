# integrated_burgers_pinns.py

"""
This script implements three variants of PINN for the 1D viscous Burgers equation:
    u_t + u u_x = ν u_xx,  x ∈ [−1,1],  t ∈ [0,1]
with initial condition u(x,0) = −sin(π x), and boundary conditions u(±1,t)=0.

Variants:
    1) basic     → all derivatives via autograd
    2) weno_z5   → compute (u^2/2)_x with 5th-order WENO-Z5, u_t and u_xx via autograd
    3) hybrid    → hybrid: u_x via WENO-Z5 only where “shock” is detected (β_max > threshold), else via autograd; u_xx and u_t via autograd

Usage:
    python integrated_burgers_pinns.py --method basic
    python integrated_burgers_pinns.py --method weno_z5
    python integrated_burgers_pinns.py --method hybrid

Each run will train for 5000 epochs, printing losses every 500 epochs.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1) PINN network definition
# ──────────────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x, t):
        X = torch.cat((x, t), dim=1)
        for i in range(len(self.net) - 1):
            X = self.activation(self.net[i](X))
        X = self.net[-1](X)
        return X

# ──────────────────────────────────────────────────────────────────────────────
# 2) Basic PINN residual (all autodiff)
# ──────────────────────────────────────────────────────────────────────────────
def pde_residual_basic(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)                                    # [N,1]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx                    # [N,1]

# ──────────────────────────────────────────────────────────────────────────────
# 3) WENO-Z5 flux derivative (vectorized, PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
def weno_z5_flux_derivative_torch(u, dx, eps=1e-4):
    """
    Vectorized 5th-order WENO-Z flux derivative for f = u^2/2.
    Input:
      u:    [N] torch tensor of u(x_i) values
      dx:   float (uniform spacing)
      eps:  small parameter for WENO
    Output:
      f_x:  [N] torch tensor of (u^2/2)_x
    """
    # 1) Compute f = u^2/2
    f = 0.5 * u.pow(2)  # [N]

    # 2) Pad f on each side by 3 (edge replicate)
    f_pad = F.pad(f.unsqueeze(0).unsqueeze(0), (3, 3), mode='replicate')  # [1,1,N+6]

    # 3) Extract all 5‐point stencils via unfold (sliding window)
    windows = f_pad.unfold(dimension=2, size=5, step=1)  # [1,1,N+2,5]
    windows = windows[:, :, 3:-2, :].view(-1, 5)         # [N,5]

    # 4) Candidate reconstructions p0, p1, p2
    p0 = (  2*windows[:,0] -   7*windows[:,1] + 11*windows[:,2]) / 6.0
    p1 = ( -1*windows[:,1] +   5*windows[:,2] +   2*windows[:,3]) / 6.0
    p2 = (  2*windows[:,2] +   5*windows[:,3] -   1*windows[:,4]) / 6.0

    # 5) Smoothness indicators β0, β1, β2
    beta0 = (13/12)*(windows[:,0] - 2*windows[:,1] + windows[:,2]).pow(2) \
          + ( 1/4)*(windows[:,0] - 4*windows[:,1] + 3*windows[:,2]).pow(2)
    beta1 = (13/12)*(windows[:,1] - 2*windows[:,2] + windows[:,3]).pow(2) \
          + ( 1/4)*(windows[:,1] - windows[:,3]).pow(2)
    beta2 = (13/12)*(windows[:,2] - 2*windows[:,3] + windows[:,4]).pow(2) \
          + ( 1/4)*(3*windows[:,2] - 4*windows[:,3] + windows[:,4]).pow(2)

    # 6) Global smoothness τ = |β0 - β2|
    tau = (beta0 - beta2).abs()

    # 7) Linear weights d0, d1, d2
    d0, d1, d2 = 0.1, 0.6, 0.3

    # 8) Nonlinear weights α_k with exponent p=1
    alpha0 = d0 * (1.0 + tau / (beta0 + eps))
    alpha1 = d1 * (1.0 + tau / (beta1 + eps))
    alpha2 = d2 * (1.0 + tau / (beta2 + eps))
    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    # 9) Reconstructed flux at i+1/2: f_half
    f_half = w0 * p0 + w1 * p1 + w2 * p2  # [N]

    # 10) f_half_{i-1}: roll by one (pad first with itself)
    f_half_prev = torch.cat((f_half[:1], f_half[:-1]), dim=0)  # [N]

    # 11) Final derivative
    f_x = (f_half - f_half_prev) / dx  # [N]
    return f_x

# ──────────────────────────────────────────────────────────────────────────────
# 4) WENO-Z5 PINN residual
# ──────────────────────────────────────────────────────────────────────────────
def pde_residual_weno_z5(model, x, t, nu):
    # Predict u
    u = model(x, t)            # [N,1]
    u_flat = u.view(-1)        # [N]

    # Sort x to compute dx and maintain mapping
    x_flat = x.view(-1)
    x_np = x_flat.detach().cpu().numpy()
    sorted_idx = np.argsort(x_np)
    x_sorted = x_np[sorted_idx]
    unique_x = np.unique(np.round(x_sorted, 8))
    unique_x.sort()
    dx = float(unique_x[1] - unique_x[0])

    # Reorder u_flat accordingly
    u_np = u_flat.detach().cpu().numpy()
    u_sorted = u_np[sorted_idx]
    u_sorted_torch = torch.tensor(u_sorted, dtype=torch.float32, device=u.device)

    # Compute f_x_sorted by WENO-Z5
    f_x_sorted = weno_z5_flux_derivative_torch(u_sorted_torch, dx, eps=1e-4)  # [N]

    # Map f_x back to original ordering
    f_x_full = torch.zeros_like(u_flat)
    f_x_full[sorted_idx] = f_x_sorted

    # Convert to [N,1]
    f_x_tensor = f_x_full.view(-1, 1)  # [N,1]

    # Compute u_t and u_xx via autograd
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_autodiff = model(x, t)  # [N,1]
    u_t = torch.autograd.grad(u_autodiff, t, torch.ones_like(u_autodiff), create_graph=True)[0]
    u_x = torch.autograd.grad(u_autodiff, x, torch.ones_like(u_autodiff), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # PDE residual: u_t + f_x - nu * u_xx
    return u_t + f_x_tensor - nu * u_xx

# ──────────────────────────────────────────────────────────────────────────────
# 5) Hybrid WENO/autograd PINN residual (vectorized)
# ──────────────────────────────────────────────────────────────────────────────
def pde_residual_hybrid(model, x, t, nu, shock_thresh=1e-3):
    # 5.1) Compute u, u_x, u_xx, u_t via autograd
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)  # [N,1]

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]  # [N,1]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]  # [N,1]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]  # [N,1]

    u_flat = u.view(-1)         # [N]
    x_flat = x.view(-1)         # [N]

    # 5.2) Compute β_max per point (vectorized)
    u_pad = F.pad(u_flat.unsqueeze(0).unsqueeze(0), (3,3), mode='replicate')  # [1,1,N+6]
    windows = u_pad.unfold(dimension=2, size=5, step=1)  # [1,1,N+2,5]
    windows = windows[:, :, 3:-2, :].view(-1, 5)         # [N,5]

    s0, s1, s2, s3, s4 = windows[:,0], windows[:,1], windows[:,2], windows[:,3], windows[:,4]
    beta0 = (13/12)*(s0 - 2*s1 + s2).pow(2) + (1/4)*(s0 - 4*s1 + 3*s2).pow(2)
    beta1 = (13/12)*(s1 - 2*s2 + s3).pow(2) + (1/4)*(s1 - s3).pow(2)
    beta2 = (13/12)*(s2 - 2*s3 + s4).pow(2) + (1/4)*(3*s2 - 4*s3 + s4).pow(2)
    beta_max = torch.max(torch.stack((beta0, beta1, beta2), dim=1), dim=1)[0]  # [N]

    # 5.3) Build shock mask: where β_max > shock_thresh
    shock_mask = (beta_max > shock_thresh)  # [N], bool

    # 5.4) If any shock points, compute WENO-Z5 flux derivative for all points
    N = x_flat.shape[0]
    f_x_full = torch.zeros(N, device=u.device)

    if shock_mask.any():
        # Sort x_flat to get dx and ordering
        x_np = x_flat.detach().cpu().numpy()
        sorted_idx = np.argsort(x_np)
        x_sorted = x_np[sorted_idx]
        unique_x = np.unique(np.round(x_sorted, 8))
        unique_x.sort()
        dx = float(unique_x[1] - unique_x[0])

        # Reorder u_flat accordingly
        u_np = u_flat.detach().cpu().numpy()
        u_sorted = u_np[sorted_idx]
        u_sorted_torch = torch.tensor(u_sorted, dtype=torch.float32, device=u.device)

        # Compute flux derivative on sorted array
        f_x_sorted = weno_z5_flux_derivative_torch(u_sorted_torch, dx, eps=1e-4)  # [N]

        # Map back
        f_x_full_sorted = torch.zeros_like(f_x_sorted)
        f_x_full_sorted[sorted_idx] = f_x_sorted
        f_x_full = f_x_full_sorted
    # else: f_x_full stays zeros

    # 5.5) Combine: if not shock, use autograd u_x; else use WENO f_x
    u_x_flat = u_x.view(-1)  # [N]
    # f_x_full now holds WENO values; override only where shock_mask is True
    u_x_comb = u_x_flat.clone()
    if shock_mask.any():
        u_x_comb[shock_mask] = f_x_full[shock_mask]

    f_x_tensor = u_x_comb.view(-1, 1)  # [N,1]

    # 5.6) Final residual
    return u_t + u * f_x_tensor - nu * u_xx  # [N,1]

# ──────────────────────────────────────────────────────────────────────────────
# 6) Initial & boundary conditions
# ──────────────────────────────────────────────────────────────────────────────
def initial_condition(x):
    return -torch.sin(np.pi * x)  # [N,1]

def boundary_condition(x, t):
    return torch.zeros_like(x)   # [N,1]

# ──────────────────────────────────────────────────────────────────────────────
# 7) Training function (shared)
# ──────────────────────────────────────────────────────────────────────────────
def train_pinn(method, device):
    """
    method ∈ {'basic', 'weno_z5', 'hybrid'}
    device = 'cpu' or 'cuda'
    """
    # PDE parameters
    nu = 0.01 / np.pi
    layers = [2, 50, 50, 50, 1]
    lr = 1e-4 if method != 'basic' else 1e-3
    epochs = 5000
    shock_thresh = 1e-3  # for hybrid

    # Instantiate model
    model = PINN(layers).to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    if method != 'basic':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler = None

    # 7.1) Initial condition points
    N_ic = 100
    x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    u_ic = initial_condition(x_ic).to(device)

    # 7.2) Boundary condition points
    N_bc = 100
    t_bc = torch.linspace(0, 1, N_bc).view(-1, 1).to(device)
    x_bc_left  = -torch.ones_like(t_bc).to(device)
    x_bc_right =  torch.ones_like(t_bc).to(device)
    u_bc_left  = boundary_condition(x_bc_left, t_bc).to(device)
    u_bc_right = boundary_condition(x_bc_right, t_bc).to(device)

    # 7.3) Collocation points
    N_f = 5000
    N_f_side = int(np.sqrt(N_f))
    x_vals = torch.linspace(-1, 1, N_f_side).view(-1, 1).to(device)
    t_vals = torch.rand((N_f_side, 1)).to(device)
    x_grid, t_grid = torch.meshgrid(x_vals.view(-1), t_vals.view(-1), indexing='xy')
    x_f = x_grid.contiguous().view(-1, 1).to(device)  # [N_f,1]
    t_f = t_grid.contiguous().view(-1, 1).to(device)  # [N_f,1]

    # 7.4) Choose residual function
    if method == 'basic':
        residual_fn = lambda mdl, xx, tt: pde_residual_basic(mdl, xx, tt, nu)
    elif method == 'weno_z5':
        residual_fn = lambda mdl, xx, tt: pde_residual_weno_z5(mdl, xx, tt, nu)
    elif method == 'hybrid':
        residual_fn = lambda mdl, xx, tt: pde_residual_hybrid(mdl, xx, tt, nu, shock_thresh)
    else:
        raise ValueError("method must be 'basic', 'weno_z5', or 'hybrid'")

    # 7.5) Training loop
    print(f"Starting {method} PINN training on {device}...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # IC loss
        u_pred_ic = model(x_ic, t_ic)
        loss_ic = nn.MSELoss()(u_pred_ic, u_ic)

        # BC loss
        u_pred_bc_left  = model(x_bc_left,  t_bc)
        u_pred_bc_right = model(x_bc_right, t_bc)
        loss_bc = nn.MSELoss()(u_pred_bc_left,  u_bc_left) + nn.MSELoss()(u_pred_bc_right, u_bc_right)

        # PDE residual loss
        res_f = residual_fn(model, x_f, t_f)
        loss_pde = nn.MSELoss()(res_f, torch.zeros_like(res_f))

        # Total loss
        loss = loss_ic + loss_bc + loss_pde
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        # Print
        if epoch % 500 == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:5d} | "
                f"LR={lr_current:.1e} | "
                f"Loss_ic={loss_ic.item():.3e}  "
                f"Loss_bc={loss_bc.item():.3e}  "
                f"Loss_pde={loss_pde.item():.3e}  "
                f"Total={loss.item():.3e}"
            )

    print(f"{method.capitalize()} PINN training completed.")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 8) Main: parse arguments and run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Basic / WENO-Z5 / Hybrid PINN for 1D Burgers")
    parser.add_argument(
        "--method", choices=["basic", "weno_z5", "hybrid"], default="hybrid",
        help="Which PINN method to run"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    args = parser.parse_args()

    train_pinn(args.method, device=args.device)
