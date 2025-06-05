# residuals.py
import torch
import torch.nn.functional as F
from torch.autograd import grad
from weno_utils import weno_z5_flux_derivative

def pde_residual_basic(model: torch.nn.Module,
                       x: torch.Tensor,
                       t: torch.Tensor,
                       nu: float) -> torch.Tensor:
    """
    Burgers PDE: u_t + u u_x − nu u_{xx} = 0
    Compute u_t, u_x, u_{xx} via pure autodiff.
    Inputs:
      - x, t each require_grad
      - model(x,t) → u_pred [N,1]
    Returns:
      - residual [N,1]
    """
    x_req = x.clone().requires_grad_(True)
    t_req = t.clone().requires_grad_(True)

    u = model(x_req, t_req)                 # [N,1]
    u_x = grad(u, x_req, torch.ones_like(u),
               create_graph=True)[0]
    u_t = grad(u, t_req, torch.ones_like(u),
               create_graph=True)[0]
    u_xx = grad(u_x, x_req, torch.ones_like(u_x),
                create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx        # [N,1]


def pde_residual_weno_z5(model: torch.nn.Module,
                         x: torch.Tensor,
                         t: torch.Tensor,
                         nu: float) -> torch.Tensor:
    """
    Burgers PDE: u_t + ∂_x(½ u^2) − nu u_{xx} = 0,
    where ∂_x(½ u^2) is computed by WENO‐Z5.
    """
    # 1) predict u at (x,t)
    u_pred = model(x, t).view(-1)  # [N]

    # 2) sort by x, compute WENO on sorted
    x_flat = x.view(-1)
    sort_idx = torch.argsort(x_flat)
    x_sorted = x_flat[sort_idx]
    u_sorted = u_pred[sort_idx]

    # 3) compute dx (assume roughly uniform once sorted)
    dx = (x_sorted[1] - x_sorted[0]).clamp(min=1e-8)

    # 4) f_x_sorted via WENO
    f_x_sorted = weno_z5_flux_derivative(u_sorted, dx)

    # 5) unsort back
    unsort_idx = torch.argsort(sort_idx)
    f_x_full = f_x_sorted[unsort_idx]  # [N]

    # 6) Now compute u_t and u_xx via autodiff for the diffusive term
    x_req = x.clone().requires_grad_(True)
    t_req = t.clone().requires_grad_(True)
    u2 = model(x_req, t_req)            # [N,1]
    u_t = grad(u2, t_req, torch.ones_like(u2),
               create_graph=True)[0]
    u_x = grad(u2, x_req, torch.ones_like(u2),
               create_graph=True)[0]
    u_xx = grad(u_x, x_req, torch.ones_like(u_x),
                create_graph=True)[0]

    return u_t + f_x_full.view(-1, 1) - nu * u_xx  # [N,1]


def pde_residual_hybrid(model: torch.nn.Module,
                        x: torch.Tensor,
                        t: torch.Tensor,
                        nu: float,
                        beta0: float = 1e-3,
                        k: int = 50) -> torch.Tensor:
    """
    Hybrid: use autodiff for u_t, u_xx, and partially for u_x
      – compute β at each sorted point → build sensor σ
      – blend u_x (autodiff) and f_x (WENO), forming u_x_comb
      – residual = u_t + u * u_x_comb − nu u_xx
    """
    # 1) set up requires_grad
    x_req = x.clone().requires_grad_(True)
    t_req = t.clone().requires_grad_(True)
    u_pred = model(x_req, t_req)  # [N,1]
    N = u_pred.shape[0]

    # 2) autodiff for u_x, u_xx, u_t
    u_x = grad(u_pred, x_req, torch.ones_like(u_pred),
               create_graph=True)[0]  # [N,1]
    u_xx = grad(u_x, x_req, torch.ones_like(u_x),
                create_graph=True)[0]  # [N,1]
    u_t = grad(u_pred, t_req, torch.ones_like(u_pred),
               create_graph=True)[0]  # [N,1]

    # 3) flatten to 1D for sorting
    u_flat = u_pred.view(-1)
    x_flat = x_req.view(-1)
    sort_idx = torch.argsort(x_flat)
    x_sorted = x_flat[sort_idx]  # [N]
    u_sorted = u_flat[sort_idx]  # [N]

    # 4) build β_max on interior (N−2)
    #    pad u_sorted by 3 for 5-point stencils
    u_pad = F.pad(u_sorted.unsqueeze(0).unsqueeze(0),
                  (3, 3), mode="replicate")  # [1,1,N+6]
    win = u_pad.unfold(2, 5, 1).squeeze(0).squeeze(0)  # [N+2, 5]
    win_interior = win[2:-2]  # [N-2,5]
    s0, s1, s2, s3, s4 = win_interior.unbind(dim=1)

    beta0_loc = (13/12) * (s0 - 2 * s1 + s2).pow(2) \
                + 0.25 * (s0 - 4 * s1 + 3 * s2).pow(2)
    beta1_loc = (13/12) * (s1 - 2 * s2 + s3).pow(2) \
                + 0.25 * (s1 - s3).pow(2)
    beta2_loc = (13/12) * (s2 - 2 * s3 + s4).pow(2) \
                + 0.25 * (3 * s2 - 4 * s3 + s4).pow(2)

    beta_max = torch.max(
        torch.stack([beta0_loc, beta1_loc, beta2_loc], dim=1), dim=1
    )[0]  # [N-2]

    # 5) pad β_max to length N (repeat first & last interior)
    beta_front = beta_max[:1]    # single value
    beta_back = beta_max[-1:]
    beta_padded = torch.cat([beta_front, beta_max, beta_back], dim=0)  # [N]

    # 6) WENO flux derivative on sorted u
    dx = (x_sorted[1] - x_sorted[0]).clamp(min=1e-8)
    f_x_sorted = weno_z5_flux_derivative(u_sorted, dx)  # [N]

    # 7) unsort f_x and β_max
    unsort_idx = torch.argsort(sort_idx)
    f_x_full = f_x_sorted[unsort_idx]        # [N]
    beta_full = beta_padded[unsort_idx]       # [N]

    # 8) build sensor σ at each point
    sigma = torch.sigmoid(k * (beta_full - beta0))  # [N]

    # 9) blend u_x (autodiff) vs WENO derivative
    u_x_flat = u_x.view(-1)             # [N]
    u_x_comb = ((1 - sigma) * u_x_flat) + (sigma * f_x_full)  # [N]
    u_x_comb = u_x_comb.view(-1, 1)     # [N,1]

    # 10) return hybrid residual
    return u_t + u_pred * u_x_comb - nu * u_xx  # [N,1]
