# train_utils.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import device, DataConfig
from residuals import (pde_residual_basic,
                       pde_residual_weno_z5,
                       pde_residual_hybrid)
from model import PINN

def train_pinn(model: torch.nn.Module,
               data_cfg: DataConfig,
               method: str = "hybrid",
               nu: float = 0.01 / 3.141592653589793,
               max_epochs: int = 1000,
               lr_basic: float = 1e-3,
               lr_weno: float = 1e-4) -> tuple[torch.nn.Module, list[float]]:
    """
    Train the given PINN model with one of ["basic","weno_z5","hybrid"].
    Returns: (trained_model, loss_history)
    """
    # 1) choose optimizer & scheduler
    lr = lr_basic if method == "basic" else lr_weno
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = None
    if method != "basic":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    # 2) unpack training points
    pts = data_cfg.generate_training_points()
    x_f, t_f = pts["collocation"]
    x_ic, t_ic = pts["initial"]
    x_bc_left, x_bc_right, t_bc = pts["boundary"]
    x_val = pts["validation"]

    # initial u_ic (u(x,0))
    u_ic = -torch.sin(torch.pi * x_ic).to(device)  # if using u(x,0) = -sin(πx)

    # boundary u = 0 (homogeneous Dirichlet at x = ±1)
    u_bc_left = torch.zeros_like(x_bc_left).to(device)
    u_bc_right = torch.zeros_like(x_bc_right).to(device)

    # 3) pick residual function
    if method == "basic":
        residual_fn = lambda m, xx, tt: pde_residual_basic(m, xx, tt, nu)
    elif method == "weno_z5":
        residual_fn = lambda m, xx, tt: pde_residual_weno_z5(m, xx, tt, nu)
    else:
        residual_fn = lambda m, xx, tt: pde_residual_hybrid(m, xx, tt, nu)

    # 4) train‐loop
    loss_history = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()

        # 4.1) IC loss
        u_pred_ic = model(x_ic, t_ic)  # [N_b,1]
        loss_ic = F.mse_loss(u_pred_ic, u_ic)

        # 4.2) BC loss
        u_pred_left = model(x_bc_left, t_bc)
        u_pred_right = model(x_bc_right, t_bc)
        loss_bc = F.mse_loss(u_pred_left, u_bc_left) + \
                  F.mse_loss(u_pred_right, u_bc_right)

        # 4.3) PDE loss at collocation
        res = residual_fn(model, x_f, t_f)  # [N_f,1]
        loss_pde = F.mse_loss(res, torch.zeros_like(res))

        loss = loss_ic + loss_bc + loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # track best model state
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()

        loss_history.append(loss.item())

        if epoch % 200 == 0 or epoch == 1:
            lr_cur = optimizer.param_groups[0]["lr"]
            print(f"{method.upper()} | Epoch {epoch:4d} | LR={lr_cur:.1e} "
                  f"IC={loss_ic:.2e} BC={loss_bc:.2e} PDE={loss_pde:.2e} TOT={loss:.2e}")

    # 5) load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, loss_history
