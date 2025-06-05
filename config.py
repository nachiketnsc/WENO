# config.py

import torch
import numpy as np
from scipy.integrate import simps as _simps

# ──────────────────────────────────────────────────────────────────────────────
# 1) Global device configuration
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(111)
np.random.seed(111)

# ──────────────────────────────────────────────────────────────────────────────
# 2) DataConfig: generate IC/BC/Collocation/Test points using torch.SobolEngine
# ──────────────────────────────────────────────────────────────────────────────
class DataConfig:
    def __init__(self,
                 n_collocation: int = 5000,
                 n_boundary: int = 100,
                 n_validation: int = 1000,
                 n_test: int = 10000,
                 x_lower: float = -1.0,
                 x_upper: float = 1.0):
        # sample sizes
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_validation = n_validation
        self.n_test = n_test

        # domain bounds
        self.x_lower = x_lower
        self.x_upper = x_upper

        # device
        self.device = device

    def generate_training_points(self):
        """
        Returns a dict with:
          - 'collocation': (x_f, t_f)
          - 'initial':     (x_ic, t_ic)
          - 'boundary':    (x_bc_left, x_bc_right, t_bc)
          - 'validation':  x_validation
          - 'test':        (x_test, t_test)
        All returned tensors live on `self.device`.
        """
        # 2.1) Collocation: Sobol in x, uniform random in t
        sobol = torch.quasirandom.SobolEngine(dimension=1,
                                              scramble=True,
                                              seed=111)
        # Draw n_collocation points in [0,1] and scale to [x_lower,x_upper]
        x_sobol_unit = sobol.draw(self.n_collocation).squeeze(dim=1)  # shape [n_collocation]
        x_collocation = (x_sobol_unit * (self.x_upper - self.x_lower)
                         + self.x_lower).view(-1, 1).to(self.device)

        # Random t ∈ [0,1]
        t_collocation = torch.rand(self.n_collocation, 1,
                                   device=self.device)

        # 2.2) Initial Condition (IC): t = 0, x ∈ [x_lower,x_upper]
        x_ic = torch.linspace(self.x_lower,
                              self.x_upper,
                              self.n_boundary,
                              device=self.device).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)

        # 2.3) Boundary Condition (BC): x = x_lower or x = x_upper, t ∈ [0,1]
        t_bc = torch.linspace(0, 1,
                              self.n_boundary,
                              device=self.device).view(-1, 1)
        x_bc_left = self.x_lower * torch.ones_like(t_bc)
        x_bc_right = self.x_upper * torch.ones_like(t_bc)

        # 2.4) Validation: x ∈ [x_lower, x_upper] at t=0 (purely for exact‐error)
        x_validation = torch.linspace(self.x_lower,
                                      self.x_upper,
                                      self.n_validation,
                                      device=self.device).view(-1, 1)

        # 2.5) Test: a larger set in x and t (for final plotting / error‐calc)
        x_test = torch.linspace(self.x_lower,
                                self.x_upper,
                                self.n_test,
                                device=self.device).view(-1, 1)
        t_test = torch.linspace(0, 1,
                                self.n_test,
                                device=self.device).view(-1, 1)

        return {
            "collocation": (x_collocation, t_collocation),
            "initial": (x_ic, t_ic),
            "boundary": (x_bc_left, x_bc_right, t_bc),
            "validation": x_validation,
            "test": (x_test, t_test),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3) Exact solution (Cole–Hopf)
# ──────────────────────────────────────────────────────────────────────────────
def exact_solution(x: torch.Tensor,
                   t: torch.Tensor,
                   nu: float = 0.01 / np.pi,
                   n_eta: int = 1000) -> torch.Tensor:
    """
    Compute the exact Burgers solution via Cole–Hopf transform on a 1D grid.
    Input: x, t ∈ [x_lower,x_upper]×[0,1], each shape [N,1]
    Output: u_exact [N,1]
    """
    x_flat = x.detach().cpu().numpy().flatten()
    t_flat = t.detach().cpu().numpy().flatten()
    u_vals = []
    eta = np.linspace(-1, 1, n_eta)

    for xi, ti in zip(x_flat, t_flat):
        ti = max(ti, 1e-6)  # avoid t=0 exactly
        exp_arg = (-(xi - eta)**2) / (4 * nu * ti) \
                  - (1 / (2 * nu)) * (-np.cos(np.pi * eta) / np.pi)
        exp_arg = np.clip(exp_arg, -100, 100)
        numerator = np.exp(exp_arg)
        denominator = np.exp((-1 / (2 * nu)) * (-np.cos(np.pi * eta) / np.pi))
        integral = _simps(numerator / (denominator + 1e-8), eta)
        u_val = -2 * nu * (np.pi * np.sin(np.pi * xi)) / (integral + 1e-8)
        u_vals.append(u_val)

    out = torch.tensor(u_vals, dtype=torch.float32,
                       device=device).view(-1, 1)
    return out
