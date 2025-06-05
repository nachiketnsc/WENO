# Hybrid WENO-PINN for 1D Burgers’ Equation

This project demonstrates a **Hybrid WENO-PINN** approach for solving the one-dimensional viscous Burgers’ equation:

\[
u_t + u\,u_x = \nu\,u_{xx}, \quad x \in [-1,1],\ t \in [0,1].
\]

## Basic Idea

- **PINN (Physics-Informed Neural Network):**  
  A neural network \(u_\theta(x,t)\) is trained to satisfy the PDE residual  
  \[
    r(x,t) \;=\; u_t + u\,u_x - \nu\,u_{xx} \;=\; 0,
  \]  
  along with initial/boundary conditions. All spatial derivatives (\(u_x,\,u_{xx}\)) are computed via automatic differentiation (AD).  
  PINNs excel in smooth regions of the solution because AD is highly accurate when \(u(x,t)\) varies smoothly.

- **WENO-Z5 (Weighted Essentially Non-Oscillatory, 5th order):**  
  A classical finite-difference scheme that reconstructs spatial derivatives using a 5-point stencil. WENO-Z5 is specifically designed to handle steep gradients or discontinuities (shocks) without introducing spurious oscillations (Gibbs phenomenon). It chooses nonlinear weights based on local “smoothness indicators” to adaptively pick stencils that avoid crossing discontinuities.

- **Why Hybridize?**  
  Burgers’ equation often develops sharp gradients (viscous shocks) even for small viscosity \(\nu\). A pure PINN tends to “smooth out” these sharp features, leading to blurred shock profiles. WENO-Z5 can sharply resolve the shock, but applying it everywhere would require a very fine mesh and sacrifices the flexibility of a mesh-free PINN. By combining them:
  1. **Smooth regions** (where \(|u_x|\) is small): use the PINN’s AD for spatial derivatives.  
  2. **Near a shock** (where \(|u_x|\) exceeds some threshold \(\varepsilon\)): switch to WENO-Z5 to compute \(\partial_x(\tfrac12u^2)\) on a small 5-point stencil.  

  This hybrid approach leverages the best of both worlds:  
  - In smooth areas, the PINN remains efficient and accurate via AD.  
  - Near discontinuities, WENO-Z5 provides non-oscillatory, high-order accuracy without needing a globally fine mesh.  

## Workflow Overview

1. **Network Architecture:**  
   A feed-forward neural network \(u_\theta(x,t)\) with several hidden layers (e.g., 4 layers × 64 neurons).
2. **PDE Residual Calculation:**  
  2. **PDE Residual Calculation**

- **Compute** \(u_t\) and \(u_x\) by automatic differentiation at each collocation point \((x_i,\,t_i)\).


- **Otherwise**:
  1. Gather a 5-point stencil around \(x_i\):
     \[
       \{\,x_{j-2},\,x_{j-1},\,x_{j},\,x_{j+1},\,x_{j+2}\}.
     \]
  2. Evaluate \(u\) at those five spatial points (all at time \(t_i\)):
3. **Loss Function:**  
   Combine mean-squared error of the PDE residual, initial condition mismatch, and boundary condition mismatch.  
4. **Training Loop:**  
   - Sample collocation points in \((x,t)\).  
   - Update network parameters \(\theta\) via Adam (and optionally LBFGS) to minimize total loss.  
   - Track how many points triggered WENO versus PINN (for diagnostics).  
5. **Result:**  
   The trained network produces a sharp shock profile at \(t=1\) that closely matches the analytic (or high-resolution) solution, while avoiding spurious oscillations.

## Why It’s Useful

- **Accurate Shock Capturing:**  
  Pure PINNs smear out discontinuities. By switching to WENO-Z5 exactly where needed, the hybrid approach recovers sharp shock profiles without oscillations.
- **Mesh-Free Flexibility + Local Stencil Accuracy:**  
  PINNs work on scattered collocation points, while WENO-Z5 only applies on local 5-point neighborhoods near a shock. You don’t need a fine mesh everywhere.
- **Adaptive Derivative Computation:**  
  The “shock sensor” (based on \(\lvert u_x\rvert\)) automatically detects regions that require WENO. In smooth areas, AD remains optimal.

In summary, this Hybrid WENO-PINN method fuses high-order, non-oscillatory finite differences near discontinuities with PINN’s mesh-free capabilities in smooth regions, making it a robust choice for problems like Burgers’ equation that exhibit both smooth and sharply varying features.  
