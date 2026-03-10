# Inverse Problem Setup Guide for PINN Material Identification

## Problem Statement

Given: observed surface displacements u_obs (from DIC, sensors, or FEM synthetic data)
Find: material parameters Θ (e.g., {μ, K} for Neo-Hookean, or spatially-varying K(x))

Jointly solve for the displacement field u and parameters Θ by minimizing:

```
J(u, Θ) = L_equil(u, Θ) + w_data · L_data(u) + w_if · L_if(u)
         + w_rbm · L_rbm(u) + w_reg · L_reg(Θ)
```

---

## Identifiability Analysis (Do This First)

Before coding: check that Θ can actually be recovered from your data.

```python
def compute_sensitivity_matrix(forward_solver, theta_true, x_obs, delta=0.01):
    """Finite-difference sensitivity: S_ij = ∂u_obs_i / ∂Θ_j"""
    u0 = forward_solver(theta_true, x_obs)
    n_obs = len(u0.flatten())
    n_params = len(theta_true)

    S = np.zeros((n_obs, n_params))
    for j, theta_j in enumerate(theta_true):
        theta_plus = theta_true.copy()
        theta_plus[j] *= (1 + delta)
        u_plus = forward_solver(theta_plus, x_obs)
        S[:, j] = (u_plus.flatten() - u0.flatten()) / (theta_j * delta)

    # Check condition number
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    kappa = sv[0] / sv[-1]
    print(f"Sensitivity matrix κ(S) = {kappa:.2e}")
    print(f"Singular values: {sv}")
    if kappa > 1e4:
        print("WARNING: ill-conditioned — add more load cases or observations")

    return S, kappa
```

Rules of thumb:
- κ(S) < 1e2: well-identified
- 1e2 < κ(S) < 1e4: marginally identified (add load cases)
- κ(S) > 1e4: ill-identified (regularization required; parameter may be unidentifiable)

---

## Multi-Load-Case Strategy

Single load case almost always fails to identify multiple parameters independently.
The minimum number of independent load cases:

| Material model | Min load cases | Recommended |
|---------------|---------------|-------------|
| Neo-Hookean (μ, K) | 2 | 3 (tension, shear, compression) |
| Arruda-Boyce (μ, N) | 2 | 3 (include large strain to sample locking) |
| TI-AB (4 params) | 3 | 4 (include fiber-aligned and transverse) |
| Viscoelastic (E_∞, τ) | 2 (different freq.) | 3 cyclic + 1 creep |
| Spatially varying K(x) | N_obs >> N_params | As many as possible |

Implementation pattern for multi-load-case:
```python
# Initialize separate u_pinn per load case; shared material params
pinn_nets = [VectorPINN(xi_dim=3, out_dim=3) for _ in load_cases]
theta = TIParams()   # shared across all load cases

for schwarz_iter in range(max_iters):
    L_total = 0.0
    for lc_idx, lc in enumerate(load_cases):
        u_pinn_lc = pinn_nets[lc_idx]
        L_equil = compute_equilibrium_loss(u_pinn_lc, theta, lc.xi_int)
        L_data  = compute_data_loss(u_pinn_lc, lc.x_obs, lc.u_obs)
        L_bc    = compute_bc_loss(u_pinn_lc, lc.xi_bc, lc.bc_values)
        L_total += L_equil + w_data * L_data + w_bc * L_bc

    L_rbm = compute_rbm_loss(pinn_nets[0])  # penalize first load case
    L_reg = compute_regularization(theta)
    L_total += w_rbm * L_rbm + w_reg * L_reg

    L_total.backward()
    optimizer.step()
```

---

## Equilibrium Loss (Finite Strain)

For a hyperelastic material with energy density Ψ(F, Θ):

```python
def compute_equilibrium_loss(u_pinn, theta, xi_pts, decoder):
    """Strong-form equilibrium residual: ∇_X · P = 0"""
    xi = xi_pts.requires_grad_(True)

    # Displacement at collocation points
    u = u_pinn(xi)           # (N, 3)
    x0 = decoder(xi)         # (N, 3) reference configuration (= X for total Lagrangian)

    # Deformation gradient: F = I + ∂u/∂X
    # Via chain rule: ∂u/∂X = ∂u/∂ξ @ (∂ξ/∂X) = ∂u/∂ξ @ J⁻¹
    J_xi = jacobian(x0, xi)          # (N, 3, 3) = ∂φ/∂ξ
    J_xi_inv = torch.linalg.inv(J_xi)
    du_dxi = jacobian(u, xi)          # (N, 3, 3) = ∂u/∂ξ

    F = torch.eye(3) + torch.bmm(du_dxi, J_xi_inv)   # (N, 3, 3)

    # First PK stress via autograd on Ψ
    F_leaf = F.detach().requires_grad_(True)
    Psi = theta.strain_energy(F_leaf)   # (N,)
    P = torch.autograd.grad(Psi.sum(), F_leaf, create_graph=True)[0]  # (N, 3, 3)

    # Divergence of P in reference config: ∇_X · P = ∂P_iJ/∂X_J
    div_P = divergence_reference(P, xi, J_xi_inv)   # (N, 3)

    L_equil = (div_P**2).mean()
    return L_equil
```

---

## Rigid-Body Motion Penalty

Without this, the inverse problem has infinite solutions (any rigid motion of the
recovered displacement is equally valid).

```python
def compute_rbm_loss(u_pinn, xi_vol, decoder):
    """Penalize mean translation and mean rotation."""
    u = u_pinn(xi_vol)    # (N, 3)
    x = decoder(xi_vol)   # (N, 3) physical positions

    # Mean translation should be zero (or match a known reference point)
    u_mean = u.mean(dim=0)
    L_trans = (u_mean**2).mean()

    # Mean skew-symmetric part of ∇u should be zero (no net rotation)
    du_dxi = jacobian(u, xi_vol)    # (N, 3, 3)
    J_inv  = torch.linalg.inv(jacobian(x, xi_vol))
    grad_u = torch.bmm(du_dxi, J_inv)   # (N, 3, 3)

    # Rotation tensor: W = (∇u - ∇uᵀ)/2
    W = 0.5 * (grad_u - grad_u.transpose(-1, -2))
    L_rot = (W**2).mean()

    return L_trans + L_rot
```

---

## Regularization for Spatially Varying Parameters

When Θ(x) is a field (e.g., heterogeneous permeability K(x)):

```python
class PermeabilityField(nn.Module):
    """Spatially varying permeability represented as a small MLP."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 9)    # symmetric 3×3 tensor
        )
        # Symmetry: K = A @ Aᵀ (positive definite by construction)

    def forward(self, x):
        A_flat = self.net(x)
        A = A_flat.reshape(-1, 3, 3)
        K = torch.bmm(A, A.transpose(-1, -2)) + 1e-4 * torch.eye(3)  # SPD
        return K

def compute_regularization(theta_field, xi_vol, decoder, order='H1'):
    """Tikhonov regularization on field-valued parameter."""
    x = decoder(xi_vol)
    Theta = theta_field(x)    # (N, ...)

    if order == 'L2':
        return (Theta**2).mean()
    elif order == 'H1':
        # Gradient penalty
        dTheta_dx = jacobian(Theta.sum(0), x)   # (N, ...)
        return (dTheta_dx**2).mean()
    elif order == 'TV':
        # Total variation (promotes piecewise-constant)
        return dTheta_dx.abs().mean()
```

---

## Convergence Monitoring for Inverse Problems

Log these quantities every Schwarz iteration:

```python
metrics = {
    'iter': schwarz_iter,
    'L_equil': L_equil.item(),
    'L_data': L_data.item(),
    'L_rbm': L_rbm.item(),
    'L_reg': L_reg.item(),
    'param_error': compute_param_error(theta, theta_true),   # if known
    'u_rel_l2': compute_rel_l2(u_pred, u_exact),             # if known
    'theta': {k: v.item() for k, v in theta.named_params()},
}
```

Convergence criteria:
- `param_rel_error < 5%` for each identified parameter
- `L_data < sigma_DIC²` (data residual at noise floor)
- `L_equil < 1e-4` (equilibrium satisfied)
- Parameters stable over last 10 Schwarz iters (std/mean < 1%)

---

## Common Failure Modes

| Failure | Diagnosis | Fix |
|---------|-----------|-----|
| Parameters diverge to ±∞ | Ill-conditioned inverse | Add regularization; add load cases |
| Parameters converge to wrong value | Local minimum | Try different initialization; use warm start from simpler model |
| u correct but Θ wrong | Non-unique inverse | More independent observations; physics-based prior |
| L_equil plateaus high | PDE not satisfied | Check Jacobian pullback; verify P formula |
| L_data plateaus above noise floor | Model mismatch | Wrong material model family; check kinematics |
| Oscillating parameter estimates | Too large LR for Θ | Use lower LR for Θ than for u |
| RBM constraint not effective | Wrong reference point | Use multiple reference constraints (translation + rotation separately) |
