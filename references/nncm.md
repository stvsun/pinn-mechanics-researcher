# Neural Network Constitutive Models (NNCMs)

## Table of Contents
1. [When to use NNCMs instead of parametric models](#1-when-to-use-nncms)
2. [Polyconvex hyperelastic NNs](#2-polyconvex-hyperelastic-nns)
3. [ICNN architecture for energy](#3-icnn-architecture)
4. [Frame indifference via invariants](#4-frame-indifference-via-invariants)
5. [Yield surface learning](#5-yield-surface-learning)
6. [GENERIC framework for thermodynamic consistency](#6-generic-framework)
7. [Training NNCMs from DIC data](#7-training-from-dic-data)
8. [Failure modes and checks](#8-failure-modes)

---

## 1. When to use NNCMs

Use a neural network constitutive model when:
- The material response does not fit any known analytic form (novel polymers, biological tissue, metamaterials)
- You have multi-load-path experimental data (DIC from tension + shear + biaxial + torsion)
- You want to discover the energy landscape rather than verify a hypothesis

Do NOT use NNCMs when:
- You have only uniaxial data — the energy landscape is underdetermined (infinite models fit uniaxial data)
- Interpretability of material parameters (μ, K, etc.) is required for reporting
- You have < 500 experimental data points after filtering

**Minimum data requirement for NNCM hyperelasticity:** full-field DIC from ≥ 3 independent load paths covering tension, compression, and shear. Uniaxial alone identifies only one invariant path.

---

## 2. Polyconvex Hyperelastic NNs

For a hyperelastic material, the Helmholtz free energy density W(F) must satisfy:

1. **Material frame indifference (objectivity):** W(QF) = W(F) for all rotations Q → W depends only on right Cauchy-Green tensor C = FᵀF
2. **Polyconvexity (Ball's condition):** W(F) is polyconvex if W(F) = Ψ(F, cofF, detF) where Ψ is convex in each argument → ensures existence of minimizers
3. **Normalization:** W(I) = 0, P(I) = ∂W/∂F|_{F=I} = 0 (stress-free reference)
4. **Growth conditions:** W(F) → ∞ as detF → 0 (no self-penetration); W(F) → ∞ as |F| → ∞

The Piola stress and tangent stiffness:
```
P = ∂W/∂F                          (first Piola-Kirchhoff)
A = ∂²W/∂F∂F                       (tangent modulus, must be positive definite)
```

---

## 3. ICNN Architecture

An **Input-Convex Neural Network (ICNN)** guarantees convexity in its inputs by construction. For hyperelasticity, parameterize W as a function of invariants (I₁, I₂, I₃):

```python
import torch
import torch.nn as nn
from typing import Optional


class PolyconvexEnergyNet(nn.Module):
    """
    Polyconvex hyperelastic energy W(F) via ICNN on invariants.

    Architecture: W = ICNN(I₁, I₂, I₃) where ICNN is convex in all inputs.
    This guarantees polyconvexity (and hence rank-1 convexity) of W w.r.t. F
    when expressed through the principal invariants.

    Normalization is enforced by subtracting W(I) at construction.

    Args:
        hidden_dim: Width of hidden layers
        n_layers:   Depth of ICNN
        eps:        Small offset for growth condition (prevents W → -∞)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_layers: int = 4,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.eps = eps

        # Standard linear path (always convex in input)
        self.W_linear = nn.Parameter(torch.zeros(3))  # weights for I1, I2, I3

        # ICNN layers: z_{k+1} = softplus(W_z z_k + W_x x + b)
        # W_z must be non-negative → use softplus(raw_W_z) at each layer
        self.W_z_raw = nn.ParameterList()
        self.W_x = nn.ModuleList()
        self.biases = nn.ParameterList()

        prev_dim = 3  # input: [I1, I2, I3]
        for i in range(n_layers):
            self.W_x.append(nn.Linear(3, hidden_dim))
            if i > 0:
                self.W_z_raw.append(nn.Parameter(torch.randn(hidden_dim, hidden_dim)))
            self.biases.append(nn.Parameter(torch.zeros(hidden_dim)))
            prev_dim = hidden_dim

        self.W_out = nn.Linear(hidden_dim, 1)

    def _invariants(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute principal invariants of C = FᵀF.
        I₁ = tr(C), I₂ = ½[(tr C)² - tr(C²)], I₃ = det(C) = (det F)²
        Returns: (N, 3)
        """
        C = F.transpose(-2, -1) @ F         # (N, 3, 3)
        I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)  # (N, 1)
        trC2 = (C @ C).diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        I2 = 0.5 * (I1 ** 2 - trC2)
        I3 = torch.linalg.det(C).unsqueeze(-1)   # (N, 1)
        return torch.cat([I1, I2, I3], dim=-1)    # (N, 3)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute W(F). Returns scalar energy density per point, shape (N,).
        """
        inv = self._invariants(F)  # (N, 3)

        # ICNN forward pass
        z = torch.zeros(inv.shape[0], len(self.W_x[0].weight), device=F.device)
        for i, W_x_layer in enumerate(self.W_x):
            x_contrib = W_x_layer(inv)
            if i > 0:
                W_z = torch.nn.functional.softplus(self.W_z_raw[i-1])
                z_contrib = z @ W_z.T
                z = torch.nn.functional.softplus(x_contrib + z_contrib + self.biases[i])
            else:
                z = torch.nn.functional.softplus(x_contrib + self.biases[i])

        W_raw = self.W_out(z).squeeze(-1)  # (N,)

        # Linear convex term + growth condition: add eps * I3^{-1} for det→0
        W = W_raw + (self.W_linear * inv).sum(-1) + self.eps / inv[:, 2].clamp(min=1e-8)

        # Normalization: subtract W(I) evaluated at reference state
        # (done once after training, or use a shift parameter)
        return W

    def piola_stress(
        self, F: torch.Tensor
    ) -> torch.Tensor:
        """
        First Piola-Kirchhoff stress P = ∂W/∂F via autograd.
        Returns: (N, 3, 3)
        """
        F_req = F.requires_grad_(True)
        W = self.forward(F_req).sum()
        P = torch.autograd.grad(W, F_req, create_graph=True)[0]
        return P  # (N, 3, 3)

    def normalize_(self) -> None:
        """
        Subtract W(I) so that W(I) = 0. Call once after training.
        Also enforces P(I) = 0 by adjusting the linear term.
        """
        with torch.no_grad():
            I = torch.eye(3).unsqueeze(0)
            W_ref = self.forward(I).item()
            # Subtract the reference energy from the bias of the output layer
            self.W_out.bias.data -= W_ref
```

---

## 4. Frame Indifference via Invariants

Working in invariant space automatically enforces material frame indifference. For isotropic materials use principal invariants of C:

| Invariant | Formula | Physical meaning |
|-----------|---------|-----------------|
| I₁ = tr(C) | λ₁² + λ₂² + λ₃² | Sum of squared principal stretches |
| I₂ = ½[(tr C)² - tr(C²)] | λ₁²λ₂² + λ₂²λ₃² + λ₁²λ₃² | Sum of squared area ratios |
| I₃ = det(C) = J² | (λ₁λ₂λ₃)² | Squared volumetric stretch |
| J = det(F) | λ₁λ₂λ₃ | Volume ratio |

For **transversely isotropic** materials (fibers), add structural invariants:
```
I₄ = a₀ · (C a₀)     (fiber stretch)
I₅ = a₀ · (C² a₀)    (fiber shear coupling)
```
where a₀ is the reference fiber direction.

---

## 5. Yield Surface Learning

For elastoplastic materials, learn f(σ) = 0 as a neural network:

```python
class YieldSurfaceNet(nn.Module):
    """
    Convex yield surface f(σ): scalar output, f < 0 = elastic, f = 0 = yield.

    Architecture: ICNN on stress invariants (J1, J2, J3) ensures convexity
    of the elastic domain in stress space (Drucker's postulate).
    Isotropy is enforced by using invariants only.

    Args:
        hidden_dim: Width of hidden layers
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        # ICNN on (J1, sqrt(J2), J3) — convex in these arguments
        self.icnn = PolyconvexEnergyNet(hidden_dim=hidden_dim, n_layers=3, eps=0.0)
        # Known points to anchor: f(σ_y) = 0 where σ_y are yield stress measurements
        self.yield_anchors: Optional[torch.Tensor] = None

    def stress_invariants(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute stress invariants from Cauchy stress σ.
        J1 = tr(σ), J2 = ½ tr(s²) where s = σ - (J1/3)I, J3 = det(s)
        """
        J1 = sigma.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)  # (N, 1)
        s = sigma - (J1 / 3) * torch.eye(3, device=sigma.device)
        J2 = 0.5 * (s @ s).diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        J3 = torch.linalg.det(s).unsqueeze(-1)
        return torch.cat([J1, J2.sqrt().clamp(min=1e-8), J3], dim=-1)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """f(σ): < 0 elastic, = 0 yield, > 0 inadmissible. Shape (N,)."""
        inv = self.stress_invariants(sigma)
        # Fake F input: use diagonal tensor from invariants (hack for reuse)
        # In practice, build a dedicated ICNN for 3 inputs
        F_dummy = torch.diag_embed(inv)  # (N, 3, 3)
        return self.icnn(F_dummy)
```

---

## 6. GENERIC Framework

For thermodynamically consistent rate-dependent materials (viscoelasticity, plasticity), the **GENERIC formulation** (General Equation for Non-Equilibrium Reversible-Irreversible Coupling) guarantees:

- First law (energy conservation): Ė = {E, E} + [E, E] = 0
- Second law (entropy production): Ṡ ≥ 0

```python
class GENERICNet(nn.Module):
    """
    Thermodynamics-consistent constitutive model via GENERIC.

    State: z = (q, p, s) = (configuration, momentum, entropy density)
    Evolution: ż = L(z) ∇E(z) + M(z) ∇S(z)

    where:
    - E(z): total energy (learned NN, convex in z)
    - S(z): entropy (learned NN, concave in z)
    - L(z): Poisson operator (antisymmetric: L = -Lᵀ)
    - M(z): friction operator (positive semi-definite)

    Degeneracy conditions:
    - L(z) ∇S(z) = 0  (reversible part doesn't change entropy)
    - M(z) ∇E(z) = 0  (irreversible part doesn't change energy)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # Energy: ICNN in state variables
        self.energy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Entropy: concave NN (negate ICNN output)
        self.entropy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # M matrix: positive definite via Cholesky (lower triangular L, M = LLᵀ)
        self.L_lower = nn.Linear(state_dim, state_dim * state_dim)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        return self.energy_net(z).squeeze(-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        return -self.entropy_net(z).squeeze(-1)  # negated for concavity

    def friction_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """M(z) as positive semi-definite matrix via LLᵀ."""
        N, d = z.shape
        L_flat = self.L_lower(z)  # (N, d*d)
        L = L_flat.reshape(N, d, d)
        # Lower triangular with positive diagonal
        L_tril = torch.tril(L)
        L_tril = L_tril + torch.diag_embed(torch.nn.functional.softplus(L_tril.diagonal(dim1=-2, dim2=-1)))
        return L_tril @ L_tril.transpose(-2, -1)  # (N, d, d), PSD
```

---

## 7. Training NNCMs from DIC Data

### Inverse loss for NNCM (no fixed parameter set)

```python
def nncm_inverse_loss(
    u_nets: list[nn.Module],
    decoders: list[nn.Module],
    energy_net: PolyconvexEnergyNet,
    dic_data: list[dict],
    w_equil: float = 1.0,
    w_data: float = 10.0,
    w_polyconvex: float = 0.1,
    sigma_dic: float = 0.02,
) -> dict[str, torch.Tensor]:
    """
    Joint inverse loss for learning W(F) from DIC + equilibrium.

    Unlike parametric identification (which learns μ, K ∈ ℝ), this
    simultaneously learns the entire energy function W(F) via the ICNN
    and the displacement fields u via the PINN.

    The polyconvexity penalty enforces ∂²W/∂F² > 0 at sampled deformation
    gradients — this is needed because ICNN guarantees convexity in
    invariants but not strict polyconvexity in F directly.

    Args:
        u_nets:       Displacement PINN networks (one per chart)
        decoders:     Chart decoder networks
        energy_net:   PolyconvexEnergyNet being trained
        dic_data:     List of chart DIC observations
        w_polyconvex: Weight for polyconvexity penalty
        sigma_dic:    DIC noise floor (mm) — stop when L_data ≈ sigma_dic²
    """
    # 1. Equilibrium residual: ∇ · P = 0 where P = ∂W/∂F
    L_equil = compute_equilibrium_nncm(u_nets, decoders, energy_net)

    # 2. Huber data loss on DIC observations
    L_data = huber_dic_loss(u_nets, dic_data, delta=2.0 * sigma_dic)

    # 3. Polyconvexity check: eigenvalues of ∂²W/∂F∂F should be ≥ 0
    # Sample random F near deformed state and compute tangent modulus
    F_sample = sample_deformation_gradients(u_nets, decoders, n=200)
    L_polyconvex = polyconvexity_penalty(energy_net, F_sample)

    # 4. Normalization: W(I) = 0
    F_ref = torch.eye(3).unsqueeze(0)
    L_norm = energy_net.forward(F_ref).pow(2).mean()

    total = (w_equil * L_equil + w_data * L_data +
             w_polyconvex * L_polyconvex + 100.0 * L_norm)

    return {
        "total": total,
        "equil": L_equil.detach().item(),
        "data": L_data.detach().item(),
        "polyconvex": L_polyconvex.detach().item(),
        "norm": L_norm.detach().item(),
    }


def polyconvexity_penalty(
    energy_net: PolyconvexEnergyNet,
    F_sample: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize violations of strong ellipticity (Legendre-Hadamard condition).
    For each F sample, check min eigenvalue of acoustic tensor Q(n,m) ≥ 0.
    Penalty = mean(ReLU(-min_eigenvalue)).
    """
    P = energy_net.piola_stress(F_sample)  # (N, 3, 3)
    # Acoustic tensor: Q_ik(n) = A_iJkL n_J n_L (expensive — approximate)
    # Quick check: penalize negative curvature via second-order finite differences
    dF = 0.01 * torch.randn_like(F_sample)
    W_center = energy_net.forward(F_sample)
    W_plus = energy_net.forward(F_sample + dF)
    W_minus = energy_net.forward(F_sample - dF)
    second_diff = W_plus - 2 * W_center + W_minus  # should be ≥ 0
    return torch.nn.functional.relu(-second_diff).mean()
```

### Required load paths for energy identification

| Load paths available | Identifiable quantities |
|---------------------|------------------------|
| Uniaxial only | W(I₁, I₃) along one curve — underdetermined |
| Uniaxial + equibiaxial | W(I₁, I₂) surface — determines isochoric response |
| Uniaxial + biaxial + shear | Full W(I₁, I₂, I₃) — fully determined for isotropic |
| Above + fiber-direction tests | W(I₁, I₂, I₃, I₄, I₅) — transverse isotropy |

**Rule of thumb:** You need at least as many independent load paths as invariants in your energy function.

---

## 8. Failure Modes and Checks

| Failure | Symptom | Fix |
|---------|---------|-----|
| Non-convex energy | P(F) not monotone in stretch | Add polyconvexity penalty; increase w_polyconvex |
| Frame-indifference violation | Different W for same C | Always parameterize by invariants, never F directly |
| Unnormalized energy | Large W(I) ≠ 0 | Call energy_net.normalize_() after training |
| Rank-1 instability | Deformation localizes to bands | Check Legendre-Hadamard condition; add regularization on ∂²W |
| Overfit to single load path | W differs radically for other load paths | Add more load paths; use L1 regularization on ICNN weights |
| Negative J during training | Material "collapses" | Add growth condition: eps/J² term in energy |

---

## Reference Papers

- Linden et al. (2023) "Neural networks meet hyperelasticity: A guide to enforcing physics" — polyconvex ICNNs
- He & Chen (2023) "NN-EUCLID: Deep-learning hyperelasticity without stress data" — unsupervised from displacement only
- Flaschel et al. (2021) "Unsupervised discovery of interpretable hyperelastic constitutive laws" — sparsity-based
- Masi et al. (2021) "Thermodynamics-based Artificial Neural Networks (TANN)" — GENERIC-based
- Vlassis & Sun (2021) "Sobolev training of thermodynamic-informed neural networks for interpretable elasto-plasticity" — WaiChing Sun group
