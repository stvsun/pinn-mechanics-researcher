# Schwarz Domain Decomposition for Multi-Chart PINNs

## Classical Schwarz (Mathematical Background)

The multiplicative Schwarz method solves an elliptic BVP on Ω = ∪ᵢ Ωᵢ (overlapping)
by iterating local solves sequentially:

```
Given u^k on Ω:
For i = 1, ..., N_charts:
    Solve local BVP on Ωᵢ with boundary data from u^k on ∂Ωᵢ ∩ (Ω \ Ωᵢ)
    Update u^k on Ωᵢ → u^{k+1} on Ωᵢ
```

**Convergence theorem** (Lions 1988, additive; Xu 1992, multiplicative):
For elliptic operators L = -∇·(a∇·), the Schwarz iteration converges geometrically
at rate ρ < 1 that depends on the overlap size δ:

```
‖u^{k+1} - u*‖ ≤ ρ ‖u^k - u*‖,    ρ = C/(1 + C δ²/H²)
```

More overlap → better convergence (up to H, the chart diameter).
This motivates generous overlap (typically 20–30% of chart radius).

---

## PINN Schwarz Implementation

The PINN variant replaces exact local solves with minimization over PINN parameters
for `local_steps` gradient steps:

```python
for schwarz_iter in range(max_schwarz_iters):

    # Colored Gauss-Seidel ordering (group non-adjacent charts first)
    for color_group in coloring:
        for chart_i in color_group:
            for local_step in range(local_steps):

                # Sample interior (PDE), boundary (BC), interface points
                xi_int = sample_interior(chart_i)
                xi_bc  = sample_boundary(chart_i)
                xi_if  = sample_interface(chart_i)   # overlap with neighbors

                # Compute per-chart loss
                L_pde = pde_loss(pinn_i, decoder_i, xi_int)
                L_bc  = bc_loss(pinn_i, decoder_i, xi_bc)

                # Interface: use FROZEN neighbor predictions
                with torch.no_grad():
                    u_neighbors = [pinn_j(xi_if_j) for j in neighbors_of_i]

                L_iv = value_interface_loss(pinn_i, u_neighbors, xi_if)
                L_if = flux_interface_loss(pinn_i, u_neighbors, xi_if)

                L_total = w_pde*L_pde + w_bc*L_bc + w_iv*L_iv + w_if*L_if
                optimizer_i.zero_grad()
                L_total.backward()
                optimizer_i.step()

    # Global evaluation (all charts together)
    rel_l2 = evaluate_global(pinn_nets, decoders, eval_pts, u_exact)
```

Key: neighbors are **frozen** during chart i's update (multiplicative, not additive).
This is why the order matters — prioritize charts with poor local errors first.

---

## Chart Coloring for Parallel Efficiency

To parallelize within a Schwarz iteration, color charts so no two same-color charts
are neighbors (share an interface). Then all same-color charts can update simultaneously
(like graph coloring for finite elements):

```python
def color_charts(adjacency: dict) -> list[list[int]]:
    """Greedy graph coloring for non-adjacent parallel groups."""
    colors = {}
    for i in sorted(adjacency.keys()):
        neighbor_colors = {colors[j] for j in adjacency[i] if j in colors}
        colors[i] = next(c for c in range(len(adjacency)) if c not in neighbor_colors)

    n_colors = max(colors.values()) + 1
    groups = [[i for i, c in colors.items() if c == k] for k in range(n_colors)]
    return groups
```

Typical: 5–8 color groups for an 8-chart Stanford Bunny atlas.

---

## Trust-Region Filter

To prevent the PINN Schwarz from accepting steps that increase global error:

```python
def trust_region_update(pinn_nets, prev_state, rel_l2_proposed,
                        best_rel_l2, threshold_factor=1.03):
    """Accept update only if global error does not increase by more than factor."""
    if rel_l2_proposed > threshold_factor * best_rel_l2:
        # Reject: restore previous state
        for net, state in zip(pinn_nets, prev_state):
            net.load_state_dict(state)
        return False, best_rel_l2

    if rel_l2_proposed < best_rel_l2:
        best_rel_l2 = rel_l2_proposed
        save_checkpoint(pinn_nets, 'best_rel_l2')

    return True, best_rel_l2
```

The factor of 1.03 (3% tolerance) prevents premature stopping due to oscillation
while protecting against large regressions.

---

## Interface Condition Types

### 1. Dirichlet Interface (default)
```
u_i(x) = u_j(x)    on  Γ_{ij} = ∂Ωᵢ ∩ Ωⱼ
L_iv = ‖u_i(ξ_if) - u_j(ξ_if)‖²
```
Simple, but can oscillate for stiff problems. Works well for Poisson.

### 2. Neumann / Flux Interface
```
∇u_i · n_{ij} = ∇u_j · n_{ij}    on  Γ_{ij}
L_if = ‖(∇u_i - ∇u_j) · n_{ij}‖²
```
Should be satisfied alongside L_iv; adds one more convergence constraint.
Typically needs lower weight early (0.1) and increasing schedule.

### 3. Robin Interface (for flow problems)
```
λ ∇p_i · n + (p_i - p_j) = 0    on  Γ_{ij}
L_robin = ‖λ ∇p_i · n + p_i - p_j‖²
```
λ = 0: Dirichlet (value match). λ → ∞: Neumann (flux match).
λ ~ h (mesh size) is typically optimal; tune between 0.1–1.0.
Reduces oscillation for convection-dominated / coupled flow problems.

---

## Adaptive Loss Weight Scheduling

```python
class AdaptiveWeightScheduler:
    """Increase interface weights when Schwarz stagnates."""

    def __init__(self, w_iv_init=0.8, w_if_init=0.2, growth_factor=1.5,
                 patience=5, max_w_if=5.0):
        self.w_iv = w_iv_init
        self.w_if = w_if_init
        self.growth = growth_factor
        self.patience = patience
        self.max_w_if = max_w_if
        self.stale_count = 0
        self.best = float('inf')

    def step(self, if_residual):
        if if_residual < self.best * 0.99:
            self.best = if_residual
            self.stale_count = 0
        else:
            self.stale_count += 1

        if self.stale_count >= self.patience:
            self.w_if = min(self.w_if * self.growth, self.max_w_if)
            self.stale_count = 0
            print(f"  [adaptive] w_if raised to {self.w_if:.3f}")

        return self.w_iv, self.w_if
```

---

## Plateau Detection and Early Stopping

```python
class PlateauDetector:
    def __init__(self, patience=15, tol=5e-5):
        self.patience = patience
        self.tol = tol
        self.best = float('inf')
        self.stale = 0

    def step(self, rel_l2):
        if rel_l2 < self.best - self.tol:
            self.best = rel_l2
            self.stale = 0
        else:
            self.stale += 1

        if self.stale >= self.patience:
            return True   # stop
        return False
```

Typical: patience=15–20 Schwarz iterations; tol=5e-5 relative L² improvement.

---

## Convergence Diagnostics Checklist

After each experiment, check:

1. **Interface value residual** L_iv ≤ 1e-3: charts are coupled
2. **Interface flux residual** L_if ≤ 1e-2: physics is continuous
3. **Per-chart rel-L² breakdown**: maximum/mean ratio < 3 (otherwise one chart dominates)
4. **Trust-region accept rate**: < 50% acceptance → reduce learning rate or add pretrain
5. **Stale counter history**: stale jumps at specific iters → chart ordering issue
6. **PDE residual ramp**: L_pde should decrease monotonically after warmup period

If L_iv stagnates while L_pde decreases: increase w_iv or add overlap.
If L_pde stagnates while L_iv decreases: network capacity is limiting; widen chart PINN.
