# pinn-mechanics-researcher

A [Claude Code](https://claude.ai/claude-code) skill that activates expert-level reasoning for physics-informed neural network (PINN) mechanics research, in the style of [WaiChing Sun's group](https://www.engineering.columbia.edu/faculty/waiching-sun) at Columbia University.

## What it does

When installed, this skill makes Claude respond as an established computational mechanics researcher who combines rigorous continuum mechanics theory with modern PyTorch programming. It covers the full pipeline from geometry-aware PINN formulation to inverse material identification from experimental data.

**Be blunt and precise** is the core response style — expect exact numbers, specific hyperparameter values, and direct diagnosis rather than hedged suggestions.

---

## Capabilities

### 1. Forward BVP Solvers on Complex 3D Geometry
- Schwarz alternating method with multi-chart coordinate atlas
- Jacobian pullback of PDE residuals through chart decoders (chain rule, autograd)
- Interior supervised pretraining (batch=2048 mandatory)
- Degenerate Jacobian filtering (`sigma_floor=1e-3`, `kappa_max=1e3`)
- N-dimensional chart framework: 3D volume, 2D thin shell, 4D space-time, 1D rod/beam

### 2. Inverse Material Identification from DIC
- Full-field displacement data from stereo DIC → material parameters
- Rigid-body motion subtraction (6-DOF least-squares)
- ICP coordinate alignment to simulation mesh
- Chart coordinate mapping via Newton inversion of decoder
- Joint loss: `L_equil + w_data·L_data + w_rbm·L_rbm + w_reg·L_reg`
- Identifiability analysis (sensitivity matrix κ check before training)

### 3. DIC Software Format Support

Loaders for all major DIC packages with automatic format detection from file extension and header:

| Software | Format | Noise floor (σ_u) |
|----------|--------|-------------------|
| Correlated Solutions Vic-3D | CSV | 0.01–0.05 mm |
| Correlated Solutions Vic-2D | CSV (pixels) | 0.01–0.05 mm |
| GOM ARAMIS | CSV (`;`-delimited) or HDF5 | 0.005–0.03 mm |
| Dantec ISTRA 4D | CSV | 0.01–0.05 mm |
| DICe (Sandia, open-source) | `.txt` | 0.02–0.1 px |
| ncorr (MATLAB) | `.mat` | 0.01–0.08 px |
| MatchID | CSV or `.xlsx` | 0.005–0.02 mm |
| Generic | any XYZ+UVW CSV | — |

### 4. Neural Network Constitutive Models (NNCMs)
- **Polyconvex hyperelastic NNs**: ICNN on principal invariants (I₁, I₂, I₃) — Ball's condition guaranteed
- **Yield surface learning**: convex ICNN on stress invariants (J₁, √J₂, J₃)
- **GENERIC framework**: thermodynamically consistent rate-dependent models (1st + 2nd law)
- Frame indifference enforced by invariant parameterization
- Growth conditions, normalization, polyconvexity penalty
- Key reference: Vlassis & Sun (2021), He & Chen (2023), Linden et al. (2023)

### 5. Material Model Library (Parametric)
| Model | Parameters | Use case |
|-------|-----------|----------|
| Neo-Hookean | μ, K | Rubbers, soft tissue |
| Arruda-Boyce | μ, N | Polymers (chain-based) |
| TI Arruda-Boyce | μ_m, N_m, κ_f, N_f | Fiber-reinforced soft tissue |
| Kelvin-Voigt | E, η | Rate-dependent viscoelastic |
| Phase-field damage | G_c, ℓ | Brittle fracture |
| Biot poromechanics | K_u, G, α, M | Saturated porous media |

### 6. Autonomous Benchmark Suite
- pytest MMS (Method of Manufactured Solutions) regression tests
- Per-material objectivity tests (W(QF) = W(F))
- Daily convergence benchmarks: `< 2% rel-L²` on sphere/cube in `< 60s CPU`
- Cron / GitHub Actions scheduling templates

### 7. Convergence Debugging
Blunt diagnosis of multi-chart PINN failure modes:

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| L_if oscillates | w_if too low; MLP arch | Raise w_if 5×; switch to CompactChartNet |
| One chart 3× worse | Dead init + degenerate Jacobian | 10k pretrain epochs, batch=2048; audit κ(J) |
| NaN in PDE loss | Degenerate chart region | Raise sigma_floor to 5e-3 |
| Params diverge (inverse) | Ill-conditioned problem | Add load cases; fix K from prior |

---

## Installation

### From this repo

```bash
git clone https://github.com/stvsun/pinn-mechanics-researcher
claude skill install pinn-mechanics-researcher/
```

### Global install (available in all Claude Code projects)

```bash
claude skill install --global pinn-mechanics-researcher/
```

### From the packaged `.skill` file

```bash
claude skill install pinn-mechanics-researcher.skill
```

---

## Trigger phrases

The skill auto-activates when your message contains any of:

- `Schwarz PINN`, `atlas charts`, `coordinate chart PINN`
- `material identification PINN`, `DIC-PINN`, `WaiChing Sun`
- `inverse elasticity PINN`, `Arruda-Boyce identification`
- `benchmark PINN`, `regression test PINN`
- `NNCM`, `polyconvex neural network`, `ICNN hyperelasticity`
- `GENERIC framework`, `data-driven constitutive`

It also triggers for contextual requests like: implementing a Schwarz solver, integrating DIC data into a PINN, debugging chart-specific convergence failures, or writing material model tests.

---

## Repository structure

```
pinn-mechanics-researcher/
├── SKILL.md                        # Main skill — response style, quick-reference tables,
│                                   # core algorithms, hyperparameter guide
└── references/
    ├── material-models.md          # Parametric constitutive laws with PyTorch code
    ├── nncm.md                     # Neural network constitutive models (ICNN, GENERIC,
    │                               # yield surface, polyconvexity, GENERIC framework)
    ├── dic-integration.md          # DIC loaders for 8 software packages, RBM subtraction,
    │                               # ICP alignment, Huber loss, noise floor table
    ├── inverse-problem-guide.md    # Identifiability analysis, multi-load-case strategy,
    │                               # spatial parameter fields, convergence monitoring
    └── schwarz-theory.md           # Convergence theory (Lions 1988), trust-region filter,
                                    # chart coloring, adaptive weight scheduling
```

---

## Example prompts

```
I need to implement a linear elasticity PINN on a 3D gear geometry (PLY mesh)
using 6 overlapping Schwarz charts. How do I pull back the PDE residual through
the chart decoder Jacobian? Give me PyTorch code.
```

```
I ran stereo DIC (Vic-3D) on a silicone cylinder under uniaxial tension.
I want to identify Neo-Hookean μ and K. Walk me through the full pipeline:
DIC loading, rigid-body subtraction, chart mapping, inverse loss, and
identifiability analysis.
```

```
My 8-chart Schwarz PINN is stuck at 8.2% rel-L². Chart 3 is at 15.6%,
L_if = 0.09 and oscillating. I'm using MLP with interior-pretrain-epochs=0
and w_if=0.2. What is wrong?
```

```
I want to learn the hyperelastic energy functional W(F) directly from
multi-load-path DIC data without assuming a parametric form.
How do I set up the ICNN and the joint inverse loss?
```

---

## Background

This skill was developed for research code in the style of WaiChing Sun's Computational Poromechanics Lab at Columbia, which has produced foundational work on:

- Data-driven and machine-learning enhanced constitutive modeling
- PINN-based solvers for solid and porous media mechanics
- Geometric deep learning for mechanics on complex domains
- Thermodynamics-informed neural network material models

Key references this skill draws on:

- Sun, W. et al. — Multiscale modeling and data-driven constitutive modeling
- Vlassis, N. & Sun, W. (2021) — Sobolev training of thermodynamic-informed neural networks
- He, X. & Chen, J.S. (2023) — NN-EUCLID unsupervised hyperelasticity
- Linden, L. et al. (2023) — Neural networks meet hyperelasticity
- Lions, P.L. (1988) — On the Schwarz alternating method

---

## Contributing

Pull requests welcome. Key areas to expand:

- Additional DIC software loaders (Dantec iSTRUCT, Imetrum, LaVision)
- Rate-dependent NNCMs (viscohyperelastic, finite-strain plasticity)
- 4D space-time chart implementation
- Poromechanics inverse problems (Biot identification from pore pressure + displacement)
- Integration with FEniCSx or JAX-FEM for FEM-PINN hybrid solvers

---

## License

MIT
