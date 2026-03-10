# Material Model Library for PINN-Based Identification

## Notation

| Symbol | Meaning |
|--------|---------|
| X, x | Reference, current position |
| F = ∂x/∂X | Deformation gradient |
| J = det(F) | Jacobian (volume ratio) |
| C = FᵀF | Right Cauchy-Green tensor |
| b = FFᵀ | Left Cauchy-Green tensor |
| I₁,I₂,I₃ | Invariants of C |
| P | First Piola-Kirchhoff stress |
| S | Second Piola-Kirchhoff stress (S = 2∂Ψ/∂C) |
| τ | Kirchhoff stress (τ = FSFᵀ) |
| σ | Cauchy stress (σ = J⁻¹τ) |

Equilibrium in reference config: ∇_X · P + ρ₀ B = 0

---

## 1. Neo-Hookean (Compressible)

```
Ψ(F) = μ/2 (I₁ - 3) - μ ln J + K/2 (ln J)²
P = μ(F - F⁻ᵀ) + K ln(J) F⁻ᵀ
```

Parameters: shear modulus μ > 0, bulk modulus K > 0

**PINN tip**: autograd through Ψ is cleanest:
```python
def neo_hookean_psi(F, mu, K):
    C = F.T @ F
    I1 = torch.trace(C)
    J = torch.det(F)
    return mu/2 * (I1 - 3) - mu * torch.log(J) + K/2 * torch.log(J)**2

# Then: P = autograd(psi, F)
```

Patch tests: uniaxial tension, simple shear, hydrostatic compression.

---

## 2. Mooney-Rivlin

```
Ψ(F) = C₁(I₁ - 3) + C₂(I₂ - 3) + K/2(J - 1)²
I₂ = (I₁² - tr(C²))/2
P = 2(C₁ + C₂ I₁) F - 2C₂ F C + K(J-1) J F⁻ᵀ
```

Parameters: C₁, C₂ > 0, K > 0

Identifiability: C₁ and C₂ only separable with both uniaxial AND biaxial data.

---

## 3. Arruda-Boyce (8-chain)

```
Ψ = μ Σ_{k=1}^{5} αₖ/N^{k-1} (I₁ᵏ - 3ᵏ) + K/2(J - 1)²
αₖ = {1/2, 1/20, 11/1050, 19/7000, 519/673750}
λ_chain = √(I₁/3),  λ_lock = √N
```

Parameters: μ (small-strain shear), N (chain length, controls locking)

**PINN approach**: implement the Padé approximant for the inverse Langevin function:
```python
def inv_langevin(x):
    # Padé [3/2] approximant, accurate for x < 0.85
    return x * (3 - x**2) / (1 - x**2)
```

---

## 4. Transversely Isotropic Arruda-Boyce (TI-AB)

Reference implementation: `run_rabbit_inverse_ti_arruda_boyce_atlas_schwarz.py`

```
Ψ = Ψ_matrix(I₁, J; μ_m, N_m) + Ψ_fiber(I₄; κ_f, N_f)
I₄ = a₀ · C a₀    (a₀ = preferred fiber direction)
Ψ_fiber only active when I₄ > 1 (fiber extension, not compression)
```

Parameters: matrix stiffness μ_m, matrix chain length N_m,
fiber stiffness κ_f, fiber chain length N_f

**Identifiability**: need load cases with fiber both loaded and sheared.
True values in reference: μ_m=1.80, N_m=4.80, κ_f=2.20, N_f=5.20.

---

## 5. Viscoelastic (Kelvin-Voigt / Maxwell)

### Kelvin-Voigt (small deformation):
```
σ = E ε + η ε̇
P_visc = ∂Ψ_el/∂F + η ∂²Ψ_el/(∂F∂t)
```

### Generalized Maxwell (finite strain):
```
Ψ = Ψ_∞(C) + Σᵢ Ψᵢ(Cᵢₑ)
Cᵢₑ = FᵢᵉᵀFᵢᵉ   (internal variable, elastic part of ith arm)
Evolution: Ḟᵢᵛ = 1/(2ηᵢ) devSᵢ Fᵢᵛ   (flow rule)
```

Parameters: long-term modulus E_∞, arm moduli {Eᵢ}, relaxation times {τᵢ = ηᵢ/Eᵢ}

**PINN for viscoelastic**: requires time integration of internal variable Fᵢᵛ.
Use RK4 over discrete time steps; feed internal variables as extra PINN inputs.
Increase identifiability by using cyclic loading (different frequencies) as load cases.

---

## 6. Phase-Field Damage (Bourdin-Francfort)

```
E(u, d) = ∫_Ω (1-d)² Ψ_+(F) dV + ∫_Ω G_c(d²/2ℓ + ℓ/2|∇d|²) dV
```

Coupled problem: solve for displacement u and phase field d simultaneously.
- u equation: ∇·((1-d)² P_+(F)) = 0
- d equation: G_c(d/ℓ - ℓ Δd) = (1-d) ∂Ψ_+/∂d ... 2(1-d)Ψ_+

Parameters: critical energy release rate G_c, regularization length ℓ

**Spectral split**: decompose Ψ into tension (Ψ_+) and compression (Ψ_-) parts
to prevent crack interpenetration (Miehe split or volumetric-deviatoric split).

For PINN: monitor d ∈ [0,1] via sigmoid output layer; add irreversibility constraint
`L_irrev = relu(d_prev - d_new)²` if running sequential loading steps.

---

## 7. Biot Poromechanics

Two fields: solid displacement u, pore pressure p

```
Mechanical: ∇·(σ_eff - α p I) = 0
           σ_eff = 2G ε + λ tr(ε) I    (linear elastic skeleton)
           ε = sym(∇u)

Flow:  α ∂(∇·u)/∂t + (1/M) ∂p/∂t = ∇·(K/η_f ∇p) + f_p
       K = permeability tensor, η_f = fluid viscosity, M = Biot modulus
```

Parameters to identify: K (permeability), G (shear modulus), α (Biot coefficient)

**PINN for Biot**: two-network approach:
```python
u_pinn = VectorPINN(xi)    # displacement
p_pinn = ScalarPINN(xi)    # pore pressure

L_mech = mean(div_sigma_eff - alpha * grad_p)**2
L_flow = mean(alpha * div_u_dot + p_dot/M - div(K/eta * grad_p))**2
```

For steady-state: drop time derivatives. Elder flow is a related convection-diffusion
problem (temperature-driven buoyancy).

---

## Adding a New Material Model: Checklist

1. **Write Ψ(F, Θ)** analytically; identify all material parameters Θ
2. **Implement in PyTorch** using autograd to derive P:
   ```python
   F.requires_grad_(True)
   psi = my_psi(F, theta)
   P = torch.autograd.grad(psi, F, create_graph=True)[0]
   ```
3. **Verify objectivity**: generate random rotation Q; assert Ψ(QF) ≈ Ψ(F)
4. **Patch tests**: run 1D uniaxial tension to closed-form solution:
   - Linear elastic limit (small strain): compare to E, ν
   - Lock-up (Arruda-Boyce): compare to √N limit
5. **Identifiability**: construct S = ∂u_obs/∂Θ, check κ(S) < 1e4
6. **Regularization**: add `L_reg = w_reg * sum(p**2 for p in theta_params)` or
   spatial smoothness for field-varying Θ(x)
