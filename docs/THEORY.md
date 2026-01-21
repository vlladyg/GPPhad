# GPPhad: Mathematical Theory and Background

This document provides the theoretical foundation for the GPPhad library, covering Gaussian Process regression for thermodynamic modeling and phase diagram construction.

---

## Table of Contents

1. [Thermodynamic Framework](#1-thermodynamic-framework)
2. [Gaussian Process Regression](#2-gaussian-process-regression)
3. [Derivative Observations](#3-derivative-observations)
4. [Multi-Phase Modeling](#4-multi-phase-modeling)
5. [Phase Coexistence](#5-phase-coexistence)
6. [Uncertainty Propagation](#6-uncertainty-propagation)
7. [Active Learning](#7-active-learning)
8. [Zero-Point Energy Treatment](#8-zero-point-energy-treatment)
9. [Numerical Considerations](#9-numerical-considerations)

---

## 1. Thermodynamic Framework

### 1.1 Free Energy Representation

GPPhad models the Helmholtz free energy $F(T, V)$ through the dimensionless ratio:

$$\phi(T, V) = \frac{F(T, V)}{T}$$

This choice has several advantages:
1. **Thermal limit**: As $T \to 0$, $\phi \to -\infty$ gracefully handles the entropy contribution
2. **Smoothness**: $\phi$ varies more smoothly with temperature than $F$
3. **Natural derivatives**: Pressure and entropy have simple expressions

### 1.2 Thermodynamic Relations

All thermodynamic quantities derive from derivatives of $\phi$:

| Property | Symbol | Formula | Units (GPPhad) |
|----------|--------|---------|----------------|
| Free energy | $F$ | $T\phi$ | eV |
| Entropy | $S$ | $-T\frac{\partial\phi}{\partial T} - \phi$ | $k_B$ |
| Pressure | $P$ | $-T\frac{\partial\phi}{\partial V}$ | eV/Å³ |
| Internal energy | $E$ | $T^2\frac{\partial\phi}{\partial T}$ | eV |
| Heat capacity | $C_V$ | $-T\frac{\partial^2 F}{\partial T^2}$ | $k_B$ |
| Bulk modulus | $B$ | $-V\frac{\partial P}{\partial V} = TV\frac{\partial^2\phi}{\partial V^2}$ | eV/Å³ |
| Thermal expansion | $\alpha$ | $\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P$ | $k_B^{-1}$ |
| Grüneisen | $\gamma$ | $\frac{V\alpha B}{C_V}$ | — |

### 1.3 Reference Functions

The GP models deviations from known analytical references:

$$\phi(T, V) = \phi_{\text{ref}}(T, V) + f(T, V)$$

where $f \sim \mathcal{GP}(0, k)$.

**Liquid reference:**
$$\phi_{\text{ref}}^{\text{liq}} = \ln V$$

This captures the ideal gas entropy contribution.

**Solid reference (with E₀):**
$$\phi_{\text{ref}}^{\text{sol}} = -1 + \frac{3}{2}\ln(2\pi T) - \frac{E_0(V)}{T}$$

This includes:
- Classical harmonic oscillator entropy: $\frac{3}{2}\ln(2\pi T)$
- Zero-point energy: $-E_0(V)/T$

---

## 2. Gaussian Process Regression

### 2.1 GP Prior

A Gaussian Process defines a distribution over functions:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

where:
- $m(\mathbf{x})$ is the mean function (we use $m = 0$)
- $k(\mathbf{x}, \mathbf{x}')$ is the covariance (kernel) function

### 2.2 GP Posterior

Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $y_i = f(\mathbf{x}_i) + \epsilon_i$, the posterior at test point $\mathbf{x}_*$ is:

$$f(\mathbf{x}_*) | \mathcal{D} \sim \mathcal{N}(\mu_*, \sigma_*^2)$$

with:
$$\mu_* = \mathbf{k}_*^T (K + \Sigma)^{-1} \mathbf{y}$$
$$\sigma_*^2 = k_{**} - \mathbf{k}_*^T (K + \Sigma)^{-1} \mathbf{k}_*$$

where:
- $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ — training covariance matrix
- $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), \ldots, k(\mathbf{x}_N, \mathbf{x}_*)]^T$ — test-train covariances
- $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$ — test variance
- $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_N^2)$ — noise variances

### 2.3 Marginal Likelihood

The log marginal likelihood for hyperparameter optimization:

$$\log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^T (K + \Sigma)^{-1} \mathbf{y} - \frac{1}{2}\log|K + \Sigma| - \frac{N}{2}\log(2\pi)$$

GPPhad maximizes this to find optimal hyperparameters $\boldsymbol{\theta}$.

---

## 3. Derivative Observations

### 3.1 Derivative Kernels

Molecular dynamics simulations provide derivatives of $\phi$:
- Energy: related to $\frac{\partial\phi}{\partial T}$
- Pressure: related to $\frac{\partial\phi}{\partial V}$

For a GP, derivatives are also GPs with derived kernels:

$$\text{Cov}\left[\frac{\partial^{m+n} f}{\partial T^m \partial V^n}(\mathbf{x}), \frac{\partial^{p+q} f}{\partial T^p \partial V^q}(\mathbf{x}')\right] = \frac{\partial^{m+n+p+q} k(\mathbf{x}, \mathbf{x}')}{\partial T^m \partial V^n \partial T'^p \partial V'^q}$$

### 3.2 Notation

GPPhad uses notation `d_m_n` for $\frac{\partial^{m+n}\phi}{\partial T^m \partial V^n}$:

| Code | Derivative |
|------|------------|
| `d_0_0` | $\phi$ |
| `d_1_0` | $\frac{\partial\phi}{\partial T}$ |
| `d_0_1` | $\frac{\partial\phi}{\partial V}$ |
| `d_2_0` | $\frac{\partial^2\phi}{\partial T^2}$ |
| `d_0_2` | $\frac{\partial^2\phi}{\partial V^2}$ |
| `d_1_1` | $\frac{\partial^2\phi}{\partial T \partial V}$ |
| ... | ... |

### 3.3 Kernel Structure

The base kernel for liquids combines:

$$k(x_1, x_2) = \sigma_0^2 + \exp\left(-\frac{(T_1-T_2)^2}{2\ell_T^2} - \frac{(V_1^{-1}-V_2^{-1})^2}{2\ell_V^2} - \frac{(c_1^{-1}-c_2^{-1})^2}{2\ell_c^2}\right)\left(\frac{\sigma_1^2}{T_1 T_2} + \sigma_2^2 e^{-\frac{(T_1-T_2)^2}{2\ell_{T2}^2}}\right)$$

Hyperparameters:
- $\sigma_0, \sigma_1, \sigma_2$ — Amplitude parameters
- $\ell_T, \ell_V, \ell_c$ — Length scales for temperature, volume, cutoff

All derivative kernels are computed symbolically from this base.

---

## 4. Multi-Phase Modeling

### 4.1 Phase-Specific Kernels

Each phase has independent GP:

$$K_{\text{full}} = \begin{pmatrix} K_{\text{liq}} & 0 & 0 \\ 0 & K_{\text{fcc}} & 0 \\ 0 & 0 & K_{\text{bcc}} \end{pmatrix}$$

Phases are uncorrelated because different atomic arrangements lead to fundamentally different free energy landscapes.

### 4.2 Input Structure

Each observation point: $\mathbf{x} = (\text{phase}, \text{derivative}, T, V, \text{cutoff}, N)$

The covariance function dispatcher:

```python
def cf(x1, x2, th, f_dict):
    if x1[0] == x2[0]:  # Same phase
        func = f_dict[x1[0]][x1[1] + '_' + x2[1]]
        return func(x1[2:], x2[2:], th)
    else:  # Different phases
        return 0.0
```

### 4.3 Melted vs Non-Melted

**Melted (unified):** All phase data in single arrays
- Covariance: block-diagonal structure enforced by kernel
- Advantage: Can add inter-phase constraints (melting points)

**Non-melted (separated):** Per-phase arrays
- Advantage: Independent optimization
- Used during initial training

---

## 5. Phase Coexistence

### 5.1 Equilibrium Conditions

At phase coexistence, two phases $\alpha$ and $\beta$ satisfy:

1. **Thermal equilibrium**: $T_\alpha = T_\beta = T$
2. **Mechanical equilibrium**: $P_\alpha(T, V_\alpha) = P_\beta(T, V_\beta)$
3. **Chemical equilibrium**: $\mu_\alpha = \mu_\beta$

For pure substances, chemical equilibrium is equivalent to:
$$G_\alpha = G_\beta \quad \Leftrightarrow \quad F_\alpha + PV_\alpha = F_\beta + PV_\beta$$

### 5.2 Coexistence Equations

**P-T coexistence** (find T and volumes at given P):

$$\begin{cases}
T\frac{\partial\phi_\alpha}{\partial V}(T, V_\alpha) = -P \\
T\frac{\partial\phi_\beta}{\partial V}(T, V_\beta) = -P \\
\phi_\alpha(T, V_\alpha) - \phi_\beta(T, V_\beta) = \frac{P(V_\beta - V_\alpha)}{T}
\end{cases}$$

**T-P coexistence** (find volumes and P at given T):

$$\begin{cases}
\frac{\partial\phi_\alpha}{\partial V}(T, V_\alpha) = \frac{\partial\phi_\beta}{\partial V}(T, V_\beta) \\
\phi_\alpha - \phi_\beta = \frac{\partial\phi_\alpha}{\partial V}(V_\alpha - V_\beta)
\end{cases}$$

**Triple point** (three phases coexist):

$$\begin{cases}
\frac{\partial\phi_\alpha}{\partial V} = \frac{\partial\phi_\beta}{\partial V} \\
\phi_\alpha - \phi_\beta = \frac{\partial\phi_\alpha}{\partial V}(V_\alpha - V_\beta) \\
\frac{\partial\phi_\beta}{\partial V} = \frac{\partial\phi_\gamma}{\partial V} \\
\phi_\beta - \phi_\gamma = \frac{\partial\phi_\beta}{\partial V}(V_\beta - V_\gamma)
\end{cases}$$

### 5.3 Numerical Solution

GPPhad uses `scipy.optimize.fsolve` with variable transformation:

$$v = v_{\max}\frac{1 + t^2}{v_{\max}/v_{\min} + t^2}$$

This maps $t \in (-\infty, \infty)$ to $v \in (v_{\min}, v_{\max})$, ensuring solutions respect bounds.

---

## 6. Uncertainty Propagation

### 6.1 Error Propagation Framework

For implicit function $\mathbf{F}(\mathbf{x}, \boldsymbol{\phi}) = 0$ where $\boldsymbol{\phi}$ has uncertainty:

$$\delta\mathbf{x} = -\left(\frac{\partial\mathbf{F}}{\partial\mathbf{x}}\right)^{-1} \frac{\partial\mathbf{F}}{\partial\boldsymbol{\phi}} \delta\boldsymbol{\phi}$$

### 6.2 Jacobian Computation

For P-T coexistence, the Jacobian $\partial\mathbf{F}/\partial\mathbf{x}$:

$$\frac{\partial\mathbf{F}}{\partial\mathbf{x}} = \begin{pmatrix}
T\frac{\partial^2\phi_\alpha}{\partial V^2} & 0 & \frac{\partial\phi_\alpha}{\partial V} + T\frac{\partial^2\phi_\alpha}{\partial T\partial V} \\
\frac{\partial^2\phi_\alpha}{\partial V^2} & -\frac{\partial^2\phi_\beta}{\partial V^2} & \frac{\partial^2\phi_\alpha}{\partial T\partial V} - \frac{\partial^2\phi_\beta}{\partial T\partial V} \\
-V_\alpha\frac{\partial^2\phi_\alpha}{\partial V^2} & V_\beta\frac{\partial^2\phi_\beta}{\partial V^2} & \frac{\partial\phi_\alpha}{\partial T} - \frac{\partial\phi_\beta}{\partial T} - V_\alpha\frac{\partial^2\phi_\alpha}{\partial T\partial V} + V_\beta\frac{\partial^2\phi_\beta}{\partial T\partial V}
\end{pmatrix}$$

### 6.3 Variance Computation

The uncertainty in solution $\mathbf{x}$ depends on GP variance:

$$\Sigma_{\mathbf{x}} = J^{-1} \Sigma_{\mathbf{F}} (J^{-1})^T$$

where $\Sigma_{\mathbf{F}}$ is the covariance matrix of the residual function values, computed from GP predictions.

### 6.4 S₀ Contribution

For solid phases with zero-point energy, uncertainty includes contribution from E₀ GP:

$$\Sigma_{\text{total}} = \Sigma_{\text{main}} + \Sigma_{S_0}$$

where $\Sigma_{S_0}$ comes from the separate E₀ Gaussian Process.

---

## 7. Active Learning

### 7.1 Objective

Find the next simulation point $\mathbf{x}^*$ that maximally reduces uncertainty in target quantity (e.g., triple point temperature).

### 7.2 Information Gain

The acquisition function measures expected variance reduction:

$$\alpha(\mathbf{x}) = -\log\left(\frac{\sigma_{\text{new}}(\mathbf{x})}{\sigma_{\text{old}}}\right)$$

where:
- $\sigma_{\text{old}}$ — Current uncertainty in target
- $\sigma_{\text{new}}(\mathbf{x})$ — Uncertainty after adding observation at $\mathbf{x}$

### 7.3 Efficient Update

Adding observation $\mathbf{x}$ requires updating $K^{-1}$. GPPhad uses rank-1 update:

$$K_{\text{new}}^{-1} = \begin{pmatrix} K^{-1} + g K^{-1}\mathbf{k}\mathbf{k}^T K^{-1} & -g K^{-1}\mathbf{k} \\ -g\mathbf{k}^T K^{-1} & g \end{pmatrix}$$

where:
- $\mathbf{k} = [k(\mathbf{x}_1, \mathbf{x}), \ldots, k(\mathbf{x}_N, \mathbf{x})]^T$
- $g = (k(\mathbf{x}, \mathbf{x}) - \mathbf{k}^T K^{-1} \mathbf{k})^{-1}$

### 7.4 Greedy Selection

GPPhad evaluates acquisition function over a grid and selects:

$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in \text{grid}} \alpha(\mathbf{x})$$

---

## 8. Zero-Point Energy Treatment

### 8.1 The Problem

At low temperatures, the harmonic approximation becomes important:
$$F \approx E_0(V) + \frac{3}{2}k_B T \ln\left(\frac{\hbar\omega}{k_B T}\right) + \ldots$$

MD simulations use finite N atoms, introducing size effects in $E_0$ and phonon frequencies.

### 8.2 Two-Stage Approach

**Stage 1**: For each volume V, fit $E_0(N)$ extrapolating to $N \to \infty$

$$E_0(N) \sim E_0^\infty + \frac{a}{N^{2/3}} + \ldots$$

Using GP with covariance:
$$k_{E_0}(N_1, N_2) = \sigma^2 \left(1 + \frac{\theta^{2\nu}}{N_1^\nu N_2^\nu}\exp\left(-\frac{(\theta/N_1)^{-(\nu-1)/2} - (\theta/N_2)^{-(\nu-1)/2}}{2}\right)\right)$$

**Stage 2**: Fit $E_0(V)$ as a GP over volume

$$k_{E_0}(V_1, V_2) = \sigma^2 \exp\left(-\frac{(V_1^{-1} - V_2^{-1})^2}{2\ell^2}\right)$$

### 8.3 Integration into Main GP

The E₀ GP predictions enter the solid reference:
$$\phi_{\text{ref}}^{\text{sol}} = -1 + \frac{3}{2}\ln(2\pi T) - \frac{E_0^{\text{GP}}(V)}{T}$$

Uncertainty in E₀ propagates to uncertainty in thermodynamic predictions.

---

## 9. Numerical Considerations

### 9.1 High Precision Arithmetic

Free energy differences between phases are small (~meV) while individual terms are large (~eV). GPPhad uses 400-bit precision via GMPY2:

```python
import gmpy2
gmpy2.get_context().precision = 400
```

### 9.2 Matrix Inversion

Standard double precision fails for large, ill-conditioned covariance matrices. GPPhad uses specialized high-precision routines from `mpinv`:

```python
from mpinv import fast_mp_matrix_inverse_symm
K_inv = fast_mp_matrix_inverse_symm(K)
```

### 9.3 Numerical Stability Tips

1. **Scale hyperparameters**: Use log-scale (e.g., `th[0] = log10(sigma)`)
2. **Add nugget**: Small diagonal term prevents singularity
3. **Check condition number**: Large condition number indicates problems
4. **Normalize inputs**: Temperature and volume to similar ranges

### 9.4 Complexity

| Operation | Complexity | Dominant Cost |
|-----------|------------|---------------|
| Matrix construction | $O(N^2)$ | Kernel evaluation |
| Matrix inversion | $O(N^3)$ | High-precision arithmetic |
| Prediction | $O(N)$ | Matrix-vector product |
| Hyperparameter optimization | $O(N^3)$ per iteration | Matrix inversion |
| Active learning step | $O(MN^2)$ | M candidate rank-1 updates |

---

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

2. Frenkel, D., & Smit, B. (2001). *Understanding Molecular Simulation*. Academic Press.

3. Allen, M. P., & Tildesley, D. J. (2017). *Computer Simulation of Liquids*. Oxford University Press.

4. Grabowski, B., et al. (2007). Ab initio up to the melting point: Anharmonicity and vacancies in aluminum. *Physical Review B*, 79(13), 134106.
