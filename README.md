# GPPhad — Gaussian Process Phase Diagrams

<p align="center">
  <img src="https://img.shields.io/badge/python-3.5+-blue.svg" alt="Python 3.5+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/version-1.1-orange.svg" alt="Version 1.1">
</p>

**GPPhad** is a Python library for constructing phase diagrams with uncertainty quantification using Gaussian Process regression. The library enables thermodynamic property prediction, phase coexistence calculations, triple point determination, and active learning for optimal experimental design from molecular dynamics simulation data.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Data Format](#-data-format)
- [Mathematical Background](#-mathematical-background)
- [Citation](#-citation)
- [Authors](#-authors)
- [License](#-license)

---

## ✨ Features

- **Multi-phase Gaussian Process Regression** — Unified GP model for liquid and solid phases
- **High Precision Arithmetic** — 400-bit precision using GMPY2 for numerical stability
- **Thermodynamic Properties** — Predict F, S, P, E, bulk modulus (B), thermal expansion (α), Grüneisen parameter (γ)
- **Phase Diagram Construction** — Automatic calculation of phase boundaries with error bars
- **Phase Coexistence** — Compute (P,T), (T,P), and triple points with uncertainty
- **Active Learning** — Greedy selection of optimal simulation points to reduce uncertainty
- **Zero-Point Energy Handling** — Two-stage GP regression for E₀ and enthalpy extrapolation
- **Flexible Covariance Functions** — Customizable kernels for different phases and derivatives

---

## 🔧 Installation

### Prerequisites

Ensure you have Python 3.5+ and the following dependencies:

```bash
pip install numpy scipy pandas gmpy2 dill
```

You also need `mpinv` for high-precision matrix inversion (contact authors or check repository).

### Install from Source

```bash
git clone https://github.com/your-repo/phase_diagrams-by-gaussian-process.git
cd phase_diagrams-by-gaussian-process
pip install -e .
```

### Verify Installation

```python
import GPPhad
print(GPPhad.__name__)  # Should print 'GPPhad'
```

---

## 🚀 Quick Start

```python
import numpy as np
from gmpy2 import mpfr, get_context
get_context().precision = 400

from GPPhad import create_from_scratch, retrain, consts, cov_real, print_point

# Define phases and fixed simulation parameters
phases = ['liq', 'sol_fcc', 'sol_bcc']
x_fixed = [5, 10**30]  # [cutoff, N_atoms]

# Define hyperparameters (pre-optimized or initial guess)
th_full = [mpfr('-6.247'), mpfr('0.293'), ...]  # 17 hyperparameters

# Create GP from molecular dynamics data files
GP_Li = create_from_scratch(cov_real, th_full, phases, x_fixed=x_fixed)

# Add melting point constraints and retrain
GP_Li.bounds = {"liq": [10, 24], "sol_fcc": [10, 15], 'sol_bcc': [10, 24]}
melt_points = [
    [['sol_fcc', 'liq'], [12/consts['Pk'], 495.5*consts['k'], 1.6*consts['k']]],
    [['sol_bcc', 'liq'], [0/consts['Pk'], 475.7*consts['k'], 1.7*consts['k']]]
]
ind_bounds = {'liq': range(0, 6), 'sol_fcc': range(6, 11), 'sol_bcc': range(11, 17)}
GP_Li, th_temp = retrain(GP_Li, melt_points, ind_bounds)

# Calculate triple point with uncertainty
k = consts['k']
bounds = [[11, 17], [11, 17], [11, 17], [440*k, 560*k]]
y, y_var, V = GP_Li.compute_var('triple', phases, bounds=bounds)
print_point('triple', phases, y, y_var)
```

**Output:**
```
sol_fcc volume: 14.487 ± 0.096
liq volume: 14.516 ± 0.109
sol_bcc volume: 14.607 ± 0.106
triple point temp: 500.185 ± 1.514
```

---

## 📚 Core Concepts

### Thermodynamic Framework

GPPhad models the Helmholtz free energy divided by temperature as a Gaussian Process:

$$\frac{F(T, V)}{T} = f(T, V) + \text{reference}$$

where $f(T, V) \sim \mathcal{GP}(0, k)$ is the GP with covariance function $k$.

The GP directly learns derivatives of $F/T$:
- `d_m_n` represents $\frac{\partial^{m+n} (F/T)}{\partial T^m \partial V^n}$

### Multi-Phase Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GP_full                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │  liq    │  │ sol_fcc │  │ sol_bcc │  ← Phase-specific    │
│  │ kernel  │  │ kernel  │  │ kernel  │    covariance        │
│  └────┬────┘  └────┬────┘  └────┬────┘                      │
│       │            │            │                            │
│       └────────────┼────────────┘                            │
│                    ▼                                         │
│         ┌──────────────────┐                                 │
│         │ Block-diagonal K │  ← Combined covariance matrix   │
│         └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

### Two-Stage Zero-Point Energy

For solid phases, GPPhad uses a two-stage approach:

1. **Stage 1**: Fit E₀(N) and P₀(N) as functions of system size N
2. **Stage 2**: Extrapolate to N → ∞ using volume-dependent GP

---

## 📖 API Reference

### Main Module (`GPPhad`)

#### `create_from_scratch(cov_dict, th, phases, melt=True, cut=0, **kwargs)`

Create a GP from molecular dynamics data files.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `cov_dict` | dict | Covariance function dictionary |
| `th` | list | Hyperparameters vector |
| `phases` | list | Phase names (e.g., `['liq', 'sol_fcc']`) |
| `melt` | bool | Whether phases share training data |
| `cut` | int | Number of random seeds to exclude |
| `x_fixed` | list | Fixed parameters `[cutoff, N_atoms]` |
| `bounds` | dict | Volume bounds per phase |

**Returns:** `GP_full` — Trained Gaussian Process object

---

### `GP_full` Class

The main Gaussian Process class for multi-phase modeling.

#### Constructor

```python
GP_full(cov_dict, th=None, X=None, Y=None, err=None, phases=None, **kwargs)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `cov_dict` | dict | Covariance function dictionary |
| `th` | list | Hyperparameters |
| `X` | list | Training inputs `[phase, derivative, T, V, cutoff, N]` |
| `Y` | array | Training targets |
| `err` | list | Error terms `['err', hash, variance]` |
| `phases` | list | Phase names |
| `S0` | dict | Zero-point entropy GP per phase |
| `H` | dict | Enthalpy GP per phase |
| `bounds` | dict | Volume bounds per phase |

#### Prediction Methods

##### `predict(X_test)`

Standard GP regression prediction.

```python
result = GP.predict([[phase, derivative, T, V, cutoff, N]])
mean, variance = result[0, 0], result[0, 1]
```

##### `d_func(t, v, phase='sol', d='d_0_0')`

Evaluate derivative of F/T with reference.

```python
# Get d(F/T)/dV at T=0.04 eV, V=14 Å³
dFdV = GP.d_func(0.04, 14, phase='sol_fcc', d='d_0_1')
```

##### Thermodynamic Properties

| Method | Returns | Units |
|--------|---------|-------|
| `predict_F(t, v, phase)` | Free energy (mean, std) | eV |
| `predict_S(t, v, phase)` | Entropy (mean, std) | kB |
| `predict_P(t, v, phase)` | Pressure (mean, std) | eV/Å³ |
| `predict_E(t, v, phase)` | Internal energy (mean, std) | eV |
| `predict_B(t, v, phase)` | Bulk modulus | eV/Å³ |
| `predict_alpha(t, v, phase)` | Thermal expansion | 1/kB |
| `predict_gamma(t, v, phase)` | Grüneisen parameter | dimensionless |

#### Phase Diagram Methods

##### `compute_mean(opt, phases, bounds, **kwargs)`

Solve phase coexistence equations.

```python
# P-T coexistence at P=0.1 eV/Å³
y = GP.compute_mean('pt', ['sol_fcc', 'liq'], 
                     bounds=[[10, 15], [10, 15], [300*k, 600*k]], 
                     P=0.1)
```

**Options (`opt`):**
| Value | Description | Returns |
|-------|-------------|---------|
| `'pt'` | Coexistence at fixed P | [V₁, V₂, T] |
| `'tp'` | Coexistence at fixed T | [V₁, V₂] |
| `'triple'` | Triple point | [V₁, V₂, V₃, T] |
| `'v'` | Volume at fixed T, P | [V] |
| `'B'` | Bulk modulus at T, P | [V, B] |
| `'alpha'` | Thermal expansion at T, P | [V, α] |
| `'gamma'` | Grüneisen at T, P | [V, γ] |

##### `compute_var(opt, phases, bounds, **kwargs)`

Compute mean and uncertainty via error propagation.

```python
y, y_var, eq_error = GP.compute_var('triple', phases, bounds=bounds)
```

**Returns:**
- `y` — Solution vector
- `y_var` — Standard deviations 
- `eq_error` — Equation residual errors

##### `ad_step(opt, net, phases, **kwargs)`

Active learning step to select optimal next simulation point.

```python
# Generate candidate points
net = []
for t in np.linspace(100, 600, 10):
    for v in np.linspace(12, 16, 5):
        net.append([phase, 'd_1_0', t*k, v, cutoff, N])
        net.append([phase, 'd_0_1', t*k, v, cutoff, N])

# Find best point
best_idx, score, var_old = GP.ad_step('triple', net, phases, 
                                       bounds=bounds, it=0)
```

#### Hyperparameter Optimization

##### `marg_like(th, phase=None, recomp=False)`

Compute negative log marginal likelihood.

##### `optimize(th, ind=None, phase=None, ...)`

Optimize hyperparameters via L-BFGS-B.

```python
result, th_new = GP.optimize(th, ind=[0, 1, 2], 
                              method='L-BFGS-B', tol=1e-12)
```

#### Persistence

##### `save(file)`

Save GP to pickle file.

```python
GP.save('trained_model.pickle')
```

##### `load(file)` (module function)

Load GP from pickle file.

```python
from GPPhad import load
GP = load('trained_model.pickle')
```

#### Data Management

##### `add_points(x, y, err, phase=None)`

Add new training points with efficient matrix update.

##### `add_melt(phases, melt_p)`

Add melting point constraint.

```python
GP.add_melt(['sol_fcc', 'liq'], [P, T, dT])
```

---

### Covariance Functions

GPPhad provides predefined covariance dictionaries:

#### `cov_real`

Full covariance for liquid and solid phases with derivatives up to 4th order.

**Structure:**
```python
cov_real = {
    "liq": {
        "d_0_0_d_0_0": lambda x1, x2, th: ...,
        "d_0_0_d_1_0": lambda x1, x2, th: ...,
        # ... all derivative combinations
    },
    "sol_fcc": { ... },
    "sol_bcc": { ... }
}
```

**Hyperparameter indices:**
| Phase | Indices | Description |
|-------|---------|-------------|
| liq | 0-5 | 6 hyperparameters |
| sol_fcc | 6-10 | 5 hyperparameters |
| sol_bcc | 11-16 | 6 hyperparameters |

#### `cov_E0_1`, `cov_E0_2`

Covariances for two-stage E₀ extrapolation.

#### `cov_H_1`

Covariance for enthalpy extrapolation.

---

### Utility Functions

#### `consts`

Physical constants dictionary:
```python
consts = {
    'Pk': 160.2176621,  # Pressure conversion: eV/Å³ to GPa
    'k': 8.617333262e-5  # Boltzmann constant in eV/K
}
```

#### `print_point(opt, phases, point, point_var)`

Pretty-print phase diagram results with units.

```python
print_point('triple', phases, y, y_var)
# Output:
# sol_fcc volume: 14.487 ± 0.096
# liq volume: 14.516 ± 0.109
# sol_bcc volume: 14.607 ± 0.106
# triple point temp: 500.185 ± 1.514
```

---

## 📁 Data Format

### Directory Structure

```
data/
├── liq/
│   └── liq.dat
├── sol_fcc/
│   ├── sol_fcc.dat
│   ├── E0.dat         # Optional: zero-point energy
│   ├── E0_ref.dat     # Reference for E0 GP
│   └── H.dat          # Enthalpy data
└── sol_bcc/
    └── sol_bcc.dat
```

### Main Data File (`phase.dat`)

Tab/space-separated file with columns:

```
step  T  V  cutoff  N  ...  E  P  seed
```

| Column | Description | Units |
|--------|-------------|-------|
| step | MD timestep | - |
| T | Temperature | K (converted internally) |
| V | Volume per atom | Å³ |
| cutoff | Interaction cutoff | Å |
| N | Number of atoms | - |
| E | Internal energy | eV/atom |
| P | Pressure | eV/Å³ |
| seed | Random seed identifier | - |

### Zero-Point Energy (`E0_ref.dat`)

```
V, N, E0, P0
```

### Enthalpy (`H.dat`)

```
V, N, H
```

---

## 🔬 Examples

### Example 1: Lithium Phase Diagram

Complete workflow for Li with BCC, FCC, and liquid phases:

```python
import numpy as np
from gmpy2 import mpfr, get_context
get_context().precision = 400

from GPPhad import create_from_scratch, retrain, consts, cov_real

# Setup
phases = ['liq', 'sol_fcc', 'sol_bcc']
x_fixed = [5, 10**30]
k = consts['k']

# Create GP from data
GP_Li = create_from_scratch(cov_real, th_full, phases, x_fixed=x_fixed)

# Define bounds and melting points
GP_Li.bounds = {"liq": [10, 24], "sol_fcc": [10, 15], 'sol_bcc': [10, 24]}
melt_points = [
    [['sol_fcc', 'liq'], [12/consts['Pk'], 495.5*k, 1.6*k]],
    [['sol_bcc', 'liq'], [0, 475.7*k, 1.7*k]]
]
ind_bounds = {'liq': range(0, 6), 'sol_fcc': range(6, 11), 'sol_bcc': range(11, 17)}

# Retrain with melting constraints
GP_Li, th_new = retrain(GP_Li, melt_points, ind_bounds)

# Calculate FCC-liquid coexistence curve
P_range = np.linspace(9, 30, 20) / consts['Pk']
T_coex = np.zeros(len(P_range))
T_err = np.zeros(len(P_range))

for i, P in enumerate(P_range):
    y, y_var, _ = GP_Li.compute_var('pt', ['sol_fcc', 'liq'],
                                     bounds=[[10, 15], [10, 15], [370*k, 530*k]],
                                     P=P)
    T_coex[i] = y[2] / k  # Convert to Kelvin
    T_err[i] = y_var[2] / k

# Plot
import matplotlib.pyplot as plt
plt.fill_between(P_range * consts['Pk'], T_coex - T_err, T_coex + T_err, alpha=0.3)
plt.plot(P_range * consts['Pk'], T_coex)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Temperature (K)')
```

### Example 2: Thermal Expansion Calculation

```python
# Calculate thermal expansion coefficient at T=300K, P=0
T = 300 * consts['k']
P = 0

# Find equilibrium volume
bounds = [[13, 16]]
y = GP.compute_mean('v', ['sol_fcc'], bounds, T=T, P=P)
V_eq = y[0]

# Calculate alpha with uncertainty
bounds_alpha = [[13, 16], [0, 0.1]]  # [V bounds, alpha bounds]
y, y_var, _ = GP.compute_var('alpha', ['sol_fcc'], bounds_alpha, T=T)

alpha = y[1] * consts['k']  # Convert to 1/K
alpha_err = y_var[1] * consts['k']

print(f"α = {alpha:.2e} ± {alpha_err:.2e} 1/K")
```

### Example 3: Active Learning

```python
# Setup for triple point refinement
phases = ['sol_fcc', 'liq', 'sol_bcc']
bounds = [[11, 17], [11, 17], [11, 17], [440*k, 560*k]]

# Generate candidate grid
V_grid = np.linspace(13, 15, 5)
T_grid = np.linspace(300, 550, 10)

net = []
for v in V_grid:
    for t in T_grid:
        net.append(['sol_fcc', 'd_1_0', t*k, v, 5, 10**30])
        net.append(['sol_fcc', 'd_0_1', t*k, v, 5, 10**30])

# Find optimal point
best_idx, score, var_old = GP.ad_step('triple', net, phases,
                                       bounds=bounds, it=0)

print(f"Best point: T={net[best_idx*2][2]/k:.0f} K, V={net[best_idx*2][3]:.2f} Å³")
print(f"Expected variance reduction: {np.exp(float(score)):.1%}")
```

---

## 📐 Mathematical Background

### Gaussian Process Regression

Given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the GP posterior at test point $x_*$ is:

$$\mu_* = k_*^T K^{-1} y$$
$$\sigma_*^2 = k_{**} - k_*^T K^{-1} k_*$$

where:
- $K_{ij} = k(x_i, x_j)$ is the covariance matrix
- $k_* = [k(x_1, x_*), \ldots, k(x_N, x_*)]^T$
- $k_{**} = k(x_*, x_*)$

### Derivative Observations

For free energy modeling, we observe derivatives. The covariance between derivatives is:

$$\text{Cov}\left[\frac{\partial^{m+n} f}{\partial T^m \partial V^n}, \frac{\partial^{p+q} f}{\partial T^p \partial V^q}\right] = \frac{\partial^{m+n+p+q} k}{\partial T_1^m \partial V_1^n \partial T_2^p \partial V_2^q}$$

### Phase Coexistence

At coexistence, phases satisfy:
1. **Thermal equilibrium**: $T_\alpha = T_\beta$
2. **Mechanical equilibrium**: $P_\alpha = P_\beta$  
3. **Chemical equilibrium**: $\mu_\alpha = \mu_\beta$ (or equivalently $G_\alpha = G_\beta$)

GPPhad solves these as a nonlinear system with uncertainty propagation.

### Active Learning

The acquisition function maximizes information gain:

$$\alpha(x) = -\log\left(\frac{\sigma_{\text{new}}(x)}{\sigma_{\text{old}}}\right)$$

where $\sigma$ is the uncertainty in the target property (e.g., triple point temperature).

---

## 📚 Citation

If you use GPPhad in your research, please cite:

```bibtex
@article{ladygin2024gpphad,
  title={Phase diagrams by Gaussian process regression},
  author={Ladygin, Vladimir and Shapeev, Alexander},
  journal={...},
  year={2024}
}
```

---

## 👥 Authors

- **Vladimir Ladygin** — Development and implementation
- **Alexander Shapeev** — Theoretical framework

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
