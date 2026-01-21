# GPPhad API Reference

Complete API documentation for the GPPhad library.

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [GP_full Class](#gp_full-class)
3. [GP_loader Class](#gp_loader-class)
4. [Covariance Dictionaries](#covariance-dictionaries)
5. [Two-Stage Modules](#two-stage-modules)
6. [Utility Functions](#utility-functions)
7. [Phase Diagram Module](#phase-diagram-module)

---

## Module Overview

```
GPPhad/
├── __init__.py          # Main exports
├── GP/
│   ├── __init__.py      # GP_full class, load function
│   ├── _kernel.py       # Kernel base class
│   ├── _cf.py           # Covariance function dispatcher
│   ├── _constr.py       # Matrix construction
│   ├── _func.py         # Thermodynamic functions
│   ├── _optimize.py     # Hyperparameter optimization
│   ├── _save_add.py     # Persistence and data addition
│   └── phase_diagram/
│       ├── __init__.py
│       ├── _cov.py      # Covariance for phase diagrams
│       ├── _dF.py       # Jacobian computations
│       ├── _X.py        # Design matrix construction
│       ├── mean.py      # Mean predictions
│       ├── var.py       # Variance computations
│       └── AL.py        # Active learning
├── loader.py            # Data loading utilities
├── retrain.py           # Retraining with constraints
├── cov_dicts.py         # Predefined covariance functions
├── two_stages/
│   ├── __init__.py
│   ├── E0.py            # Zero-point energy GP
│   └── H.py             # Enthalpy GP
└── utils.py             # Helper functions
```

---

## GP_full Class

```python
class GPPhad.GP.GP_full(kernel)
```

Main Gaussian Process class for multi-phase thermodynamic modeling.

### Inheritance

```
kernel → GP_full
```

### Constructor

```python
GP_full(cov_dict, th=None, X=None, Y=None, err=None, phases=None, **kwargs)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_dict` | `dict` | Required | Covariance function dictionary mapping phase names to derivative covariance functions |
| `th` | `list[mpfr]` | `None` | Hyperparameters vector |
| `X` | `list` or `dict` | `None` | Training inputs |
| `Y` | `np.ndarray` | `None` | Training targets |
| `err` | `list` | `None` | Error/noise terms |
| `phases` | `list[str]` | `None` | Phase names |

#### Keyword Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `x_fixed` | `list` | Fixed parameters `[cutoff, N_atoms]` |
| `S0` | `dict` | Zero-point entropy GP per phase |
| `H` | `dict` | Enthalpy GP per phase |
| `bounds` | `dict` | Volume bounds per phase |
| `cluster` | `dict` | Cluster configuration |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `X` | `list` or `dict` | Training inputs |
| `Y` | `np.ndarray` | Training targets |
| `err` | `list` | Error terms |
| `K` | `np.ndarray` or `dict` | Covariance matrix |
| `K_inv` | `np.ndarray` or `dict` | Inverse covariance matrix |
| `err_m` | `np.ndarray` or `dict` | Error matrix |
| `th` | `list[mpfr]` | Hyperparameters |
| `phases` | `list[str]` | Phase names |
| `melt` | `bool` | Whether using unified (melted) dataset |
| `bounds` | `dict` | Volume bounds |
| `x_fixed` | `list` | Fixed parameters |
| `S0` | `dict` | Zero-point entropy GPs |
| `H` | `dict` | Enthalpy GPs |
| `prev` | `dict` | Cache for previous calculations |
| `crit_t` | `float` | Critical temperature |
| `crit_v` | `float` | Critical volume |

---

### Methods

#### `predict(X_test)`

Standard Gaussian Process prediction.

```python
def predict(self, X_test) -> np.ndarray
```

**Parameters:**
- `X_test` (`list`): Test points, each of form `[phase, derivative, T, V, ...]`

**Returns:**
- `np.ndarray` of shape `(L, 2)` where `result[i, 0]` is mean and `result[i, 1]` is variance

**Example:**
```python
# Simple prediction
result = GP.predict([['sol_fcc', 'd_0_0', 0.04, 14.0, 5, 10**30]])
mean = float(result[0, 0])
std = float(result[0, 1])**(0.5)

# Linear combination prediction
X_combo = [(0.5, ['sol_fcc', 'd_0_0', 0.04, 14.0, 5, 10**30]),
           (0.5, ['liq', 'd_0_0', 0.04, 15.0, 5, 10**30])]
result = GP.predict([X_combo])
```

---

#### `d_func(t, v, phase='sol', d='d_0_0')`

Evaluate derivative of F/T including reference terms.

```python
def d_func(self, t, v, phase='sol', d='d_0_0') -> float
```

**Parameters:**
- `t` (`mpfr`): Temperature in energy units (kB·T in eV)
- `v` (`mpfr`): Volume per atom in Å³
- `phase` (`str`): Phase name
- `d` (`str`): Derivative specification `d_m_n` for ∂^(m+n)(F/T)/∂T^m∂V^n

**Returns:**
- `float`: Negative of the derivative value (for thermodynamic convention)

**Derivative Codes:**

| Code | Derivative | Physical Meaning |
|------|------------|------------------|
| `d_0_0` | F/T | Dimensionless free energy |
| `d_1_0` | ∂(F/T)/∂T | Related to entropy |
| `d_0_1` | ∂(F/T)/∂V | Related to pressure |
| `d_2_0` | ∂²(F/T)/∂T² | Heat capacity contribution |
| `d_0_2` | ∂²(F/T)/∂V² | Bulk modulus contribution |
| `d_1_1` | ∂²(F/T)/∂T∂V | Thermal expansion contribution |

---

### Thermodynamic Property Methods

#### `predict_F(t, v, phase='sol')`

Predict Helmholtz free energy.

```python
def predict_F(self, t, v, phase='sol') -> np.ndarray
```

**Returns:** `[mean, std]` in eV

**Thermodynamic relation:** $F = T \cdot (F/T)$

---

#### `predict_S(t, v, phase='sol')`

Predict entropy.

```python
def predict_S(self, t, v, phase='sol') -> np.ndarray
```

**Returns:** `[mean, std]` in kB units

**Thermodynamic relation:** $S = -\frac{\partial F}{\partial T} = -T\frac{\partial(F/T)}{\partial T} - \frac{F}{T}$

---

#### `predict_P(t, v, phase='sol')`

Predict pressure.

```python
def predict_P(self, t, v, phase='sol') -> np.ndarray
```

**Returns:** `[mean, std]` in eV/Å³

**Thermodynamic relation:** $P = -\frac{\partial F}{\partial V} = -T\frac{\partial(F/T)}{\partial V}$

**Unit conversion:** `P_GPa = P_eV * consts['Pk']`

---

#### `predict_E(t, v, phase='sol')`

Predict internal energy.

```python
def predict_E(self, t, v, phase='sol') -> np.ndarray
```

**Returns:** `[mean, std]` in eV

**Thermodynamic relation:** $E = F + TS = T^2\frac{\partial(F/T)}{\partial T}$

---

#### `predict_B(t, v, phase='sol')`

Predict isothermal bulk modulus (mean only).

```python
def predict_B(self, t, v, phase='sol') -> float
```

**Returns:** Bulk modulus in eV/Å³

**Thermodynamic relation:** $B = -V\frac{\partial P}{\partial V} = TV\frac{\partial^2(F/T)}{\partial V^2}$

---

#### `predict_alpha(t, v, phase='sol')`

Predict volumetric thermal expansion coefficient (mean only).

```python
def predict_alpha(self, t, v, phase='sol') -> float
```

**Returns:** Thermal expansion in kB units

**Thermodynamic relation:** $\alpha = \frac{1}{V}\frac{\partial V}{\partial T}\bigg|_P$

**Unit conversion:** `alpha_per_K = alpha * consts['k']`

---

#### `predict_gamma(t, v, phase='sol')`

Predict Grüneisen parameter (mean only).

```python
def predict_gamma(self, t, v, phase='sol') -> float
```

**Returns:** Dimensionless Grüneisen parameter

**Thermodynamic relation:** $\gamma = \frac{V\alpha B}{C_V}$

---

### Phase Diagram Methods

#### `compute_mean(opt, phases, bounds, **kwargs)`

Solve phase coexistence equations to find mean solution.

```python
def compute_mean(self, opt, phases, bounds, **kwargs) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `opt` | `str` | Calculation type |
| `phases` | `list[str]` | Phases involved |
| `bounds` | `list[list]` | Search bounds for each variable |

**Options:**

| `opt` | Description | Required kwargs | Returns |
|-------|-------------|-----------------|---------|
| `'pt'` | P-T coexistence | `P` (pressure) | `[V₁, V₂, T]` |
| `'tp'` | T-P coexistence | `T` (temperature) | `[V₁, V₂]` |
| `'triple'` | Triple point | — | `[V₁, V₂, V₃, T]` |
| `'v'` | Volume at (T,P) | `T`, `P` | `[V]` |
| `'B'` | Bulk modulus | `T` | `[V, B]` |
| `'alpha'` | Thermal expansion | `T` | `[V, α]` |
| `'gamma'` | Grüneisen | `T` | `[V, γ]` |

**Example:**
```python
# Find melting curve at P = 10 GPa
P = 10 / consts['Pk']  # Convert to eV/Å³
k = consts['k']
bounds = [[10, 15], [10, 15], [400*k, 600*k]]  # [V_sol, V_liq, T]
y = GP.compute_mean('pt', ['sol_fcc', 'liq'], bounds, P=P)
V_sol, V_liq, T = y[0], y[1], y[2]/k  # T in Kelvin
```

---

#### `compute_var(opt, phases, bounds, **kwargs)`

Compute solution and uncertainty via error propagation.

```python
def compute_var(self, opt, phases, bounds, **kwargs) -> tuple
```

**Parameters:** Same as `compute_mean`, plus:

| kwarg | Type | Description |
|-------|------|-------------|
| `point` | `np.ndarray` | Pre-computed solution (skip solving) |
| `adapt` | `dict` | Adaptive learning info |

**Returns:**
- `x` (`np.ndarray`): Solution vector
- `sigmas` (`list`): Standard deviations for each variable
- `eq_error` (`np.ndarray`): Equation residual errors

**Example:**
```python
y, y_var, eq_err = GP.compute_var('triple', phases, bounds=bounds)
print(f"T = {y[3]/k:.1f} ± {y_var[3]/k:.1f} K")
```

---

#### `ad_step(opt, net, phases, **kwargs)`

Perform one step of active learning to select optimal next point.

```python
def ad_step(self, opt, net, phases, **kwargs) -> tuple
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `opt` | `str` | Target calculation type |
| `net` | `list` | Candidate points (pairs for E and P) |
| `phases` | `list[str]` | Phases for target calculation |

**Keyword Arguments:**

| kwarg | Type | Description |
|-------|------|-------------|
| `bounds` | `list` | Bounds for target calculation |
| `it` | `int` | Iteration number (for saving) |
| `w` | `callable` | Weight function `w(T, V)` |

**Returns:**
- `best_ind` (`int`): Index of best point in `net//2`
- `best_score` (`mpfr`): Information gain score
- `VAR` (`float`): Current variance

**Example:**
```python
# Generate candidate grid
net = []
for t in temperatures:
    for v in volumes:
        net.append(['sol_fcc', 'd_1_0', t, v, 5, N])  # Energy derivative
        net.append(['sol_fcc', 'd_0_1', t, v, 5, N])  # Pressure derivative

best_idx, score, var = GP.ad_step('triple', net, phases, bounds=bounds, it=0)
print(f"Next point: T={net[best_idx*2][2]}, V={net[best_idx*2][3]}")
```

---

### Optimization Methods

#### `marg_like(th, phase=None, recomp=False)`

Compute negative log marginal likelihood.

```python
def marg_like(self, th, phase=None, recomp=False) -> float
```

**Parameters:**
- `th` (`list`): Hyperparameters
- `phase` (`str`): Phase for per-phase optimization
- `recomp` (`bool`): Whether to recompute covariance matrix

**Returns:** Negative log marginal likelihood (for minimization)

---

#### `optimize(th, ind=None, phase=None, optimizer=minimize, method='L-BFGS-B', grad=False, bounds=None, tol=1e-12, rm=True)`

Optimize hyperparameters by maximizing marginal likelihood.

```python
def optimize(self, th, ind=None, phase=None, ...) -> tuple
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `th` | `list` | Required | Starting hyperparameters |
| `ind` | `list[int]` | `None` | Indices to optimize (others fixed) |
| `phase` | `str` | `None` | Phase for per-phase optimization |
| `optimizer` | `callable` | `scipy.optimize.minimize` | Optimizer function |
| `method` | `str` | `'L-BFGS-B'` | Optimization method |
| `grad` | `bool` | `False` | Use analytical gradient |
| `bounds` | `list` | `None` | Parameter bounds |
| `tol` | `float` | `1e-12` | Tolerance |

**Returns:**
- `result` (`float`): Final marginal likelihood
- `th` (`list`): Optimized hyperparameters

**Example:**
```python
# Optimize only liquid phase parameters (indices 0-5)
result, th_new = GP.optimize(th, ind=list(range(6)), phase='liq')
```

---

### Persistence Methods

#### `save(file, **kwargs)`

Save GP to pickle file.

```python
def save(self, file, **kwargs) -> None
```

**Saved attributes:** `X`, `Y`, `err`, `th`, `phases`, `f_dict`, `bounds`, `S0`, `H`, `x_fixed`, `cluster`

---

#### `GPPhad.load(file)` (module function)

Load GP from pickle file.

```python
def load(file) -> GP_full
```

**Example:**
```python
from GPPhad import load
GP = load('trained_model.pickle')
```

---

### Data Management Methods

#### `add_points(x, y, err, phase=None)`

Add new training points with efficient rank-1 matrix update.

```python
def add_points(self, x, y, err, phase=None) -> None
```

**Parameters:**
- `x` (`list`): New input points
- `y` (`list`): New target values
- `err` (`list`): New error terms
- `phase` (`str`): Phase (for non-melted GP)

---

#### `add_melt(phases, melt_p)`

Add melting point constraint.

```python
def add_melt(self, phases, melt_p) -> None
```

**Parameters:**
- `phases` (`list[str]`): `[phase1, phase2]`
- `melt_p` (`list`): `[P, T, dT]` — pressure, temperature, temperature error

---

#### `add_H(phase, V=None)`

Add zero-point energy constraint.

```python
def add_H(self, phase, V=None) -> None
```

---

## GP_loader Class

```python
class GPPhad.loader.GP_loader
```

Utility class for loading and preprocessing molecular dynamics data.

### Constructor

```python
GP_loader(phases, melt=True)
```

### Methods

#### `_init_data(folder='./data', cut=0)`

Initialize data from files.

#### `_init_dataset()`

Convert raw data to GP format.

#### `read_pt(phase, seed, threshold=10000)`

Read single simulation point.

#### `lmp_to_GP(phase, seed)`

Convert LAMMPS data to GP format.

---

## Covariance Dictionaries

### `cov_real`

Full covariance function dictionary for real materials.

**Structure:**
```python
cov_real = {
    "liq": {
        "d_m_n_d_p_q": lambda x1, x2, th: ...,
        # All derivative combinations up to 4th order
    },
    "sol_fcc": { ... },
    "sol_bcc": { ... }
}
```

**Hyperparameter structure:**
- `th[0:6]` — Liquid phase (6 parameters)
- `th[6:11]` — FCC solid (5 parameters)
- `th[11:17]` — BCC solid (6 parameters)

### `cov_E0_1`

First-stage covariance for E₀ fitting.

### `cov_E0_2`

Second-stage covariance for E₀ volume dependence.

### `cov_H_1`

Covariance for enthalpy extrapolation.

---

## Two-Stage Modules

### `constr_E0(folder)`

Construct zero-point energy GP.

```python
def constr_E0(folder: str) -> GP_full
```

Reads `E0_ref.dat` and creates GP for E₀(V).

### `constr_H(folder)`

Construct enthalpy GP.

```python
def constr_H(folder: str) -> GP_full
```

Reads `H.dat` and creates GP for H(N).

---

## Utility Functions

### `consts`

Physical constants dictionary.

```python
consts = {
    'Pk': 160.2176621,      # eV/Å³ to GPa
    'k': 8.617333262e-5     # Boltzmann constant eV/K
}
```

### `mean(slice_)`

High-precision mean calculation.

### `std(slice_)`

High-precision standard deviation.

### `print_point(opt, phases, point, point_var)`

Format and print phase diagram results.

```python
def print_point(opt, phases, point, point_var) -> tuple
```

**Returns:** `(point_converted, point_var_converted)` in physical units

---

## Retrain Function

```python
def retrain(GP, melt_points, ind_bounds) -> tuple
```

Retrain GP with melting point constraints.

**Parameters:**
- `GP`: GP_full object
- `melt_points`: List of `[[phases], [P, T, dT]]`
- `ind_bounds`: Dict mapping phase to hyperparameter indices

**Returns:** `(GP_new, th_new)`

---

## Error Handling

GPPhad uses assertions and prints for debugging. Common issues:

| Error | Cause | Solution |
|-------|-------|----------|
| `Matrix inversion failed` | Singular covariance matrix | Check data for duplicates, adjust nugget |
| `AssertionError in read_pt` | Empty data slice | Verify data file format |
| `KeyError in cov_dict` | Missing derivative covariance | Add missing kernel derivative |

---

## Thread Safety

GPPhad is **not thread-safe** due to mutable state in `GP.prev` cache and global GMPY2 context. For parallel computation, use separate processes.
