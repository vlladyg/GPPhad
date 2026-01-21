# GPPhad Tutorial: Step-by-Step Guide

A comprehensive walkthrough for computing phase diagrams with uncertainty using GPPhad.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Preparing Your Data](#2-preparing-your-data)
3. [Creating Your First GP](#3-creating-your-first-gp)
4. [Training and Optimization](#4-training-and-optimization)
5. [Predicting Thermodynamic Properties](#5-predicting-thermodynamic-properties)
6. [Computing Phase Boundaries](#6-computing-phase-boundaries)
7. [Finding Triple Points](#7-finding-triple-points)
8. [Active Learning](#8-active-learning)
9. [Visualizing Results](#9-visualizing-results)
10. [Advanced Topics](#10-advanced-topics)

---

## 1. Prerequisites

### 1.1 Installation

```bash
# Clone repository
git clone https://github.com/your-repo/phase_diagrams-by-gaussian-process.git
cd phase_diagrams-by-gaussian-process

# Install dependencies
pip install numpy scipy pandas gmpy2 dill

# Install GPPhad
pip install -e .
```

### 1.2 Required Imports

```python
import numpy as np
import matplotlib.pyplot as plt

# High-precision arithmetic
from gmpy2 import mpfr, get_context
get_context().precision = 400

# GPPhad modules
from GPPhad import (
    create_from_scratch,
    load,
    retrain,
    consts,
    cov_real,
    GP_full,
    print_point
)

# Physical constants
k = consts['k']      # Boltzmann constant: 8.617e-5 eV/K
Pk = consts['Pk']    # Pressure conversion: 160.22 eV/Å³ → GPa
```

### 1.3 Understanding Units

| Quantity | GPPhad Units | SI Conversion |
|----------|--------------|---------------|
| Temperature | kT (eV) | T(K) × k |
| Volume | Å³/atom | — |
| Pressure | eV/Å³ | P(GPa) / Pk |
| Energy | eV/atom | — |
| Entropy | kB/atom | S × k for J/(K·atom) |

---

## 2. Preparing Your Data

### 2.1 Directory Structure

Create your data directory:

```
project/
├── data/
│   ├── liq/
│   │   └── liq.dat
│   ├── sol_fcc/
│   │   ├── sol_fcc.dat
│   │   ├── E0.dat          # Optional: zero-point energies
│   │   ├── E0_ref.dat      # For E0 GP fitting
│   │   └── H.dat           # Enthalpy data
│   └── sol_bcc/
│       └── sol_bcc.dat
└── analysis.py
```

### 2.2 Main Data File Format

Create `phase.dat` with your MD results:

```
# step T V cutoff N E_kinetic E_potential virial pressure seed
10000 500 14.5 5.0 2048 0.0645 -1.234 0.156 0.00123 1
10100 500 14.5 5.0 2048 0.0651 -1.231 0.148 0.00121 1
...
```

**Columns:**
1. Step number (for equilibration filtering)
2. Temperature (K)
3. Volume per atom (Å³)
4. Cutoff radius (Å)
5. Number of atoms
6. Kinetic energy (eV/atom)
7. Potential energy (eV/atom)
8. Virial
9. Pressure (eV/Å³)
10. Random seed (groups related runs)

**Important:** Remove comment lines or let GPPhad handle them.

### 2.3 Zero-Point Energy Files (Optional)

For accurate solid phase modeling, provide E₀ data:

**E0_ref.dat:**
```
14.0,256,E0_value,P0_value
14.0,500,E0_value,P0_value
14.0,864,E0_value,P0_value
14.5,256,E0_value,P0_value
...
```

**H.dat:**
```
14.0,256,H_value
14.0,500,H_value
...
```

---

## 3. Creating Your First GP

### 3.1 Using `create_from_scratch`

The simplest way to create a GP from data files:

```python
# Define phases
phases = ['liq', 'sol_fcc', 'sol_bcc']

# Fixed simulation parameters [cutoff, N_atoms]
# Use large N to simulate infinite system
x_fixed = [5.0, 10**30]

# Initial hyperparameters (17 total for 3 phases)
th_init = [
    # Liquid phase (0-5): 6 parameters
    mpfr(-6.0), mpfr(0.3), mpfr(0.5), mpfr(-0.05), mpfr(0.05), mpfr(0.1),
    # FCC solid (6-10): 5 parameters
    mpfr(-5.0), mpfr(-0.9), mpfr(0.03), mpfr(-0.01), mpfr(11.0),
    # BCC solid (11-16): 6 parameters
    mpfr(-4.0), mpfr(0.3), mpfr(0.5), mpfr(-0.05), mpfr(-0.03), mpfr(-1.2),
]

# Create GP from data
GP = create_from_scratch(
    cov_real,      # Covariance function dictionary
    th_init,       # Hyperparameters
    phases,        # Phase names
    x_fixed=x_fixed,
    melt=True      # Unified dataset
)

print(f"Training points: {len(GP.X)}")
print(f"Marginal likelihood: {GP.marg_like(GP.th, recomp=False):.2f}")
```

### 3.2 Manual GP Creation

For more control:

```python
# Prepare data manually
X = [
    ['liq', 'd_1_0', 0.043, 15.0, 5.0, 2048],  # Energy derivative
    ['liq', 'd_0_1', 0.043, 15.0, 5.0, 2048],  # Pressure derivative
    # ... more points
]

Y = np.array([
    [energy_value],
    [pressure_value],
    # ...
], dtype=object)

err = [
    ['err', hash(tuple(X[0])), error_variance_1],
    ['err', hash(tuple(X[1])), error_variance_2],
    # ...
]

# Create GP
GP = GP_full(
    cov_real,
    th=th_init,
    X=X,
    Y=Y,
    err=err,
    phases=phases,
    x_fixed=x_fixed
)
```

---

## 4. Training and Optimization

### 4.1 Single-Phase Optimization

Optimize hyperparameters for one phase at a time:

```python
# Optimize liquid phase (parameters 0-5)
result, th_new = GP.optimize(
    GP.th,
    ind=list(range(6)),      # Indices to optimize
    phase='liq',             # Phase name
    method='L-BFGS-B',
    tol=1e-8
)
print(f"Liquid optimized: marginal likelihood = {result:.2f}")

# Update GP hyperparameters
GP.th = th_new
```

### 4.2 Adding Melting Point Constraints

Incorporate experimental melting data:

```python
# Define volume bounds for each phase
GP.bounds = {
    "liq": [10, 24],      # Å³
    "sol_fcc": [10, 15],
    "sol_bcc": [10, 24]
}

# Melting point data: [phases], [P (eV/Å³), T (eV), dT (eV)]
melt_points = [
    # FCC-Liquid at P=12 GPa, T=495.5±1.6 K
    [['sol_fcc', 'liq'], [12/Pk, 495.5*k, 1.6*k]],
    # BCC-Liquid at P=0, T=475.7±1.7 K  
    [['sol_bcc', 'liq'], [0, 475.7*k, 1.7*k]],
]

# Hyperparameter indices for each phase
ind_bounds = {
    'liq': range(0, 6),
    'sol_fcc': range(6, 11),
    'sol_bcc': range(11, 17)
}

# Retrain with constraints
GP_trained, th_final = retrain(GP, melt_points, ind_bounds)
```

### 4.3 Saving and Loading

```python
# Save trained GP
GP_trained.save('lithium_trained.pickle')

# Load later
from GPPhad import load
GP = load('lithium_trained.pickle')
```

---

## 5. Predicting Thermodynamic Properties

### 5.1 Basic Predictions

```python
# Point specification
T = 500 * k    # Temperature in eV
V = 14.5       # Volume in Å³
phase = 'sol_fcc'

# Free energy
F_mean, F_std = GP.predict_F(T, V, phase)
print(f"F = {F_mean:.4f} ± {F_std:.4f} eV")

# Entropy
S_mean, S_std = GP.predict_S(T, V, phase)
print(f"S = {S_mean:.4f} ± {S_std:.4f} kB")

# Pressure
P_mean, P_std = GP.predict_P(T, V, phase)
print(f"P = {P_mean*Pk:.2f} ± {P_std*Pk:.2f} GPa")

# Internal energy
E_mean, E_std = GP.predict_E(T, V, phase)
print(f"E = {E_mean:.4f} ± {E_std:.4f} eV")
```

### 5.2 Derived Properties

```python
# Bulk modulus
B = GP.predict_B(T, V, phase)
print(f"B = {B*Pk:.1f} GPa")

# Thermal expansion
alpha = GP.predict_alpha(T, V, phase)
print(f"α = {alpha*k:.2e} /K")

# Grüneisen parameter
gamma = GP.predict_gamma(T, V, phase)
print(f"γ = {gamma:.3f}")
```

### 5.3 Computing Equation of State

```python
# P-V isotherm at T = 500 K
T = 500 * k
V_range = np.linspace(12, 18, 50)
P_vals = np.zeros(len(V_range))
P_errs = np.zeros(len(V_range))

for i, V in enumerate(V_range):
    result = GP.predict_P(T, V, 'sol_fcc')
    P_vals[i] = result[0] * Pk   # GPa
    P_errs[i] = result[1] * Pk

# Plot
plt.figure(figsize=(8, 5))
plt.fill_between(V_range, P_vals - P_errs, P_vals + P_errs, alpha=0.3)
plt.plot(V_range, P_vals)
plt.xlabel('Volume (Å³/atom)')
plt.ylabel('Pressure (GPa)')
plt.title('FCC Isotherm at 500 K')
plt.show()
```

---

## 6. Computing Phase Boundaries

### 6.1 Two-Phase Coexistence at Fixed Pressure

Find melting temperature at given pressure:

```python
# P-T coexistence
P = 10 / Pk  # 10 GPa in eV/Å³

# Bounds: [V_solid, V_liquid, Temperature]
bounds = [
    [10, 15],      # FCC volume range
    [10, 18],      # Liquid volume range  
    [400*k, 600*k] # Temperature range
]

# Compute with uncertainty
y, y_var, eq_err = GP.compute_var(
    'pt',                      # P-T coexistence
    ['sol_fcc', 'liq'],        # Phases
    bounds=bounds,
    P=P
)

# Extract results
V_fcc = y[0]
V_liq = y[1]
T_melt = y[2] / k  # Convert to Kelvin

V_fcc_err = y_var[0]
V_liq_err = y_var[1]
T_melt_err = y_var[2] / k

print(f"At P = 10 GPa:")
print(f"  T_melt = {T_melt:.1f} ± {T_melt_err:.1f} K")
print(f"  V_fcc = {V_fcc:.3f} ± {V_fcc_err:.3f} Å³")
print(f"  V_liq = {V_liq:.3f} ± {V_liq_err:.3f} Å³")
```

### 6.2 Two-Phase Coexistence at Fixed Temperature

Find pressure at given temperature:

```python
# T-P coexistence
T = 400 * k  # 400 K

bounds = [
    [12, 16],  # Phase 1 volume
    [12, 17]   # Phase 2 volume
]

y, y_var, eq_err = GP.compute_var(
    'tp',
    ['sol_fcc', 'sol_bcc'],
    bounds=bounds,
    T=T
)

# For 'tp', pressure is computed from the solution
V1, V2 = y[0], y[1]
P_coex = y[2] * Pk  # Returned in the extended result
P_err = y_var[2] * Pk

print(f"At T = 400 K:")
print(f"  FCC-BCC coexistence at P = {P_coex:.2f} ± {P_err:.2f} GPa")
```

### 6.3 Computing Full Melting Curve

```python
# Pressure range
P_range = np.linspace(0, 30, 20) / Pk

# Storage
T_curve = np.zeros(len(P_range))
T_error = np.zeros(len(P_range))

for i, P in enumerate(P_range):
    # Adjust bounds based on expected behavior
    bounds = [[10, 15], [10, 18], [350*k, 600*k]]
    
    try:
        y, y_var, _ = GP.compute_var('pt', ['sol_fcc', 'liq'], 
                                      bounds=bounds, P=P)
        T_curve[i] = y[2] / k
        T_error[i] = y_var[2] / k
    except:
        T_curve[i] = np.nan
        T_error[i] = np.nan

# Plot
plt.figure(figsize=(10, 6))
valid = ~np.isnan(T_curve)
plt.fill_between(P_range[valid]*Pk, 
                 T_curve[valid] - T_error[valid],
                 T_curve[valid] + T_error[valid], 
                 alpha=0.3, color='red')
plt.plot(P_range[valid]*Pk, T_curve[valid], 'r-', lw=2)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Temperature (K)')
plt.title('FCC-Liquid Melting Curve')
plt.show()
```

---

## 7. Finding Triple Points

### 7.1 Three-Phase Equilibrium

```python
# Triple point: three phases coexist
phases = ['sol_fcc', 'liq', 'sol_bcc']

# Initial guess and bounds
x0 = [14.3, 14.4, 14.3, 500]  # [V_fcc, V_liq, V_bcc, T(K)]
dx = 3

bounds = [
    [x0[0]-dx, x0[0]+dx],  # V_fcc
    [x0[1]-dx, x0[1]+dx],  # V_liq
    [x0[2]-dx, x0[2]+dx],  # V_bcc
    [(x0[3]-60)*k, (x0[3]+60)*k]  # Temperature
]

# Compute
y, y_var, eq_err = GP.compute_var('triple', phases, bounds=bounds)

# Pretty print
point, point_var = print_point('triple', phases, y, y_var)

# Extract triple point conditions
V_fcc = point[0]
V_liq = point[1]
V_bcc = point[2]
T_triple = point[3]  # Already in K

# Compute corresponding pressure
P_triple = -GP.d_func(y[3], y[0], phase='sol_fcc', d='d_0_1') * y[3] * Pk

print(f"\nTriple Point Summary:")
print(f"  T = {T_triple:.1f} ± {point_var[3]:.1f} K")
print(f"  P = {P_triple:.2f} GPa")
```

---

## 8. Active Learning

### 8.1 Generating Candidate Points

```python
# Define search grid
V_grid = np.linspace(12, 16, 5)
T_grid = np.linspace(300, 550, 10)

# Phase to sample
phase = 'sol_fcc'
cutoff = 5
N = 2048

# Build candidate list (pairs: energy + pressure derivatives)
net = []
for V in V_grid:
    for T in T_grid:
        t = T * k  # Convert to eV
        net.append([phase, 'd_1_0', t, V, cutoff, N])  # dE/dT related
        net.append([phase, 'd_0_1', t, V, cutoff, N])  # dP/dV related

print(f"Generated {len(net)//2} candidate points")
```

### 8.2 Finding Optimal Next Point

```python
# Target: reduce triple point uncertainty
phases = ['sol_fcc', 'liq', 'sol_bcc']
bounds = [[11, 17], [11, 17], [11, 17], [440*k, 560*k]]

# Run active learning step
best_idx, score, var_old = GP.ad_step(
    'triple',    # Target calculation
    net,         # Candidate points
    phases,
    bounds=bounds,
    it=0         # Saves iteration data
)

# Extract best point
best_T = net[best_idx * 2][2] / k  # Temperature in K
best_V = net[best_idx * 2][3]      # Volume

print(f"Best next point:")
print(f"  T = {best_T:.0f} K")
print(f"  V = {best_V:.2f} Å³")
print(f"  Expected variance reduction: {np.exp(float(score)):.1%}")
```

### 8.3 Iterative Improvement

```python
for iteration in range(5):
    # Find optimal point
    best_idx, score, var_old = GP.ad_step('triple', net, phases,
                                           bounds=bounds, it=iteration)
    
    # Run MD simulation at suggested point (external)
    T_new = net[best_idx * 2][2]
    V_new = net[best_idx * 2][3]
    
    # Suppose you get these results from MD:
    # E_new, P_new, E_err, P_err = run_md(T_new, V_new)
    
    # Add to GP
    X_add = [
        ['sol_fcc', 'd_1_0', T_new, V_new, 5, 2048],
        ['sol_fcc', 'd_0_1', T_new, V_new, 5, 2048]
    ]
    Y_add = [[E_new], [P_new]]
    err_add = [
        ['err', hash(tuple(X_add[0])), E_err**2],
        ['err', hash(tuple(X_add[1])), P_err**2]
    ]
    
    GP.add_points(X_add, Y_add, err_add)
    
    print(f"Iteration {iteration}: Added point at T={T_new/k:.0f}K, V={V_new:.2f}Å³")
```

---

## 9. Visualizing Results

### 9.1 Complete Phase Diagram

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))

# Color scheme
colors = {'liq': '#E91E63', 'sol_fcc': '#4CAF50', 'sol_bcc': '#2196F3'}

# Plot training points
for phase in phases:
    X_phase = np.array([x for x in GP.X if x[0] == phase])
    T_points = X_phase[::2, 2] / k  # Every other point (skip duplicates)
    V_points = X_phase[::2, 3]
    # Compute pressure at each point
    P_points = np.array([
        GP.predict_P(X_phase[i, 2], X_phase[i, 3], phase)[0] * Pk
        for i in range(0, len(X_phase), 2)
    ])
    ax.scatter(P_points, T_points, c=colors[phase], label=f'{phase} data', 
               alpha=0.6, s=50)

# FCC-Liquid boundary
P_fcc = np.linspace(9, 30, 15) / Pk
T_fcc = np.zeros(len(P_fcc))
T_fcc_err = np.zeros(len(P_fcc))
for i, P in enumerate(P_fcc):
    y, y_var, _ = GP.compute_var('pt', ['sol_fcc', 'liq'],
                                  bounds=[[10,15], [10,18], [370*k, 600*k]], P=P)
    T_fcc[i] = y[2] / k
    T_fcc_err[i] = y_var[2] / k

ax.fill_between(P_fcc*Pk, T_fcc-T_fcc_err, T_fcc+T_fcc_err, 
                alpha=0.3, color='red')
ax.plot(P_fcc*Pk, T_fcc, 'r-', lw=2, label='FCC-Liquid')

# BCC-Liquid boundary
P_bcc = np.linspace(0, 9, 15) / Pk
T_bcc = np.zeros(len(P_bcc))
T_bcc_err = np.zeros(len(P_bcc))
for i, P in enumerate(P_bcc):
    y, y_var, _ = GP.compute_var('pt', ['sol_bcc', 'liq'],
                                  bounds=[[12,22], [12,22], [350*k, 550*k]], P=P)
    T_bcc[i] = y[2] / k
    T_bcc_err[i] = y_var[2] / k

ax.fill_between(P_bcc*Pk, T_bcc-T_bcc_err, T_bcc+T_bcc_err, 
                alpha=0.3, color='red')
ax.plot(P_bcc*Pk, T_bcc, 'r-', lw=2)

# Triple point
ax.errorbar(P_triple, T_triple, xerr=0.5, yerr=point_var[3],
            fmt='ko', ms=10, capsize=5, label='Triple point')

# Labels
ax.set_xlabel('Pressure (GPa)', fontsize=14)
ax.set_ylabel('Temperature (K)', fontsize=14)
ax.set_title('Phase Diagram with Uncertainty', fontsize=16)
ax.legend(loc='upper left')
ax.set_xlim(0, 30)
ax.set_ylim(300, 700)

# Phase labels
ax.text(5, 400, 'BCC', fontsize=20, color=colors['sol_bcc'])
ax.text(22, 400, 'FCC', fontsize=20, color=colors['sol_fcc'])
ax.text(12, 600, 'Liquid', fontsize=20, color=colors['liq'])

plt.tight_layout()
plt.savefig('phase_diagram.pdf', dpi=300)
plt.show()
```

---

## 10. Advanced Topics

### 10.1 Custom Covariance Functions

Create your own kernel:

```python
import gmpy2 as gp

my_cov = {
    "my_phase": {
        "d_0_0_d_0_0": lambda x1, x2, th: (
            10**(2*th[0]) * gp.exp(-((x1[0]-x2[0])**2)/(2*th[1]**2))
        ),
        "d_0_0_d_0_1": lambda x1, x2, th: (
            10**(2*th[0]) * (x1[0]-x2[0]) / th[1]**2 
            * gp.exp(-((x1[0]-x2[0])**2)/(2*th[1]**2))
        ),
        # ... more derivatives
    }
}
```

### 10.2 Handling Convergence Issues

```python
# If optimization fails, try:

# 1. Better initial guess
th_init = GP.th.copy()
th_init[0] = mpfr(-5.0)  # Adjust amplitude

# 2. Bounded optimization
param_bounds = [(−10, 0), (0.01, 2), ...]  # Min, max for each param
result, th = GP.optimize(th_init, bounds=param_bounds)

# 3. Two-stage optimization
# First optimize amplitudes, then length scales
result1, th1 = GP.optimize(th_init, ind=[0, 6, 11])  # Amplitudes
result2, th2 = GP.optimize(th1, ind=[1, 2, 3, 4, 5])  # Scales
```

### 10.3 Debugging Tips

```python
# Check data loading
print(f"X shape: {len(GP.X)} points")
print(f"Y shape: {GP.Y.shape}")
print(f"Phases: {GP.phases}")

# Check covariance matrix condition
import numpy as np
K_float = np.array(GP.K, dtype=float)
cond = np.linalg.cond(K_float)
print(f"Condition number: {cond:.2e}")
if cond > 1e15:
    print("Warning: Ill-conditioned matrix!")

# Check predictions make sense
for phase in GP.phases:
    T_test = 500 * k
    V_test = 14.0
    P = GP.predict_P(T_test, V_test, phase)[0] * Pk
    print(f"{phase} at 500K, 14Å³: P = {P:.2f} GPa")
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `nan` in predictions | Extrapolating outside data | Check bounds, add data |
| Optimization stuck | Poor initial guess | Try multiple starting points |
| Large uncertainties | Sparse data | Add more training points |
| Negative variance | Numerical precision | Increase GMPY2 precision |
| Coexistence not found | Bounds too tight | Expand search bounds |

### Getting Help

1. Check example notebooks in `Examples/`
2. Read docstrings: `help(GP.compute_var)`
3. Open an issue on GitHub

---

## Next Steps

1. **Explore Examples**: See `Examples/Li/` for full lithium phase diagram
2. **Read Theory**: Check `docs/THEORY.md` for mathematical background
3. **API Details**: Full reference in `docs/API_REFERENCE.md`
