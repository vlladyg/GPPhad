# GPPhad Documentation

Welcome to the GPPhad documentation — a Python library for computing phase diagrams with uncertainty quantification using Gaussian Process regression.

---

## Documentation Overview

| Document | Description |
|----------|-------------|
| [README](../README.md) | Project overview, quick start, and installation |
| [Tutorial](TUTORIAL.md) | Step-by-step guide for new users |
| [API Reference](API_REFERENCE.md) | Complete API documentation |
| [Theory](THEORY.md) | Mathematical background and derivations |

---

## Quick Links

### Getting Started
- [Installation](../README.md#-installation)
- [Quick Start](../README.md#-quick-start)
- [Creating Your First GP](TUTORIAL.md#3-creating-your-first-gp)

### Core Functionality
- [GP_full Class](API_REFERENCE.md#gp_full-class)
- [Thermodynamic Predictions](API_REFERENCE.md#thermodynamic-property-methods)
- [Phase Coexistence](API_REFERENCE.md#phase-diagram-methods)

### Examples
- [Lithium Phase Diagram](../Examples/Li/Li_test.ipynb)
- [Thermal Expansion](../Examples/NaBr/NaBr_thermal_exp.ipynb)

### Advanced Topics
- [Active Learning](TUTORIAL.md#8-active-learning)
- [Custom Covariance Functions](TUTORIAL.md#101-custom-covariance-functions)
- [Numerical Considerations](THEORY.md#9-numerical-considerations)

---

## Package Structure

```
GPPhad/
├── __init__.py          # Main exports
│   ├── create_from_scratch()
│   ├── load()
│   ├── retrain()
│   └── ...
├── GP/
│   ├── __init__.py      # GP_full class
│   ├── _kernel.py       # Kernel base class
│   ├── _func.py         # Thermodynamic functions
│   ├── _optimize.py     # Hyperparameter optimization
│   └── phase_diagram/
│       ├── mean.py      # Phase coexistence solver
│       ├── var.py       # Uncertainty propagation
│       └── AL.py        # Active learning
├── two_stages/
│   ├── E0.py            # Zero-point energy GP
│   └── H.py             # Enthalpy GP
├── loader.py            # Data loading utilities
├── cov_dicts.py         # Predefined kernels
└── utils.py             # Helper functions
```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPPhad Workflow                          │
└─────────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ┌──────────────┐
   │   MD Data    │  → liq.dat, sol_fcc.dat, ...
   │ (LAMMPS/MD)  │  → E0.dat, H.dat (optional)
   └──────┬───────┘
          │
          ▼
2. GP CONSTRUCTION
   ┌──────────────┐
   │create_from_  │  → Load & preprocess data
   │  scratch()   │  → Build covariance matrix
   └──────┬───────┘
          │
          ▼
3. TRAINING
   ┌──────────────┐
   │  optimize()  │  → Maximize marginal likelihood
   │  retrain()   │  → Add experimental constraints
   └──────┬───────┘
          │
          ▼
4. PREDICTION
   ┌──────────────┐
   │ predict_*()  │  → F, S, P, E, B, α, γ
   │compute_var() │  → Phase boundaries + uncertainty
   └──────┬───────┘
          │
          ▼
5. ANALYSIS
   ┌──────────────┐
   │  ad_step()   │  → Active learning (optional)
   │   save()     │  → Persistence
   └──────────────┘
```

---

## Citation

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

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Examples**: See `Examples/` directory
- **Contact**: Vladimir Ladygin, Alexander Shapeev
