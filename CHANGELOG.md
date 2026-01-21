# Changelog

All notable changes to GPPhad will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1] - Current Release

### Added
- Active learning module (`AL.py`) for optimal experimental design
- Grüneisen parameter calculation (`predict_gamma`)
- Thermal expansion coefficient (`predict_alpha`)
- Two-stage zero-point energy extrapolation
- Enthalpy GP for solid phases
- Comprehensive covariance dictionaries (`cov_real`)
- Support for melting point constraints
- Pretty-print function for phase diagram results

### Changed
- Improved numerical stability with 400-bit precision
- Refactored phase diagram computation into modular components
- Enhanced error propagation for coexistence calculations

### Fixed
- Matrix conditioning issues at extreme temperatures
- Volume bound handling in optimization

---

## [1.0] - Initial Release

### Added
- Core Gaussian Process regression framework
- Multi-phase support (liquid, solid phases)
- Derivative observation handling
- Basic thermodynamic predictions (F, S, P, E)
- Phase coexistence calculations (P-T, T-P, triple points)
- Marginal likelihood optimization
- Data loading from MD simulation files
- Model persistence (save/load)

---

## Roadmap

### Planned Features
- [ ] GPU acceleration for large datasets
- [ ] Automatic differentiation for gradient computation
- [ ] Support for multi-component systems
- [ ] Web interface for visualization
- [ ] Integration with ASE/pymatgen

### Under Consideration
- Sparse GP approximations for scaling
- Neural network kernel extensions
- Bayesian optimization interface
