# Algorithm Reference Library

Clean-room reimplementations of algorithms from GPL-licensed projects.
Each document contains:
1. Mathematical description from published papers (not from GPL source)
2. Our implementation pseudocode
3. Verification test cases
4. Paper citation and DOI

## Algorithms

### Collision Operators
- [BGK Collision](bgk_collision.md) - Single-relaxation-time
- [TRT Collision](trt_collision.md) - Two-relaxation-time
- [MRT Collision](mrt_collision.md) - Multiple-relaxation-time

### Turbulence Models
- [Smagorinsky SGS](smagorinsky_sgs.md) - Subgrid-scale turbulence
- [WALE Model](wale_model.md) - Wall-Adapting Local Eddy-viscosity

### Boundary Conditions
- [Zou-He BC](zou_he_bc.md) - Velocity/pressure boundary conditions

### Thermal LBM
- [Boussinesq Approximation](boussinesq_approx.md) - Natural convection

### Non-Newtonian
- [Power-Law Model](power_law.md) - Shear-thinning/thickening
- [Carreau Model](carreau_model.md) - Polymer viscosity

## References

Each algorithm is documented with its original paper citation.
We only reference algorithms from published papers, not from GPL source code.

## License Compliance

These reimplementations are based on:
- Published peer-reviewed papers
- Mathematical derivations from textbooks
- Publicly available algorithmic descriptions

We do NOT copy code from GPL-licensed projects.
We read GPL source to understand the algorithm, close the source,
and implement from mathematical description and published papers.
