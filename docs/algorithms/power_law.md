# Power-Law (Ostwald-de Waele) Viscosity Model

## Mathematical Description

The power-law model describes shear-thinning (pseudoplastic) and shear-thickening
(dilatant) fluids:

```
ν = ν₀ * |γ̇|^(n-1)
```

where:
- `ν₀` is the reference viscosity
- `γ̇` is the shear rate magnitude
- `n` is the flow behavior index

## Behavior by Index

| Index | Type | Example |
|-------|------|---------|
| n < 1 | Shear-thinning | Blood, paint, ketchup |
| n = 1 | Newtonian | Water, air |
| n > 1 | Shear-thickening | Cornstarch, quicksand |

## Shear Rate in LBM

The shear rate is computed from the strain rate tensor:

```
|γ̇| = √(2 * S_ij * S_ij)
```

where `S_ij` is computed from non-equilibrium distributions.

## Implementation

```python
def power_law_viscosity(strain_rate_mag, base_viscosity, n=0.5):
    """Compute local viscosity for power-law fluid."""
    safe_shear = max(strain_rate_mag, 1e-10)  # Avoid division by zero
    return base_viscosity * safe_shear**(n - 1)
```

## Verification

- Poiseuille flow for power-law: Analytical velocity profile
  - n < 1: Plug-like profile (flatter center)
  - n > 1: Pointed profile (sharper center)

## References

1. Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). *Dynamics of Polymeric
   Liquids, Volume 1: Fluid Mechanics*. Wiley.
2. Chhabra, R. P., & Richardson, J. F. (2008). *Non-Newtonian Flow and Applied
   Rheology*. Butterworth-Heinemann. [DOI: 10.1016/B978-0-7506-8692-1.X5011-0](https://doi.org/10.1016/B978-0-7506-8692-1.X5011-0)
