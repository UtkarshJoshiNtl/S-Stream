# Carreau Viscosity Model

## Mathematical Description

The Carreau model is more realistic than power-law for polymer fluids.
It captures three regimes:

```
ν = ν∞ + (ν₀ - ν∞) * (1 + (λ * |γ̇|)²)^((n-1)/2)
```

where:
- `ν₀` is zero-shear viscosity (Newtonian plateau at low shear)
- `ν∞` is infinite-shear viscosity (Newtonian plateau at high shear)
- `λ` is the relaxation time (controls shear-thinning onset)
- `n` is the power-law index

## Regimes

1. **Low shear** (λ|γ̇| << 1): ν ≈ ν₀ (Newtonian plateau)
2. **Moderate shear** (λ|γ̇| ~ 1): Power-law region
3. **High shear** (λ|γ̇| >> 1): ν ≈ ν∞ (Newtonian plateau)

## Implementation

```python
def carreau_viscosity(strain_rate_mag, base_viscosity, n=0.5, lambda_val=1.0, nu_inf_ratio=0.01):
    """Compute local viscosity for Carreau fluid."""
    nu_inf = base_viscosity * nu_inf_ratio
    nu_0 = base_viscosity
    lambda_shear = lambda_val * strain_rate_mag
    viscosity = nu_inf + (nu_0 - nu_inf) * (1 + lambda_shear**2)**((n - 1) / 2)
    return viscosity
```

## Verification

- Poiseuille flow for Carreau: Velocity profile transitions from Newtonian
  (parabolic) to power-law (plug-like) depending on shear rate

## References

1. Carreau, P. J. (1972). "Rheological equations from molecular network theories."
   *Transactions of the Society of Rheology*, 16(1), 99-127.
2. Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). *Dynamics of Polymeric
   Liquids, Volume 1: Fluid Mechanics*. Wiley.
