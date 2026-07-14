# BGK Collision (Single-Relaxation-Time)

## Mathematical Description

The BGK (Bhatnagar-Gross-Krook) collision operator is the simplest collision model.
It relaxes the distribution function toward equilibrium at a single rate:

```
f_i(x + c_i, t + 1) = f_i(x, t) - (1/τ) * (f_i - f_eq_i)
```

where:
- `τ` is the relaxation time
- `f_eq_i` is the equilibrium distribution
- `ω = 1/τ` is the relaxation rate

## Relation to Viscosity

The kinematic viscosity is related to relaxation time by:
```
ν = (τ - 0.5) * c_s²
```

where `c_s² = 1/3` for D2Q9/D3Q19.

## Implementation

```python
def bgk_collide(f, rho, u, omega):
    """BGK collision step."""
    feq = compute_equilibrium(rho, u)
    f_new = f - omega * (f - feq)
    return f_new
```

## Verification

- Poiseuille flow: Velocity profile matches analytical solution
- Couette flow: Linear velocity profile
- Taylor-Green vortex: Exponential decay matches analytical solution

## References

1. Bhatnagar, P. L., Gross, E. P., & Krook, M. (1954). "A model for collision
   processes in gases. I. Small amplitude processes in charged and neutral
   one-component systems." *Physical Review*, 94(3), 511.
2. d'Humières, D. (1992). "Generalized lattice-Boltzmann equations."
   *Progress in Astronautics and Aeronautics*, 159, 450.
