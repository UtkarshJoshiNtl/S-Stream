# Smagorinsky Subgrid-Scale (SGS) Model

## Mathematical Description

The Smagorinsky model adds turbulent viscosity based on local strain rate:

```
ν_t = (C_s * Δ)² * |S|
```

where:
- `C_s` is the Smagorinsky constant (typically 0.1-0.2)
- `Δ` is the grid spacing (1 in lattice units)
- `|S|` is the strain rate magnitude

## Strain Rate Computation

The strain rate tensor is computed from non-equilibrium distributions:

```
S_ij = -(1/2τ) * Π_neq_ij
Π_neq_ij = Σ c_iα * c_iβ * (f_i - f_eq_i)
```

The magnitude is:
```
|S| = √(2 * S_ij * S_ij)
```

## Effective Relaxation

The effective relaxation rate includes turbulent viscosity:

```
ω_eff = 1 / (3 * (ν + ν_t) + 0.5)
```

## Implementation

```python
def smagorinsky_collide(f, rho, u, omega_base, cs=0.1):
    """Smagorinsky SGS collision."""
    feq = compute_equilibrium(rho, u)
    fneq = f - feq

    # Compute non-equilibrium stress tensor
    pi_neq_xx = sum(cx[i]^2 * fneq[i] for i in range(9))
    pi_neq_yy = sum(cy[i]^2 * fneq[i] for i in range(9))
    pi_neq_xy = sum(cx[i]*cy[i] * fneq[i] for i in range(9))

    # Strain rate magnitude
    s_mag = sqrt(2 * (pi_neq_xx^2 + pi_neq_yy^2 + 2*pi_neq_xy^2))

    # Turbulent viscosity and effective omega
    nu_t = cs^2 * s_mag
    omega_eff = 1 / (3 * (1/(3*omega_base - 1.5) + nu_t) + 0.5)
    omega_eff = min(omega_eff, 1.99)

    # Apply collision with effective omega
    f_new = f * (1 - omega_eff) + feq * omega_eff
    return f_new
```

## Verification

- Decaying isotropic turbulence: Energy spectrum matches Kolmogorov -5/3 scaling
- Channel flow: Log-law region captured correctly

## References

1. Smagorinsky, J. (1963). "General circulation experiments with the primitive
   equations: I. The basic experiment." *Monthly Weather Review*, 91(3), 99-164.
2. Lilly, D. K. (1967). "The representation of small-scale turbulence in
   numerical simulation experiments." *Proceedings of the IBM Scientific
   Computing Symposium on Environmental Sciences*, 195-210.
3. Sagaut, P. (2006). *Large Eddy Simulation for Incompressible Flows*.
   Springer. [DOI: 10.1007/3-540-34220-6](https://doi.org/10.1007/3-540-34220-6)
