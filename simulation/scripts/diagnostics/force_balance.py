"""Independent force check from a control-volume momentum balance.

The momentum-exchange drag is what the solver reports.  This script takes the
final velocity/density field saved by ``resolution_sweep.py`` and integrates
the steady x-momentum balance over a box containing the sphere:

    F_body_on_fluid = integral over the outer box faces of
                      [rho u_x (u.n) + p n_x - (tau.n)_x] dS

so the drag on the sphere is the negative of that integral.  The floor, when
the sphere rests on it, is a no-slip wall whose shear is evaluated exactly from
the saved first layer of populations, using the same half-way bounce-back
momentum exchange the solver uses.

Two caveats are reported rather than hidden:

* The tunnel in these tests is not a developed periodic channel and its flow is
  still decaying at 8000-12000 steps (see ``tunnel_drift.py``).  The balance
  therefore has a large unsteady term, and the absolute pressure (~0.34) cancels
  between inlet and outlet leaving an ill-conditioned small difference.
* The balance is therefore reported for two boxes, a wide one and a tight one
  ending at the rear of the sphere.  Their spread is part of the diagnostic.

Run after the sweep::

    .venv/bin/python simulation/scripts/diagnostics/force_balance.py \
        /tmp/opencode/sstream_diag/fields_D30_Cs012.npz \
        --reference /tmp/opencode/sstream_diag/empty_D30.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CS2, DATA_DIR, fmt, print_table  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sstream.lattice import DIRECTIONS, OPPOSITE  # noqa: E402

# (name, axis, n_x, n_y, n_z).  The face index is the lower or upper CV index
# depending on the sign of the normal component on that axis.
FACES = (
    ("xmin", 0, -1, 0, 0),
    ("xmax", 0, 1, 0, 0),
    ("ymin", 1, 0, -1, 0),
    ("ymax", 1, 0, 1, 0),
    ("zmax", 2, 0, 0, 1),
    ("zmin", 2, 0, 0, -1),
)


def _face_slice(axis: int, at: int, normal: tuple[int, int, int], lo: dict, hi: dict):
    idx = [slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1)]
    idx[axis] = at
    return tuple(idx)


def _face_terms(vel, rho, p, grads, nu_field, face, normal):
    """Convective, pressure and viscous x-flux through one cell-centred face."""
    ux = vel[..., 0][face]
    uy = vel[..., 1][face]
    uz = vel[..., 2][face]
    rho_f = rho[face]
    nx, ny, nz = normal
    conv = float(np.sum(rho_f * ux * (ux * nx + uy * ny + uz * nz), dtype=np.float64))
    press = float(np.sum(p[face] * nx, dtype=np.float64))

    dudx, dudy, dudz, dvdx, dwdx = grads
    nu_f = nu_field[face]
    tau_xx = 2.0 * rho_f * nu_f * dudx[face]
    tau_xy = rho_f * nu_f * (dudy[face] + dvdx[face])
    tau_xz = rho_f * nu_f * (dudz[face] + dwdx[face])
    visc = -float(np.sum(tau_xx * nx + tau_xy * ny + tau_xz * nz, dtype=np.float64))
    return conv, press, visc


def _floor_wall_force(f_floor):
    """Floor shear on the wall from the z=0 bounce-back populations."""
    force = np.zeros(3, dtype=np.float64)
    for i, c in enumerate(DIRECTIONS):
        if c[2] != 1:
            continue
        force -= 2.0 * np.asarray(c, dtype=np.float64) * float(
            np.sum(f_floor[OPPOSITE[i]], dtype=np.float64)
        )
    return force


def _pressure_drag(solid, rho):
    """Pressure-only drag on the sphere from all x-normal wetted faces."""
    p = CS2 * rho
    lower = float(np.sum(p[:-1, :, :][solid[1:, :, :] & ~solid[:-1, :, :]], dtype=np.float64))
    upper = float(np.sum(p[1:, :, :][solid[:-1, :, :] & ~solid[1:, :, :]], dtype=np.float64))
    return lower - upper


def _cv_terms(vel, rho, *, crop, cv, floor_contact, f_floor, tau, cs):
    """Compute the box-surface integrals from one velocity/density field."""
    x0, x1, y0, y1, z0, z1 = (int(v) for v in cv)
    xs, xe, ys, ye, zs, ze = (int(v) for v in crop)
    lo = {0: x0 - xs, 1: y0 - ys, 2: z0 - zs}
    hi = {0: x1 - xs, 1: y1 - ys, 2: z1 - zs}

    ux, uy, uz = (vel[..., i] for i in range(3))
    dudx = np.gradient(ux, axis=0)
    dudy = np.gradient(ux, axis=1)
    dudz = np.gradient(ux, axis=2)
    dvdx = np.gradient(uy, axis=0)
    dvdy = np.gradient(uy, axis=1)
    dvdz = np.gradient(uy, axis=2)
    dwdx = np.gradient(uz, axis=0)
    dwdy = np.gradient(uz, axis=1)
    dwdz = np.gradient(uz, axis=2)
    grads = (dudx, dudy, dudz, dvdx, dwdx)

    sxx, syy, szz = dudx, dvdy, dwdz
    sxy = 0.5 * (dudy + dvdx)
    sxz = 0.5 * (dudz + dwdx)
    syz = 0.5 * (dvdz + dwdy)
    strain2 = sxx**2 + syy**2 + szz**2 + 2.0 * (sxy**2 + sxz**2 + syz**2)
    nu_t = (cs * cs) * np.sqrt(np.maximum(2.0 * strain2, 0.0))
    nu_mol = CS2 * (tau - 0.5)
    nu_mol_field = np.full_like(nu_t, nu_mol)
    nu_eff = nu_mol_field + nu_t
    p = CS2 * rho

    terms = {"conv": 0.0, "press": 0.0, "visc_mol": 0.0, "visc_eff": 0.0,
             "floor": 0.0, "floor_wall_on_wall": 0.0}
    per_face = []
    for name, axis, nx, ny, nz in FACES:
        if axis == 2 and nz == -1 and floor_contact:
            continue  # the floor is not an x-momentum flux face; handled below
        normal = (nx, ny, nz)
        at = lo[axis] if (nx + ny + nz) < 0 else hi[axis]
        face = _face_slice(axis, at, normal, lo, hi)
        conv, press, visc_mol = _face_terms(vel, rho, p, grads, nu_mol_field, face, normal)
        _, _, visc_eff = _face_terms(vel, rho, p, grads, nu_eff, face, normal)
        terms["conv"] += conv
        terms["press"] += press
        terms["visc_mol"] += visc_mol
        terms["visc_eff"] += visc_eff
        per_face.append((name, normal, conv, press, visc_mol, visc_eff))

    if floor_contact:
        wall_force = _floor_wall_force(f_floor)
        # The floor is part of the outer boundary: ``wall_force[0]`` is the
        # fluid-on-wall momentum, so -tau_xj n_j integrated over the floor face
        # equals +wall_force[0] in the body-on-fluid balance Q.
        terms["floor_wall_on_wall"] = float(wall_force[0])
        terms["floor"] = float(wall_force[0])

    q_mol = terms["conv"] + terms["press"] + terms["visc_mol"] + terms["floor"]
    q_eff = terms["conv"] + terms["press"] + terms["visc_eff"] + terms["floor"]
    return {
        "terms": terms,
        "per_face": per_face,
        "q_mol": q_mol,
        "q_eff": q_eff,
        "nu_t_mean": float(nu_t.mean()),
        "nu_t_max": float(nu_t.max()),
        "nu_mol": nu_mol,
    }


def analyse(path: Path):
    data = np.load(path, allow_pickle=True)
    vel = np.asarray(data["velocity"], dtype=np.float64)
    rho = np.asarray(data["density"], dtype=np.float64)
    solid = np.asarray(data["solid"], dtype=bool)
    crop = np.asarray(data["crop"], dtype=int)
    cv = np.asarray(data["cv"], dtype=int)
    force_me = np.asarray(data["force_lattice"], dtype=np.float64)
    floor_contact = bool(data["floor_contact"][0])
    tau = float(data["tau"][0])
    u = float(data["u_lattice"][0])
    area = int(data["area_lattice"][0])
    cs = float(data["smagorinsky"][0])
    label = str(data["label"][0])
    f_floor = np.asarray(data["f_floor"], dtype=np.float64) if floor_contact else None

    main = _cv_terms(
        vel, rho, crop=crop, cv=cv, floor_contact=floor_contact,
        f_floor=f_floor, tau=tau, cs=cs,
    )
    # The saved box reaches 4D downstream.  When the sphere is close to the
    # outlet that face is clipped to nx-2, where the outlet boundary condition
    # distorts the velocity gradients.  A second, tight box ending at the rear
    # of the solid is much less sensitive to that; the spread between the two
    # is itself a measure of how ill-conditioned the balance is in a tunnel
    # that is still draining.
    solid_idx = np.argwhere(solid)
    rear_local = int(solid_idx[:, 0].max()) if solid_idx.size else int(cv[1] - crop[0])
    cv_tight = cv.copy()
    cv_tight[1] = min(int(cv[1]), int(crop[0]) + rear_local)
    tight = _cv_terms(
        vel, rho, crop=crop, cv=cv_tight, floor_contact=floor_contact,
        f_floor=f_floor, tau=tau, cs=cs,
    )

    scale = 2.0 / (u * u * area)
    cd_me = float(force_me[0] * scale)
    cd_press = float(_pressure_drag(solid, rho) * scale)
    result = {
        "label": label,
        "path": str(path),
        "floor_contact": floor_contact,
        "u_lattice": u,
        "tau": tau,
        "nu_mol": main["nu_mol"],
        "nu_t_mean": main["nu_t_mean"],
        "nu_t_max": main["nu_t_max"],
        "area_lattice": area,
        "force_me": force_me,
        "cd_me": cd_me,
        "cd_pressure_only": cd_press,
        "terms": main["terms"],
        "per_face": main["per_face"],
        "q_mol": main["q_mol"],
        "q_eff": main["q_eff"],
        "cd_cv_mol": float(-main["q_mol"] * scale),
        "cd_cv_eff": float(-main["q_eff"] * scale),
        "residual_mol": (-main["q_mol"] * scale - cd_me) / max(abs(cd_me), 1e-12),
        "residual_eff": (-main["q_eff"] * scale - cd_me) / max(abs(cd_me), 1e-12),
        "cv_tight": [int(v) for v in cv_tight],
        "cd_cv_tight": float(-tight["q_mol"] * scale),
        "terms_tight": tight["terms"],
        "residual_tight": (-tight["q_mol"] * scale - cd_me) / max(abs(cd_me), 1e-12),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="npz crop(s) from resolution_sweep.py")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths]
    if not paths:
        default = Path("/tmp/opencode/sstream_diag/fields_D30_Cs012.npz")
        if not default.exists():
            raise SystemExit(f"no crop at {default}; pass an npz path")
        paths = [default]
    results = []
    for path in paths:
        print(f"\n=== {path} ===")
        r = analyse(path)
        results.append(r)
        print(f"  label: {r['label']}  floor contact: {r['floor_contact']}")
        print(
            f"  u={r['u_lattice']:.4f} tau={r['tau']:.6f} nu_mol={r['nu_mol']:.6f} "
            f"nu_t mean={r['nu_t_mean']:.3e} area={r['area_lattice']} cells^2"
        )
        print(f"  momentum exchange: Fx={r['force_me'][0]:.6f}  Cd={r['cd_me']:.5f}")
        print(
            f"  raw CV (mol nu):   Fx={r['q_mol']:.6f}  Cd={r['cd_cv_mol']:.5f}  "
            f"residual={r['residual_mol']:.2%}"
        )
        print(
            f"  tight CV (x1={r['cv_tight'][1]}): Cd={r['cd_cv_tight']:.5f}  "
            f"residual={r['residual_tight']:.2%}"
        )
        print(f"  pressure-only sphere surface: Cd={r['cd_pressure_only']:.5f}")
        t = r["terms"]
        print(
            "  raw terms: "
            f"conv={t['conv']:+.6f} press={t['press']:+.6f} "
            f"visc_mol={t['visc_mol']:+.6f} visc_eff={t['visc_eff']:+.6f} "
            f"floor={t['floor']:+.6f}"
        )
        print("  per face (name, normal, conv, press, visc_mol, visc_eff):")
        for name, normal, conv, press, vm, ve in r["per_face"]:
            print(
                f"    {name:5s} n={normal}  {conv:+.6f}  {press:+.6f}  {vm:+.6f}  {ve:+.6f}"
            )

    headers = [
        "case", "floor", "Cd ME", "Cd CV wide", "residual wide",
        "Cd CV tight", "residual tight", "Cd pressure only",
        "conv", "press", "visc mol", "floor",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["label"],
                "yes" if r["floor_contact"] else "no",
                fmt(r["cd_me"], 5),
                fmt(r["cd_cv_mol"], 5),
                fmt(r["residual_mol"], 4),
                fmt(r["cd_cv_tight"], 5),
                fmt(r["residual_tight"], 4),
                fmt(r["cd_pressure_only"], 5),
                fmt(r["terms"]["conv"], 5),
                fmt(r["terms"]["press"], 5),
                fmt(r["terms"]["visc_mol"], 5),
                fmt(r["terms"]["floor"], 5),
            ]
        )
    print()
    print_table(rows, headers)
    out = DATA_DIR / "force_balance.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
