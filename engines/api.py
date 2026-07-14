"""REST API for S-Stream simulation engine.

Provides a lightweight HTTP interface for running simulations headless,
querying fields, and managing probes — enabling web frontends, CI/CD
integration, and cloud deployment.

Usage::

    python main.py --serve --port 8080

Endpoints:
    GET  /                    — API info
    GET  /engine              — Current engine metadata
    POST /run                 — Run N steps
    GET  /field?type=speed    — Get field as JSON array
    GET  /field/png?type=smoke — Get field as PNG image
    GET  /probe?id=0          — Get probe data
    GET  /obstacles           — Get obstacle mask
    POST /obstacle            — Add obstacle
    DELETE /obstacles         — Clear all obstacles
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from engines.base import SimEngine


class RunRequest(BaseModel):
    steps: int = 100


class ObstacleRequest(BaseModel):
    x: int = 64
    y: int = 64
    radius: int = 8


class EmitterRequest(BaseModel):
    x: int = 2
    y: int = 32
    strength: float = 0.05


def create_app(sim: SimEngine) -> Any:
    """Create a FastAPI app wrapping the given simulation engine."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for REST API mode. "
            "Install with: pip install sstream[api]"
        )

    app = FastAPI(
        title="S-Stream API",
        description="Lattice Boltzmann fluid simulation REST API",
        version="0.3.0",
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "S-Stream API",
            "version": "0.3.0",
            "engine": type(sim).__name__,
            "grid": list(sim.grid_shape),
            "fields": sim.get_field_names(),
        }

    @app.get("/engine")
    def engine_info() -> dict[str, Any]:
        return {
            "type": type(sim).__name__,
            "ndim": sim.ndim,
            "grid_shape": list(sim.grid_shape),
            "viscosity": sim.viscosity,
            "u_inflow": sim.u_inflow,
            "field_names": sim.get_field_names(),
            "obstacle_count": int(np.sum(sim.get_obstacles())),
            "emitter_count": sim.get_emitter_count(),
        }

    @app.post("/run")
    def run_steps(req: RunRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        sim.run(req.steps)
        elapsed = time.perf_counter() - t0
        h, w = sim.grid_shape[-2], sim.grid_shape[-1]
        mlups = (h * w * req.steps) / elapsed / 1e6 if elapsed > 0 else 0
        return {
            "steps": req.steps,
            "elapsed_s": round(elapsed, 4),
            "mlups": round(mlups, 1),
            "grid": list(sim.grid_shape),
        }

    @app.get("/field")
    def get_field(type: str = Query("speed", description="Field name")) -> dict[str, Any]:
        try:
            field = sim.get_field(type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "type": type,
            "shape": list(field.shape),
            "min": float(field.min()),
            "max": float(field.max()),
            "mean": float(field.mean()),
        }

    @app.get("/field/array")
    def get_field_array(type: str = Query("speed")) -> dict[str, Any]:
        try:
            field = sim.get_field(type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "type": type,
            "shape": list(field.shape),
            "data": field.tolist(),
        }

    @app.get("/field/png")
    def get_field_png(type: str = Query("smoke")) -> Response:
        try:
            import matplotlib.pyplot as plt

            field = sim.get_field(type)
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.imshow(field, cmap="viridis", origin="lower")
            ax.set_title(type.capitalize())
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return Response(content=buf.read(), media_type="image/png")
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="matplotlib required for PNG export. Install with: pip install sstream[notebook]",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/velocity")
    def get_velocity() -> dict[str, Any]:
        vel = sim.get_velocity()
        speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
        return {
            "shape": list(vel.shape),
            "speed_mean": float(speed.mean()),
            "speed_max": float(speed.max()),
        }

    @app.get("/pressure")
    def get_pressure() -> dict[str, Any]:
        p = sim.get_pressure()
        return {
            "shape": list(p.shape),
            "mean": float(p.mean()),
            "min": float(p.min()),
            "max": float(p.max()),
        }

    @app.get("/obstacles")
    def get_obstacles() -> dict[str, Any]:
        obs = sim.get_obstacles()
        return {
            "shape": list(obs.shape),
            "count": int(obs.sum()),
        }

    @app.post("/obstacle")
    def add_obstacle(req: ObstacleRequest) -> dict[str, Any]:
        sim.add_obstacle(req.x, req.y, req.radius)
        return {"status": "ok", "x": req.x, "y": req.y, "radius": req.radius}

    @app.delete("/obstacles")
    def clear_obstacles() -> dict[str, str]:
        sim.clear_obstacles()
        return {"status": "cleared"}

    @app.post("/emitter")
    def add_emitter(req: EmitterRequest) -> dict[str, Any]:
        sim.add_emitter(req.x, req.y, req.strength)
        return {"status": "ok", "x": req.x, "y": req.y, "strength": req.strength}

    @app.delete("/emitters")
    def clear_emitters() -> dict[str, str]:
        sim.clear_emitters()
        return {"status": "cleared"}

    @app.get("/probe")
    def get_probe(type: str = Query("smoke")) -> dict[str, Any]:
        try:
            field = sim.get_field(type)
            return {
                "type": type,
                "center": float(field[field.shape[0] // 2, field.shape[1] // 2]),
                "mean": float(field.mean()),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app
