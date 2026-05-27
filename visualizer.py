from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import numpy as np

if TYPE_CHECKING:
    from cpu_lbm import CPULBM2D


class FluidVisualizer:
    """PyGame-based 2D renderer for smoke density, obstacles, emitters, and HUD."""

    def __init__(self, width: int, height: int, scale: int = 5) -> None:
        self.width = width
        self.height = height
        self.scale = scale

        pygame.init()
        self.display_width = width * scale
        self.display_height = height * scale
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("CuFloda - Fluid Simulation")

        self.surface = pygame.Surface((width, height)).convert()

        self.obstacle_surf = pygame.Surface(
            (self.display_width, self.display_height), pygame.SRCALPHA
        )

        self.bg_color = (0, 0, 0)
        self.obstacle_color = (100, 100, 100)
        self.emitter_color = (255, 100, 50)

        self.font = pygame.font.Font(None, 36)
        self._cache_control_surfaces()

        self.paused = False
        self.running = True
        self.drawing_obstacle = False
        self.emitter_mode = False
        self._emitter_count = 0

    def _cache_control_surfaces(self) -> None:
        self.controls_surface = self.font.render(
            "O: obstacle mode | E: emitter mode | C: clear emitters",
            True,
            (180, 180, 180),
        )
        self.mode_obstacle_surface = self.font.render(
            "MODE: OBSTACLE (drag to draw)", True, (255, 200, 100)
        )
        self.mode_emitter_surface = self.font.render(
            "MODE: EMITTER (click to place)", True, (100, 200, 255)
        )

    def render_smoke(self, smoke: np.ndarray) -> None:
        # Fire-like color map: black→red→yellow→white as density increases.
        # Empirically tuned thresholds create a flame gradient.
        s_norm = np.clip(smoke / 0.3, 0, 1)

        r = np.clip(s_norm * 4.0 - 0.2, 0, 1) * 255
        g = np.clip((s_norm - 0.15) * 3.0, 0, 1) * 255
        b = np.clip((s_norm - 0.5) * 2.5, 0, 1) * 255

        r = r.astype(np.uint8)
        g = g.astype(np.uint8)
        b = b.astype(np.uint8)

        rgb = np.stack([r, g, b], axis=2)
        # PyGame blit_array expects (W, H, C) — transpose from (H, W, C)
        pygame.surfarray.blit_array(self.surface, np.transpose(rgb, (1, 0, 2)))

        scaled = pygame.transform.scale(
            self.surface, (self.display_width, self.display_height)
        )
        self.screen.blit(scaled, (0, 0))

    def render_obstacles(self, obstacles: np.ndarray) -> None:
        if not np.any(obstacles):
            return
        self.obstacle_surf.fill((0, 0, 0, 0))
        ys, xs = np.where(obstacles)
        for y, x in zip(ys, xs):
            self.obstacle_surf.fill(
                self.obstacle_color,
                (x * self.scale, y * self.scale, self.scale, self.scale),
            )
        self.screen.blit(self.obstacle_surf, (0, 0))

    def render_emitters(self, emitters: list[tuple[int, int, float]]) -> None:
        for x, y, _ in emitters:
            cx = x * self.scale + self.scale // 2
            cy = y * self.scale + self.scale // 2
            pygame.draw.circle(
                self.screen, self.emitter_color, (cx, cy), self.scale // 2
            )
            pygame.draw.circle(self.screen, (255, 255, 200), (cx, cy), self.scale // 4)

    def render_info(self, fps: float, step_count: int) -> None:
        info_text = (
            f"FPS: {fps:.1f} | Steps: {step_count}"
            f" | {'PAUSED' if self.paused else 'RUNNING'}"
            f" | Emitters: {self._emitter_count}"
        )
        info_surf = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surf, (10, 10))

        mode_surf = (
            self.mode_emitter_surface
            if self.emitter_mode
            else self.mode_obstacle_surface
        )
        self.screen.blit(mode_surf, (10, 50))

        self.screen.blit(self.controls_surface, (10, self.display_height - 40))

    def update(
        self,
        smoke: np.ndarray,
        obstacles: np.ndarray,
        emitters: list[tuple[int, int, float]],
        fps: float,
        step_count: int,
    ) -> None:
        self._emitter_count = len(emitters)
        self.screen.fill(self.bg_color)
        self.render_smoke(smoke)
        self.render_obstacles(obstacles)
        self.render_emitters(emitters)
        self.render_info(fps, step_count)
        pygame.display.flip()

    def handle_events(self, sim: CPULBM2D) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    sim.initialize(rho=1.0, u=0.1, v=0.0)
                    sim.clear_obstacles()
                elif event.key == pygame.K_e:
                    self.emitter_mode = True
                elif event.key == pygame.K_o:
                    self.emitter_mode = False
                elif event.key == pygame.K_c:
                    sim.clear_emitters()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = pygame.mouse.get_pos()
                    gx = x // self.scale
                    gy = y // self.scale
                    if self.emitter_mode:
                        sim.add_emitter(gx, gy, strength=0.05)
                    else:
                        self.drawing_obstacle = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing_obstacle = False

            elif event.type == pygame.MOUSEMOTION:
                if self.drawing_obstacle and not self.emitter_mode:
                    x, y = pygame.mouse.get_pos()
                    sim.add_obstacle(x // self.scale, y // self.scale, radius=3)

        return True

    def close(self) -> None:
        pygame.quit()
