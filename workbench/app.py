from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from analysis.regimes import detect_flow_regime
from analysis.sanity import check_sanity
from analysis.scorecard import compute_scorecard
from engines.base import SimEngine
from export.data import export_field_snapshot, export_probe_csv
from export.image import export_image
from export.report import export_markdown_report
from export.video import VideoRecorder
from presets.loader import find_preset, load_preset, validate_presets
from resources.theme import APP_STYLESHEET
from scene.probe import Probe, validate_scene_probes
from scene.scene import ProbeSpec, Scene, apply_to_sim, default_scene
from scene.serializer import load as scene_load
from scene.serializer import save as scene_save
from workbench.dialogs.export_dialog import ExportDialog
from workbench.dialogs.sweep_dialog import SweepDialog
from workbench.dialogs.wizard_dialog import StartDialog, WizardTemplate
from workbench.panels.analysis_panel import AnalysisPanel
from workbench.panels.outcome_panel import OutcomePanel
from workbench.panels.scene_panel import ScenePanel
from workbench.panels.sidebar import Sidebar, SidebarSection
from workbench.viewport import Viewport
from workbench.widgets.tool_strip import ToolStrip
from workbench.widgets.transport_bar import TransportBar

logger = logging.getLogger(__name__)

# Field names are not listed here on purpose. Every engine declares its own via
# `SimEngine.get_field_names()`, and the UI reads that list, so a field cannot be
# offered by the UI unless the active engine can actually produce it.

_DEFAULT_FIELD = "smoke"


class MainWindow(QMainWindow):
    def __init__(self, sim: SimEngine) -> None:
        super().__init__()
        self.sim = sim
        self.scene: Scene = default_scene()
        self._file_path: Path | None = None
        self.paused = False
        self.step_count = 0
        self._fps_timer = QTimer(self)
        self._fps_count = 0
        self._fps_value = 0.0
        self._recorder: VideoRecorder | None = None
        self._demo_target = 0
        self._demo_running = False
        self._expert_mode = False
        self._frame_start = 0.0
        self._speed_factor = 1.0
        self._state_history: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._max_history_steps = 300
        self._scrubbing = False

        self.setWindowTitle("S-Stream - Fluid Workbench")
        self.resize(1320, 840)
        self.setStyleSheet(APP_STYLESHEET)

        self._configure_domain_for_scene()
        apply_to_sim(self.scene, self.sim)

        self.viewport = Viewport()
        self.viewport.set_sim(sim)
        self.viewport.set_scene(self.scene)
        self.viewport.obstacle_created.connect(self._on_viewport_obstacle)
        self.viewport.probe_placed.connect(self._on_viewport_probe)

        # Build Main QSplitter (Left Sidebar + Right Viewport Area)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Right Container (Canvas + Timeline)
        right_container = QWidget(self.main_splitter)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        vp_frame = QFrame()
        vp_frame.setFrameStyle(QFrame.Shape.NoFrame)
        vp_frame.setStyleSheet(
            "QFrame { border: 1px solid #30363d; border-radius: 6px; "
            "background: #0d1117; }"
        )
        vp_layout = QVBoxLayout(vp_frame)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        vp_layout.setSpacing(0)
        self.tool_strip = ToolStrip()
        self.tool_strip.setStyleSheet(
            "QFrame#toolStrip { background: #161b22; "
            "border-bottom: 1px solid #30363d; "
            "border-top-left-radius: 6px; border-top-right-radius: 6px; }"
        )
        self.tool_strip.tool_selected.connect(self._set_draw_mode)
        vp_layout.addWidget(self.tool_strip)

        self.sanity_banner = QLabel("")
        self.sanity_banner.setVisible(False)
        vp_layout.addWidget(self.sanity_banner)
        vp_layout.addWidget(self.viewport, 1)

        right_layout.addWidget(vp_frame, 1)

        # --- Bottom Transport Bar (full width) ---
        self.transport_bar = TransportBar()
        self.transport_bar.play_toggled.connect(lambda _: self.toggle_pause())
        self.transport_bar.step_requested.connect(self.step_once)
        self.transport_bar.reset_requested.connect(self.reset)
        self.transport_bar.speed_changed.connect(self._on_speed_changed)
        self.transport_bar.timeline_slider.sliderPressed.connect(
            self._on_timeline_pressed
        )
        self.transport_bar.timeline_slider.sliderMoved.connect(self._on_timeline_moved)
        self.transport_bar.timeline_slider.sliderReleased.connect(
            self._on_timeline_released
        )
        self.timeline_slider = self.transport_bar.timeline_slider
        self.timeline_label = self.transport_bar.step_label
        right_layout.addWidget(self.transport_bar)

        # Create panels before adding to left sidebar
        self.runtime_probes: list[Probe] = []
        self._rebuild_probes()

        self.scene_panel = ScenePanel(sim, self.scene)
        self.scene_panel.scene_changed.connect(self._on_scene_changed)
        self.scene_panel.parameters_changed.connect(self._on_params_changed)

        self.analysis_panel = AnalysisPanel(sim)
        self.outcome_panel = OutcomePanel(sim, self.scene)

        # Left accordion sidebar
        self.sidebar = Sidebar(self.main_splitter)
        self.sidebar.setMinimumWidth(280)
        self.sidebar.setMaximumWidth(400)
        self.sidebar.add_section(
            SidebarSection("scene", "SCENE", self.scene_panel, expanded=True)
        )
        self.sidebar.add_section(
            SidebarSection(
                "parameters",
                "PARAMETERS",
                self._build_parameters_widget(),
                expanded=True,
            )
        )
        self.sidebar.add_section(
            SidebarSection(
                "visualize",
                "VISUALIZE",
                self._build_visualize_widget(),
                expanded=True,
            )
        )
        self.sidebar.add_section(
            SidebarSection("analysis", "ANALYSIS", self.analysis_panel, expanded=True)
        )
        self.sidebar.add_section(
            SidebarSection("outcome", "OUTCOME", self.outcome_panel, expanded=True)
        )

        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(right_container)
        self.main_splitter.setSizes([280, 1040])

        self.setCentralWidget(self.main_splitter)

        self._sync_analysis_probes()
        # All four holders exist by now, so the initial sharing goes through the
        # same call as every later adoption rather than repeating the list.
        self._share_scene()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_shortcuts()
        self.set_expert_mode(False)  # Default to Beginner mode
        self._show_welcome_if_first()

        self._auto_detect_view()
        self._sync_physics_mode_combo()

        self._fps_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self._update_re_label()
        self._update_sanity_banner()
        self._report_recipe_problems()

    def _report_recipe_problems(self) -> None:
        problems = self._check_recipe_actions()
        if not problems:
            return
        for problem in problems:
            logger.error("Preset library problem: %s", problem)
        self.status.showMessage(
            f"{len(problems)} preset problem(s): {problems[0]}", 12000
        )

    def _setup_toolbar(self) -> None:
        self.toolbar = QToolBar("Simulation")
        self.addToolBar(self.toolbar)

        # --- Scene management ---
        self.start_btn = QPushButton("Start…")
        self.start_btn.setToolTip(
            "Open guided flow stories and presets\n"
            "Why: Quick access to learning scenarios"
        )
        self.start_btn.clicked.connect(self._open_wizard)
        self.toolbar.addWidget(self.start_btn)

        self.toolbar.addSeparator()

        # --- Workflow tools ---
        self.demo_btn = QPushButton("Run Demo")
        self.demo_btn.setToolTip(
            "Auto-run preset to settled state\n"
            "Why: See the final flow pattern without waiting"
        )
        self.demo_btn.clicked.connect(self._run_guided_demo)
        self.toolbar.addWidget(self.demo_btn)

        self.presets_btn = QPushButton("Presets")
        self.presets_btn.setToolTip(
            "Load saved scene files\nWhy: Reuse or share your experiments"
        )
        self.presets_btn.clicked.connect(self._open_preset_dialog)
        self.toolbar.addWidget(self.presets_btn)

        self.export_fig_btn = QPushButton("Export Figure")
        self.export_fig_btn.setToolTip(
            "Save high-res image or report\nWhy: Include in lab reports or"
            " presentations"
        )
        self.export_fig_btn.clicked.connect(self._quick_export_figure)
        self.toolbar.addWidget(self.export_fig_btn)

        self.toolbar.addSeparator()

        # --- Mode toggle ---
        self.mode_btn = QPushButton("Beginner")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setToolTip(
            "Toggle Beginner/Expert mode\n"
            "Why: Beginner hides advanced tools for learning"
        )
        self.mode_btn.clicked.connect(self._toggle_mode)
        self.toolbar.addWidget(self.mode_btn)

        self.toolbar.addSeparator()

        # --- Advanced tools (hidden in Beginner mode) ---
        self.recipes_btn = QPushButton("Recipes")
        self.recipes_btn.setToolTip(
            "Guided workflow recipes\nWhy: Step-by-step instructions for common tasks"
        )
        self.recipes_btn.clicked.connect(self._open_recipes_dialog)
        self.toolbar.addWidget(self.recipes_btn)

        self.sweep_re_btn = QPushButton("Sweep Re")
        self.sweep_re_btn.setToolTip(
            "Parameter sweep across Reynolds numbers\n"
            "Why: See how flow changes with Re (generates plots)"
        )
        self.sweep_re_btn.clicked.connect(self._open_sweep_dialog)
        self.toolbar.addWidget(self.sweep_re_btn)

        self.ai_btn = QPushButton("AI")
        self.ai_btn.setCheckable(True)
        self.ai_btn.setEnabled(False)
        self.ai_btn.setToolTip(
            "AI tutor (coming soon)\nWhy: Get explanations and guidance"
        )
        self.ai_btn.clicked.connect(self._toggle_ai_preview)
        self.toolbar.addWidget(self.ai_btn)

        self.perf_btn = QPushButton("Perf")
        self.perf_btn.setCheckable(True)
        self.perf_btn.setToolTip(
            "Show performance metrics (FPS, MLUPs)\nWhy: Monitor simulation speed"
        )
        self.perf_btn.clicked.connect(self._toggle_perf)
        self.toolbar.addWidget(self.perf_btn)

        self.physics_combo = QComboBox()
        self.physics_combo.addItem("Standard", "standard")
        self.physics_combo.addItem("Liquid  [Experimental]", "liquid")
        self.physics_combo.addItem("Oil-water  [Experimental]", "oil-water")
        self.physics_combo.setToolTip(
            "Physics mode (recreates the simulation engine)\n"
            "Why: Standard for single-phase, Liquid for multiphase"
        )
        self.physics_combo.currentIndexChanged.connect(self._on_physics_mode_changed)
        self.physics_combo_label = QLabel("Physics:")
        self.toolbar.addWidget(self.physics_combo_label)
        self.toolbar.addWidget(self.physics_combo)

        self.gpu_btn = QPushButton("GPU")
        self.gpu_btn.setCheckable(True)
        self.gpu_btn.setToolTip(
            "Enable GPU acceleration (requires CuPy)\n"
            "Why: Faster simulation on larger grids"
        )
        self.gpu_btn.clicked.connect(self._toggle_gpu)
        self.toolbar.addWidget(self.gpu_btn)

        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip(
            "Record video (MP4/GIF)\nWhy: Create animations for presentations"
        )
        self.record_btn.clicked.connect(self._toggle_recording)
        self.toolbar.addWidget(self.record_btn)

    def _setup_statusbar(self) -> None:
        self.status = QStatusBar()
        self.re_label = QLabel("Re: -")
        self.status.addWidget(self.re_label)
        hints = QLabel(
            "Space: Play/Pause | .: Step | J/K/L: Speed | 1-5: Tools | "
            "V: Field | S: Streams | Esc: Cancel"
        )
        hints.setStyleSheet("color: #64748b;")
        self.status.addWidget(hints)
        self.status_label = QLabel("Step 0  |  FPS: -")
        self.status.addPermanentWidget(self.status_label)
        self.grid_label = QLabel(self._grid_label_text())
        self.status.addPermanentWidget(self.grid_label)
        self.setStatusBar(self.status)

    def _setup_shortcuts(self) -> None:
        pause_shortcut = QAction(self)
        pause_shortcut.setShortcut(QKeySequence(Qt.Key.Key_Space))
        pause_shortcut.triggered.connect(self.toggle_pause)
        self.addAction(pause_shortcut)

        reset_shortcut = QAction(self)
        reset_shortcut.setShortcut(QKeySequence(Qt.Key.Key_R))
        reset_shortcut.triggered.connect(self.reset)
        self.addAction(reset_shortcut)

        quit_shortcut = QAction(self)
        quit_shortcut.setShortcut(QKeySequence("Ctrl+Q"))
        quit_shortcut.triggered.connect(self.close)
        self.addAction(quit_shortcut)

        # Viewport-focused shortcuts (only fire while the canvas has focus,
        # so typing in sidebar spinboxes/comboes is never intercepted).
        vp = self.viewport
        viewport_map = {
            Qt.Key_Period: self.step_once,
            Qt.Key_K: self.toggle_pause,
            Qt.Key_J: lambda: self._shift_speed(-1),
            Qt.Key_L: lambda: self._shift_speed(+1),
            Qt.Key_V: self._cycle_field,
            Qt.Key_S: lambda: self._on_vis_toggle(
                "streamlines", 0 if vp._show_streamlines else 2
            ),
            Qt.Key_1: lambda: self._set_draw_mode("select"),
            Qt.Key_2: lambda: self._set_draw_mode("circle"),
            Qt.Key_3: lambda: self._set_draw_mode("rect"),
            Qt.Key_4: lambda: self._set_draw_mode("polygon"),
            Qt.Key_5: lambda: self._set_draw_mode("probe"),
        }
        for key, handler in viewport_map.items():
            sc = QShortcut(QKeySequence(key), vp)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(handler)

    def _shift_speed(self, delta: int) -> None:
        combo = self.transport_bar.speed_combo
        idx = combo.currentIndex()
        idx = max(0, min(combo.count() - 1, idx + delta))
        combo.setCurrentIndex(idx)

    def _cycle_field(self) -> None:
        idx = self.field_combo.currentIndex()
        self.field_combo.setCurrentIndex((idx + 1) % self.field_combo.count())

    def _setup_menus(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._file_new)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._file_open)
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._file_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._file_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._open_export_dialog)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = menu.addMenu("&Tools")
        wizard_action = QAction("&Start...", self)
        wizard_action.setShortcut(QKeySequence("Ctrl+W"))
        wizard_action.triggered.connect(self._open_wizard)
        tools_menu.addAction(wizard_action)

        open_preset_action = QAction("Open &Preset...", self)
        open_preset_action.triggered.connect(self._open_preset_dialog)
        tools_menu.addAction(open_preset_action)

        self.recipes_action = QAction("&Recipes...", self)
        self.recipes_action.triggered.connect(self._open_recipes_dialog)
        tools_menu.addAction(self.recipes_action)

        self.sweep_action = QAction("&Sweep...", self)
        self.sweep_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.sweep_action.triggered.connect(self._open_sweep_dialog)
        tools_menu.addAction(self.sweep_action)

    def _file_new(self) -> None:
        self._adopt_scene(default_scene())
        self._file_path = None
        self._apply_and_refresh()

    def _file_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene", "", "Scene Files (*.json)"
        )
        if not path:
            return
        try:
            self._adopt_scene(scene_load(path))
            self._file_path = Path(path)
            self._apply_and_refresh()
        except Exception as e:
            QMessageBox.warning(self, "Open Failed", str(e))

    def _file_save(self) -> None:
        if self._file_path:
            scene_save(self.scene, self._file_path)
        else:
            self._file_save_as()

    def _file_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", "", "Scene Files (*.json)"
        )
        if not path:
            return
        self._file_path = Path(path)
        scene_save(self.scene, self._file_path)
        self._update_title()

    def _rebuild_probes(self) -> None:
        self.runtime_probes = [Probe(spec) for spec in self.scene.probes]
        self.viewport.set_probes(self.runtime_probes)

    def _adopt_scene(self, scene: Scene) -> None:
        """Take ownership of a Scene.

        The window is the single owner of scene identity. Four other objects hold a
        reference -- the viewport, the scene panel, the analysis panel and the
        outcome panel -- and a load that updated only `self.scene` left them reading
        the document that was just discarded. This is the one place a new Scene
        enters the app; `_share_scene()` re-points the holders.
        """
        self.scene = scene
        # The stacks hold JSON snapshots of the document being replaced, so undo
        # after a load would reinstate the previous scene with nothing to show that
        # it had. There is no meaningful undo across a document boundary.
        self.scene_panel.clear_history()
        self._share_scene()

    def _share_scene(self) -> None:
        """Re-point every holder at `self.scene`.

        Called by `_adopt_scene` and by `_on_scene_changed` whenever a holder turns
        out to be looking at a different object. The scene panel restores undo
        history by replacing its own reference, so without this the panel would keep
        editing a scene the viewport, probes, save and export cannot see.
        """
        self.viewport.set_scene(self.scene)
        self.scene_panel.set_scene(self.scene)
        self.analysis_panel.set_scene(self.scene)
        self.outcome_panel.set_scene(self.scene)

    def _sync_analysis_probes(self) -> None:
        self.analysis_panel.set_probes(self.runtime_probes)
        if hasattr(self, "outcome_panel"):
            self.outcome_panel.set_probes(self.runtime_probes)

    def _apply_and_refresh(self) -> None:
        self._configure_domain_for_scene()
        # Before apply_to_sim: it seeds the initial condition but cannot resize,
        # and a scene loaded onto a mismatched engine silently mis-places every
        # obstacle and probe.
        self._ensure_engine_matches_scene()
        apply_to_sim(self.scene, self.sim)
        self._report_probe_field_problems()
        self._rebuild_probes()
        self._sync_analysis_probes()
        # Probes are rebuilt above, so this clears panel-side state rather than
        # the (now fresh) probe objects: stability trace, tick phase, timeline,
        # scrubbing, demo and recorder.
        self._reset_runtime_state()
        self._share_scene()
        self.scene_panel.sync_params_from_scene()
        self.grid_label.setText(self._grid_label_text())
        cmap = self.scene.product.recommended_colormap
        self._set_colormap(cmap)
        self._demo_target = self.scene.product.autorun_steps
        self.outcome_panel.set_demo_target(self._demo_target)
        # `force`: update_outcome refreshes every fifteenth step, so without it the
        # regime card kept describing the scene that was just discarded for up to
        # fifteen steps after the load -- and a paused window never ticks at all.
        self.outcome_panel.update_outcome(self.step_count, force=True)
        self._update_title()
        self._update_sanity_banner()
        self._update_re_label()
        self._focus_outcome()

    def _report_probe_field_problems(self) -> None:
        """Surface probe fields the engine cannot record.

        Reached whenever a scene is applied -- loading a preset, opening a saved
        file, or switching engine -- so a probe that will never record anything
        says so now instead of showing a permanently flat trace.
        """
        problems = validate_scene_probes(self.sim, self.scene)
        for problem in problems:
            logger.warning("Probe field problem: %s", problem)
        if problems:
            self.status.showMessage(
                f"{len(problems)} probe field(s) unavailable on "
                f"{type(self.sim).__name__} — see log",
                8000,
            )

    def _configure_domain_for_scene(self) -> None:
        """Enable cavity lid when the Start template / scene name matches."""
        if not hasattr(self.sim, "domain_mode"):
            return
        from engines.lbm2d import DOMAIN_CAVITY, DOMAIN_CHANNEL

        if self.scene.name == "Lid-Driven Cavity":
            self.sim.domain_mode = DOMAIN_CAVITY
            self.sim.lid_velocity = 0.1
            self.sim.u_inflow = 0.0
            self.scene.u_inflow = 0.0
        elif getattr(self.sim, "domain_mode", None) == DOMAIN_CAVITY:
            self.sim.domain_mode = DOMAIN_CHANNEL

    def _grid_label_text(self) -> str:
        gs = self.sim.grid_shape
        return f"{gs[1]}x{gs[0]}  |  nu {self.sim.viscosity:.4f}"

    def _update_title(self) -> None:
        name = self._file_path.stem if self._file_path else self.scene.name
        self.setWindowTitle(f"S-Stream - {name}")

    def _update_re_label(self) -> None:
        from analysis.physics import characteristic_length, reynolds_number

        length = characteristic_length(self.scene)
        re = reynolds_number(self.sim, length)
        self.re_label.setText(f"Re: {re:.1f}")
        self._update_re_pill(re)

    def _update_re_pill(self, re: float) -> None:
        if not hasattr(self, "re_pill_label"):
            return
        if re < 500:
            color = "#3fb950"
        elif re < 10000:
            color = "#d29922"
        else:
            color = "#f85149"
        self.re_pill_label.setText(f"Re: {re:.1f}" if re > 0 else "Re: -")
        self.re_pill_label.setStyleSheet(
            f"QLabel {{ background: #21262d; color: {color}; "
            f"border: 1px solid {color}; "
            "border-radius: 9px; padding: 2px 10px; "
            "font-family: 'JetBrains Mono', monospace; "
            "font-size: 11px; font-weight: 600; }"
        )

    def _focus_outcome(self) -> None:
        if hasattr(self, "sidebar") and hasattr(self.sidebar, "expand_section"):
            self.sidebar.expand_section("outcome")

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.transport_bar.set_play_state(not self.paused)

    def _record_state_history(self) -> None:
        if self._scrubbing:
            return
        f_copy = np.copy(getattr(self.sim, "f", self.sim.get_velocity_view()))
        rho_copy = np.copy(getattr(self.sim, "rho", self.sim.get_pressure()))
        self._state_history.append((self.step_count, f_copy, rho_copy))
        if len(self._state_history) > self._max_history_steps:
            self._state_history.pop(0)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, len(self._state_history) - 1)
        self.timeline_slider.setValue(len(self._state_history) - 1)
        self.timeline_slider.blockSignals(False)
        self.timeline_label.setText(f"Step {self.step_count}")

    def _on_timeline_pressed(self) -> None:
        self._scrubbing = True
        self.paused = True
        self.transport_bar.set_play_state(False)

    def _on_timeline_moved(self, index: int) -> None:
        if 0 <= index < len(self._state_history):
            step, f_snap, rho_snap = self._state_history[index]
            if hasattr(self.sim, "f"):
                np.copyto(self.sim.f, f_snap)
            if hasattr(self.sim, "rho"):
                np.copyto(self.sim.rho, rho_snap)
            self.timeline_label.setText(f"Step {step} (History)")
            self.viewport.update()

    def _on_timeline_released(self) -> None:
        self._scrubbing = False

    def step_once(self) -> None:
        self.sim.step()
        self.step_count += 1
        self._record_state_history()
        self.analysis_panel.tick(1.0)
        self.outcome_panel.update_outcome(self.step_count)
        self.viewport.update()
        self._update_re_label()

    def _reset_runtime_state(self) -> None:
        """Clear every piece of state derived from the previous run.

        Reset, scene load, file open and preset load all invalidate the same
        derived state, and each used to clear a different subset of it. The
        residue was visible: after a reset the stability trace still drew the
        previous run's convergence curve, probe plots still carried old samples,
        `_tick_counter` stayed offset so probe sampling no longer lined up with
        the step count it is plotted against, and a reset triggered mid-drag left
        `_scrubbing` True -- at which point `tick()` stops stepping forever.

        One path, so a new piece of derived state is cleared by construction
        rather than by remembering to add it to four call sites.
        """
        self.step_count = 0

        # Timeline / history scrubbing.
        self._scrubbing = False
        self._state_history.clear()
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.timeline_label.setText("Step 0")

        # Guided demo.
        self._demo_running = False

        # Recording must not continue into a fresh run's first frame.
        if self._recorder is not None:
            self._stop_video_recording()

        # Panel-side derived state: probe histories, stability trace, tick phase.
        self.analysis_panel.reset_runtime()
        self.viewport.invalidate_caches()

    def reset(self, silent: bool = False) -> None:
        if not silent and self.step_count > 0:
            confirm = QMessageBox.question(
                self,
                "Reset Simulation",
                "Reset will clear all simulation state. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        self._configure_domain_for_scene()
        # reset() applies directly rather than via _apply_and_refresh, so it
        # needs the same grid guard: the scene's width/height can have been
        # edited in the properties editor since the engine was built.
        self._ensure_engine_matches_scene()
        apply_to_sim(self.scene, self.sim)
        self._reset_runtime_state()
        self.outcome_panel.update_outcome(self.step_count, force=True)
        self._update_re_label()
        self._update_sanity_banner()

    def tick(self) -> None:
        if not self.paused and not self._scrubbing:
            self.sim.step()
            self.step_count += 1
            self._record_state_history()
            self.analysis_panel.tick(1.0)
            if self._demo_running and self._demo_target > 0:
                if self.step_count >= self._demo_target:
                    self._demo_running = False
                    self.paused = True
                    self.transport_bar.set_play_state(False)
                    self.demo_btn.setText("Run Demo")
        self.outcome_panel.update_outcome(self.step_count)
        self.viewport.update()
        if self._recorder is not None and self._recorder.recording:
            if not self._recorder.add_frame(self.viewport.grab().toImage()):
                self._stop_video_recording()
        self._fps_count += 1
        fps_str = f"{self._fps_value:.0f}"
        self.status_label.setText(f"Step {self.step_count}  |  FPS: {fps_str}")
        self.transport_bar.set_fps(self._fps_value)
        self._update_re_label()
        elapsed = time.perf_counter() - self._frame_start
        interval = int(33 / self._speed_factor) - int(elapsed * 1000)
        self.timer.start(max(1, interval))
        self._frame_start = time.perf_counter()

    def _on_speed_changed(self, factor: float) -> None:
        self._speed_factor = factor

    def _update_sanity_banner(self) -> None:
        if not hasattr(self, "sanity_banner"):
            return
        warnings = check_sanity(
            self.sim,
            self.scene,
            probes=self.runtime_probes,
            step_count=self.step_count,
        )
        if not warnings:
            text = "Setup looks stable"
            color = "#3fb950"
        else:
            danger = [w for w in warnings if w.level == "danger"]
            warn = [w for w in warnings if w.level == "warn"]
            if danger:
                color = "#f85149"
                text = "Critical: " + "; ".join(w.title for w in danger)
            elif warn:
                color = "#d29922"
                text = "Caution: " + "; ".join(w.title for w in warn)
            else:
                color = "#8b949e"
                text = "Notes: " + "; ".join(w.title for w in warnings)
        self.sanity_banner.setText(text)
        self.sanity_banner.setToolTip("\n".join(w.message for w in warnings))
        self.sanity_banner.setVisible(True)
        self.sanity_banner.setStyleSheet(
            f"QLabel {{ background: #161b22; color: {color}; "
            "border-bottom: 1px solid #30363d; "
            "padding: 4px 12px; font-size: 11px; font-weight: 600; }"
        )

    def _set_draw_mode(self, mode: str | None) -> None:
        self.tool_strip.set_mode(mode)
        self.viewport.set_draw_mode(mode)

    def _on_viewport_obstacle(self, obs) -> None:
        self.scene_panel.add_obstacle_from_viewport(obs)

    def _on_viewport_probe(self, spec: ProbeSpec) -> None:
        # The panel owns the snapshot, so a probe placed on the canvas is undoable
        # like every other scene edit.
        self.scene_panel.add_probe_from_viewport(spec)
        self._update_title()

    def _on_scene_changed(self) -> None:
        # The scene panel restores undo history by replacing its own reference, so
        # this is where a divergent object is re-adopted rather than left for the
        # viewport, probes, save and export to disagree about. MainWindow stays the
        # sole owner: if a holder ever points elsewhere, it is corrected here
        # instead of becoming a second source of truth.
        if self.scene_panel.scene is not self.scene:
            logger.debug(
                "Adopting scene re-pointed by %s: %r -> %r",
                type(self.scene_panel).__name__,
                id(self.scene),
                id(self.scene_panel.scene),
            )
            self.scene = self.scene_panel.scene
        self.viewport.set_scene(self.scene)
        self.analysis_panel.set_scene(self.scene)
        self.outcome_panel.set_scene(self.scene)
        self._rebuild_probes()
        self._sync_analysis_probes()
        self.scene_panel.sync_params_from_scene()
        self.grid_label.setText(self._grid_label_text())
        self._update_title()

    def _on_params_changed(self) -> None:
        self.grid_label.setText(self._grid_label_text())
        self.outcome_panel.update_outcome(self.step_count, force=True)

    def _update_fps(self) -> None:
        self._fps_value = self._fps_count
        self._fps_count = 0

    def _auto_detect_view(self) -> None:
        if not hasattr(self, "field_combo"):
            return
        available = self._available_fields()
        preferred = (
            getattr(self.scene.product, "recommended_colormap", "") or _DEFAULT_FIELD
        )
        self._set_colormap(preferred if preferred in available else available[0])

    def _build_parameters_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        pill_row = QHBoxLayout()
        self.re_pill_label = QLabel("Re: -")
        self.re_pill_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pill_row.addWidget(self.re_pill_label)
        pill_row.addStretch(1)
        layout.addLayout(pill_row)

        layout.addWidget(self.scene_panel.parameter_group)
        layout.addStretch(1)
        self._update_re_pill(0.0)
        return widget

    def _build_visualize_widget(self) -> QWidget:
        from resources.colormaps import FIELD_REGISTRY

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        field_row = QHBoxLayout()
        field_lbl = QLabel("Field")
        field_lbl.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: 600;")
        field_row.addWidget(field_lbl)
        self.field_combo = QComboBox()
        for name in self._available_fields():
            info = FIELD_REGISTRY.get(name)
            label = info.label if info else name.capitalize()
            self.field_combo.addItem(label, name)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_row.addWidget(self.field_combo, 1)
        layout.addLayout(field_row)

        self._vis_checkboxes: list[tuple[QCheckBox, str]] = []
        for key, text in (
            ("streamlines", "Streamlines"),
            ("quiver", "Velocity arrows"),
            ("contours", "Pressure contours"),
            ("force", "Force arrows"),
            ("particles", "Particles"),
            ("hud", "Physics HUD"),
        ):
            cb = QCheckBox(text)
            cb.setChecked(key == "hud")
            cb.stateChanged.connect(lambda state, k=key: self._on_vis_toggle(k, state))
            layout.addWidget(cb)
            self._vis_checkboxes.append((cb, key))
        layout.addStretch(1)
        return widget

    def _on_field_changed(self, index: int) -> None:
        name = self.field_combo.itemData(index)
        if name:
            self._set_colormap(name)

    def _on_vis_toggle(self, key: str, state: int) -> None:
        checked = bool(state)
        setters = {
            "streamlines": self.viewport.set_show_streamlines,
            "quiver": self.viewport.set_show_quiver,
            "contours": self.viewport.set_show_contours,
            "force": self.viewport.set_show_force_arrows,
            "particles": self.viewport.set_show_particles,
            "hud": self.viewport.set_show_hud,
        }
        setters[key](checked)

    def _available_fields(self) -> list[str]:
        """Field names the active engine can actually produce.

        Always includes "smoke" as a fallback so the combo is never empty; if an
        engine ever fails to declare its fields we would rather show smoke than
        leave the user with nothing.
        """
        names = list(self.sim.get_field_names())
        if _DEFAULT_FIELD not in names:
            names.insert(0, _DEFAULT_FIELD)
        return names

    def _set_colormap(self, name: str) -> None:
        available = self._available_fields()
        if name not in available:
            fallback = available[0]
            logger.warning(
                "Field %r is not available on %s; falling back to %r",
                name,
                type(self.sim).__name__,
                fallback,
            )
            name = fallback
        self.viewport.set_colormap(name)
        self.analysis_panel.set_colormap(name)
        if hasattr(self, "field_combo"):
            idx = self.field_combo.findData(name)
            if idx >= 0:
                self.field_combo.blockSignals(True)
                self.field_combo.setCurrentIndex(idx)
                self.field_combo.blockSignals(False)

    def _show_welcome_if_first(self) -> None:
        from PySide6.QtCore import QSettings

        settings = QSettings("S-Stream", "S-Stream")
        if settings.value("welcome_shown", False, type=bool):
            return
        settings.setValue("welcome_shown", True)
        self._open_wizard()

    def _open_wizard(self) -> None:
        dialog = StartDialog(self, tab=0)
        dialog.template_selected.connect(self._on_wizard_template)
        dialog.preset_selected.connect(self._load_preset_file)
        self._active_dialog = dialog
        dialog.open()

    def _prompt_poe_prediction(self, scene_name: str) -> None:
        poe_msg = QMessageBox(self)
        poe_msg.setWindowTitle(f"Predict-Observe-Explain — {scene_name}")
        poe_msg.setIcon(QMessageBox.Icon.Question)
        poe_msg.setText(
            f"<b>Predict-Observe-Explain (POE)</b><br><br>"
            f"Before starting <i>{scene_name}</i>, what flow behavior "
            f"do you predict will form downstream?"
        )
        poe_msg.setInformativeText(
            "Formulating a mental hypothesis before running significantly "
            "enhances spatial physics understanding."
        )
        poe_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        poe_msg.setDefaultButton(QMessageBox.StandardButton.Ok)
        poe_msg.exec()

    def _on_wizard_template(self, template: WizardTemplate) -> None:
        self._adopt_scene(template.scene)
        self._file_path = None
        self._apply_and_refresh()
        self._prompt_poe_prediction(template.name)
        if self.scene.product.autorun_steps:
            self._run_guided_demo()
        if template.tips:
            self.status.showMessage(f"Tips: {template.tips[0]}", 8000)

    def _open_preset_dialog(self) -> None:
        dialog = StartDialog(self, tab=1)
        dialog.template_selected.connect(self._on_wizard_template)
        dialog.preset_selected.connect(self._load_preset_file)
        self._active_dialog = dialog
        dialog.open()

    def _load_preset_file(self, path: str) -> None:
        try:
            self._adopt_scene(load_preset(path))
            self._file_path = None
            self._apply_and_refresh()
            self._prompt_poe_prediction(self.scene.name)
            if self.scene.product.autorun_steps:
                self._run_guided_demo()
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", str(e))

    def _run_guided_demo(self) -> None:
        target = self.scene.product.autorun_steps or 3000
        self.reset(silent=True)
        self._demo_target = target
        self._demo_running = True
        self.paused = False
        self.transport_bar.set_play_state(True)
        self.demo_btn.setText("Running...")
        self.outcome_panel.set_demo_target(target)
        self._set_colormap(self.scene.product.recommended_colormap or "vorticity")
        self._focus_outcome()

    def _toggle_ai_preview(self, checked: bool) -> None:
        self.ai_btn.setChecked(False)
        self._focus_outcome()
        self.outcome_panel.refresh_ai_preview(has_api_key=False)
        self.status.showMessage("AI tutor — Coming soon", 4000)

    def _toggle_mode(self, checked: bool) -> None:
        self.set_expert_mode(checked)

    def set_expert_mode(self, expert: bool) -> None:
        """Beginner shows: Play/Step/Reset, Start, View, Demo/Presets/Export,
        Mode, Drawing tools, Visualization tools."""
        self._expert_mode = expert
        self.mode_btn.blockSignals(True)
        self.mode_btn.setChecked(expert)
        self.mode_btn.setText("Expert" if expert else "Beginner")
        self.mode_btn.blockSignals(False)
        self.scene_panel.set_expert_mode(expert)

        # Advanced tools hidden in Beginner mode
        for w in (
            self.recipes_btn,
            self.sweep_re_btn,
            self.ai_btn,
            self.perf_btn,
            self.gpu_btn,
            self.physics_combo_label,
            self.physics_combo,
            self.record_btn,
        ):
            w.setVisible(expert)
        self.tool_strip.set_polygon_visible(expert)
        if hasattr(self, "recipes_action"):
            self.recipes_action.setVisible(expert)
        if hasattr(self, "sweep_action"):
            self.sweep_action.setVisible(expert)

        if not expert:
            for cb, key in self._vis_checkboxes:
                if key != "hud":
                    cb.setChecked(False)
                    self._on_vis_toggle(key, 0)
            self.perf_btn.setChecked(False)
            self.viewport.set_perf_mode(False)
            if self.viewport.draw_mode == "polygon":
                self._set_draw_mode(None)

    def _sync_physics_mode_combo(self) -> None:
        name = type(self.sim).__name__
        key = "standard"
        if name == "LBM2DLiquid":
            key = "liquid"
        elif name == "LBM2DMultiComponent":
            key = "oil-water"
        idx = self.physics_combo.findData(key)
        if idx >= 0:
            self.physics_combo.blockSignals(True)
            self.physics_combo.setCurrentIndex(idx)
            self.physics_combo.blockSignals(False)

    def _on_physics_mode_changed(self, _index: int) -> None:
        mode = self.physics_combo.currentData()
        if mode:
            self._set_physics_mode(str(mode))

    def _adopt_engine(self, sim: SimEngine) -> None:
        """Install a freshly built engine everywhere that holds a reference.

        Four objects capture the engine at construction time, so replacing it
        means re-pointing all of them or they keep reporting from the old
        instance. Engine recreation (physics-mode switch, engine switch, grid
        resize) all funnel through here so that list cannot drift.
        """
        self.sim = sim
        self.viewport.set_sim(sim)
        self.scene_panel.sim = sim
        self.analysis_panel.sim = sim
        self.outcome_panel.sim = sim

    def _engine_factories(self) -> dict[str, type[SimEngine]]:
        from engines import LBM2D, LBM2DLiquid, LBM2DMultiComponent

        return {
            "standard": LBM2D,
            "liquid": LBM2DLiquid,
            "oil-water": LBM2DMultiComponent,
        }

    def _ensure_engine_matches_scene(self) -> bool:
        """Rebuild the engine when the scene's grid differs from the engine's.

        `apply_to_sim` seeds the initial condition but never resizes: an engine
        built at 128x128 stays 128x128 when handed a 512x256 scene. Nothing
        errors. Obstacles land on a different part of the domain than the scene
        describes, and probes past the edge are silently clamped into the last
        column -- so a Karman Street probe meant to sit at x=300 measures the
        same cell as one at x=180. Rebuilding is the only correct response,
        since the field arrays are allocated in the engine constructor.
        """
        h, w = self.sim.grid_shape
        if (w, h) == (self.scene.width, self.scene.height):
            return False

        old = f"{w}x{h}"
        factory = type(self.sim)
        sim = factory(width=self.scene.width, height=self.scene.height)
        self._adopt_engine(sim)
        logger.info(
            "Rebuilt %s from %s to %dx%d to match the scene",
            factory.__name__,
            old,
            self.scene.width,
            self.scene.height,
        )
        self.status.showMessage(
            f"Grid resized {old} to {self.scene.width}x{self.scene.height} "
            "(engine rebuilt)",
            6000,
        )
        return True

    _MODE_LABELS = {
        "standard": "Standard",
        "liquid": "Liquid",
        "oil-water": "Oil-water",
    }

    def _set_physics_mode(self, mode: str) -> None:
        """Recreate LBM2D / Liquid / MultiComponent and rewire panels."""
        factories = self._engine_factories()
        current = type(self.sim).__name__
        target = factories.get(mode)
        if target is not None and target.__name__ == current:
            return

        factory = target or factories["standard"]
        label = self._MODE_LABELS.get(mode, "Standard")
        sim = factory(
            width=self.scene.width,
            height=self.scene.height,
            viscosity=self.scene.viscosity,
        )
        self._adopt_engine(sim)
        self._apply_and_refresh()
        self._auto_detect_view()
        self.status.showMessage(f"Physics mode: {label} (engine recreated)", 6000)

    def _toggle_perf(self, checked: bool) -> None:
        self.viewport.set_perf_mode(checked)

    def _toggle_gpu(self, checked: bool) -> None:
        """Toggle GPU acceleration (CuPy)."""
        from engines import LBM2D, LBM2DGPU

        w, h = self.scene.width, self.scene.height
        nu = self.scene.viscosity
        current = type(self.sim).__name__

        # Determine target engine
        if checked:
            if LBM2DGPU is None:
                self.gpu_btn.setChecked(False)
                QMessageBox.warning(
                    self,
                    "GPU Not Available",
                    "CuPy is not installed. Install with: pip install cupy-cuda12x",
                )
                return
            target_engine = LBM2DGPU
            label = "GPU (CuPy)"
        else:
            target_engine = LBM2D
            label = "CPU"

        # Skip if already using target engine
        if current == target_engine.__name__:
            return

        # Recreate engine
        sim: SimEngine = target_engine(width=w, height=h, viscosity=nu)
        self._adopt_engine(sim)
        self._apply_and_refresh()
        self._auto_detect_view()
        self.status.showMessage(f"Engine: {label} (recreated)", 6000)

    def _open_recipes_dialog(self) -> None:
        dialog = StartDialog(self, tab=2)
        dialog.recipe_selected.connect(self._execute_recipe)
        self._active_dialog = dialog
        dialog.open()

    _RECIPE_ACTIONS: dict[str, tuple[str, str] | None] = {
        "Show vortex shedding": ("karman_street", "vorticity"),
        "Compare drag of two shapes": ("bluff_body_drag", "pressure"),
        "Generate Cd vs Re": None,
        "Explain Reynolds number": ("channel_flow", "speed"),
        "Create a lab-report figure": ("cylinder", "vorticity"),
    }

    def _execute_recipe(self, name: str) -> None:
        action = self._RECIPE_ACTIONS.get(name)
        if action is None:
            if "sweep" in name.lower() or "cd vs re" in name.lower():
                self._open_sweep_dialog()
            else:
                self.status.showMessage(
                    f"Recipe: {name} — use the preset gallery to find a scene.",
                    6000,
                )
            return
        preset_id, colormap = action
        preset = find_preset(preset_id)
        if preset is None:
            self.status.showMessage(
                f"Recipe '{name}' refers to preset id '{preset_id}', "
                "which is not in the library.",
                8000,
            )
            logger.error("Recipe %r references unknown preset id %r", name, preset_id)
            return
        self._load_preset_file(preset["file"])
        if colormap:
            self._set_colormap(colormap)

    def _check_recipe_actions(self) -> list[str]:
        """Report recipes whose preset id is missing from the library.

        Called once at startup. Without this a dangling id is invisible until a
        user clicks the recipe, and the only symptom is a status-bar message.
        """
        problems = [
            f"Recipe '{label}' -> preset id '{preset_id}' not found"
            for label, action in self._RECIPE_ACTIONS.items()
            if action is not None
            for preset_id in [action[0]]
            if find_preset(preset_id) is None
        ]
        problems.extend(validate_presets())
        return problems

    def _open_sweep_dialog(self) -> None:
        dialog = SweepDialog(self.scene, self)
        dialog.exec()
        if dialog.result is not None:
            sweep_dict = dialog.result.to_dict()
            self.scene.sweeps.append(sweep_dict)
            scene_save(self.scene, self._file_path) if self._file_path else None

    def _open_export_dialog(self) -> None:
        dialog = ExportDialog(self)
        dialog.export_image_requested.connect(self._export_image)
        dialog.export_data_requested.connect(self._export_data)
        dialog.start_recording.connect(self._start_video_recording)
        dialog.exec()

    def _quick_export_figure(self) -> None:
        base = self.scene.name.lower().replace(" ", "_") or "sstream_flow"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report Figure",
            f"{base}.png",
            "PNG Image (*.png)",
        )
        if not path:
            return
        self._export_image(
            path,
            scale=3,
            colorbar=True,
            annotations=True,
            field_name=self.viewport.get_colormap(),
        )
        try:
            export_markdown_report(
                Path(path).with_suffix(".md"),
                self.scene,
                self.sim,
                self.step_count,
                regime=detect_flow_regime(
                    self.sim, self.scene, self.runtime_probes, self.step_count
                ),
                warnings=check_sanity(
                    self.sim, self.scene, self.runtime_probes, self.step_count
                ),
                scorecard=compute_scorecard(
                    self.sim, self.scene, self.runtime_probes, self.step_count
                ),
            )
        except Exception as e:
            QMessageBox.warning(self, "Report Export Failed", str(e))

    def _export_image(
        self,
        path: str,
        scale: int,
        colorbar: bool,
        annotations: bool,
        field_name: str = "smoke",
    ) -> None:
        try:
            export_image(
                sim=self.sim,
                scene=self.scene,
                path=path,
                scale=scale,
                include_colorbar=colorbar,
                include_annotations=annotations,
                step_count=self.step_count,
                colormap=field_name,
            )
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    def _export_data(self, typ: str, path: str) -> None:
        try:
            if typ == "csv":
                export_probe_csv(self.runtime_probes, path)
            elif typ == "npz":
                export_field_snapshot(self.sim, path)
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    def _toggle_recording(self) -> None:
        if self._recorder is not None:
            self._stop_video_recording()
        else:
            self._open_export_dialog()

    def _start_video_recording(self, path: str, fps: int, max_frames: int) -> None:
        try:
            max_f = max_frames if max_frames > 0 else None
            self._recorder = VideoRecorder(path, fps=fps, max_frames=max_f)
            self.record_btn.setText("Stop")
            self.record_btn.setChecked(True)
        except Exception as e:
            QMessageBox.warning(self, "Recording Failed", str(e))
            self._recorder = None

    def _stop_video_recording(self) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.close()
        except Exception as e:
            QMessageBox.warning(
                self, "Recording Error", f"Failed to finalize video: {e}"
            )
        self._recorder = None
        self.record_btn.setText("Record")
        self.record_btn.setChecked(False)
