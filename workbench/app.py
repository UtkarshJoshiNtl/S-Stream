from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
)

from analysis.regimes import detect_flow_regime
from analysis.sanity import check_sanity
from analysis.scorecard import compute_scorecard
from engines.base import SimEngine
from export.data import export_field_snapshot, export_probe_csv
from export.image import export_image
from export.report import export_markdown_report
from export.video import VideoRecorder
from presets.loader import list_presets, load_preset
from scene.probe import Probe
from scene.scene import ProbeSpec, Scene, apply_to_sim, default_scene
from scene.serializer import load as scene_load
from scene.serializer import save as scene_save
from workbench.dialogs.export_dialog import ExportDialog
from workbench.dialogs.sweep_dialog import SweepDialog
from workbench.dialogs.wizard_dialog import StartDialog, WizardTemplate
from workbench.panels.analysis_panel import AnalysisPanel
from workbench.panels.outcome_panel import OutcomePanel
from workbench.panels.scene_panel import ScenePanel
from workbench.viewport import Viewport

_COLORMAPS = [
    "speed",
    "smoke",
    "vorticity",
    "pressure",
    "density",
    "phase",
    "temperature",
    "component1",
    "component2",
    "color",
]


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

        self.setWindowTitle("S-Stream - Fluid Workbench")
        self.resize(1320, 840)

        self._configure_domain_for_scene()
        apply_to_sim(self.scene, self.sim)

        self.viewport = Viewport()
        self.viewport.set_sim(sim)
        self.viewport.set_scene(self.scene)
        self.viewport.obstacle_created.connect(self._on_viewport_obstacle)
        self.viewport.probe_placed.connect(self._on_viewport_probe)
        vp_frame = QFrame()
        vp_frame.setFrameStyle(QFrame.Shape.NoFrame)
        vp_frame.setStyleSheet(
            "QFrame { border: 2px solid #1e293b; border-radius: 6px; "
            "background: transparent; }"
        )
        vp_layout = QVBoxLayout(vp_frame)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        vp_layout.addWidget(self.viewport)
        self.setCentralWidget(vp_frame)

        self.runtime_probes: list[Probe] = []
        self._rebuild_probes()

        self.scene_panel = ScenePanel(sim, self.scene)
        self.scene_panel.scene_changed.connect(self._on_scene_changed)
        self.scene_panel.parameters_changed.connect(self._on_params_changed)
        self.scene_dock = QDockWidget("Scene", self)
        self.scene_dock.setWidget(self.scene_panel)
        self.scene_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_dock)

        self.analysis_panel = AnalysisPanel(sim)
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setWidget(self.analysis_panel)
        self.analysis_dock.setMinimumWidth(310)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        self.outcome_panel = OutcomePanel(sim, self.scene)
        self.outcome_dock = QDockWidget("What am I seeing?", self)
        self.outcome_dock.setWidget(self.outcome_panel)
        self.outcome_dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.outcome_dock)
        self.tabifyDockWidget(self.analysis_dock, self.outcome_dock)
        self.outcome_dock.raise_()

        self._sync_analysis_probes()
        self.analysis_panel.set_scene(self.scene)
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_shortcuts()
        self.set_expert_mode(False)
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

    def _setup_toolbar(self) -> None:
        self.toolbar = QToolBar("Simulation")
        self.addToolBar(self.toolbar)

        self.play_btn = QPushButton("Pause")
        self.play_btn.setFixedWidth(80)
        self.play_btn.clicked.connect(self.toggle_pause)
        self.toolbar.addWidget(self.play_btn)

        step_btn = QPushButton("Step")
        step_btn.setToolTip("Advance one frame")
        step_btn.clicked.connect(self.step_once)
        self.toolbar.addWidget(step_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        self.toolbar.addWidget(reset_btn)

        self.toolbar.addSeparator()

        self.start_btn = QPushButton("Start…")
        self.start_btn.setToolTip("Open templates, presets, and recipes")
        self.start_btn.clicked.connect(self._open_wizard)
        self.toolbar.addWidget(self.start_btn)

        self.toolbar.addSeparator()

        self.colormap_combo = QPushButton(self._colormap_label("speed"))
        self.colormap_combo.setMenu(self._build_colormap_menu())
        self.toolbar.addWidget(self.colormap_combo)

        self.toolbar.addSeparator()

        self.demo_btn = QPushButton("Run Demo")
        self.demo_btn.clicked.connect(self._run_guided_demo)
        self.toolbar.addWidget(self.demo_btn)

        self.presets_btn = QPushButton("Presets")
        self.presets_btn.clicked.connect(self._open_preset_dialog)
        self.toolbar.addWidget(self.presets_btn)

        self.export_fig_btn = QPushButton("Export Figure")
        self.export_fig_btn.clicked.connect(self._quick_export_figure)
        self.toolbar.addWidget(self.export_fig_btn)

        self.recipes_btn = QPushButton("Recipes")
        self.recipes_btn.clicked.connect(self._open_recipes_dialog)
        self.toolbar.addWidget(self.recipes_btn)

        self.sweep_re_btn = QPushButton("Sweep Re")
        self.sweep_re_btn.clicked.connect(self._open_sweep_dialog)
        self.toolbar.addWidget(self.sweep_re_btn)

        self.ai_btn = QPushButton("AI")
        self.ai_btn.setCheckable(True)
        self.ai_btn.setEnabled(False)
        self.ai_btn.setToolTip("AI tutor — Coming soon")
        self.ai_btn.clicked.connect(self._toggle_ai_preview)
        self.toolbar.addWidget(self.ai_btn)

        self.mode_btn = QPushButton("Beginner")
        self.mode_btn.setCheckable(True)
        self.mode_btn.clicked.connect(self._toggle_mode)
        self.toolbar.addWidget(self.mode_btn)

        self.perf_btn = QPushButton("Perf")
        self.perf_btn.setCheckable(True)
        self.perf_btn.clicked.connect(self._toggle_perf)
        self.toolbar.addWidget(self.perf_btn)

        self.physics_combo = QComboBox()
        self.physics_combo.addItem("Standard", "standard")
        self.physics_combo.addItem("Liquid  [Experimental]", "liquid")
        self.physics_combo.addItem("Oil-water  [Experimental]", "oil-water")
        self.physics_combo.setToolTip("Physics mode (recreates the simulation engine)")
        self.physics_combo.currentIndexChanged.connect(self._on_physics_mode_changed)
        self.physics_combo_label = QLabel("Physics:")
        self.toolbar.addWidget(self.physics_combo_label)
        self.toolbar.addWidget(self.physics_combo)

        self.gpu_btn = QPushButton("GPU")
        self.gpu_btn.setCheckable(True)
        self.gpu_btn.setToolTip(
            "Enable GPU acceleration (requires CuPy)\n" "Recreates engine on toggle"
        )
        self.gpu_btn.clicked.connect(self._toggle_gpu)
        self.toolbar.addWidget(self.gpu_btn)

        self.toolbar.addSeparator()

        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.toolbar.addWidget(self.record_btn)

        self.toolbar.addSeparator()

        self._draw_group: list[QPushButton] = []
        self._freehand_btn: QPushButton | None = None
        for mode, label in [
            ("circle", "Circle"),
            ("rect", "Rect"),
            ("polygon", "Polygon"),
            ("probe", "Probe"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda checked, m=mode, b=btn: self._set_draw_mode(m, b)
            )
            self.toolbar.addWidget(btn)
            self._draw_group.append(btn)
            if mode == "polygon":
                self._freehand_btn = btn

        select_btn = QPushButton("Select")
        select_btn.setCheckable(True)
        select_btn.setChecked(True)
        select_btn.clicked.connect(lambda: self._set_draw_mode(None, select_btn))
        self.toolbar.addWidget(select_btn)
        self._draw_group.append(select_btn)

        self.toolbar.addSeparator()

        self.arrows_btn = QPushButton("Arrows")
        self.arrows_btn.setCheckable(True)
        self.arrows_btn.clicked.connect(self._toggle_arrows)
        self.toolbar.addWidget(self.arrows_btn)

        self.streams_btn = QPushButton("Streams")
        self.streams_btn.setCheckable(True)
        self.streams_btn.clicked.connect(self._toggle_streams)
        self.toolbar.addWidget(self.streams_btn)

        self.contours_btn = QPushButton("Contours")
        self.contours_btn.setCheckable(True)
        self.contours_btn.clicked.connect(self._toggle_contours)
        self.toolbar.addWidget(self.contours_btn)

        self.force_btn = QPushButton("Force")
        self.force_btn.setCheckable(True)
        self.force_btn.clicked.connect(self._toggle_force_arrows)
        self.toolbar.addWidget(self.force_btn)

        self.particles_btn = QPushButton("Particles")
        self.particles_btn.setCheckable(True)
        self.particles_btn.clicked.connect(self._toggle_particles)
        self.toolbar.addWidget(self.particles_btn)

    def _setup_statusbar(self) -> None:
        self.status = QStatusBar()
        self.re_label = QLabel("Re: -")
        self.status.addWidget(self.re_label)
        hints = QLabel("Space: Play/Pause  |  R: Reset  |  Esc: Cancel draw")
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

        open_preset_action = QAction("Open &Preset...", self)
        open_preset_action.triggered.connect(self._open_preset_dialog)
        file_menu.addAction(open_preset_action)

        wizard_action = QAction("&Start...", self)
        wizard_action.setShortcut(QKeySequence("Ctrl+W"))
        wizard_action.triggered.connect(self._open_wizard)
        file_menu.addAction(wizard_action)

        self.recipes_action = QAction("&Recipes...", self)
        self.recipes_action.triggered.connect(self._open_recipes_dialog)
        file_menu.addAction(self.recipes_action)

        file_menu.addSeparator()

        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._open_export_dialog)
        file_menu.addAction(export_action)

        self.sweep_action = QAction("&Sweep...", self)
        self.sweep_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.sweep_action.triggered.connect(self._open_sweep_dialog)
        file_menu.addAction(self.sweep_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _file_new(self) -> None:
        self.scene = default_scene()
        self._file_path = None
        self._apply_and_refresh()

    def _file_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene", "", "Scene Files (*.json)"
        )
        if not path:
            return
        try:
            self.scene = scene_load(path)
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

    def _sync_analysis_probes(self) -> None:
        self.analysis_panel.set_probes(self.runtime_probes)
        if hasattr(self, "outcome_panel"):
            self.outcome_panel.set_probes(self.runtime_probes)

    def _apply_and_refresh(self) -> None:
        self._configure_domain_for_scene()
        apply_to_sim(self.scene, self.sim)
        self.step_count = 0
        self._rebuild_probes()
        self._sync_analysis_probes()
        self.analysis_panel.set_scene(self.scene)
        self.viewport.set_scene(self.scene)
        self.scene_panel.scene = self.scene
        self.scene_panel.refresh()
        self.scene_panel.sync_params_from_scene()
        self.grid_label.setText(self._grid_label_text())
        cmap = self.scene.product.recommended_colormap
        if type(self.sim).__name__ == "LBM2DLiquid" and cmap == "smoke":
            cmap = "density"
        if type(self.sim).__name__ == "LBM2DMultiComponent" and cmap == "smoke":
            cmap = "component1"
        self._set_colormap(cmap)
        self.outcome_panel.set_scene(self.scene)
        self._demo_target = self.scene.product.autorun_steps
        self.outcome_panel.set_demo_target(self._demo_target)
        self._update_title()
        self._update_re_label()
        self.outcome_dock.raise_()

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

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.play_btn.setText("Play" if self.paused else "Pause")

    def step_once(self) -> None:
        self.sim.step()
        self.step_count += 1
        self.analysis_panel.tick(1.0)
        self.outcome_panel.update_outcome(self.step_count)
        self.viewport.update()
        self._update_re_label()

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
        apply_to_sim(self.scene, self.sim)
        self.step_count = 0
        self.outcome_panel.update_outcome(self.step_count, force=True)
        self._update_re_label()

    def tick(self) -> None:
        if not self.paused:
            self.sim.step()
            self.step_count += 1
            self.analysis_panel.tick(1.0)
            if self._demo_running and self._demo_target > 0:
                if self.step_count >= self._demo_target:
                    self._demo_running = False
                    self.paused = True
                    self.play_btn.setText("Play")
                    self.demo_btn.setText("Run Demo")
        self.outcome_panel.update_outcome(self.step_count)
        self.viewport.update()
        if self._recorder is not None and self._recorder.recording:
            if not self._recorder.add_frame(self.viewport.grab().toImage()):
                self._stop_video_recording()
        self._fps_count += 1
        fps_str = f"{self._fps_value:.0f}"
        self.status_label.setText(f"Step {self.step_count}  |  FPS: {fps_str}")
        self._update_re_label()
        elapsed = time.perf_counter() - self._frame_start
        self.timer.start(max(1, 33 - int(elapsed * 1000)))
        self._frame_start = time.perf_counter()

    def _set_draw_mode(self, mode: str | None, sender: QPushButton) -> None:
        for btn in self._draw_group:
            btn.setChecked(btn is sender)
        self.viewport.set_draw_mode(mode)

    def _on_viewport_obstacle(self, obs) -> None:
        self.scene_panel.add_obstacle_from_viewport(obs)

    def _on_viewport_probe(self, spec: ProbeSpec) -> None:
        self.scene.probes.append(spec)
        self._rebuild_probes()
        self._sync_analysis_probes()
        self.scene_panel.refresh()
        self._update_title()

    def _on_scene_changed(self) -> None:
        self._rebuild_probes()
        self._sync_analysis_probes()
        self.scene_panel.sync_params_from_scene()
        self.grid_label.setText(self._grid_label_text())
        self.outcome_panel.set_scene(self.scene)
        self._update_title()

    def _on_params_changed(self) -> None:
        self.grid_label.setText(self._grid_label_text())
        self.outcome_panel.update_outcome(self.step_count, force=True)

    def _update_fps(self) -> None:
        self._fps_value = self._fps_count
        self._fps_count = 0

    def _auto_detect_view(self) -> None:
        name = type(self.sim).__name__
        if name == "LBM2DLiquid":
            self.viewport.set_colormap("density")
            self.colormap_combo.setText(self._colormap_label("density"))
        elif name == "LBM2DMultiComponent":
            self.viewport.set_colormap("component1")
            self.colormap_combo.setText(self._colormap_label("component1"))

    @staticmethod
    def _colormap_label(name: str) -> str:
        from resources.colormaps import FIELD_REGISTRY

        info = FIELD_REGISTRY.get(name)
        label = info.label if info else name.capitalize()
        return f"View: {label}"

    def _build_colormap_menu(self):
        from resources.colormaps import FIELD_REGISTRY

        menu = QMenu(self)
        for name in _COLORMAPS:
            info = FIELD_REGISTRY.get(name)
            label = info.label if info else name.capitalize()
            action = menu.addAction(label)
            action.triggered.connect(lambda checked, n=name: self._set_colormap(n))
        return menu

    def _toggle_arrows(self, checked: bool) -> None:
        self.viewport.set_show_quiver(checked)

    def _toggle_streams(self, checked: bool) -> None:
        self.viewport.set_show_streamlines(checked)

    def _toggle_contours(self, checked: bool) -> None:
        self.viewport.set_show_contours(checked)

    def _toggle_force_arrows(self, checked: bool) -> None:
        self.viewport.set_show_force_arrows(checked)

    def _toggle_particles(self, checked: bool) -> None:
        self.viewport.set_show_particles(checked)

    def _set_colormap(self, name: str) -> None:
        if name not in _COLORMAPS:
            name = _COLORMAPS[0]
        self.viewport.set_colormap(name)
        self.analysis_panel.set_colormap(name)
        self.colormap_combo.setText(self._colormap_label(name))

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

    def _on_wizard_template(self, template: WizardTemplate) -> None:
        self.scene = template.scene
        self._file_path = None
        self._apply_and_refresh()
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
            self.scene = load_preset(path)
            self._file_path = None
            self._apply_and_refresh()
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
        self.play_btn.setText("Pause")
        self.demo_btn.setText("Running...")
        self.outcome_panel.set_demo_target(target)
        self._set_colormap(self.scene.product.recommended_colormap or "vorticity")
        self.outcome_dock.raise_()

    def _toggle_ai_preview(self, checked: bool) -> None:
        self.ai_btn.setChecked(False)
        self.outcome_dock.raise_()
        self.outcome_panel.refresh_ai_preview(has_api_key=False)
        self.status.showMessage("AI tutor — Coming soon", 4000)

    def _toggle_mode(self, checked: bool) -> None:
        self.set_expert_mode(checked)

    def set_expert_mode(self, expert: bool) -> None:
        """Beginner shows: Play/Step/Reset, Start, View, Circle/Rect/Probe/Select."""
        self._expert_mode = expert
        self.mode_btn.blockSignals(True)
        self.mode_btn.setChecked(expert)
        self.mode_btn.setText("Expert" if expert else "Beginner")
        self.mode_btn.blockSignals(False)
        self.scene_panel.set_expert_mode(expert)

        for w in (
            self.demo_btn,
            self.presets_btn,
            self.export_fig_btn,
            self.recipes_btn,
            self.sweep_re_btn,
            self.ai_btn,
            self.perf_btn,
            self.gpu_btn,
            self.record_btn,
            self.physics_combo_label,
            self.physics_combo,
            self.arrows_btn,
            self.streams_btn,
            self.contours_btn,
            self.force_btn,
            self.particles_btn,
        ):
            w.setVisible(expert)
        if self._freehand_btn is not None:
            self._freehand_btn.setVisible(expert)
        if hasattr(self, "recipes_action"):
            self.recipes_action.setVisible(expert)
        if hasattr(self, "sweep_action"):
            self.sweep_action.setVisible(expert)

        if not expert:
            for btn, setter in (
                (self.arrows_btn, self.viewport.set_show_quiver),
                (self.streams_btn, self.viewport.set_show_streamlines),
                (self.contours_btn, self.viewport.set_show_contours),
                (self.force_btn, self.viewport.set_show_force_arrows),
                (self.particles_btn, self.viewport.set_show_particles),
            ):
                btn.setChecked(False)
                setter(False)
            self.perf_btn.setChecked(False)
            self.viewport.set_perf_mode(False)
            if self.viewport.draw_mode == "polygon":
                self._set_draw_mode(None, self._draw_group[-1])

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

    def _set_physics_mode(self, mode: str) -> None:
        """Recreate LBM2D / Liquid / MultiComponent and rewire panels."""
        from engines import LBM2D, LBM2DLiquid, LBM2DMultiComponent

        w, h = self.scene.width, self.scene.height
        nu = self.scene.viscosity
        current = type(self.sim).__name__
        if mode == "liquid" and current == "LBM2DLiquid":
            return
        if mode == "oil-water" and current == "LBM2DMultiComponent":
            return
        if mode == "standard" and current == "LBM2D":
            return

        if mode == "liquid":
            sim: SimEngine = LBM2DLiquid(width=w, height=h, viscosity=nu)
            label = "Liquid"
        elif mode == "oil-water":
            sim = LBM2DMultiComponent(width=w, height=h, viscosity=nu)
            label = "Oil-water"
        else:
            sim = LBM2D(width=w, height=h, viscosity=nu)
            label = "Standard"

        self.sim = sim
        self.viewport.set_sim(sim)
        self.scene_panel.sim = sim
        self.analysis_panel.sim = sim
        self.outcome_panel.sim = sim
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
        self.sim = sim
        self.viewport.set_sim(sim)
        self.scene_panel.sim = sim
        self.analysis_panel.sim = sim
        self.outcome_panel.sim = sim
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
        preset_name, colormap = action
        presets = list_presets()
        matched = next((p for p in presets if preset_name in p["name"].lower()), None)
        if matched is None:
            name_lower = preset_name.replace("_", " ")
            matched = next(
                (p for p in presets if name_lower in p["name"].lower()),
                None,
            )
        if matched is not None:
            self._load_preset_file(matched["file"])
            if colormap:
                self._set_colormap(colormap)
        else:
            self.status.showMessage(
                f"Preset '{preset_name}' not found. Check presets/ folder.",
                6000,
            )

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
