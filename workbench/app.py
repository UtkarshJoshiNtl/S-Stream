from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
)

from engines.base import SimEngine
from export.data import export_field_snapshot, export_probe_csv
from export.image import export_image
from export.video import VideoRecorder
from presets.loader import list_presets, load_preset
from scene.probe import Probe
from scene.scene import ProbeSpec, Scene, apply_to_sim, default_scene
from scene.serializer import load as scene_load
from scene.serializer import save as scene_save
from workbench.dialogs.export_dialog import ExportDialog
from workbench.dialogs.presets_dialog import PresetsDialog
from workbench.dialogs.sweep_dialog import SweepDialog
from workbench.panels.analysis_panel import AnalysisPanel
from workbench.panels.scene_panel import ScenePanel
from workbench.viewport import Viewport


_COLORMAPS = ["smoke", "speed", "vorticity", "pressure"]


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
        self._frame_times: list[float] = []
        self._recorder: VideoRecorder | None = None

        self.setWindowTitle("S-Stream — Fluid Workbench")
        self.resize(1200, 800)

        # Apply initial scene to sim
        apply_to_sim(self.scene, self.sim)

        # --- central viewport ---
        self.viewport = Viewport()
        self.viewport.set_sim(sim)
        self.viewport.set_scene(self.scene)
        self.viewport.obstacle_created.connect(self._on_viewport_obstacle)
        self.viewport.probe_placed.connect(self._on_viewport_probe)
        self.setCentralWidget(self.viewport)

        # --- runtime probes (after viewport exists) ---
        self.runtime_probes: list[Probe] = []
        self._rebuild_probes()

        # --- left dock: scene panel ---
        self.scene_panel = ScenePanel(sim, self.scene)
        self.scene_panel.scene_changed.connect(self._on_scene_changed)
        self.scene_panel.parameters_changed.connect(self._on_params_changed)
        self.scene_dock = QDockWidget("Scene", self)
        self.scene_dock.setWidget(self.scene_panel)
        self.scene_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_dock)

        # --- right dock: analysis panel ---
        self.analysis_panel = AnalysisPanel(sim)
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setWidget(self.analysis_panel)
        self.analysis_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        self._sync_analysis_probes()

        # --- file menu ---
        self._setup_menus()

        # --- toolbar ---
        self.toolbar = QToolBar("Simulation")
        self.addToolBar(self.toolbar)

        self.play_btn = QPushButton("Pause")
        self.play_btn.setFixedWidth(80)
        self.play_btn.clicked.connect(self.toggle_pause)
        self.toolbar.addWidget(self.play_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        self.toolbar.addWidget(reset_btn)

        self.toolbar.addSeparator()

        self.record_btn = QPushButton("● Record")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.toolbar.addWidget(self.record_btn)

        self.toolbar.addSeparator()

        self._draw_group: list[QPushButton] = []
        for mode, label in [
            ("circle", "○ Circle"),
            ("rect", "▭ Rect"),
            ("polygon", "✎ Freehand"),
            ("probe", "✚ Probe"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda checked, m=mode, b=btn: self._set_draw_mode(m, b)
            )
            self.toolbar.addWidget(btn)
            self._draw_group.append(btn)

        select_btn = QPushButton("▸ Select")
        select_btn.setCheckable(True)
        select_btn.setChecked(True)
        select_btn.clicked.connect(lambda: self._set_draw_mode(None, select_btn))
        self.toolbar.addWidget(select_btn)
        self._draw_group.append(select_btn)

        self.toolbar.addSeparator()

        self.colormap_combo = QPushButton(self._colormap_label("smoke"))
        self.colormap_combo.setMenu(self._build_colormap_menu())
        self.toolbar.addWidget(self.colormap_combo)

        # --- status bar ---
        self.status = QStatusBar()
        self.re_label = QLabel("Re: —")
        self.status.addWidget(self.re_label)
        self.status_label = QLabel("Step 0  |  FPS: —")
        self.status.addPermanentWidget(self.status_label)
        self.grid_label = QLabel(self._grid_label_text())
        self.status.addPermanentWidget(self.grid_label)
        self.setStatusBar(self.status)

        # --- keyboard shortcuts ---
        pause_shortcut = QAction(self)
        pause_shortcut.setShortcut(QKeySequence(Qt.Key.Key_Space))
        pause_shortcut.triggered.connect(self.toggle_pause)
        self.addAction(pause_shortcut)

        reset_shortcut = QAction(self)
        reset_shortcut.setShortcut(QKeySequence(Qt.Key.Key_R))
        reset_shortcut.triggered.connect(self.reset)
        self.addAction(reset_shortcut)

        quit_shortcut = QAction(self)
        quit_shortcut.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        quit_shortcut.triggered.connect(self.close)
        self.addAction(quit_shortcut)

        # --- Welcome dialog ---
        self._show_welcome_if_first()

        # --- FPS counter (once per second) ---
        self._fps_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        # --- simulation timer ---
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # --- file menu ---

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

        file_menu.addSeparator()

        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._open_export_dialog)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        sweep_action = QAction("&Sweep...", self)
        sweep_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        sweep_action.triggered.connect(self._open_sweep_dialog)
        file_menu.addAction(sweep_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
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

    def _apply_and_refresh(self) -> None:
        apply_to_sim(self.scene, self.sim)
        self.step_count = 0
        self._rebuild_probes()
        self._sync_analysis_probes()
        self.viewport.set_scene(self.scene)
        self.scene_panel.scene = self.scene
        self.scene_panel.refresh()
        self.scene_panel.sync_params_from_scene()
        self.grid_label.setText(self._grid_label_text())
        self._update_title()

    def _grid_label_text(self) -> str:
        gs = self.sim.grid_shape
        return f"{gs[1]}×{gs[0]}  |  ν {self.sim.viscosity}"

    def _update_title(self) -> None:
        name = self._file_path.stem if self._file_path else self.scene.name
        self.setWindowTitle(f"S-Stream — {name}")

    # --- simulation control ---

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.play_btn.setText("Play" if self.paused else "Pause")

    def reset(self) -> None:
        apply_to_sim(self.scene, self.sim)
        self.step_count = 0

    def tick(self) -> None:
        if not self.paused:
            self.sim.step()
            self.step_count += 1
            self.analysis_panel.tick(1.0)
        self.viewport.update()
        if self._recorder is not None and self._recorder.recording:
            if not self._recorder.add_frame(self.viewport.grab().toImage()):
                self._stop_video_recording()
        self._fps_count += 1
        fps_str = f"{self._fps_value:.0f}"
        self.status_label.setText(f"Step {self.step_count}  |  FPS: {fps_str}")

    # --- drawing modes ---

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
        self._update_title()

    def _on_params_changed(self) -> None:
        self.grid_label.setText(self._grid_label_text())

    def _update_fps(self) -> None:
        self._fps_value = self._fps_count
        self._fps_count = 0

    # --- colormap ---

    @staticmethod
    def _colormap_label(name: str) -> str:
        return f"🎨 {name.capitalize()}"

    def _build_colormap_menu(self):
        menu = QMenu(self)
        for name in _COLORMAPS:
            action = menu.addAction(name.capitalize())
            action.triggered.connect(lambda checked, n=name: self._set_colormap(n))
        return menu

    def _set_colormap(self, name: str) -> None:
        self.viewport.set_colormap(name)
        self.colormap_combo.setText(self._colormap_label(name))

    # --- presets ---

    def _show_welcome_if_first(self) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings("S-Stream", "S-Stream")
        if settings.value("welcome_shown", False, type=bool):
            return
        settings.setValue("welcome_shown", True)
        self._open_preset_dialog()

    def _open_preset_dialog(self) -> None:
        presets = list_presets()
        if not presets:
            return
        dialog = PresetsDialog(presets, self)
        dialog.preset_selected.connect(self._load_preset_file)
        dialog.exec()

    def _load_preset_file(self, path: str) -> None:
        try:
            self.scene = load_preset(path)
            self._file_path = None
            self._apply_and_refresh()
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", str(e))

    # --- sweep ---

    def _open_sweep_dialog(self) -> None:
        dialog = SweepDialog(self.scene, self)
        dialog.exec()
        if dialog.result is not None:
            sweep_dict = dialog.result.to_dict()
            self.scene.sweeps.append(sweep_dict)
            scene_save(self.scene, self._file_path) if self._file_path else None

    # --- export ---

    def _open_export_dialog(self) -> None:
        dialog = ExportDialog(self)
        dialog.export_image_requested.connect(self._export_image)
        dialog.export_data_requested.connect(self._export_data)
        dialog.start_recording.connect(self._start_video_recording)
        dialog.exec()

    def _export_image(
        self,
        path: str,
        scale: int,
        colorbar: bool,
        annotations: bool,
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
                colormap=self.viewport._colormap,
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

    def _start_video_recording(
        self, path: str, fps: int, max_frames: int
    ) -> None:
        try:
            max_f = max_frames if max_frames > 0 else None
            self._recorder = VideoRecorder(path, fps=fps, max_frames=max_f)
            self.record_btn.setText("■ Stop")
            self.record_btn.setChecked(True)
        except Exception as e:
            QMessageBox.warning(self, "Recording Failed", str(e))
            self._recorder = None

    def _stop_video_recording(self) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.close()
        except Exception:
            pass
        self._recorder = None
        self.record_btn.setText("● Record")
        self.record_btn.setChecked(False)
