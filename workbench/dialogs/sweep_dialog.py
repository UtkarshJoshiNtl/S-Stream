from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from analysis.sweep import SweepResult, run_sweep
from scene.scene import Scene

_SWEEP_PARAMS = [
    "viscosity",
    "u_inflow",
    "smoke_diffusion",
    "smoke_decay",
]


class _SweepWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(object)

    def __init__(self, scene, parameter, values, measurements, steps):
        super().__init__()
        self.scene = scene
        self.parameter = parameter
        self.values = values
        self.measurements = measurements
        self.steps = steps

    def run(self) -> None:
        result = run_sweep(
            self.scene,
            self.parameter,
            self.values,
            self.measurements,
            self.steps,
            progress_callback=lambda c, t: self.progress.emit(c, t),
        )
        self.finished.emit(result)


class SweepDialog(QDialog):
    def __init__(self, scene: Scene, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter Sweep")
        self.setMinimumSize(700, 500)
        self._scene = scene
        self._result: SweepResult | None = None

        layout = QVBoxLayout(self)

        # --- Configuration ---
        config_group = QGroupBox("Configuration")
        form = QFormLayout()

        self._param_combo = QComboBox()
        for p in _SWEEP_PARAMS:
            self._param_combo.addItem(p)
        for obs in scene.obstacles:
            if hasattr(obs, "radius"):
                self._param_combo.addItem(f"{obs.name}.radius")
            if hasattr(obs, "w"):
                self._param_combo.addItem(f"{obs.name}.w")
            if hasattr(obs, "h"):
                self._param_combo.addItem(f"{obs.name}.h")
        form.addRow("Parameter:", self._param_combo)

        self._min_val = QDoubleSpinBox()
        self._min_val.setRange(0.0, 10.0)
        self._min_val.setSingleStep(0.01)
        self._min_val.setDecimals(5)
        self._min_val.setValue(0.001)
        form.addRow("Min value:", self._min_val)

        self._max_val = QDoubleSpinBox()
        self._max_val.setRange(0.0, 10.0)
        self._max_val.setSingleStep(0.01)
        self._max_val.setDecimals(5)
        self._max_val.setValue(0.2)
        form.addRow("Max value:", self._max_val)

        self._num_steps = QSpinBox()
        self._num_steps.setRange(2, 50)
        self._num_steps.setValue(8)
        form.addRow("Steps:", self._num_steps)

        self._sim_steps = QSpinBox()
        self._sim_steps.setRange(100, 50000)
        self._sim_steps.setSingleStep(500)
        self._sim_steps.setValue(5000)
        form.addRow("Steps per run:", self._sim_steps)

        self._measure_combo = QComboBox()
        self._measure_combo.addItems(
            ["reynolds_number", "drag_coefficient", "max_speed", "max_vorticity"]
        )
        form.addRow("Measurement:", self._measure_combo)

        config_group.setLayout(form)
        layout.addWidget(config_group)

        # --- Run / Status ---
        status_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Sweep")
        self._run_btn.clicked.connect(self._run_sweep)
        status_row.addWidget(self._run_btn)

        self._est_label = QLabel("")
        status_row.addWidget(self._est_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # --- Results plot ---
        plot_group = QGroupBox("Results")
        plot_layout = QVBoxLayout()
        self._plot = pg.PlotWidget()
        self._plot.showGrid(True, True, 0.3)
        self._plot.setLabel("bottom", "parameter value")
        self._plot.setLabel("left", "measurement")
        self._curve = self._plot.plot(pen="#4fc3f7", symbol="o", symbolSize=6)
        plot_layout.addWidget(self._plot)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group, 1)

        # --- Buttons ---
        buttons = QDialogButtonBox()
        close_btn = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet(
            "QDialog { background: #11111b; } "
            "QLabel { color: #cdd6f4; } "
            "QGroupBox { color: #cdd6f4; border: 1px solid #313244; "
            "border-radius: 4px; margin-top: 12px; padding-top: 8px; } "
            "QGroupBox::title { color: #cdd6f4; } "
            "QPushButton { color: #cdd6f4; background: #313244; "
            "border: 1px solid #45475a; padding: 6px 16px; border-radius: 4px; } "
            "QPushButton:hover { background: #45475a; } "
            "QComboBox { color: #cdd6f4; background: #313244; "
            "border: 1px solid #45475a; padding: 4px; } "
            "QSpinBox { color: #cdd6f4; background: #313244; "
            "border: 1px solid #45475a; padding: 4px; }"
        )

        self._update_estimate()

    def _get_values(self) -> list[float]:
        min_v = self._min_val.value()
        max_v = self._max_val.value()
        n = self._num_steps.value()
        if n <= 1:
            return [min_v]
        return [min_v + (max_v - min_v) * i / (n - 1) for i in range(n)]

    def _update_estimate(self) -> None:
        n = self._num_steps.value()
        steps = self._sim_steps.value()
        est = n * steps * 0.00001
        self._est_label.setText(f"~{est:.1f}s estimated")

    def _run_sweep(self) -> None:
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        param = self._param_combo.currentText()
        values = self._get_values()
        meas = [self._measure_combo.currentText()]
        steps = self._sim_steps.value()

        self._worker = _SweepWorker(self._scene, param, values, meas, steps)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _on_result(self, result: SweepResult) -> None:
        self._result = result
        self._run_btn.setEnabled(True)

        if result.data:
            for m in result.measurements:
                vals = result.data[m]
                xs = result.values[: len(vals)]
                self._curve.setData(xs, vals)
                self._plot.setLabel("left", m)
                self._plot.setLabel("bottom", result.parameter)

        self._progress.setVisible(False)

    @property
    def result(self) -> SweepResult | None:
        return self._result
