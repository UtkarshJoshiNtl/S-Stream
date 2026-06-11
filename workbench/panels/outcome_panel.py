from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.ai_context import build_ai_context, local_ai_response
from analysis.physics import characteristic_length, drag_coefficient, reynolds_number
from analysis.regimes import detect_flow_regime
from analysis.sanity import check_sanity
from analysis.scorecard import compute_scorecard
from engines.base import SimEngine
from scene.probe import Probe
from scene.scene import Scene


class OutcomePanel(QWidget):
    def __init__(self, sim: SimEngine, scene: Scene, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.scene = scene
        self.probes: list[Probe] = []
        self.step_count = 0
        self.demo_target = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.headline = QLabel("<b>What am I seeing?</b>")
        self.headline.setWordWrap(True)
        layout.addWidget(self.headline)

        self.summary = QLabel("Load a preset or draw a shape to get an explanation.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.readouts = QLabel("")
        self.readouts.setWordWrap(True)
        layout.addWidget(self.readouts)

        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        layout.addWidget(self.warnings)

        self.ai_box = QTextEdit()
        self.ai_box.setReadOnly(True)
        self.ai_box.setMinimumHeight(160)
        self.ai_box.setPlaceholderText("AI tutor preview appears here.")
        layout.addWidget(self.ai_box, 1)

        self.setStyleSheet(
            "QWidget { background: #111827; color: #e5e7eb; } "
            "QLabel { color: #e5e7eb; } "
            "QTextEdit { background: #0b1020; color: #d1d5db; border: 1px solid #374151; }"
        )

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene
        self.demo_target = scene.product.autorun_steps
        self.update_outcome(force=True)

    def set_probes(self, probes: list[Probe]) -> None:
        self.probes = probes

    def set_demo_target(self, target: int) -> None:
        self.demo_target = max(0, target)
        self.progress.setVisible(self.demo_target > 0)
        self.progress.setMaximum(max(1, self.demo_target))

    def update_outcome(self, step_count: int | None = None, force: bool = False) -> None:
        if step_count is not None:
            self.step_count = step_count
        if not force and self.step_count % 15 != 0:
            return

        regime = detect_flow_regime(self.sim, self.scene, self.probes, self.step_count)
        warnings = check_sanity(self.sim, self.scene, self.probes, self.step_count)
        score = compute_scorecard(self.sim, self.scene, self.probes, self.step_count)
        length = characteristic_length(self.scene)
        re = reynolds_number(self.sim, length)
        cd = drag_coefficient(self.sim)

        headline = self.scene.product.lesson_headline or regime.label
        self.headline.setText(f"<b>{headline}</b>")
        self.summary.setText(regime.explanation)
        st_text = f"{regime.strouhal:.3f}" if regime.strouhal else "settling"
        self.readouts.setText(
            f"Re {re:.1f} | Cd {cd:.3f} | St {st_text}<br>"
            f"Scorecard: wake {score.wake_strength:.4f}, pressure-drop proxy {score.pressure_drop:.4f}<br>"
            f"{score.summary}"
        )
        if warnings:
            self.warnings.setText(
                "<br>".join(f"<b>{w.title}</b>: {w.message}" for w in warnings[:3])
            )
        else:
            self.warnings.setText("No major sanity warnings.")

        if self.demo_target > 0:
            self.progress.setVisible(True)
            self.progress.setValue(min(self.step_count, self.demo_target))
            self.progress.setFormat(
                "Building the flow story... %v/%m steps"
                if self.step_count < self.demo_target
                else "Demo ready"
            )

    def refresh_ai_preview(self, has_api_key: bool = False) -> None:
        regime = detect_flow_regime(self.sim, self.scene, self.probes, self.step_count)
        warnings = check_sanity(self.sim, self.scene, self.probes, self.step_count)
        context = build_ai_context(
            self.scene,
            self.sim,
            regime=regime,
            warnings=warnings,
            step_count=self.step_count,
        )
        self.ai_box.setPlainText(local_ai_response(context, has_api_key) + "\n\n" + context)
