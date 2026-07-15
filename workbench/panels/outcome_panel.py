from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from analysis.physics import characteristic_length, drag_coefficient, reynolds_number
from analysis.regimes import detect_flow_regime
from analysis.sanity import check_sanity
from analysis.scorecard import compute_scorecard
from engines.base import SimEngine
from scene.probe import Probe
from scene.scene import Scene


class _Badge(QLabel):
    """Small colored pill badge."""

    def __init__(self, text: str, color: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"QLabel {{ background: {color}; color: white; padding: 2px 8px; "
            f"border-radius: 8px; font-size: 11px; font-weight: bold; }}"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


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

        # --- Zone A: Regime headline + confidence badge ---
        zone_a = QHBoxLayout()
        zone_a.setSpacing(8)
        self.headline = QLabel("<b>What am I seeing?</b>")
        self.headline.setWordWrap(True)
        zone_a.addWidget(self.headline, 1)
        self.confidence_badge = _Badge("", "#374151")
        zone_a.addWidget(self.confidence_badge)
        layout.addLayout(zone_a)

        self.summary = QLabel("Load a preset or draw a shape to get an explanation.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        # --- Progress bar ---
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # --- Zone B: Scorecard grid ---
        score_frame = QFrame()
        score_frame.setStyleSheet(
            "QFrame { background: #0f172a; border: 1px solid #1e293b; "
            "border-radius: 6px; padding: 4px; }"
        )
        score_layout = QFormLayout(score_frame)
        score_layout.setContentsMargins(8, 6, 8, 6)
        self.re_value = QLabel("—")
        self.cd_value = QLabel("—")
        self.st_value = QLabel("—")
        self.pdrop_value = QLabel("—")
        score_layout.addRow("Re", self.re_value)
        score_layout.addRow("Cd", self.cd_value)
        score_layout.addRow("St", self.st_value)
        score_layout.addRow("Δp", self.pdrop_value)
        layout.addWidget(score_frame)

        # --- Zone C: Sanity warnings ---
        self.warnings_layout = QVBoxLayout()
        self.warnings_layout.setSpacing(4)
        layout.addLayout(self.warnings_layout)

        layout.addStretch()

        self.setStyleSheet(
            "QWidget { background: #111827; color: #e5e7eb; } "
            "QLabel { color: #e5e7eb; } "
            "QFormLayout QLabel { color: #94a3b8; }"
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

    def update_outcome(
        self, step_count: int | None = None, force: bool = False
    ) -> None:
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
        st_text = f"{regime.strouhal:.3f}" if regime.strouhal else "settling"

        # --- Zone A: headline + confidence badge ---
        headline = self.scene.product.lesson_headline or regime.label
        self.headline.setText(f"<b>{headline}</b>")
        self.summary.setText(regime.explanation)

        conf = regime.confidence
        if conf >= 0.8:
            badge_color = "#16a34a"
            badge_text = f"Confident {conf:.0%}"
        elif conf >= 0.6:
            badge_color = "#ca8a04"
            badge_text = f"Moderate {conf:.0%}"
        else:
            badge_color = "#dc2626"
            badge_text = f"Uncertain {conf:.0%}"
        self.confidence_badge.setText(badge_text)
        self.confidence_badge.setStyleSheet(
            f"QLabel {{ background: {badge_color}; color: white; padding: 2px 8px; "
            f"border-radius: 8px; font-size: 11px; font-weight: bold; }}"
        )

        # --- Zone B: scorecard with range checks ---
        expected = self.scene.product.expected_ranges
        self.re_value.setText(self._range_text(re, "Re", expected))
        self.cd_value.setText(self._range_text(cd, "Cd", expected))
        self.st_value.setText(st_text if regime.strouhal is not None else "—")
        self.pdrop_value.setText(f"{score.pressure_drop:.4f}")

        # --- Zone C: sanity warnings as colored pills ---
        self._clear_warnings()
        for w in warnings[:3]:
            self._add_warning_pill(w.title, w.message, w.level)

    def _range_text(
        self, value: float, key: str, expected: dict[str, list[float]]
    ) -> str:
        if key not in expected:
            return f"{value:.2f}"
        lo, hi = expected[key]
        in_range = lo <= value <= hi
        indicator = "✓" if in_range else "✗"
        color = "#16a34a" if in_range else "#dc2626"
        return (
            f"{value:.2f}  "
            f"<span style='color:{color}; font-weight:bold'>{indicator}</span>"
            f" <span style='color:#64748b'>[{lo:.1f}–{hi:.1f}]</span>"
        )

    def _clear_warnings(self) -> None:
        while self.warnings_layout.count():
            item = self.warnings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_warning_pill(self, title: str, message: str, level: str = "warn") -> None:
        color_map = {
            "danger": "#7f1d1d",
            "error": "#7f1d1d",
            "warn": "#78350f",
            "warning": "#78350f",
            "info": "#1e3a5f",
        }
        bg = color_map.get(level, "#78350f")
        pill = QLabel(f"<b>{title}</b>: {message}")
        pill.setWordWrap(True)
        pill.setStyleSheet(
            f"QLabel {{ background: {bg}; color: #e5e7eb; padding: 4px 8px; "
            f"border-radius: 4px; font-size: 11px; }}"
        )
        self.warnings_layout.addWidget(pill)

    def refresh_ai_preview(self, has_api_key: bool = False) -> None:
        _ = has_api_key
