from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class _PresetCard(QWidget):
    clicked = Signal(str)

    def __init__(self, preset: dict, parent=None):
        super().__init__(parent)
        self._file = preset["file"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.setMinimumSize(220, 190)
        self.setMaximumSize(260, 230)
        self.setStyleSheet(
            "QWidget { background: #111827; border: 1px solid #334155; "
            "border-radius: 10px; } "
            "QWidget:hover { border: 1px solid #38bdf8; background: #172033; }"
        )

        thumb_label = QLabel()
        thumb_path = preset.get("thumbnail", "")
        if thumb_path and Path(thumb_path).exists():
            pix = QPixmap(thumb_path).scaled(
                230,
                86,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setText("flow")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet("font-size: 24px; color: #38bdf8;")
        thumb_label.setMinimumHeight(72)
        layout.addWidget(thumb_label)

        name_label = QLabel(f"<b>{preset['name']}</b>")
        name_label.setStyleSheet("color: #f8fafc; font-size: 14px;")
        layout.addWidget(name_label)

        headline = (
            preset.get("headline") or preset.get("recipe") or "Ready-made flow story"
        )
        headline_label = QLabel(headline)
        headline_label.setWordWrap(True)
        headline_label.setStyleSheet("color: #7dd3fc; font-size: 11px;")
        layout.addWidget(headline_label)

        desc = preset["description"]
        desc_label = QLabel(desc[:92] + ("..." if len(desc) > 92 else ""))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(desc_label)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._file)


class PresetsDialog(QDialog):
    preset_selected = Signal(str)

    def __init__(self, presets: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to SStream")
        self.setMinimumSize(820, 560)
        self.setModal(True)

        layout = QVBoxLayout(self)

        title = QLabel("<h1>Pick a beautiful flow story</h1>")
        title.setStyleSheet("color: #f8fafc;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Start with a guided demo: SStream will choose the view, run the flow, "
            "explain what is happening, and help you export a figure."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #cbd5e1;")
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(14)
        for i, preset in enumerate(presets):
            card = _PresetCard(preset)
            card.clicked.connect(self._on_card_clicked)
            row, col = divmod(i, 3)
            grid.addWidget(card, row, col)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        close_btn = buttons.addButton("Skip", QDialogButtonBox.ButtonRole.RejectRole)
        close_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet(
            "QDialog { background: #020617; } "
            "QLabel { color: #e5e7eb; } "
            "QPushButton { color: #f8fafc; background: #1e293b; "
            "border: 1px solid #475569; padding: 7px 18px; border-radius: 5px; } "
            "QPushButton:hover { background: #334155; }"
        )

    def _on_card_clicked(self, file_path: str) -> None:
        self.preset_selected.emit(file_path)
        self.accept()
