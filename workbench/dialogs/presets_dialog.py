from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class _PresetCard(QWidget):
    clicked = Signal(str)

    def __init__(self, name: str, desc: str, thumb_path: str, parent=None):
        super().__init__(parent)
        self._file = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.setMinimumSize(180, 140)
        self.setMaximumSize(220, 180)
        self.setStyleSheet(
            "QWidget { background: #1e1e2e; border: 1px solid #3a3a4e; "
            "border-radius: 6px; } "
            "QWidget:hover { border: 1px solid #7c7cf0; }"
        )

        thumb_label = QLabel()
        if thumb_path and Path(thumb_path).exists():
            pix = QPixmap(thumb_path).scaled(
                200, 80, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setText("⚙")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet("font-size: 32px; color: #7c7cf0;")
        thumb_label.setMinimumHeight(60)
        layout.addWidget(thumb_label)

        name_label = QLabel(f"<b>{name}</b>")
        name_label.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(name_label)

        desc_label = QLabel(desc[:60] + ("…" if len(desc) > 60 else ""))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(desc_label)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._file)

    def set_file(self, path: str) -> None:
        self._file = path


class PresetsDialog(QDialog):
    preset_selected = Signal(str)

    def __init__(self, presets: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to S-Stream")
        self.setMinimumSize(620, 420)
        self.setModal(True)

        layout = QVBoxLayout(self)

        title = QLabel("<h2>Choose a Preset</h2>")
        title.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select a ready-made experiment to get started. "
            "You can also open a saved scene from the File menu."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #6c7086;")
        layout.addWidget(subtitle)

        scroll = QWidget()
        grid = QGridLayout(scroll)
        grid.setSpacing(12)
        for i, preset in enumerate(presets):
            card = _PresetCard(
                preset["name"],
                preset["description"],
                preset.get("thumbnail", ""),
            )
            card.set_file(preset["file"])
            card.clicked.connect(self._on_card_clicked)
            row, col = divmod(i, 3)
            grid.addWidget(card, row, col)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        close_btn = buttons.addButton("Skip", QDialogButtonBox.ButtonRole.RejectRole)
        close_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet(
            "QDialog { background: #11111b; } "
            "QLabel { color: #cdd6f4; } "
            "QPushButton { color: #cdd6f4; background: #313244; "
            "border: 1px solid #45475a; padding: 6px 16px; border-radius: 4px; } "
            "QPushButton:hover { background: #45475a; }"
        )

    def _on_card_clicked(self, file_path: str) -> None:
        self.preset_selected.emit(file_path)
        self.accept()
