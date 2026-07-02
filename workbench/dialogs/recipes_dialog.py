from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

RECIPES = {
    "Show vortex shedding": (
        "Open Cylinder Wake, run the demo, switch to vorticity, "
        "and watch alternating wake structures."
    ),
    "Compare drag of two shapes": (
        "Draw or load two obstacle scenes, run each to a settled wake, "
        "then compare Cd and wake strength."
    ),
    "Generate Cd vs Re": (
        "Use Sweep Re to vary inlet/viscosity " "and export the plotted drag trend."
    ),
    "Explain Reynolds number": (
        "Start with Channel Flow, change viscosity, "
        "and watch the Re readout and sanity notes."
    ),
    "Create a lab-report figure": (
        "Run a flagship preset, wait for Demo ready, "
        "then export a report PNG and Markdown summary."
    ),
}


class RecipesDialog(QDialog):
    recipe_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Recipes")
        self.setMinimumSize(460, 360)

        layout = QVBoxLayout(self)
        title = QLabel("<h2>Pick a flow story</h2>")
        title.setWordWrap(True)
        layout.addWidget(title)

        self.list_widget = QListWidget()
        for name in RECIPES:
            self.list_widget.addItem(name)
        self.list_widget.currentTextChanged.connect(self._show_recipe)
        layout.addWidget(self.list_widget)

        self.details = QLabel("Choose a recipe to see the guided workflow.")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)

        use_btn = QPushButton("Use This Recipe")
        use_btn.clicked.connect(self._accept_recipe)
        layout.addWidget(use_btn)

        self.setStyleSheet(
            "QDialog { background: #111827; color: #e5e7eb; } "
            "QLabel { color: #e5e7eb; } "
            "QListWidget { background: #0b1020; color: #e5e7eb; "
            "border: 1px solid #374151; } "
            "QPushButton { background: #2563eb; color: white; "
            "padding: 8px; border-radius: 5px; }"
        )

    def _show_recipe(self, name: str) -> None:
        self.details.setText(RECIPES.get(name, ""))

    def _accept_recipe(self) -> None:
        item = self.list_widget.currentItem()
        if item:
            self.recipe_selected.emit(item.text())
        self.accept()
