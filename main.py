from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from engines import LBM2D
from workbench.app import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    sim = LBM2D(width=128, height=128)
    window = MainWindow(sim)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
