APP_STYLESHEET = """
QMainWindow { background: #0f172a; }
QDockWidget {
    background: #0f172a;
    color: #e2e8f0;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background: #1e293b;
    padding: 6px 12px;
    font-weight: 600;
    font-size: 12px;
}
QToolBar {
    background: #1e293b;
    border: none;
    padding: 4px 8px;
    spacing: 4px;
}
QToolBar QPushButton {
    background: #334155;
    color: #e2e8f0;
    border: 1px solid #475569;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
}
QToolBar QPushButton:hover {
    background: #475569;
    border-color: #64748b;
}
QToolBar QPushButton:pressed {
    background: #1e40af;
}
QToolBar QPushButton:checked {
    background: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
}
QToolBar QPushButton:disabled {
    background: #1e293b;
    color: #64748b;
    border-color: #334155;
}
QMenuBar {
    background: #0f172a;
    color: #e2e8f0;
    border-bottom: 1px solid #1e293b;
    padding: 2px;
}
QMenuBar::item:selected { background: #334155; }
QMenu {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    padding: 4px;
}
QMenu::item {
    padding: 6px 28px 6px 16px;
    border-radius: 4px;
}
QMenu::item:selected { background: #2563eb; }
QMenu::separator {
    height: 1px;
    background: #334155;
    margin: 4px 8px;
}
QStatusBar {
    background: #0f172a;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
    font-size: 12px;
}
QLabel {
    color: #e2e8f0;
    background: transparent;
}
QGroupBox {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    color: #e2e8f0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: #94a3b8;
    font-size: 11px;
}
QTreeWidget {
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    outline: none;
    font-size: 12px;
}
QTreeWidget::item {
    padding: 4px 6px;
    min-height: 22px;
}
QTreeWidget::item:selected {
    background: #2563eb;
    color: #ffffff;
}
QTreeWidget::item:hover {
    background: #1e293b;
}
QListWidget {
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    outline: none;
    font-size: 12px;
}
QListWidget::item {
    padding: 6px 8px;
}
QListWidget::item:selected {
    background: #2563eb;
    color: #ffffff;
}
QTextEdit, QPlainTextEdit {
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    font-size: 12px;
}
QLineEdit {
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QLineEdit:focus {
    border-color: #3b82f6;
}
QDoubleSpinBox, QSpinBox {
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 12px;
    min-height: 20px;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #3b82f6;
}
QDoubleSpinBox::up-button, QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
}
QDoubleSpinBox::down-button, QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
}
QProgressBar {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: #e2e8f0;
    font-size: 11px;
    min-height: 18px;
}
QProgressBar::chunk {
    background: #2563eb;
    border-radius: 3px;
}
QTabWidget::pane {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
}
QTabBar::tab {
    background: #1e293b;
    color: #94a3b8;
    padding: 8px 16px;
    border: 1px solid #334155;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #0f172a;
    color: #e2e8f0;
    border-bottom: 1px solid #0f172a;
}
QTabBar::tab:hover:!selected {
    background: #334155;
}
QScrollBar:vertical {
    background: #0f172a;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #475569; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0f172a;
    height: 10px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #334155;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #475569; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QDialog {
    background: #0f172a;
    color: #e2e8f0;
}
QMessageBox {
    background: #0f172a;
    color: #e2e8f0;
}
QMessageBox QLabel { color: #e2e8f0; }
QMessageBox QPushButton {
    background: #334155;
    color: #e2e8f0;
    border: 1px solid #475569;
    padding: 6px 20px;
    border-radius: 4px;
    min-width: 60px;
}
QMessageBox QPushButton:hover { background: #475569; }
QFormLayout { margin: 0; }
"""
