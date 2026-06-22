from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ExportDialog(QDialog):
    """Tabbed dialog for exporting images, video, and data."""

    # Emitted when the user wants to start video recording
    start_recording = Signal(str, int, int)  # path, fps, max_frames
    # Emitted when image export is configured
    export_image_requested = Signal(str, int, bool, bool)
    # Emitted when data export is configured
    export_data_requested = Signal(str, str)  # type ("csv" | "npz"), path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._build_image_tab(), "Image")
        tabs.addTab(self._build_video_tab(), "Video")
        tabs.addTab(self._build_data_tab(), "Data")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # --- Image Tab ---

    def _build_image_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form = QFormLayout()
        self._img_path = QLineEdit()
        self._img_path.setPlaceholderText("Path…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_img)
        path_row = QHBoxLayout()
        path_row.addWidget(self._img_path)
        path_row.addWidget(browse_btn)
        form.addRow("File:", path_row)

        self._img_scale = QSpinBox()
        self._img_scale.setRange(1, 8)
        self._img_scale.setValue(2)
        self._img_scale.setSuffix("×")
        form.addRow("Resolution:", self._img_scale)

        self._img_colorbar = QCheckBox("Include colorbar")
        self._img_colorbar.setChecked(True)
        form.addRow(self._img_colorbar)

        self._img_annotations = QCheckBox("Include annotations")
        self._img_annotations.setChecked(True)
        form.addRow(self._img_annotations)

        self._img_field = QComboBox()
        self._img_field.addItems(["smoke", "speed", "vorticity", "pressure"])
        form.addRow("Field:", self._img_field)

        layout.addLayout(form)
        layout.addStretch()

        export_btn = QPushButton("Export Image")
        export_btn.clicked.connect(self._do_export_image)
        layout.addWidget(export_btn)
        return w

    def _browse_img(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG (*.png)"
        )
        if path:
            self._img_path.setText(path)

    def _do_export_image(self) -> None:
        path = self._img_path.text().strip()
        if not path:
            return
        self.export_image_requested.emit(
            path,
            self._img_scale.value(),
            self._img_colorbar.isChecked(),
            self._img_annotations.isChecked(),
        )

    # --- Video Tab ---

    def _build_video_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form = QFormLayout()
        self._vid_path = QLineEdit()
        self._vid_path.setPlaceholderText("Path…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_vid)
        path_row = QHBoxLayout()
        path_row.addWidget(self._vid_path)
        path_row.addWidget(browse_btn)
        form.addRow("File:", path_row)

        self._vid_fps = QSpinBox()
        self._vid_fps.setRange(1, 120)
        self._vid_fps.setValue(30)
        self._vid_fps.setSuffix(" fps")
        form.addRow("Frame rate:", self._vid_fps)

        self._vid_max = QSpinBox()
        self._vid_max.setRange(0, 99999)
        self._vid_max.setValue(900)
        self._vid_max.setSpecialValueText("Unlimited")
        form.addRow("Max frames:", self._vid_max)

        layout.addLayout(form)
        layout.addStretch()

        record_btn = QPushButton("Start Recording")
        record_btn.clicked.connect(self._do_start_recording)
        layout.addWidget(record_btn)
        return w

    def _browse_vid(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Record Video", "", "MP4 (*.mp4);;GIF (*.gif)"
        )
        if path:
            self._vid_path.setText(path)

    def _do_start_recording(self) -> None:
        path = self._vid_path.text().strip()
        if not path:
            return
        max_frames = self._vid_max.value() if self._vid_max.value() > 0 else 0
        self.start_recording.emit(path, self._vid_fps.value(), max_frames)

    # --- Data Tab ---

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        group = QGroupBox("Export type")
        rlayout = QVBoxLayout(group)
        self._data_csv = QRadioButton("Probe CSV")
        self._data_csv.setChecked(True)
        self._data_npz = QRadioButton("Field snapshot (.npz)")
        rlayout.addWidget(self._data_csv)
        rlayout.addWidget(self._data_npz)
        layout.addWidget(group)

        form = QFormLayout()
        self._data_path = QLineEdit()
        self._data_path.setPlaceholderText("Path…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_data)
        path_row = QHBoxLayout()
        path_row.addWidget(self._data_path)
        path_row.addWidget(browse_btn)
        form.addRow("File:", path_row)
        layout.addLayout(form)
        layout.addStretch()

        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self._do_export_data)
        layout.addWidget(export_btn)
        return w

    def _browse_data(self) -> None:
        filt = "CSV (*.csv)" if self._data_csv.isChecked() else "NPZ (*.npz)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", filt
        )
        if path:
            self._data_path.setText(path)

    def _do_export_data(self) -> None:
        path = self._data_path.text().strip()
        if not path:
            return
        typ = "csv" if self._data_csv.isChecked() else "npz"
        self.export_data_requested.emit(typ, path)
