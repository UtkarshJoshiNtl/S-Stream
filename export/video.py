from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage


class VideoRecorder:
    """Record simulation frames to an MP4 or GIF file via imageio.

    Parameters
    ----------
    path: Output file path (must end in .mp4 or .gif).
    fps: Frames per second in the output.
    max_frames: Maximum number of frames to record (None = unlimited).
    """

    def __init__(
        self,
        path: str | Path,
        fps: int = 30,
        max_frames: int | None = 900,
    ) -> None:
        import imageio.v3 as iio

        self._path = Path(path)
        self._fps = fps
        self._max_frames = max_frames
        self._count = 0
        ext = self._path.suffix.lower()
        if ext not in (".mp4", ".gif"):
            raise ValueError(f"Unsupported format: {ext} (use .mp4 or .gif)")
        self._writer = iio.imopen(
            str(self._path),
            "w",
            extension=ext,
        )

    @property
    def recording(self) -> bool:
        return self._count < (self._max_frames or float("inf"))

    @property
    def frame_count(self) -> int:
        return self._count

    def add_frame(self, qimage: QImage) -> bool:
        """Append a frame from a QImage.

        Returns True if the frame was recorded, False if max_frames reached.
        """
        if self._max_frames is not None and self._count >= self._max_frames:
            return False
        arr = self._qimage_to_array(qimage)
        self._writer.write(arr)
        self._count += 1
        return True

    def close(self) -> None:
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @staticmethod
    def _qimage_to_array(img: QImage) -> np.ndarray:
        img = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3).copy()
        return arr
