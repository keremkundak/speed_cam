"""Abstract base class for video frame sources.

All decode backends (Jetson NVDEC, CUDA, CPU) inherit from FrameSource
and implement the same interface so the rest of the pipeline is
hardware-agnostic.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract base class for video frame decode backends.

    Each implementation wraps a specific decode strategy (GStreamer NVDEC,
    OpenCV CUDA, or ffmpeg-python) and exposes a uniform ``read()`` method
    that returns ``(timestamp, frame)`` tuples.

    Parameters
    ----------
    source : str
        RTSP URL (``rtsp://…``) or local file path.
    width : int, optional
        Desired output frame width.  ``None`` keeps the native resolution.
    height : int, optional
        Desired output frame height.  ``None`` keeps the native resolution.
    """

    def __init__(
        self,
        source: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._opened = False
        self._log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Open the video source and prepare for frame reads.

        Raises
        ------
        RuntimeError
            If the source cannot be opened.
        """

    @abstractmethod
    def read(self) -> Tuple[float, np.ndarray]:
        """Read the next frame.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(timestamp, frame)`` where *timestamp* is seconds since epoch
            (``time.time()`` at capture) and *frame* is a BGR ``uint8``
            ndarray of shape ``(H, W, 3)``.

        Raises
        ------
        StopIteration
            When the source is exhausted (end of file / stream closed).
        RuntimeError
            On unrecoverable decode errors.
        """

    @abstractmethod
    def release(self) -> None:
        """Release all resources (handles, GPU memory, subprocesses)."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Return the original source URL / file path."""
        return self._source

    @property
    def is_opened(self) -> bool:
        """Return ``True`` if the source is currently open."""
        return self._opened

    @property
    def is_rtsp(self) -> bool:
        """Return ``True`` when the source is an RTSP stream."""
        return self._source.lower().startswith("rtsp://")

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        """Return ``(width, height)`` if known, else ``None``."""
        if self._width and self._height:
            return (self._width, self._height)
        return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> float:
        """Return the current wall-clock time in seconds since epoch."""
        return time.time()

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize *frame* to ``(self._width, self._height)`` if set.

        Import of ``cv2`` is deferred so the ABC module itself does not
        require OpenCV at import time.
        """
        if self._width and self._height:
            h, w = frame.shape[:2]
            if w != self._width or h != self._height:
                import cv2
                frame = cv2.resize(
                    frame, (self._width, self._height),
                    interpolation=cv2.INTER_LINEAR,
                )
        return frame

    def __repr__(self) -> str:
        status = "open" if self._opened else "closed"
        return f"{self.__class__.__name__}(source={self._source!r}, {status})"
