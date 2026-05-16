"""CUDA-accelerated decode backend using OpenCV.

Uses ``cv2.VideoCapture`` with CUDA back-ends (when available) and
``cv2.cuda_GpuMat`` for GPU-resident colour-space conversion.  Falls
back transparently to the default OpenCV CPU path if CUDA support is
not compiled into the local OpenCV build.

Requires:
    - ``opencv-contrib-python`` (or a CUDA-enabled OpenCV build)
    - numpy
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from .frame_source import FrameSource

logger = logging.getLogger(__name__)

# Probe OpenCV CUDA support once at import time.
try:
    import cv2

    _CUDA_DEVICE_COUNT: int = 0
    try:
        _CUDA_DEVICE_COUNT = cv2.cuda.getCudaEnabledDeviceCount()
    except AttributeError:
        pass
    HAS_OPENCV_CUDA: bool = _CUDA_DEVICE_COUNT > 0
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_OPENCV_CUDA = False


class CUDAFrameSource(FrameSource):
    """OpenCV CUDA frame source for desktop NVIDIA GPUs.

    When OpenCV is built with CUDA support the backend attempts to use
    ``cv2.cudacodec.VideoReader`` for hardware-accelerated decode.  If
    that is not available it falls back to ``cv2.VideoCapture`` (which
    may still use hardware decode via GStreamer / FFmpeg backends under
    the hood).

    Parameters
    ----------
    source : str
        RTSP URL or local file path.
    width : int, optional
        Desired output width.
    height : int, optional
        Desired output height.
    prefer_cuda_codec : bool
        Prefer ``cv2.cudacodec`` when available (default ``True``).
    """

    def __init__(
        self,
        source: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        prefer_cuda_codec: bool = True,
    ) -> None:
        super().__init__(source, width, height)
        self._prefer_cuda = prefer_cuda_codec
        self._cap = None  # cv2.VideoCapture or cv2.cudacodec.VideoReader
        self._using_cuda_codec = False

    # ------------------------------------------------------------------
    # FrameSource interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video source with the best available OpenCV backend."""
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed.  "
                "Install via: pip install opencv-contrib-python"
            )

        # 1. Try cv2.cudacodec.VideoReader (GPU-resident decode)
        if self._prefer_cuda and HAS_OPENCV_CUDA:
            try:
                reader = cv2.cudacodec.createVideoReader(self._source)
                # Verify we can actually read from it
                ok, _ = reader.nextFrame()
                if ok:
                    self._cap = reader
                    self._using_cuda_codec = True
                    self._opened = True
                    self._log.info(
                        "Opened CUDA codec reader: %s", self._source
                    )
                    return
            except Exception as exc:
                self._log.warning(
                    "cv2.cudacodec failed (%s) — falling back to VideoCapture.",
                    exc,
                )

        # 2. Fallback: cv2.VideoCapture
        self._cap = self._open_video_capture()
        self._using_cuda_codec = False
        self._opened = True
        self._log.info(
            "Opened CUDA VideoCapture (hw_cuda=%s): %s",
            HAS_OPENCV_CUDA,
            self._source,
        )

    def read(self) -> Tuple[float, np.ndarray]:
        """Read the next decoded frame.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(timestamp, bgr_frame)``

        Raises
        ------
        StopIteration
            End of stream / file.
        RuntimeError
            Source not opened.
        """
        if not self._opened or self._cap is None:
            raise RuntimeError("Frame source is not open — call open() first")

        if self._using_cuda_codec:
            return self._read_cuda_codec()
        return self._read_video_capture()

    def release(self) -> None:
        """Release the underlying capture / reader."""
        if self._cap is not None:
            if not self._using_cuda_codec and hasattr(self._cap, "release"):
                self._cap.release()
            self._cap = None
        self._opened = False
        self._log.info("CUDA frame source released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_video_capture(self) -> "cv2.VideoCapture":
        """Open a ``cv2.VideoCapture`` with the best backend available."""
        # Prefer CAP_FFMPEG for RTSP, CAP_ANY for files
        backend = cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._source, backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"cv2.VideoCapture failed to open: {self._source}"
            )
        return cap

    def _read_video_capture(self) -> Tuple[float, np.ndarray]:
        """Read from ``cv2.VideoCapture``, optionally upload to GPU."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise StopIteration("No more frames from VideoCapture")

        ts = self._timestamp()

        # GPU-accelerated colour-space conversion when CUDA is available
        if HAS_OPENCV_CUDA:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                # Ensure BGR output (most captures already return BGR)
                frame = gpu_frame.download()
            except Exception:
                pass  # Silently fall back to CPU frame

        frame = self._resize_if_needed(frame)
        return ts, frame

    def _read_cuda_codec(self) -> Tuple[float, np.ndarray]:
        """Read from ``cv2.cudacodec.VideoReader``."""
        ok, gpu_frame = self._cap.nextFrame()
        if not ok:
            raise StopIteration("No more frames from CUDA codec reader")

        ts = self._timestamp()

        # gpu_frame is a cv2.cuda_GpuMat — download to host memory
        frame = gpu_frame.download()

        # cudacodec may return BGRA; convert to BGR if needed
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame = self._resize_if_needed(frame)
        return ts, frame
