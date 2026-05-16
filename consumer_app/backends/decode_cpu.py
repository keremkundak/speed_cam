"""CPU-only decode backend using ffmpeg-python.

Uses ``ffmpeg-python`` (a thin Python wrapper around the ``ffmpeg`` CLI)
for pure-software video decode.  Works on **any** system where ``ffmpeg``
is installed — no GPU required.

Requires:
    - ``ffmpeg`` binary on ``$PATH``
    - ``ffmpeg-python``  (``pip install ffmpeg-python``)
    - numpy
"""

import logging
import subprocess
import time
from typing import Optional, Tuple

import numpy as np

from .frame_source import FrameSource

logger = logging.getLogger(__name__)

try:
    import ffmpeg as ffmpeg_lib
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
    ffmpeg_lib = None  # type: ignore[assignment]


class CPUFrameSource(FrameSource):
    """Pure-software frame source backed by ffmpeg-python.

    Spawns an ``ffmpeg`` subprocess that decodes the video and pipes raw
    BGR24 frames to ``stdout``.  This is the most portable backend — it
    runs everywhere ``ffmpeg`` is available.

    Parameters
    ----------
    source : str
        RTSP URL or local file path.
    width : int, optional
        Desired output width.  When *both* width and height are given the
        frames are scaled by ffmpeg itself (avoiding a second resize in
        Python).
    height : int, optional
        Desired output height.
    """

    def __init__(
        self,
        source: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        super().__init__(source, width, height)
        self._process: Optional[subprocess.Popen] = None
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None

    # ------------------------------------------------------------------
    # FrameSource interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Probe the source for resolution, then spawn the decode process."""
        if not HAS_FFMPEG:
            raise RuntimeError(
                "ffmpeg-python is not installed.  "
                "Install via: pip install ffmpeg-python"
            )

        # Verify the ffmpeg binary is reachable
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg binary not found on $PATH.  "
                "Install via: apt install ffmpeg"
            )

        # Probe source to discover native resolution
        self._probe_resolution()

        # Build and launch the ffmpeg decode process
        self._process = self._spawn_ffmpeg()
        self._opened = True
        self._log.info(
            "Opened CPU decode: %s (%dx%d)",
            self._source,
            self._frame_width,
            self._frame_height,
        )

    def read(self) -> Tuple[float, np.ndarray]:
        """Read one raw BGR24 frame from the ffmpeg pipe.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(timestamp, bgr_frame)``

        Raises
        ------
        StopIteration
            When ffmpeg closes stdout (end-of-file / stream closed).
        RuntimeError
            If the source was not opened.
        """
        if not self._opened or self._process is None:
            raise RuntimeError("Frame source is not open — call open() first")

        frame_bytes = self._frame_width * self._frame_height * 3  # BGR24
        raw = self._process.stdout.read(frame_bytes)

        if not raw or len(raw) < frame_bytes:
            raise StopIteration("No more frames from ffmpeg process")

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._frame_height, self._frame_width, 3)
        )

        ts = self._timestamp()
        # Note: resize is already handled by ffmpeg via -s filter when
        # self._width/self._height are set; no extra resize needed.
        return ts, frame.copy()

    def release(self) -> None:
        """Terminate the ffmpeg subprocess and close pipes."""
        if self._process is not None:
            try:
                self._process.stdout.close()
            except Exception:
                pass
            try:
                self._process.stderr.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._opened = False
        self._log.info("CPU frame source released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_resolution(self) -> None:
        """Use ffmpeg.probe to discover the source's native resolution."""
        try:
            probe = ffmpeg_lib.probe(self._source)
        except Exception as exc:
            raise RuntimeError(
                f"ffmpeg probe failed for {self._source}: {exc}"
            ) from exc

        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"),
            None,
        )
        if video_stream is None:
            raise RuntimeError(
                f"No video stream found in source: {self._source}"
            )

        native_w = int(video_stream["width"])
        native_h = int(video_stream["height"])

        # Use requested dimensions or fall back to native
        self._frame_width = self._width or native_w
        self._frame_height = self._height or native_h

    def _spawn_ffmpeg(self) -> subprocess.Popen:
        """Spawn the ffmpeg subprocess that pipes raw BGR24 to stdout."""
        input_kwargs = {}
        if self.is_rtsp:
            input_kwargs["rtsp_transport"] = "tcp"

        stream = ffmpeg_lib.input(self._source, **input_kwargs)

        # Scale if user-requested dimensions differ from native
        stream = stream.output(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{self._frame_width}x{self._frame_height}",
        )

        # Run quietly (suppress banner / info)
        stream = stream.global_args("-loglevel", "error")

        process = stream.run_async(pipe_stdout=True, pipe_stderr=True)
        return process
