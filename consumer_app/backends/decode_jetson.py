"""Jetson NVDEC decode backend using GStreamer.

Uses a GStreamer pipeline with ``nvv4l2decoder`` (NVDEC) for
hardware-accelerated H.264 / H.265 decode on NVIDIA Jetson platforms.
Falls back to software decode (``avdec_h264``) when NVDEC is unavailable.

Requires:
    - GStreamer 1.0 with Python bindings (``gi``)
    - ``gstreamer1.0-tools``, ``gstreamer1.0-plugins-*`` (ships with JetPack)
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from .frame_source import FrameSource

logger = logging.getLogger(__name__)

# GStreamer + GLib imports are optional — guarded so the module can be
# imported on systems without GStreamer for factory introspection.
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp, GLib  # noqa: F401
    HAS_GSTREAMER = True
except (ImportError, ValueError):
    HAS_GSTREAMER = False
    Gst = None  # type: ignore[assignment]
    GLib = None  # type: ignore[assignment]


class JetsonFrameSource(FrameSource):
    """GStreamer NVDEC frame source for NVIDIA Jetson.

    Constructs a GStreamer pipeline that decodes video using the Jetson
    hardware decoder (``nvv4l2decoder``) and presents decoded BGR frames
    via an ``appsink``.

    Parameters
    ----------
    source : str
        RTSP URL or local file path.
    width : int, optional
        Desired output width (default: keep native).
    height : int, optional
        Desired output height (default: keep native).
    use_hw_decoder : bool
        When ``True`` (default), attempt NVDEC.  When ``False``, use the
        software fallback pipeline directly.
    """

    def __init__(
        self,
        source: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        use_hw_decoder: bool = True,
    ) -> None:
        super().__init__(source, width, height)
        self._use_hw = use_hw_decoder
        self._pipeline = None
        self._appsink = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline_string(self) -> str:
        """Return the GStreamer pipeline description string."""
        if self.is_rtsp:
            src = (
                f"rtspsrc location={self._source} latency=200 "
                "! rtph264depay ! h264parse"
            )
        else:
            src = f"filesrc location={self._source} ! qtdemux ! h264parse"

        if self._use_hw:
            decoder = "nvv4l2decoder ! nvvidconv"
        else:
            decoder = "avdec_h264 ! videoconvert"

        caps = "video/x-raw, format=BGR"
        if self._width and self._height:
            caps += f", width={self._width}, height={self._height}"

        sink = (
            f"! {caps} "
            "! appsink name=sink emit-signals=true sync=false "
            "max-buffers=2 drop=true"
        )
        pipeline_str = f"{src} ! {decoder} {sink}"
        self._log.debug("GStreamer pipeline: %s", pipeline_str)
        return pipeline_str

    # ------------------------------------------------------------------
    # FrameSource interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialise GStreamer and start the pipeline."""
        if not HAS_GSTREAMER:
            raise RuntimeError(
                "GStreamer Python bindings (PyGObject + Gst) are not installed. "
                "Install via: apt install python3-gi gstreamer1.0-tools "
                "gstreamer1.0-plugins-base gstreamer1.0-plugins-good "
                "gstreamer1.0-plugins-bad"
            )

        Gst.init(None)

        pipeline_str = self._build_pipeline_string()
        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except Exception as exc:
            # If HW decoder fails, retry with software decoder
            if self._use_hw:
                self._log.warning(
                    "NVDEC pipeline failed (%s) — falling back to software decode.",
                    exc,
                )
                self._use_hw = False
                pipeline_str = self._build_pipeline_string()
                self._pipeline = Gst.parse_launch(pipeline_str)
            else:
                raise RuntimeError(f"GStreamer pipeline creation failed: {exc}") from exc

        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("Could not retrieve appsink from pipeline")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Failed to start GStreamer pipeline")

        self._opened = True
        self._log.info(
            "Opened Jetson decode: %s (hw=%s)", self._source, self._use_hw
        )

    def read(self) -> Tuple[float, np.ndarray]:
        """Pull the next decoded frame from the appsink.

        Returns
        -------
        tuple[float, numpy.ndarray]
            ``(timestamp, bgr_frame)``

        Raises
        ------
        StopIteration
            End of stream.
        RuntimeError
            Pipeline error or not opened.
        """
        if not self._opened or self._appsink is None:
            raise RuntimeError("Frame source is not open — call open() first")

        sample = self._appsink.emit("pull-sample")
        if sample is None:
            # Check for EOS
            bus = self._pipeline.get_bus()
            msg = bus.peek()
            if msg and msg.type == Gst.MessageType.EOS:
                raise StopIteration("End of stream")
            raise StopIteration("No more frames available")

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)

        width = struct.get_value("width")
        height = struct.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Failed to map GStreamer buffer")

        try:
            frame = np.ndarray(
                (height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data,
            ).copy()
        finally:
            buf.unmap(map_info)

        ts = self._timestamp()
        frame = self._resize_if_needed(frame)
        return ts, frame

    def release(self) -> None:
        """Stop the pipeline and free resources."""
        if self._pipeline is not None:
            try:
                if Gst is not None:
                    self._pipeline.set_state(Gst.State.NULL)
                else:
                    self._pipeline.set_state(0)  # GST_STATE_NULL = 0
            except Exception:
                pass
            self._pipeline = None
        self._appsink = None
        self._opened = False
        self._log.info("Jetson frame source released")
