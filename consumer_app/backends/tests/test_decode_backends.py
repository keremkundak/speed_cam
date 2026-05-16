"""Tests for decode backends (Jetson NVDEC, CUDA, CPU).

All hardware-specific dependencies are mocked so tests run on any system.
"""

import subprocess
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Tuple

import numpy as np

from consumer_app.backends.frame_source import FrameSource


# -----------------------------------------------------------------------
# FrameSource ABC tests
# -----------------------------------------------------------------------

class ConcreteFrameSource(FrameSource):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, source, width=None, height=None):
        super().__init__(source, width, height)
        self._frames = []
        self._idx = 0

    def open(self):
        self._opened = True

    def read(self) -> Tuple[float, np.ndarray]:
        if self._idx >= len(self._frames):
            raise StopIteration("No more frames")
        frame = self._frames[self._idx]
        self._idx += 1
        return self._timestamp(), frame

    def release(self):
        self._opened = False


class TestFrameSourceABC(unittest.TestCase):
    """Test the FrameSource abstract base class."""

    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            FrameSource("rtsp://test")

    def test_concrete_instantiation(self):
        src = ConcreteFrameSource("rtsp://host/cam0")
        self.assertEqual(src.source, "rtsp://host/cam0")
        self.assertTrue(src.is_rtsp)
        self.assertFalse(src.is_opened)

    def test_file_source_not_rtsp(self):
        src = ConcreteFrameSource("/path/to/video.mp4")
        self.assertFalse(src.is_rtsp)

    def test_resolution_property(self):
        src = ConcreteFrameSource("file.mp4", width=1280, height=720)
        self.assertEqual(src.resolution, (1280, 720))

    def test_resolution_none_when_unset(self):
        src = ConcreteFrameSource("file.mp4")
        self.assertIsNone(src.resolution)

    def test_context_manager(self):
        src = ConcreteFrameSource("file.mp4")
        with src as s:
            self.assertTrue(s.is_opened)
        self.assertFalse(src.is_opened)

    def test_repr(self):
        src = ConcreteFrameSource("test.mp4")
        self.assertIn("ConcreteFrameSource", repr(src))
        self.assertIn("closed", repr(src))
        src.open()
        self.assertIn("open", repr(src))

    def test_timestamp_returns_float(self):
        ts = FrameSource._timestamp()
        self.assertIsInstance(ts, float)
        self.assertGreater(ts, 0)

    def test_resize_if_needed(self):
        mock_cv2 = MagicMock()
        mock_cv2.INTER_LINEAR = 1
        mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            src = ConcreteFrameSource("f.mp4", width=640, height=480)
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            result = src._resize_if_needed(frame)
            mock_cv2.resize.assert_called_once()

    def test_resize_skipped_when_matching(self):
        src = ConcreteFrameSource("f.mp4", width=640, height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = src._resize_if_needed(frame)
        np.testing.assert_array_equal(result, frame)

    def test_resize_skipped_when_no_target(self):
        src = ConcreteFrameSource("f.mp4")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = src._resize_if_needed(frame)
        np.testing.assert_array_equal(result, frame)


# -----------------------------------------------------------------------
# Jetson decode backend tests
# -----------------------------------------------------------------------

class TestJetsonFrameSource(unittest.TestCase):
    """Test JetsonFrameSource with mocked GStreamer."""

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", False)
    def test_open_raises_without_gstreamer(self):
        from consumer_app.backends.decode_jetson import JetsonFrameSource
        src = JetsonFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError) as ctx:
            src.open()
        self.assertIn("GStreamer", str(ctx.exception))

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", True)
    @patch("consumer_app.backends.decode_jetson.Gst")
    @patch("consumer_app.backends.decode_jetson.GLib")
    def test_open_rtsp_hw_success(self, mock_glib, mock_gst):
        from consumer_app.backends.decode_jetson import JetsonFrameSource

        mock_pipeline = MagicMock()
        mock_gst.parse_launch.return_value = mock_pipeline
        mock_pipeline.get_by_name.return_value = MagicMock()
        mock_pipeline.set_state.return_value = mock_gst.StateChangeReturn.SUCCESS

        src = JetsonFrameSource("rtsp://host/cam0")
        src.open()
        self.assertTrue(src.is_opened)
        mock_gst.init.assert_called_once_with(None)

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", True)
    @patch("consumer_app.backends.decode_jetson.Gst")
    @patch("consumer_app.backends.decode_jetson.GLib")
    def test_open_file_source(self, mock_glib, mock_gst):
        from consumer_app.backends.decode_jetson import JetsonFrameSource

        mock_pipeline = MagicMock()
        mock_gst.parse_launch.return_value = mock_pipeline
        mock_pipeline.get_by_name.return_value = MagicMock()
        mock_pipeline.set_state.return_value = mock_gst.StateChangeReturn.SUCCESS

        src = JetsonFrameSource("/videos/test.mp4")
        src.open()
        self.assertTrue(src.is_opened)
        pipeline_str = mock_gst.parse_launch.call_args[0][0]
        self.assertIn("filesrc", pipeline_str)

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", True)
    @patch("consumer_app.backends.decode_jetson.Gst")
    @patch("consumer_app.backends.decode_jetson.GLib")
    def test_open_pipeline_failure_raises(self, mock_glib, mock_gst):
        from consumer_app.backends.decode_jetson import JetsonFrameSource

        mock_pipeline = MagicMock()
        mock_gst.parse_launch.return_value = mock_pipeline
        mock_pipeline.get_by_name.return_value = MagicMock()
        mock_pipeline.set_state.return_value = mock_gst.StateChangeReturn.FAILURE

        src = JetsonFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError):
            src.open()

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", True)
    @patch("consumer_app.backends.decode_jetson.Gst")
    @patch("consumer_app.backends.decode_jetson.GLib")
    def test_open_hw_fallback_to_sw(self, mock_glib, mock_gst):
        """Test that HW decode failure triggers SW fallback."""
        from consumer_app.backends.decode_jetson import JetsonFrameSource

        call_count = [0]
        def parse_side_effect(pipeline_str):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("NVDEC unavailable")
            mock_pipe = MagicMock()
            mock_pipe.get_by_name.return_value = MagicMock()
            mock_pipe.set_state.return_value = mock_gst.StateChangeReturn.SUCCESS
            return mock_pipe

        mock_gst.parse_launch.side_effect = parse_side_effect

        src = JetsonFrameSource("rtsp://host/cam0", use_hw_decoder=True)
        src.open()
        self.assertTrue(src.is_opened)
        self.assertFalse(src._use_hw)

    def test_read_before_open_raises(self):
        from consumer_app.backends.decode_jetson import JetsonFrameSource
        src = JetsonFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError):
            src.read()

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", True)
    @patch("consumer_app.backends.decode_jetson.Gst")
    @patch("consumer_app.backends.decode_jetson.GLib")
    def test_read_frame(self, mock_glib, mock_gst):
        from consumer_app.backends.decode_jetson import JetsonFrameSource

        mock_pipeline = MagicMock()
        mock_gst.parse_launch.return_value = mock_pipeline
        mock_appsink = MagicMock()
        mock_pipeline.get_by_name.return_value = mock_appsink
        mock_pipeline.set_state.return_value = mock_gst.StateChangeReturn.SUCCESS

        # Mock sample with frame data
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_sample = MagicMock()
        mock_buf = MagicMock()
        mock_caps = MagicMock()
        mock_struct = MagicMock()
        mock_struct.get_value.side_effect = lambda k: 640 if k == "width" else 480
        mock_caps.get_structure.return_value = mock_struct
        mock_sample.get_buffer.return_value = mock_buf
        mock_sample.get_caps.return_value = mock_caps
        mock_map_info = MagicMock()
        mock_map_info.data = fake_frame.tobytes()
        mock_buf.map.return_value = (True, mock_map_info)
        mock_appsink.emit.return_value = mock_sample

        src = JetsonFrameSource("rtsp://host/cam0")
        src.open()
        ts, frame = src.read()

        self.assertIsInstance(ts, float)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    @patch("consumer_app.backends.decode_jetson.HAS_GSTREAMER", True)
    @patch("consumer_app.backends.decode_jetson.Gst")
    @patch("consumer_app.backends.decode_jetson.GLib")
    def test_read_eos_raises_stop_iteration(self, mock_glib, mock_gst):
        from consumer_app.backends.decode_jetson import JetsonFrameSource

        mock_pipeline = MagicMock()
        mock_gst.parse_launch.return_value = mock_pipeline
        mock_appsink = MagicMock()
        mock_pipeline.get_by_name.return_value = mock_appsink
        mock_pipeline.set_state.return_value = mock_gst.StateChangeReturn.SUCCESS
        mock_appsink.emit.return_value = None

        mock_bus = MagicMock()
        mock_msg = MagicMock()
        mock_msg.type = mock_gst.MessageType.EOS
        mock_bus.peek.return_value = mock_msg
        mock_pipeline.get_bus.return_value = mock_bus

        src = JetsonFrameSource("rtsp://host/cam0")
        src.open()
        with self.assertRaises(StopIteration):
            src.read()

    @patch("consumer_app.backends.decode_jetson.Gst")
    def test_release(self, mock_gst):
        from consumer_app.backends.decode_jetson import JetsonFrameSource
        src = JetsonFrameSource("rtsp://host/cam0")
        src._pipeline = MagicMock()
        src._appsink = MagicMock()
        src._opened = True
        src.release()
        self.assertFalse(src.is_opened)
        self.assertIsNone(src._pipeline)


# -----------------------------------------------------------------------
# CUDA decode backend tests
# -----------------------------------------------------------------------

class TestCUDAFrameSource(unittest.TestCase):
    """Test CUDAFrameSource with mocked OpenCV."""

    @patch("consumer_app.backends.decode_cuda.cv2", None)
    def test_open_raises_without_opencv(self):
        from consumer_app.backends.decode_cuda import CUDAFrameSource
        src = CUDAFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError) as ctx:
            src.open()
        self.assertIn("OpenCV", str(ctx.exception))

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", False)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_open_video_capture_fallback(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_FFMPEG = 1900

        src = CUDAFrameSource("rtsp://host/cam0")
        src.open()
        self.assertTrue(src.is_opened)
        self.assertFalse(src._using_cuda_codec)

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", True)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_open_cuda_codec_success(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        mock_reader = MagicMock()
        mock_reader.nextFrame.return_value = (True, MagicMock())
        mock_cv2.cudacodec.createVideoReader.return_value = mock_reader

        src = CUDAFrameSource("rtsp://host/cam0")
        src.open()
        self.assertTrue(src.is_opened)
        self.assertTrue(src._using_cuda_codec)

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", True)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_open_cuda_codec_fallback(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        mock_cv2.cudacodec.createVideoReader.side_effect = Exception("fail")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_ANY = 0

        src = CUDAFrameSource("/video.mp4")
        src.open()
        self.assertTrue(src.is_opened)
        self.assertFalse(src._using_cuda_codec)

    def test_read_before_open_raises(self):
        from consumer_app.backends.decode_cuda import CUDAFrameSource
        src = CUDAFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError):
            src.read()

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", False)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_read_video_capture(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_ANY = 0

        src = CUDAFrameSource("/video.mp4")
        src.open()
        ts, frame = src.read()

        self.assertIsInstance(ts, float)
        self.assertEqual(frame.shape, (480, 640, 3))

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", False)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_read_eof_raises_stop_iteration(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_ANY = 0

        src = CUDAFrameSource("/video.mp4")
        src.open()
        with self.assertRaises(StopIteration):
            src.read()

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", True)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_read_cuda_codec_bgra_conversion(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        bgra_frame = np.zeros((480, 640, 4), dtype=np.uint8)
        bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_reader = MagicMock()
        mock_gpu_frame = MagicMock()
        mock_gpu_frame.download.return_value = bgra_frame
        mock_reader.nextFrame.side_effect = [(True, MagicMock()), (True, mock_gpu_frame)]
        mock_cv2.cudacodec.createVideoReader.return_value = mock_reader
        mock_cv2.cvtColor.return_value = bgr_frame
        mock_cv2.COLOR_BGRA2BGR = 3

        src = CUDAFrameSource("rtsp://host/cam0")
        src.open()
        ts, frame = src.read()
        mock_cv2.cvtColor.assert_called_once()
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_release(self):
        from consumer_app.backends.decode_cuda import CUDAFrameSource
        src = CUDAFrameSource("rtsp://host/cam0")
        src._cap = MagicMock()
        src._opened = True
        src._using_cuda_codec = False
        src.release()
        self.assertFalse(src.is_opened)
        self.assertIsNone(src._cap)

    @patch("consumer_app.backends.decode_cuda.HAS_OPENCV_CUDA", False)
    @patch("consumer_app.backends.decode_cuda.cv2")
    def test_open_failure_raises(self, mock_cv2):
        from consumer_app.backends.decode_cuda import CUDAFrameSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_ANY = 0

        src = CUDAFrameSource("/nonexistent.mp4")
        with self.assertRaises(RuntimeError):
            src.open()


# -----------------------------------------------------------------------
# CPU decode backend tests
# -----------------------------------------------------------------------

class TestCPUFrameSource(unittest.TestCase):
    """Test CPUFrameSource with mocked ffmpeg-python."""

    @patch("consumer_app.backends.decode_cpu.HAS_FFMPEG", False)
    def test_open_raises_without_ffmpeg_python(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource
        src = CPUFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError) as ctx:
            src.open()
        self.assertIn("ffmpeg-python", str(ctx.exception))

    @patch("consumer_app.backends.decode_cpu.HAS_FFMPEG", True)
    @patch("consumer_app.backends.decode_cpu.ffmpeg_lib")
    @patch("subprocess.run")
    def test_open_success(self, mock_run, mock_ffmpeg):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_ffmpeg.probe.return_value = {
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}]
        }
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_stream.output.return_value = mock_stream
        mock_stream.global_args.return_value = mock_stream
        mock_stream.run_async.return_value = MagicMock()

        src = CPUFrameSource("/videos/test.mp4")
        src.open()
        self.assertTrue(src.is_opened)
        self.assertEqual(src._frame_width, 1280)
        self.assertEqual(src._frame_height, 720)

    @patch("consumer_app.backends.decode_cpu.HAS_FFMPEG", True)
    @patch("consumer_app.backends.decode_cpu.ffmpeg_lib")
    @patch("subprocess.run")
    def test_open_custom_resolution(self, mock_run, mock_ffmpeg):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_ffmpeg.probe.return_value = {
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080}]
        }
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_stream.output.return_value = mock_stream
        mock_stream.global_args.return_value = mock_stream
        mock_stream.run_async.return_value = MagicMock()

        src = CPUFrameSource("/videos/test.mp4", width=640, height=480)
        src.open()
        self.assertEqual(src._frame_width, 640)
        self.assertEqual(src._frame_height, 480)

    @patch("consumer_app.backends.decode_cpu.HAS_FFMPEG", True)
    @patch("subprocess.run")
    def test_open_ffmpeg_binary_missing(self, mock_run):
        from consumer_app.backends.decode_cpu import CPUFrameSource
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")
        src = CPUFrameSource("/videos/test.mp4")
        with self.assertRaises(RuntimeError) as ctx:
            src.open()
        self.assertIn("ffmpeg binary", str(ctx.exception))

    def test_read_before_open_raises(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource
        src = CPUFrameSource("rtsp://host/cam0")
        with self.assertRaises(RuntimeError):
            src.read()

    def test_read_frame(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_process = MagicMock()
        mock_process.stdout.read.return_value = fake_frame.tobytes()

        src = CPUFrameSource("/videos/test.mp4")
        src._opened = True
        src._process = mock_process
        src._frame_width = 640
        src._frame_height = 480

        ts, frame = src.read()
        self.assertIsInstance(ts, float)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_read_eof_raises_stop_iteration(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_process = MagicMock()
        mock_process.stdout.read.return_value = b""

        src = CPUFrameSource("/videos/test.mp4")
        src._opened = True
        src._process = mock_process
        src._frame_width = 640
        src._frame_height = 480

        with self.assertRaises(StopIteration):
            src.read()

    def test_read_short_read_raises_stop_iteration(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_process = MagicMock()
        mock_process.stdout.read.return_value = b"\x00" * 100

        src = CPUFrameSource("/videos/test.mp4")
        src._opened = True
        src._process = mock_process
        src._frame_width = 640
        src._frame_height = 480

        with self.assertRaises(StopIteration):
            src.read()

    def test_release(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_process = MagicMock()
        src = CPUFrameSource("rtsp://host/cam0")
        src._process = mock_process
        src._opened = True
        src.release()

        self.assertFalse(src.is_opened)
        self.assertIsNone(src._process)
        mock_process.terminate.assert_called_once()

    def test_release_kills_on_timeout(self):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_process = MagicMock()
        mock_process.wait.side_effect = Exception("timeout")
        src = CPUFrameSource("rtsp://host/cam0")
        src._process = mock_process
        src._opened = True
        src.release()

        mock_process.kill.assert_called_once()
        self.assertFalse(src.is_opened)

    @patch("consumer_app.backends.decode_cpu.HAS_FFMPEG", True)
    @patch("consumer_app.backends.decode_cpu.ffmpeg_lib")
    @patch("subprocess.run")
    def test_rtsp_uses_tcp_transport(self, mock_run, mock_ffmpeg):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_ffmpeg.probe.return_value = {
            "streams": [{"codec_type": "video", "width": 640, "height": 480}]
        }
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_stream.output.return_value = mock_stream
        mock_stream.global_args.return_value = mock_stream
        mock_stream.run_async.return_value = MagicMock()

        src = CPUFrameSource("rtsp://host/cam0")
        src.open()
        mock_ffmpeg.input.assert_called_with(
            "rtsp://host/cam0", rtsp_transport="tcp"
        )

    @patch("consumer_app.backends.decode_cpu.HAS_FFMPEG", True)
    @patch("consumer_app.backends.decode_cpu.ffmpeg_lib")
    @patch("subprocess.run")
    def test_probe_no_video_stream_raises(self, mock_run, mock_ffmpeg):
        from consumer_app.backends.decode_cpu import CPUFrameSource

        mock_ffmpeg.probe.return_value = {
            "streams": [{"codec_type": "audio", "sample_rate": 44100}]
        }

        src = CPUFrameSource("/audio_only.mp3")
        with self.assertRaises(RuntimeError) as ctx:
            src.open()
        self.assertIn("No video stream", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
