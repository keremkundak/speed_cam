"""Tests for hardware detection backend selector."""

import unittest
from unittest.mock import patch, MagicMock

from consumer_app.backends.detector import (
    detect_hardware,
    get_backend,
    clear_backend_cache,
    _detect_jetson,
    _detect_cuda_gpu,
    _detect_torch_cuda,
)
from consumer_app.backends.hw_backend import HWBackend


class TestHWBackendEnum(unittest.TestCase):
    """Test HWBackend enum properties."""

    def test_jetson_enum(self):
        """Test Jetson enum value and properties."""
        self.assertEqual(HWBackend.JETSON.value, "jetson")
        self.assertTrue(HWBackend.JETSON.has_gpu)
        self.assertTrue(HWBackend.JETSON.is_jetson)
        self.assertFalse(HWBackend.JETSON.is_cuda)
        self.assertFalse(HWBackend.JETSON.is_cpu)

    def test_cuda_enum(self):
        """Test CUDA enum value and properties."""
        self.assertEqual(HWBackend.CUDA.value, "cuda")
        self.assertTrue(HWBackend.CUDA.has_gpu)
        self.assertFalse(HWBackend.CUDA.is_jetson)
        self.assertTrue(HWBackend.CUDA.is_cuda)
        self.assertFalse(HWBackend.CUDA.is_cpu)

    def test_cpu_enum(self):
        """Test CPU enum value and properties."""
        self.assertEqual(HWBackend.CPU.value, "cpu")
        self.assertFalse(HWBackend.CPU.has_gpu)
        self.assertFalse(HWBackend.CPU.is_jetson)
        self.assertFalse(HWBackend.CPU.is_cuda)
        self.assertTrue(HWBackend.CPU.is_cpu)

    def test_enum_string_representation(self):
        """Test string representations."""
        self.assertEqual(str(HWBackend.JETSON), "JETSON")
        self.assertEqual(str(HWBackend.CUDA), "CUDA")
        self.assertEqual(str(HWBackend.CPU), "CPU")


class TestJetsonDetection(unittest.TestCase):
    """Test Jetson detection logic."""

    @patch("consumer_app.backends.detector.HAS_JETSON_STATS", False)
    def test_jetson_detection_no_jetson_stats(self):
        """Test returns None if jetson_stats not installed."""
        result = _detect_jetson()
        self.assertIsNone(result)

    @patch("consumer_app.backends.detector.HAS_JETSON_STATS", True)
    @patch("consumer_app.backends.detector.jetson_stats")
    def test_jetson_detection_success(self, mock_jetson_stats):
        """Test successful Jetson detection."""
        mock_js = MagicMock()
        mock_js.ok = True
        mock_js.board = {
            "name": "Jetson Orin NX",
            "l4t": "36.2.0",
        }
        mock_js.ram = {"total": 8589934592}  # 8GB in bytes
        mock_jetson_stats.return_value = mock_js

        result = _detect_jetson()
        self.assertIsNotNone(result)
        self.assertEqual(result["platform"], "jetson")
        self.assertEqual(result["model"], "Jetson Orin NX")

    @patch("consumer_app.backends.detector.HAS_JETSON_STATS", True)
    @patch("consumer_app.backends.detector.jetson_stats")
    def test_jetson_detection_failure(self, mock_jetson_stats):
        """Test graceful failure if jetson_stats raises."""
        mock_jetson_stats.side_effect = Exception("jetson_stats error")

        result = _detect_jetson()
        self.assertIsNone(result)


class TestCUDADetection(unittest.TestCase):
    """Test CUDA GPU detection logic."""

    @patch("consumer_app.backends.detector.HAS_PYNVML", False)
    @patch("subprocess.check_output")
    def test_cuda_detection_via_nvidia_smi(self, mock_subprocess):
        """Test CUDA detection via nvidia-smi."""
        mock_subprocess.return_value = "NVIDIA A100, 40GB\nNVIDIA A100, 40GB\n"

        result = _detect_cuda_gpu()
        self.assertIsNotNone(result)
        self.assertEqual(result["platform"], "cuda")
        self.assertIn("A100", result["model"])

    @patch("consumer_app.backends.detector.HAS_PYNVML", True)
    @patch("consumer_app.backends.detector.pynvml")
    def test_cuda_detection_via_pynvml(self, mock_pynvml):
        """Test CUDA detection via pynvml."""
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 3090"
        mock_memory = MagicMock()
        mock_memory.total = 24576 * 1024 * 1024  # 24GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        result = _detect_cuda_gpu()
        self.assertIsNotNone(result)
        self.assertEqual(result["platform"], "cuda")
        self.assertEqual(result["device_count"], 1)

    @patch("consumer_app.backends.detector.HAS_PYNVML", False)
    @patch("subprocess.check_output")
    def test_cuda_detection_no_gpu(self, mock_subprocess):
        """Test returns None if nvidia-smi not found."""
        mock_subprocess.side_effect = FileNotFoundError("nvidia-smi not found")

        result = _detect_cuda_gpu()
        self.assertIsNone(result)


class TestTorchCUDADetection(unittest.TestCase):
    """Test PyTorch CUDA detection fallback."""

    @patch("consumer_app.backends.detector.torch")
    def test_torch_cuda_available(self, mock_torch):
        """Test torch.cuda.is_available() returns True."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A10"

        result = _detect_torch_cuda()
        self.assertTrue(result)

    @patch("consumer_app.backends.detector.torch")
    def test_torch_cuda_unavailable(self, mock_torch):
        """Test torch.cuda.is_available() returns False."""
        mock_torch.cuda.is_available.return_value = False

        result = _detect_torch_cuda()
        self.assertFalse(result)


class TestDetectHardware(unittest.TestCase):
    """Test the main hardware detection function."""

    def setUp(self):
        """Clear cache before each test."""
        clear_backend_cache()

    def tearDown(self):
        """Clear cache after each test."""
        clear_backend_cache()

    @patch("consumer_app.backends.detector._detect_jetson")
    @patch("consumer_app.backends.detector._detect_cuda_gpu")
    @patch("consumer_app.backends.detector._detect_torch_cuda")
    def test_detect_jetson_priority(self, mock_torch, mock_cuda, mock_jetson):
        """Test Jetson detection has priority."""
        mock_jetson.return_value = {"platform": "jetson", "model": "Orin"}
        mock_cuda.return_value = None
        mock_torch.return_value = False

        backend = detect_hardware()
        self.assertEqual(backend, HWBackend.JETSON)
        mock_jetson.assert_called_once()

    @patch("consumer_app.backends.detector._detect_jetson")
    @patch("consumer_app.backends.detector._detect_cuda_gpu")
    @patch("consumer_app.backends.detector._detect_torch_cuda")
    def test_detect_cuda_gpu_second(self, mock_torch, mock_cuda, mock_jetson):
        """Test CUDA GPU detection when Jetson unavailable."""
        mock_jetson.return_value = None
        mock_cuda.return_value = {"platform": "cuda", "model": "RTX 3090"}
        mock_torch.return_value = False

        backend = detect_hardware()
        self.assertEqual(backend, HWBackend.CUDA)

    @patch("consumer_app.backends.detector._detect_jetson")
    @patch("consumer_app.backends.detector._detect_cuda_gpu")
    @patch("consumer_app.backends.detector._detect_torch_cuda")
    def test_detect_torch_cuda_fallback(self, mock_torch, mock_cuda, mock_jetson):
        """Test PyTorch CUDA as fallback."""
        mock_jetson.return_value = None
        mock_cuda.return_value = None
        mock_torch.return_value = True

        backend = detect_hardware()
        self.assertEqual(backend, HWBackend.CUDA)

    @patch("consumer_app.backends.detector._detect_jetson")
    @patch("consumer_app.backends.detector._detect_cuda_gpu")
    @patch("consumer_app.backends.detector._detect_torch_cuda")
    def test_detect_cpu_fallback(self, mock_torch, mock_cuda, mock_jetson):
        """Test CPU fallback when no GPU found."""
        mock_jetson.return_value = None
        mock_cuda.return_value = None
        mock_torch.return_value = False

        backend = detect_hardware()
        self.assertEqual(backend, HWBackend.CPU)

    @patch("consumer_app.backends.detector._detect_jetson")
    @patch("consumer_app.backends.detector._detect_cuda_gpu")
    @patch("consumer_app.backends.detector._detect_torch_cuda")
    def test_detection_cached(self, mock_torch, mock_cuda, mock_jetson):
        """Test that detection result is cached."""
        mock_jetson.return_value = None
        mock_cuda.return_value = {"platform": "cuda", "model": "RTX"}
        mock_torch.return_value = False

        # First call
        backend1 = detect_hardware()
        # Second call (should use cache)
        backend2 = detect_hardware()

        self.assertEqual(backend1, backend2)
        # _detect_cuda_gpu should only be called once (cached on first call)
        mock_cuda.assert_called_once()

    @patch("consumer_app.backends.detector._detect_jetson")
    @patch("consumer_app.backends.detector._detect_cuda_gpu")
    @patch("consumer_app.backends.detector._detect_torch_cuda")
    def test_get_backend_alias(self, mock_torch, mock_cuda, mock_jetson):
        """Test get_backend() is an alias for detect_hardware()."""
        mock_jetson.return_value = None
        mock_cuda.return_value = {"platform": "cuda", "model": "RTX"}
        mock_torch.return_value = False

        backend = get_backend()
        self.assertEqual(backend, HWBackend.CUDA)


if __name__ == "__main__":
    unittest.main()