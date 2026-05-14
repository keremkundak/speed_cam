"""Backend abstraction for hardware-agnostic inference and frame decoding.
 
Supports:
- Jetson: GStreamer NVDEC + TensorRT
- CUDA GPU: OpenCV CUDA + PyTorch/ONNX
- CPU: ffmpeg-python + PyTorch/ONNX
"""

from .hw_backend import HWBackend
from .detector import (
    detect_hardware,
    get_backend,
    clear_backend_cache,
)

__all__ = [
    "HWBackend",
    "detect_hardware",
    "get_backend",
    "clear_backend_cache",
]
