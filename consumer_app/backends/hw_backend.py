"""Hardware backend enumeration for cross-platform support."""

from enum import Enum


class HWBackend(str, Enum):
    """Detected hardware backend for inference and decoding.
    
    Values:
        JETSON: NVIDIA Jetson platform (Orin, Xavier, etc.)
        CUDA: NVIDIA CUDA-capable GPU (desktop, server)
        CPU: CPU-only (no GPU acceleration)
    """

    JETSON = "jetson"
    CUDA = "cuda"
    CPU = "cpu"

    def __str__(self) -> str:
        """Return uppercase string representation."""
        return self.value.upper()

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"HWBackend.{self.name}"

    @property
    def has_gpu(self) -> bool:
        """True if backend uses GPU acceleration."""
        return self in (HWBackend.JETSON, HWBackend.CUDA)

    @property
    def is_jetson(self) -> bool:
        """True if backend is Jetson."""
        return self == HWBackend.JETSON

    @property
    def is_cuda(self) -> bool:
        """True if backend is CUDA."""
        return self == HWBackend.CUDA

    @property
    def is_cpu(self) -> bool:
        """True if backend is CPU."""
        return self == HWBackend.CPU