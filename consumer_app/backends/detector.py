"""Hardware detection and backend selector.

Probes the system at startup to determine available hardware accelerators.
Falls back gracefully through Jetson → CUDA → CPU.
"""

import logging
import subprocess
import sys
from functools import lru_cache
from typing import Optional

try:
    from jetson_stats import jetson_stats
    HAS_JETSON_STATS = True
except ImportError:
    HAS_JETSON_STATS = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from .hw_backend import HWBackend

logger = logging.getLogger(__name__)


def _detect_jetson() -> Optional[dict]:
    """Detect Jetson platform using jetson_stats.
    
    Returns:
        dict with Jetson info (model, cuda_arch, etc.) if Jetson detected, else None
    """
    if not HAS_JETSON_STATS:
        return None

    try:
        js = jetson_stats()
        if js.ok:
            logger.info(f"Detected Jetson: {js.board['name']} (L4T {js.board['l4t']})")
            return {
                "platform": "jetson",
                "model": js.board.get("name", "Unknown"),
                "l4t_version": js.board.get("l4t", "Unknown"),
                "memory_gb": js.ram.get("total", 0) / (1024 ** 3),
            }
    except Exception as e:
        logger.debug(f"Jetson detection failed: {e}")

    return None


def _detect_cuda_gpu() -> Optional[dict]:
    """Detect NVIDIA CUDA GPU using nvidia-smi or pynvml.
    
    Returns:
        dict with GPU info if CUDA GPU detected, else None
    """
    # Try pynvml first (more reliable)
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()

                logger.info(
                    f"Detected CUDA GPU: {device_name} ({device_count} device(s))"
                )
                return {
                    "platform": "cuda",
                    "model": device_name,
                    "device_count": device_count,
                    "memory_gb": memory.total / (1024 ** 3),
                }
        except Exception as e:
            logger.debug(f"pynvml detection failed: {e}")

    # Fallback: nvidia-smi
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if output.strip():
            lines = output.strip().split("\n")
            device_name = lines[0].split(",")[0].strip()
            logger.info(f"Detected CUDA GPU: {device_name} ({len(lines)} device(s))")
            return {
                "platform": "cuda",
                "model": device_name,
                "device_count": len(lines),
            }
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.debug(f"nvidia-smi detection failed: {e}")

    return None


def _detect_torch_cuda() -> bool:
    """Fallback: check if torch is available with CUDA support.
    
    Returns:
        True if torch.cuda is available and has devices
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"Detected CUDA via PyTorch ({torch.cuda.get_device_name(0)})")
            return True
    except ImportError:
        pass

    return False


@lru_cache(maxsize=1)
def detect_hardware() -> HWBackend:
    """Detect available hardware and return appropriate backend.
    
    Probes in order:
    1. Jetson (via jetson_stats)
    2. CUDA GPU (via nvidia-smi or pynvml)
    3. Fallback to CPU
    
    Result is cached after first call.
    
    Returns:
        HWBackend enum (JETSON, CUDA, or CPU)
    """
    logger.info("Starting hardware detection...")

    # Step 1: Check for Jetson
    jetson_info = _detect_jetson()
    if jetson_info:
        logger.info(f"Hardware backend: JETSON ({jetson_info['model']})")
        return HWBackend.JETSON

    # Step 2: Check for CUDA GPU
    cuda_info = _detect_cuda_gpu()
    if cuda_info:
        logger.info(f"Hardware backend: CUDA ({cuda_info['model']})")
        return HWBackend.CUDA

    # Step 3: Fallback to torch.cuda check (in case nvidia-smi is unavailable)
    if _detect_torch_cuda():
        logger.info("Hardware backend: CUDA (via PyTorch)")
        return HWBackend.CUDA

    # Step 4: CPU fallback
    logger.info("No GPU detected. Hardware backend: CPU")
    return HWBackend.CPU


def get_backend() -> HWBackend:
    """Alias for detect_hardware() for consistent API.
    
    Returns:
        HWBackend enum (cached result)
    """
    return detect_hardware()


def clear_backend_cache() -> None:
    """Clear the cached hardware detection result.
    
    Useful for testing or if hardware changes at runtime (unlikely).
    """
    detect_hardware.cache_clear()
    logger.info("Hardware backend cache cleared")


if __name__ == "__main__":
    # Quick test: python -m consumer_app.backends.detector
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    backend = detect_hardware()
    print(f"\nDetected Backend: {backend}")
    print(f"  Backend value: {backend.value}")
    print(f"  Has GPU: {backend.has_gpu}")
    print(f"  Is Jetson: {backend.is_jetson}")
    print(f"  Is CUDA: {backend.is_cuda}")
    print(f"  Is CPU: {backend.is_cpu}")