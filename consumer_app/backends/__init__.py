"""Backend abstraction for hardware-agnostic inference and frame decoding.
 
Supports:
- Jetson: GStreamer NVDEC + TensorRT
- CUDA GPU: OpenCV CUDA + PyTorch/ONNX
- CPU: ffmpeg-python + PyTorch/ONNX
"""
