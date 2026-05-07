# Build for Jetson Orin hardware (SM 8.7, CUDA 12.x, JetPack 6 series).
# The ultralytics/ultralytics:*-jetson-jetpack6 images bundle a PyTorch that
# causes cudaErrorInvalidKernelImage on JetPack 6.0 (L4T R36.3.0).
# Using NVIDIA's verified L4T PyTorch build.
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /app

# ultralytics and redis — torch/cuda/tensorrt already provided by the base image.
# numpy is pinned <2 in the same resolver call: the l4t-ml base OpenCV .so is
# compiled for NumPy 1.x; upgrading to 2.x breaks ABI and cv2 import fails.
# Install GStreamer plugins and EGL libraries required for Jetson hardware encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgles2 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "numpy>=1.23.0,<2" ultralytics redis onnxslim lap

COPY consumer.py .

# models/ is mounted at runtime via docker-compose volume (./consumer_app/models:/app/models)
# This keeps the image lean and allows the exported .engine to persist across restarts.

# Ensure output directory exists so VideoWriter never fails on a missing path
RUN mkdir -p /app/outputs

# Allow up to 5 minutes for first-run TRT engine export before health checks begin
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python3 -c "import redis; redis.Redis(host='127.0.0.1').ping()" || exit 1

CMD ["python3", "consumer.py"]