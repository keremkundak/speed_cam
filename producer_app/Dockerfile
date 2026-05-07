FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi \
    python3-gst-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir redis

COPY producer.py .

# Videos are mounted at runtime via docker-compose volume (not baked in)
CMD ["python", "producer.py"]