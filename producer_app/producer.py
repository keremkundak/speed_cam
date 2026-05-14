import redis
import time
import logging
import os
import signal
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_PATH = os.environ.get("VIDEO_PATH", "/app/videos/traffic_long_720p.mp4")
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
STREAM_KEY = os.environ.get("STREAM_KEY", "video:frames")
MAX_LEN = int(os.environ.get("MAX_LEN", "500"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
REDIS_STARTUP_DELAY = float(os.environ.get("REDIS_STARTUP_DELAY", "5"))

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — stopping producer.")
    _shutdown = True


def produce_frames() -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r.delete(STREAM_KEY)

    Gst.init(None)

    # Hardware-accelerated pipeline:
    # 1. filesrc + qtdemux + h264parse: Extract H.264 stream
    # 2. nvv4l2decoder: Hardware video decoding (NVDEC)
    # 3. nvvidconv: Hardware color/format conversion (VIC) to I420
    # 4. nvjpegenc: Hardware JPEG encoding (NVJPG)
    # 5. appsink: Deliver compressed JPEG bytes to Python
    pipeline_str = (
        f"filesrc location={VIDEO_PATH} ! "
        f"qtdemux ! h264parse ! "
        f"nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! "
        f"nvjpegenc quality={JPEG_QUALITY} ! "
        f"appsink name=sink emit-signals=True max-buffers=2 drop=True sync=True"
    )
    
    logger.info("Starting GStreamer hardware pipeline: %s", pipeline_str)
    
    loop_count = 0
    frame_idx = 0

    while not _shutdown:
        loop_count += 1
        logger.info("Starting video loop #%d", loop_count)
        
        pipeline = Gst.parse_launch(pipeline_str)
        appsink = pipeline.get_by_name("sink")
        
        # Start pipeline
        pipeline.set_state(Gst.State.PLAYING)

        try:
            while not _shutdown:
                # pull-sample blocks until a frame is ready (sync=True paces it to video FPS)
                sample = appsink.emit("pull-sample")
                if not sample:
                    # End of Stream (EOS) reached
                    break

                buf = sample.get_buffer()
                success, map_info = buf.map(Gst.MapFlags.READ)
                
                if success:
                    jpeg_bytes = map_info.data
                    frame_idx += 1
                    timestamp = time.time()

                    r.xadd(
                        STREAM_KEY,
                        {
                            "frame_id": frame_idx,
                            "timestamp": str(timestamp),
                            "data": jpeg_bytes,
                        },
                        maxlen=MAX_LEN,
                        approximate=True,
                    )
                    
                    if frame_idx % 500 == 0:
                        logger.info("Frame %d pushed to Redis (loop #%d).", frame_idx, loop_count)
                        
                    buf.unmap(map_info)

        except Exception:
            logger.exception("Error in producer loop #%d", loop_count)
        finally:
            pipeline.set_state(Gst.State.NULL)

    logger.info("Producer stopped after %d total frames.", frame_idx)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logger.info("Waiting %.1f s for Redis to be ready...", REDIS_STARTUP_DELAY)
    time.sleep(REDIS_STARTUP_DELAY)
    produce_frames()