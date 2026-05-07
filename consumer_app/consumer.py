import csv
import glob
import logging
import os
import signal
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import redis
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """All runtime configuration loaded from environment variables."""

    MODEL_PATH: str = os.environ.get("MODEL_PATH", "/app/models/yolo26n.engine")
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "127.0.0.1")
    REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
    STREAM_KEY: str = os.environ.get("STREAM_KEY", "video:frames")

    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "/app/outputs")
    FRAME_WIDTH: int = int(os.environ.get("FRAME_WIDTH", "1280"))
    FRAME_HEIGHT: int = int(os.environ.get("FRAME_HEIGHT", "720"))

    # Speed zone: "x1,y1;x2,y2;x3,y3;x4,y4"  (bottom-left, top-left, top-right, bottom-right)
    SPEED_ZONE_STR: str = os.environ.get(
        "SPEED_ZONE", "435,420;590,202;938,225;1055,464"
    )

    # Real-world zone dimensions in metres
    ZONE_LENGTH_M: float = float(os.environ.get("ZONE_LENGTH_M", "4.2"))
    ZONE_WIDTH_M: float = float(os.environ.get("ZONE_WIDTH_M", "3.5"))

    # Rolling output
    SEGMENT_DURATION_S: int = int(os.environ.get("SEGMENT_DURATION_S", "300"))
    MAX_SEGMENTS: int = int(os.environ.get("MAX_SEGMENTS", "3"))
    CSV_WINDOW_S: int = int(os.environ.get("CSV_WINDOW_S", "3600"))
    # Video codec fourcc: avc1 (H.264, ~5-10x smaller than mp4v), mp4v as fallback
    VIDEO_CODEC: str = os.environ.get("VIDEO_CODEC", "avc1")

    # Speed estimation
    SPEED_SMOOTH_WINDOW: int = int(os.environ.get("SPEED_SMOOTH_WINDOW", "5"))
    MIN_ZONE_TIME_S: float = float(os.environ.get("MIN_ZONE_TIME_S", "0.3"))

    # Detection
    TRACKER_CONFIG: str = os.environ.get("TRACKER_CONFIG", "bytetrack.yaml")
    DETECT_CLASSES: list = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    INFERENCE_SIZE: int = int(os.environ.get("INFERENCE_SIZE", "640"))
    DEVICE: str = os.environ.get("DEVICE", "cuda:0")

    @classmethod
    def parse_speed_zone(cls) -> np.ndarray:
        points = [
            [int(v) for v in pair.split(",")]
            for pair in cls.SPEED_ZONE_STR.strip().split(";")
        ]
        return np.array(points, dtype=np.int32)


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Loads YOLO model, auto-exporting to TensorRT engine when needed."""

    def __init__(self, model_path: str, device: str, imgsz: int):
        self._model_path = model_path
        self._device = device
        self._imgsz = imgsz
        self._log = logging.getLogger(self.__class__.__name__)

    def _engine_path_for(self, src: str) -> str:
        return str(Path(src).with_suffix(".engine"))

    def _export(self, src: str) -> str:
        engine_path = self._engine_path_for(src)
        self._log.info("Exporting %s → TensorRT engine (this may take several minutes)...", src)
        m = YOLO(src, task="detect")
        m.export(format="engine", device=self._device, imgsz=self._imgsz, half=True)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"Engine export finished but {engine_path} was not found. "
                "Check Ultralytics export output path."
            )
        self._log.info("Engine saved: %s", engine_path)
        return engine_path

    def load(self) -> YOLO:
        mp = self._model_path
        engine_path = mp if mp.endswith(".engine") else self._engine_path_for(mp)

        # 1. Try existing engine
        if os.path.exists(engine_path):
            try:
                self._log.info("Loading TensorRT engine: %s", engine_path)
                model = YOLO(engine_path, task="detect")
                self._log.info("Engine loaded successfully.")
                return model
            except Exception as exc:
                self._log.warning("Engine load failed (%s) — will re-export.", exc)

        # 2. Find source model (.pt preferred, then .onnx) and export
        base = str(Path(engine_path).with_suffix(""))
        for ext in (".pt", ".onnx"):
            candidate = base + ext
            if os.path.exists(candidate):
                exported = self._export(candidate)
                return YOLO(exported, task="detect")

        # 3. Last resort: load whatever path was given (may be slow on CPU)
        self._log.warning(
            "No engine or source model found at expected paths. "
            "Loading directly: %s", mp
        )
        return YOLO(mp, task="detect")


# ---------------------------------------------------------------------------
# Speed Estimator
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Converts pixel positions to real-world metres via a homography built from
    the speed zone trapezoid, then computes per-vehicle speed with smoothing.

    zone_pts order: bottom-left, top-left, top-right, bottom-right (image coords).
    """

    def __init__(
        self,
        zone_pts: np.ndarray,
        zone_length_m: float,
        zone_width_m: float,
        smooth_window: int,
        min_zone_time_s: float,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self.zone_pts = zone_pts
        self.min_zone_time_s = min_zone_time_s

        src = zone_pts.astype(np.float32)
        # World rect: near-left(0,0), far-left(0,L), far-right(W,L), near-right(W,0)
        dst = np.float32([
            [0.0,          0.0],
            [0.0,          zone_length_m],
            [zone_width_m, zone_length_m],
            [zone_width_m, 0.0],
        ])
        self._H, _ = cv2.findHomography(src, dst)
        self._log.info("Homography matrix computed.")

        self._entry: dict[int, dict] = {}
        self._pos_buf: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smooth_window * 4)
        )
        self._speed_buf: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smooth_window)
        )
        self._final: dict[int, float] = {}

    def is_in_zone(self, px: tuple) -> bool:
        return cv2.pointPolygonTest(
            self.zone_pts, (float(px[0]), float(px[1])), False
        ) >= 0

    def pixel_to_world(self, px: tuple) -> tuple:
        src = np.array([[[float(px[0]), float(px[1])]]], dtype=np.float32)
        w = cv2.perspectiveTransform(src, self._H)
        return float(w[0][0][0]), float(w[0][0][1])

    def update(self, track_id: int, pixel_pt: tuple, timestamp: float, in_zone: bool) -> float | None:
        """
        Call once per frame per tracked vehicle.
        Returns smoothed speed (km/h) or None if not yet computable.
        """
        if in_zone:
            world = self.pixel_to_world(pixel_pt)
            self._pos_buf[track_id].append((world, timestamp))

            if track_id not in self._entry:
                self._entry[track_id] = {"time": timestamp, "world": world}

            speed = self._live_speed(track_id)
            if speed is not None:
                self._speed_buf[track_id].append(speed)
                return self._smooth(track_id)
            return None

        # Vehicle left zone — finalise
        if track_id in self._entry and track_id not in self._final:
            fs = self._final_speed(track_id)
            if fs is not None:
                self._final[track_id] = fs

        return self._final.get(track_id)

    def get_display_speed(self, track_id: int) -> float | None:
        if track_id in self._final:
            return self._final[track_id]
        return self._smooth(track_id) if self._speed_buf[track_id] else None

    def get_final_speed(self, track_id: int) -> float | None:
        return self._final.get(track_id)

    def get_entry_time(self, track_id: int) -> float | None:
        return self._entry.get(track_id, {}).get("time")

    def cleanup(self, active_ids: set) -> None:
        stale = set(self._entry) - active_ids
        for tid in stale:
            self._entry.pop(tid, None)
            self._pos_buf.pop(tid, None)
            self._speed_buf.pop(tid, None)
            self._final.pop(tid, None)

    # ------------------------------------------------------------------
    def _live_speed(self, track_id: int) -> float | None:
        buf = self._pos_buf[track_id]
        if len(buf) < 2:
            return None
        (wx0, wy0), t0 = buf[0]
        (wx1, wy1), t1 = buf[-1]
        dt = t1 - t0
        if dt < 0.1:
            return None
        dist = np.hypot(wx1 - wx0, wy1 - wy0)
        return (dist / dt) * 3.6

    def _final_speed(self, track_id: int) -> float | None:
        buf = self._pos_buf[track_id]
        entry = self._entry.get(track_id)
        if not buf or not entry:
            return None
        (wxL, wyL), tL = buf[-1]
        dt = tL - entry["time"]
        if dt < self.min_zone_time_s:
            return None
        wx0, wy0 = entry["world"]
        dist = np.hypot(wxL - wx0, wyL - wy0)
        speed = (dist / dt) * 3.6
        if not (1.0 < speed < 250.0):
            return None
        return speed

    def _smooth(self, track_id: int) -> float | None:
        buf = self._speed_buf[track_id]
        return float(np.mean(list(buf))) if buf else None


# ---------------------------------------------------------------------------
# Rolling Video Writer
# ---------------------------------------------------------------------------

class RollingVideoWriter:
    """
    Writes annotated video in fixed-duration segments.
    Deletes oldest segments when max_segments is exceeded.
    """

    def __init__(
        self,
        output_dir: str,
        fps: float,
        frame_size: tuple,
        segment_duration_s: int,
        max_segments: int,
        codec: str = "avc1",
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._size = frame_size
        self._frames_per_seg = max(1, int(fps * segment_duration_s))
        self._max_segments = max_segments
        self._codec = codec
        self._fourcc = cv2.VideoWriter_fourcc(*codec) if codec != "hw" else 0
        self._log.info("Video codec: %s", codec)

        self._writer: cv2.VideoWriter | None = None
        self._seg_idx = 0
        self._frame_count = 0
        self._start_segment()

    def write(self, frame: np.ndarray) -> None:
        if self._frame_count >= self._frames_per_seg:
            self._start_segment()
        self._writer.write(frame)  # type: ignore[union-attr]
        self._frame_count += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._log.info("Video writer released.")

    def _start_segment(self) -> None:
        if self._writer is not None:
            self._writer.release()

        self._seg_idx += 1
        self._frame_count = 0
        path = self._dir / f"segment_{self._seg_idx:04d}.mp4"
        
        if self._codec.lower() == "hw":
            pipeline = (
                f"appsrc ! video/x-raw, format=BGR ! videoconvert ! video/x-raw, format=BGRx ! "
                f"nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! "
                f"nvv4l2h264enc bitrate=8000000 ! h264parse ! qtmux ! filesink location={path}"
            )
            self._writer = cv2.VideoWriter(
                pipeline, cv2.CAP_GSTREAMER, 0, self._fps, self._size
            )
        else:
            self._writer = cv2.VideoWriter(str(path), self._fourcc, self._fps, self._size)
            
        self._log.info("New video segment: %s", path)
        self._evict_old_segments()

    def _evict_old_segments(self) -> None:
        segments = sorted(self._dir.glob("segment_*.mp4"))
        while len(segments) > self._max_segments:
            oldest = segments.pop(0)
            try:
                oldest.unlink()
                self._log.info("Deleted old segment: %s", oldest)
            except OSError as exc:
                self._log.warning("Could not delete %s: %s", oldest, exc)


# ---------------------------------------------------------------------------
# Rolling CSV Writer
# ---------------------------------------------------------------------------

class RollingCSVWriter:
    """
    Appends speed detection events to a CSV file.
    Periodically purges rows older than csv_window_s to cap storage.

    Columns: timestamp, track_id, speed_kmh, entry_time, exit_time

    Timestamps are written as 'YYYY-MM-DD HH:MM:SS' (local time) for readability.
    Internally, _row_ts parses them back to Unix floats for the rolling purge.
    """

    _COLUMNS = ("timestamp", "track_id", "speed_kmh", "entry_time", "exit_time")
    _PURGE_INTERVAL = 200  # purge check every N writes

    def __init__(self, output_dir: str, csv_window_s: int):
        self._log = logging.getLogger(self.__class__.__name__)
        self._path = Path(output_dir) / "detections.csv"
        self._window = csv_window_s
        self._write_count = 0
        self._ensure_header()

    def append(
        self,
        track_id: int,
        speed_kmh: float,
        entry_time: float,
        exit_time: float,
    ) -> None:
        now = time.time()
        with open(self._path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                self._fmt_ts(now),
                track_id,
                f"{speed_kmh:.1f}",
                self._fmt_ts(entry_time),
                self._fmt_ts(exit_time),
            ])
        self._write_count += 1
        self._log.info(
            "CSV — track %d  %.1f km/h", track_id, speed_kmh
        )
        if self._write_count % self._PURGE_INTERVAL == 0:
            self._purge()

    def _ensure_header(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="") as fh:
                csv.writer(fh).writerow(self._COLUMNS)

    def _purge(self) -> None:
        """Remove rows older than csv_window_s."""
        cutoff = time.time() - self._window
        try:
            with open(self._path, newline="") as fh:
                rows = list(csv.reader(fh))
            if not rows:
                return
            header = rows[0]
            fresh = [r for r in rows[1:] if self._row_ts(r) >= cutoff]
            with open(self._path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(fresh)
            self._log.info("CSV purge: kept %d / %d rows.", len(fresh), len(rows) - 1)
        except Exception:
            self._log.exception("CSV purge failed.")

    @staticmethod
    def _fmt_ts(unix_ts: float) -> str:
        """Format a Unix timestamp as a human-readable local datetime string."""
        from datetime import datetime
        return datetime.fromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _row_ts(row: list) -> float:
        """Parse a timestamp cell back to Unix float for purge cutoff comparison."""
        from datetime import datetime
        try:
            return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timestamp()
        except (IndexError, ValueError):
            try:
                return float(row[0])   # fallback: handle legacy raw Unix rows
            except ValueError:
                return 0.0


# ---------------------------------------------------------------------------
# Frame Processor
# ---------------------------------------------------------------------------

class FrameProcessor:
    """
    Runs YOLO detection + ByteTrack on a frame, updates SpeedEstimator,
    draws annotations, and triggers CSV writes on zone exits.
    """

    _COLORS = {
        "in_zone":  (0, 255, 0),
        "out_zone": (0, 0, 255),
        "zone_poly": (0, 255, 255),
        "speed_text": (255, 255, 255),
    }

    def __init__(
        self,
        model: YOLO,
        estimator: SpeedEstimator,
        csv_writer: RollingCSVWriter,
        config: type,
    ):
        self._model = model
        self._estimator = estimator
        self._csv = csv_writer
        self._cfg = config
        self._prev_in_zone: set[int] = set()
        self._log = logging.getLogger(self.__class__.__name__)

    def process(self, img: np.ndarray, msg_timestamp: float) -> np.ndarray:
        results = self._model.track(
            img,
            classes=self._cfg.DETECT_CLASSES,
            persist=True,
            tracker=self._cfg.TRACKER_CONFIG,
            verbose=False,
            imgsz=self._cfg.INFERENCE_SIZE,
            device=self._cfg.DEVICE,
        )

        current_in_zone: set[int] = set()
        active_ids: set[int] = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                foot = ((x1 + x2) / 2.0, y2)
                active_ids.add(tid)

                in_zone = self._estimator.is_in_zone(foot)
                speed = self._estimator.update(tid, foot, msg_timestamp, in_zone)

                if in_zone:
                    current_in_zone.add(tid)

                # Trigger CSV on zone exit
                just_left = self._prev_in_zone - current_in_zone
                if tid in just_left:
                    self._on_zone_exit(tid, msg_timestamp)

                self._draw_vehicle(img, box, tid, in_zone, speed)

        # Handle any remaining exits (tracked in prev but not in current detections)
        for tid in self._prev_in_zone - current_in_zone - active_ids:
            self._on_zone_exit(tid, msg_timestamp)

        self._prev_in_zone = current_in_zone
        self._estimator.cleanup(active_ids)

        cv2.polylines(img, [self._estimator.zone_pts], True, self._COLORS["zone_poly"], 2)
        return img

    def _on_zone_exit(self, track_id: int, exit_time: float) -> None:
        final = self._estimator.get_final_speed(track_id)
        entry_t = self._estimator.get_entry_time(track_id)
        if final is not None and entry_t is not None:
            self._csv.append(track_id, final, entry_t, exit_time)

    def _draw_vehicle(
        self,
        img: np.ndarray,
        box: np.ndarray,
        tid: int,
        in_zone: bool,
        speed: float | None,
    ) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        color = self._COLORS["in_zone"] if in_zone else self._COLORS["out_zone"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        display_speed = self._estimator.get_display_speed(tid)
        label = f"ID:{tid}"
        if display_speed is not None:
            label += f"  {display_speed:.1f} km/h"

        cv2.putText(
            img, label,
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            self._COLORS["speed_text"], 2,
        )


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, frame) -> None:
    global _shutdown
    logger.info("Shutdown signal received — flushing outputs and stopping.")
    _shutdown = True


def consume_frames() -> None:
    cfg = Config

    # Model
    loader = ModelLoader(cfg.MODEL_PATH, cfg.DEVICE, cfg.INFERENCE_SIZE)
    model = loader.load()

    # Speed estimation
    zone_pts = cfg.parse_speed_zone()
    estimator = SpeedEstimator(
        zone_pts,
        cfg.ZONE_LENGTH_M,
        cfg.ZONE_WIDTH_M,
        cfg.SPEED_SMOOTH_WINDOW,
        cfg.MIN_ZONE_TIME_S,
    )

    # Outputs
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    video_writer = RollingVideoWriter(
        cfg.OUTPUT_DIR,
        fps=30.0,           # Display FPS — independent of inference rate
        frame_size=(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
        segment_duration_s=cfg.SEGMENT_DURATION_S,
        max_segments=cfg.MAX_SEGMENTS,
        codec=cfg.VIDEO_CODEC,
    )
    csv_writer = RollingCSVWriter(cfg.OUTPUT_DIR, cfg.CSV_WINDOW_S)

    processor = FrameProcessor(model, estimator, csv_writer, cfg)

    # Redis
    r = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, decode_responses=False)
    last_id = "0-0"
    logger.info("Consumer ready — waiting for frames on stream '%s'.", cfg.STREAM_KEY)

    try:
        while not _shutdown:
            response = r.xread({cfg.STREAM_KEY: last_id}, count=1, block=5000)
            if not response:
                logger.info("No frames received for 5 s — stream may have ended.")
                continue

            for _stream, messages in response:
                for msg_id, data in messages:
                    last_id = msg_id

                    msg_ts = float(data.get(b"timestamp", b"0").decode())
                    img = cv2.imdecode(
                        np.frombuffer(data[b"data"], dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if img is None:
                        logger.warning("Failed to decode frame %s — skipping.", msg_id)
                        continue

                    annotated = processor.process(img, msg_ts)
                    video_writer.write(annotated)

    finally:
        video_writer.release()
        logger.info("Consumer stopped.")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    consume_frames()