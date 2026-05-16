# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### v1.0 Development (Branch: `v1.0`)

#### Phase 1: Foundation & Hardware Abstraction

**Checkpoint A Status:** In Progress (2 of 4 complete)

- [x] v1-0001: Repository restructure and directory layout
  - Renamed producer_app logical home → mediamtx/
  - Created consumer_app/backends/ for hardware abstraction
  - Created api/ for FastAPI REST service
  - Created dashboard/ for web UI
  - Updated .gitignore and added CHANGELOG.md

- [x] v1-0002: Hardware detection and backend selector
  - `consumer_app/backends/detector.py`: detects Jetson (jtop), CUDA (nvidia-smi/pynvml), CPU fallback
  - `consumer_app/backends/hw_backend.py`: HWBackend enum with helper properties
  - Result cached via @lru_cache for single detection at startup
  - Graceful fallback chain with proper logging
  - Comprehensive unit tests (13 test cases)

- [x] v1-0003: Decode backends (Jetson NVDEC, CUDA, CPU)
  - GStreamer NVDEC pipeline for Jetson (preserve alpha code)
  - OpenCV CUDA decode for NVIDIA GPUs
  - ffmpeg-python fallback for CPU-only systems
  - Common FrameSource ABC interface

- [ ] v1-0004: Inference backends (TensorRT, PyTorch, ONNX Runtime)
  - TensorRT backend (existing alpha code refactored)
  - PyTorch backend with device='cuda' or 'cpu'
  - ONNX Runtime backend with GPU/CPU execution providers
  - Factory function to select backend at runtime

**Checkpoint A Planned After:** v1-0004
  - All hardware backends functional
  - Tested on Jetson, CUDA desktop, CPU-only laptop
  - PR to main for review

#### Phase 2: MediaMTX RTSP Pipeline
- [ ] v1-0005: MediaMTX service in docker-compose
- [ ] v1-0006: Video-to-RTSP looping with ffmpeg
- [ ] v1-0007: Consumer refactored to read RTSP frames
- [ ] v1-0008: Multi-stream support with per-stream config

**Checkpoint B Planned After:** v1-0008

#### Phase 3: REST API
- [ ] v1-0009: FastAPI scaffold with health and versioning
- [ ] v1-0010: GET /api/v1/detections with history
- [ ] v1-0011: GET /api/v1/detections/stream with SSE
- [ ] v1-0012: Stream and speed zone management endpoints
- [ ] v1-0013: POST /api/v1/calibrate homography helper

**Checkpoint C Planned After:** v1-0013

#### Phase 4: Dashboard
- [ ] v1-0014: Dashboard scaffold with nginx and dark mode
- [ ] v1-0015: Live events panel with SSE
- [ ] v1-0016: History panel with date-range picker and charts
- [ ] v1-0017: Stream previews and interactive zone editor

**Checkpoint D Planned After:** v1-0017

#### Phase 5: Hardening & Release
- [ ] v1-0018: Docker Compose profiles (jetson, gpu, cpu)
- [ ] v1-0019: Structured JSON logging and Prometheus metrics
- [ ] v1-0020: Documentation updates and architecture diagram
- [ ] v1-0021: Final v1.0.0 release tag and PR to main

**Checkpoint E Planned After:** v1-0021

---

## [0.9.0] - Alpha (Current on `main`)

### Added
- Hardware-accelerated YOLO26n tracking via TensorRT on NVIDIA Jetson Orin
- GStreamer Python pipeline for H.264 NVDEC decoding and NVJPG encoding
- Homography-based vehicle speed estimation in real-world meters
- Redis stream for decoupled frame buffering between producer and consumer
- Rolling video segment recording using NVENC hardware encoder
- CSV event logging with entry/exit speeds

### Features
- Ultra-low latency edge inference (sub-50ms per frame on Jetson Orin)
- Customizable speed zone via polygon coordinates
- Real-world zone calibration (width and length in meters)

### Requirements
- NVIDIA Jetson Orin (Nano, NX, or AGX)
- JetPack 6.x and Docker Compose

---

## Notes

- **v1.0 branch strategy:** All development happens on branch `v1.0`. At each checkpoint (A–E), a PR is opened to `main` for review and merge.
- **Hardware abstraction:** v1.0 introduces support for Jetson, CUDA GPU, and CPU-only inference. Backends are selected at runtime based on detected hardware.
- **MediaMTX transition:** The Python GStreamer producer (producer_app/) is replaced by MediaMTX server in Phase 2, simplifying RTSP stream management and adding HLS/WebRTC support.
- **ONNX model export:** For CPU and CUDA-only deployments, users can export the YOLOv8 model to ONNX format: `yolo export format=onnx device=cpu`. The ONNX backend will auto-detect and use the fastest available provider.