"""
IgnisEdge - Edge Hybrid Node (Production Flight Orchestrator) — HARDENED

Robust version with full exception handling for in-flight conditions:
  - P3 USB disconnection / timeout recovery with exponential backoff
  - Frame watchdog (detects stalled camera)
  - Consecutive-error circuit breaker (fails safe after N errors)
  - Heartbeat telemetry so ground station knows the node is alive
  - Graceful degradation: if YOLO fails, physics pipeline continues solo

Pipelines:
  Physics:  P3 radiometric -> hysteresis -> morphological classifier -> Tracker
  Learning: P3 radiometric -> preprocess_radiometric -> YOLOv8s-seg (2 classes)

Arbiter rules (containment on normalized coords):
  Tracker FIRE + YOLO wildfire        -> RED
  Tracker FIRE + YOLO false_positive  -> WHITE (conflict)
  Tracker FIRE + no YOLO              -> ORANGE (physics-only)
  No tracker   + YOLO wildfire        -> YELLOW (ML-only)
  Otherwise                            -> NONE

Design decisions for robustness:
  - Each failure mode has a DEFINED RESPONSE (reconnect, skip, abort)
  - The flight loop NEVER exits on a transient error, only on:
      (a) user SIGINT/SIGTERM
      (b) N_CONSECUTIVE_FATAL_ERRORS exceeded (circuit breaker)
      (c) max_frames reached (debug mode only)
  - Camera lifecycle ownership is explicit: ONE place opens, ONE place closes.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# usb.core for explicit exception typing (soft import — tolerate missing on dev)
try:
    import usb.core  # type: ignore
    USB_ERRORS: tuple[type[Exception], ...] = (usb.core.USBError, usb.core.USBTimeoutError)
except ImportError:
    USB_ERRORS = ()

# Project modules
from p3_camera import P3Camera, raw_to_celsius, GainMode  # type: ignore
from preprocess import preprocess_radiometric  # type: ignore
from morphological_classifier_v3 import (  # type: ignore
    ThermalTrackingPipeline,
    TrackedDetection,
    ThreatClass,
    AlertLevel,
    reset_default_pipeline,
)

logger = logging.getLogger("ignisedge.edge")


# ============================================================================
# YOLO class mapping (MUST match training)
# ============================================================================

YOLO_CLASS_WILDFIRE: int = 0
YOLO_CLASS_FALSE_POSITIVE: int = 1


# ============================================================================
# Robustness constants
# ============================================================================

MAX_CAMERA_RECONNECT_ATTEMPTS: int = 10       # Before circuit breaker trips
CAMERA_RECONNECT_BACKOFF_MIN_S: float = 1.0   # Initial backoff
CAMERA_RECONNECT_BACKOFF_MAX_S: float = 30.0  # Cap on exponential growth
FRAME_WATCHDOG_TIMEOUT_S: float = 5.0         # If no frame in this window -> reset
MAX_CONSECUTIVE_FRAME_ERRORS: int = 50        # Circuit breaker: abort after this
YOLO_ERROR_TOLERANCE: int = 10                # After N YOLO failures -> disable YOLO
HEARTBEAT_EVERY_N_FRAMES: int = 250           # ~10s at 25Hz


# ============================================================================
# Enums / DTOs
# ============================================================================

class FinalAlert(Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    WHITE = "white"
    NONE = "none"


class NodeHealth(Enum):
    OK = "ok"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    CRITICAL = "critical"


@dataclass
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_norm: tuple[float, float, float, float]

    @property
    def centroid_norm(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_norm
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def contains_point_norm(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.bbox_norm
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class ArbitrationResult:
    frame_id: int
    timestamp: float
    final_alert: FinalAlert
    tracker_confirmed: bool
    tracker_threat: Optional[ThreatClass]
    tracker_centroid_norm: Optional[tuple[float, float]]
    yolo_best_class: Optional[str]
    yolo_best_confidence: Optional[float]
    reasoning: str
    n_trackers_confirmed: int = 0
    n_yolo_detections: int = 0
    node_health: NodeHealth = NodeHealth.OK


# ============================================================================
# Arbiter
# ============================================================================

def _tracker_says_fire(det: TrackedDetection) -> bool:
    return det.is_confirmed and det.tracker.stable_threat() == ThreatClass.WILDFIRE


def arbitrate(
    frame_id: int,
    timestamp: float,
    frame_shape: tuple[int, int],
    tracker_detections: list[TrackedDetection],
    yolo_detections: list[YoloDetection],
    node_health: NodeHealth,
) -> ArbitrationResult:
    """Combine physics + ML verdicts. See module docstring for rule table."""
    H, W = frame_shape

    fire_trackers = [d for d in tracker_detections if _tracker_says_fire(d)]
    yolo_fires = [y for y in yolo_detections if y.class_id == YOLO_CLASS_WILDFIRE]
    yolo_fps = [y for y in yolo_detections if y.class_id == YOLO_CLASS_FALSE_POSITIVE]

    # Rule 1: RED
    for det in fire_trackers:
        cx_px, cy_px = det.predicted_centroid
        cx_norm, cy_norm = cx_px / W, cy_px / H
        for yf in yolo_fires:
            if yf.contains_point_norm(cx_norm, cy_norm):
                return ArbitrationResult(
                    frame_id=frame_id, timestamp=timestamp,
                    final_alert=FinalAlert.RED,
                    tracker_confirmed=True,
                    tracker_threat=ThreatClass.WILDFIRE,
                    tracker_centroid_norm=(cx_norm, cy_norm),
                    yolo_best_class="wildfire",
                    yolo_best_confidence=yf.confidence,
                    reasoning=f"Tracker#{det.tracker.track_id} WILDFIRE at "
                              f"({cx_norm:.2f},{cy_norm:.2f}) in YOLO wildfire "
                              f"(conf={yf.confidence:.2f})",
                    n_trackers_confirmed=len(fire_trackers),
                    n_yolo_detections=len(yolo_detections),
                    node_health=node_health,
                )

    # Rule 2: WHITE (conflict)
    for det in fire_trackers:
        cx_px, cy_px = det.predicted_centroid
        cx_norm, cy_norm = cx_px / W, cy_px / H
        for yfp in yolo_fps:
            if yfp.contains_point_norm(cx_norm, cy_norm):
                return ArbitrationResult(
                    frame_id=frame_id, timestamp=timestamp,
                    final_alert=FinalAlert.WHITE,
                    tracker_confirmed=True,
                    tracker_threat=ThreatClass.WILDFIRE,
                    tracker_centroid_norm=(cx_norm, cy_norm),
                    yolo_best_class="false_positive",
                    yolo_best_confidence=yfp.confidence,
                    reasoning=f"CONFLICT: Tracker#{det.tracker.track_id} says WILDFIRE, "
                              f"YOLO says false_positive (conf={yfp.confidence:.2f})",
                    n_trackers_confirmed=len(fire_trackers),
                    n_yolo_detections=len(yolo_detections),
                    node_health=node_health,
                )

    # Rule 3: ORANGE (physics-only)
    if fire_trackers:
        det = fire_trackers[0]
        cx_px, cy_px = det.predicted_centroid
        cx_norm, cy_norm = cx_px / W, cy_px / H
        return ArbitrationResult(
            frame_id=frame_id, timestamp=timestamp,
            final_alert=FinalAlert.ORANGE,
            tracker_confirmed=True,
            tracker_threat=ThreatClass.WILDFIRE,
            tracker_centroid_norm=(cx_norm, cy_norm),
            yolo_best_class=None,
            yolo_best_confidence=None,
            reasoning=f"Tracker#{det.tracker.track_id} WILDFIRE (physics-only; "
                      f"node_health={node_health.value})",
            n_trackers_confirmed=len(fire_trackers),
            n_yolo_detections=len(yolo_detections),
            node_health=node_health,
        )

    # Rule 4: YELLOW (ML-only)
    if yolo_fires:
        best = max(yolo_fires, key=lambda y: y.confidence)
        return ArbitrationResult(
            frame_id=frame_id, timestamp=timestamp,
            final_alert=FinalAlert.YELLOW,
            tracker_confirmed=False,
            tracker_threat=None,
            tracker_centroid_norm=None,
            yolo_best_class="wildfire",
            yolo_best_confidence=best.confidence,
            reasoning=f"YOLO wildfire (conf={best.confidence:.2f}) no confirmed tracker",
            n_trackers_confirmed=0,
            n_yolo_detections=len(yolo_detections),
            node_health=node_health,
        )

    # Rule 5: NONE
    return ArbitrationResult(
        frame_id=frame_id, timestamp=timestamp,
        final_alert=FinalAlert.NONE,
        tracker_confirmed=False,
        tracker_threat=None,
        tracker_centroid_norm=None,
        yolo_best_class=None,
        yolo_best_confidence=None,
        reasoning="No fire evidence",
        n_trackers_confirmed=0,
        n_yolo_detections=len(yolo_detections),
        node_health=node_health,
    )


# ============================================================================
# YOLO wrapper (with graceful failure)
# ============================================================================

class YoloInference:
    """YOLOv8s-seg wrapper, normalized-coord output, graceful failure handling."""

    def __init__(
        self,
        weights_path: Path,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "0",
        class_names: Optional[dict[int, str]] = None,
    ) -> None:
        from ultralytics import YOLO  # type: ignore
        self.model = YOLO(str(weights_path))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_names = class_names or {
            YOLO_CLASS_WILDFIRE: "wildfire",
            YOLO_CLASS_FALSE_POSITIVE: "false_positive",
        }
        self.error_count: int = 0
        self.disabled: bool = False
        logger.info(f"YOLO loaded: {weights_path.name}, classes={self.class_names}")

    def infer(self, yolo_input: np.ndarray) -> list[YoloDetection]:
        """Run inference. On repeated errors, self-disable to preserve flight."""
        if self.disabled:
            return []

        try:
            H, W = yolo_input.shape[:2]
            results = self.model.predict(
                yolo_input,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            if not results:
                return []

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return []

            detections: list[YoloDetection] = []
            for i in range(len(r.boxes)):
                xyxy = r.boxes.xyxy[i].cpu().numpy()
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                x1, y1, x2, y2 = xyxy
                bbox_norm = (
                    float(x1 / W), float(y1 / H),
                    float(x2 / W), float(y2 / H),
                )
                detections.append(YoloDetection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox_norm=bbox_norm,
                ))
            self.error_count = 0
            return detections

        except Exception as e:
            self.error_count += 1
            logger.warning(f"YOLO inference error ({self.error_count}/{YOLO_ERROR_TOLERANCE}): {e}")
            if self.error_count >= YOLO_ERROR_TOLERANCE:
                logger.error(f"YOLO disabled after {YOLO_ERROR_TOLERANCE} errors. "
                             f"Node will continue physics-only.")
                self.disabled = True
            return []


# ============================================================================
# Camera manager (with reconnect + watchdog)
# ============================================================================

class CameraManager:
    """
    Owns the P3 lifecycle. Handles:
      - Initial connect / init / gain / stream-start
      - Reconnection on USB errors with exponential backoff
      - Explicit teardown (safe to call multiple times)

    Raises CameraFatalError if reconnection attempts exhaust.
    """

    class CameraFatalError(RuntimeError):
        pass

    def __init__(self, stabilization_frames: int = 5) -> None:
        self.camera: Optional[P3Camera] = None
        self.stabilization_frames = stabilization_frames
        self.reconnect_attempts: int = 0
        self.last_frame_time: float = time.time()

    def _open(self) -> None:
        """Single connect sequence. Raises on failure."""
        logger.info("Opening P3 camera...")
        self.camera = P3Camera()
        self.camera.connect()
        self.camera.init()
        time.sleep(0.5)
        self.camera.set_gain_mode(GainMode.LOW)
        time.sleep(0.5)
        self.camera.start_streaming()
        for _ in range(self.stabilization_frames):
            self.camera.read_frame_both()
            time.sleep(0.04)
        logger.info("P3 streaming OK")
        self.last_frame_time = time.time()

    def _close(self) -> None:
        """Safe close (swallows all exceptions)."""
        if self.camera is None:
            return
        try:
            self.camera.stop_streaming()
        except Exception as e:
            logger.debug(f"stop_streaming ignored: {e}")
        try:
            self.camera.disconnect()
        except Exception as e:
            logger.debug(f"disconnect ignored: {e}")
        self.camera = None

    def open(self) -> None:
        """Public entry: initial camera open with retry + backoff."""
        backoff = CAMERA_RECONNECT_BACKOFF_MIN_S
        for attempt in range(1, MAX_CAMERA_RECONNECT_ATTEMPTS + 1):
            try:
                self._open()
                self.reconnect_attempts = 0
                return
            except Exception as e:
                logger.warning(f"Camera open failed (attempt {attempt}/"
                               f"{MAX_CAMERA_RECONNECT_ATTEMPTS}): {e}")
                self._close()
                time.sleep(backoff)
                backoff = min(backoff * 2.0, CAMERA_RECONNECT_BACKOFF_MAX_S)

        raise CameraManager.CameraFatalError(
            f"Could not open P3 after {MAX_CAMERA_RECONNECT_ATTEMPTS} attempts"
        )

    def reconnect(self) -> None:
        """Called during flight when USB error detected."""
        self.reconnect_attempts += 1
        if self.reconnect_attempts > MAX_CAMERA_RECONNECT_ATTEMPTS:
            raise CameraManager.CameraFatalError(
                f"Camera reconnection exhausted ({self.reconnect_attempts} attempts). "
                f"Node aborting — RECOMMEND AUTOLAND."
            )
        logger.warning(f"Attempting camera reconnect "
                       f"({self.reconnect_attempts}/{MAX_CAMERA_RECONNECT_ATTEMPTS})")
        self._close()
        time.sleep(min(
            CAMERA_RECONNECT_BACKOFF_MIN_S * (2 ** (self.reconnect_attempts - 1)),
            CAMERA_RECONNECT_BACKOFF_MAX_S,
        ))
        self._open()

    def read_frame(self) -> np.ndarray:
        """Read one thermal frame in Celsius. Raises on USB error."""
        if self.camera is None:
            raise CameraManager.CameraFatalError("Camera not opened")
        _, thermal_raw = self.camera.read_frame_both()
        self.last_frame_time = time.time()
        return raw_to_celsius(thermal_raw)

    def check_watchdog(self) -> bool:
        """Return True if no frame received within watchdog window."""
        elapsed = time.time() - self.last_frame_time
        return elapsed > FRAME_WATCHDOG_TIMEOUT_S

    def close(self) -> None:
        """Public entry for lifecycle shutdown."""
        self._close()


# ============================================================================
# Telemetry stubs (integrate with MAVLink when available)
# ============================================================================

def _emit_telemetry(result: ArbitrationResult) -> None:
    """Forward alerts to the ground station. STUB — integrate with comms stack."""
    if result.final_alert in (FinalAlert.RED, FinalAlert.ORANGE):
        logger.warning(
            f"[TELEMETRY] frame={result.frame_id} alert={result.final_alert.value.upper()} "
            f"health={result.node_health.value} | {result.reasoning}"
        )
    elif result.final_alert == FinalAlert.WHITE:
        logger.info(f"[CONFLICT_LOG] frame={result.frame_id} {result.reasoning}")


def _emit_heartbeat(
    frame_count: int,
    health: NodeHealth,
    tracker_count: int,
    yolo_disabled: bool,
) -> None:
    """Periodic liveness ping to ground station."""
    logger.info(
        f"[HEARTBEAT] frame={frame_count} health={health.value} "
        f"trackers={tracker_count} yolo_disabled={yolo_disabled}"
    )


# ============================================================================
# Main orchestrator
# ============================================================================

@dataclass
class NodeConfig:
    yolo_weights: Path
    yolo_imgsz: int = 640
    yolo_conf: float = 0.25
    device: str = "0"
    log_every_n_frames: int = 25
    max_frames: int = -1
    stabilization_frames: int = 5


class EdgeHybridNode:
    """Top-level orchestrator. Manages camera lifecycle, pipelines, arbitration."""

    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        self.cam_manager: CameraManager = CameraManager(
            stabilization_frames=cfg.stabilization_frames
        )
        self.physics_pipeline: ThermalTrackingPipeline = ThermalTrackingPipeline()
        self.yolo: Optional[YoloInference] = None

        self._should_stop: bool = False
        self._frame_count: int = 0
        self._consecutive_frame_errors: int = 0

        # Perf counters (moving window)
        self._total_physics_ms: float = 0.0
        self._total_yolo_ms: float = 0.0
        self._total_arbitration_ms: float = 0.0
        self._perf_window_frames: int = 0

    # ---- Health ----

    @property
    def health(self) -> NodeHealth:
        if self.cam_manager.reconnect_attempts > 0:
            return NodeHealth.RECOVERING
        if self.yolo is None or self.yolo.disabled:
            return NodeHealth.DEGRADED
        if self._consecutive_frame_errors >= MAX_CONSECUTIVE_FRAME_ERRORS // 2:
            return NodeHealth.CRITICAL
        return NodeHealth.OK

    # ---- Lifecycle ----

    def setup(self) -> None:
        logger.info(f"Loading YOLO: {self.cfg.yolo_weights}")
        try:
            self.yolo = YoloInference(
                weights_path=self.cfg.yolo_weights,
                imgsz=self.cfg.yolo_imgsz,
                conf=self.cfg.yolo_conf,
                device=self.cfg.device,
            )
        except Exception as e:
            logger.error(f"YOLO load failed: {e}. Node will run physics-only.")
            self.yolo = None

        logger.info("Opening camera...")
        self.cam_manager.open()
        logger.info("Node setup complete")

    def teardown(self) -> None:
        logger.info("Tearing down node...")
        self.cam_manager.close()
        reset_default_pipeline()
        logger.info("Teardown complete")

    def request_stop(self) -> None:
        logger.info("Stop requested by signal")
        self._should_stop = True

    # ---- Per-frame ----

    def _process_one_frame(self) -> Optional[ArbitrationResult]:
        """Execute one full pipeline cycle. Raises on fatal errors only."""
        celsius = self.cam_manager.read_frame()
        frame_shape = celsius.shape
        self._frame_count += 1
        timestamp = time.time()

        # Physics
        t0 = time.perf_counter()
        tracker_detections = self.physics_pipeline.step(celsius)
        t_physics = (time.perf_counter() - t0) * 1000.0

        # YOLO (optional)
        yolo_detections: list[YoloDetection] = []
        t_yolo = 0.0
        if self.yolo is not None and not self.yolo.disabled:
            t1 = time.perf_counter()
            try:
                yolo_input = preprocess_radiometric(celsius)
                yolo_detections = self.yolo.infer(yolo_input)
            except Exception as e:
                logger.warning(f"YOLO wrapper error (non-fatal): {e}")
            t_yolo = (time.perf_counter() - t1) * 1000.0

        # Arbiter
        t2 = time.perf_counter()
        result = arbitrate(
            frame_id=self._frame_count,
            timestamp=timestamp,
            frame_shape=frame_shape,
            tracker_detections=tracker_detections,
            yolo_detections=yolo_detections,
            node_health=self.health,
        )
        t_arb = (time.perf_counter() - t2) * 1000.0

        # Telemetry
        if result.final_alert != FinalAlert.NONE:
            _emit_telemetry(result)

        # Perf accounting
        self._total_physics_ms += t_physics
        self._total_yolo_ms += t_yolo
        self._total_arbitration_ms += t_arb
        self._perf_window_frames += 1

        # Periodic diagnostics
        if self._frame_count % self.cfg.log_every_n_frames == 0:
            n = max(self._perf_window_frames, 1)
            logger.info(
                f"frame={self._frame_count}  "
                f"avg_physics={self._total_physics_ms / n:.1f}ms  "
                f"avg_yolo={self._total_yolo_ms / n:.1f}ms  "
                f"avg_arb={self._total_arbitration_ms / n:.2f}ms  "
                f"trackers={len(self.physics_pipeline.trackers)}  "
                f"health={self.health.value}"
            )
            self._total_physics_ms = 0.0
            self._total_yolo_ms = 0.0
            self._total_arbitration_ms = 0.0
            self._perf_window_frames = 0

        # Heartbeat
        if self._frame_count % HEARTBEAT_EVERY_N_FRAMES == 0:
            _emit_heartbeat(
                frame_count=self._frame_count,
                health=self.health,
                tracker_count=len(self.physics_pipeline.trackers),
                yolo_disabled=self.yolo is None or self.yolo.disabled,
            )

        return result

    def _handle_frame_error(self, exc: BaseException) -> None:
        """Categorize frame-loop exception and decide response."""
        self._consecutive_frame_errors += 1

        is_usb = isinstance(exc, USB_ERRORS) if USB_ERRORS else False
        is_fatal = isinstance(exc, CameraManager.CameraFatalError)

        if is_fatal:
            raise exc

        if is_usb:
            logger.warning(f"USB error on frame {self._frame_count}: {exc}. Reconnecting...")
            try:
                self.cam_manager.reconnect()
                self._consecutive_frame_errors = 0
            except CameraManager.CameraFatalError:
                raise
            return

        if self._consecutive_frame_errors % 10 == 1:
            logger.error(f"Transient frame error #{self._consecutive_frame_errors}: {exc}")
            logger.debug(traceback.format_exc())

        if self._consecutive_frame_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
            raise CameraManager.CameraFatalError(
                f"Transient errors exceeded {MAX_CONSECUTIVE_FRAME_ERRORS}. Aborting."
            )

    # ---- Main loop ----

    def run(self) -> None:
        """Main flight loop. Only returns on graceful stop or fatal abort."""
        try:
            self.setup()
        except CameraManager.CameraFatalError as e:
            logger.error(f"Cannot start node: {e}")
            return

        logger.info("Entering main loop")
        try:
            while not self._should_stop:
                if self.cfg.max_frames > 0 and self._frame_count >= self.cfg.max_frames:
                    logger.info(f"max_frames={self.cfg.max_frames} reached, exiting")
                    break

                # Watchdog check before read
                if self.cam_manager.check_watchdog() and self._frame_count > 0:
                    logger.warning("Frame watchdog expired — forcing reconnect")
                    try:
                        self.cam_manager.reconnect()
                        self._consecutive_frame_errors = 0
                    except CameraManager.CameraFatalError as e:
                        logger.error(f"FATAL: {e}")
                        break

                try:
                    self._process_one_frame()
                    self._consecutive_frame_errors = 0
                except CameraManager.CameraFatalError as e:
                    logger.error(f"FATAL (circuit breaker): {e}")
                    break
                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt in frame loop")
                    self._should_stop = True
                    break
                except Exception as e:
                    try:
                        self._handle_frame_error(e)
                    except CameraManager.CameraFatalError as fatal:
                        logger.error(f"FATAL (from handler): {fatal}")
                        break

        finally:
            self.teardown()

        logger.info(f"Node exited. Total frames processed: {self._frame_count}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisEdge Edge Hybrid Node (hardened)")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to YOLOv8s-seg best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Stop after N frames (-1 = infinite)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.weights.exists():
        logger.error(f"YOLO weights not found: {args.weights}")
        sys.exit(1)

    cfg = NodeConfig(
        yolo_weights=args.weights,
        yolo_imgsz=args.imgsz,
        yolo_conf=args.conf,
        device=args.device,
        max_frames=args.max_frames,
    )

    node = EdgeHybridNode(cfg)

    def _sigint_handler(signum: int, frame: object) -> None:
        node.request_stop()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    node.run()


if __name__ == "__main__":
    main()
