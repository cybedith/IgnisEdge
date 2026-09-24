"""
ignis_vision.py -- Motor de Visión Embebida de Producción para Jetson Orin Nano.
================================================================================

Características de optimización:
  1. Pipeline de Detección C++ con OpenCV (SIMD/ARM NEON).
  2. cv2.connectedComponentsWithStats con conectividad 4 (16x más rápido que SciPy).
  3. Extracción localizada de features en sub-ventanas (bounding box slicing).
  4. Hilo de captura USB desacoplado (Producer-Consumer) para prevenir desbordes de buffer en la P3.
  5. Modo --headless para operación en vuelo (0% consumo en GUI, sin dependencia X11).
  6. Emisión IPC por Unix Domain Socket (/tmp/ignis_vision.sock) hacia ignis_mission.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
try:
    from scipy import ndimage
    HAS_SCIPY = True
except (ImportError, AttributeError, Exception):
    ndimage = None  # type: ignore
    HAS_SCIPY = False

# Intentar importar msgpack para IPC de alta velocidad; fallback a json
try:
    import msgpack  # type: ignore
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Importar driver P3 si está disponible
try:
    from p3_camera import GainMode, P3Camera, raw_to_celsius  # type: ignore
    HAS_P3_DRIVER = True
except ImportError:
    HAS_P3_DRIVER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ignis_vision] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ignis_vision")


# ============================================================================
# Dataclasses de Detección y Telemetría
# ============================================================================

@dataclass(slots=True)
class DetectionCandidate:
    cx: int
    cy: int
    features: Dict[str, float]
    flicker_peak: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass(slots=True)
class TrackSnapshot:
    id: int
    state: str       # "eval" | "FUEGO" | "no"
    cx: int
    cy: int
    confidence: float
    last_flicker: float
    last_T: float
    area_px: float
    hits: int
    obs: int


@dataclass(slots=True)
class VisionFrameResult:
    timestamp: float
    status: str       # "OBSERVANDO" | "ACTIVO"
    num_fires: int
    max_temp_c: float
    fps: float
    min_temp_c: float = 0.0
    tracks: List[TrackSnapshot] = field(default_factory=list)
    thermal_grid_16x12: List[int] = field(default_factory=list)


# ============================================================================
# Algoritmos de Detección Acelerados (OpenCV + NEON)
# ============================================================================

_KERNEL_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def downsample_thermal(frame_c: np.ndarray) -> Tuple[float, float, List[int]]:
    """
    Max pooling vectorizado: reduce 192x256 -> 12x16 preservando los picos térmicos.
    Normaliza el rango a uint8 para transmisión ligera por MAVLink/radio.
    """
    t_min = frame_c.min()
    t_max = frame_c.max()

    h, w = frame_c.shape              # 192, 256
    bh, bw = 16, 16                   # Bloques de 16x16 (192/16=12, 256/16=16)
    
    # Reducción matemática pura (vista 4D sin copia)
    pooled = frame_c.reshape(h // bh, bh, w // bw, bw).max(axis=(1, 3))

    span = t_max - t_min
    if span < 1e-6:
        norm = np.zeros_like(pooled, dtype=np.uint8)
    else:
        norm = ((pooled - t_min) * (255.0 / span)).astype(np.uint8)

    return float(t_min), float(t_max), norm.flatten().tolist()


def detect_fast(
    frame_c: np.ndarray,
    k: float = 4.0,
    t_abs: float = 80.0,
    t_floor: float = 45.0,
    sigma_floor: float = 0.05,
    merge_px: int = 3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Etapa 1 optimizada:
      - Estimación de fondo local vía downsample 4x + mediana
      - Detección de contraste contextual y umbral absoluto
      - Dilatación morfológica acelerada en C++ con OpenCV (kernel cruz)
    """
    h, w = frame_c.shape
    small = cv2.resize(frame_c, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    if HAS_SCIPY and ndimage is not None:
        try:
            bg_s = ndimage.median_filter(small, size=11, mode="nearest")
        except Exception:
            bg_s = cv2.blur(small, (11, 11))
    else:
        bg_s = cv2.blur(small, (11, 11))
    bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)
    delta = frame_c - bg

    # Escala robusta global MAD (Median Absolute Deviation)
    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    sigma = max(1.4826 * mad, sigma_floor)

    mask = ((delta > k * sigma) | (frame_c > t_abs)) & (frame_c >= t_floor)
    if merge_px > 0 and mask.any():
        mask_u8 = cv2.dilate(mask.astype(np.uint8), _KERNEL_CROSS, iterations=merge_px)
        mask = mask_u8.astype(bool)

    return mask, delta, sigma


def extract_blobs_fast(
    mask: np.ndarray,
    delta: np.ndarray,
    sigma: float,
    tc: np.ndarray,
    flicker: np.ndarray,
    min_area: int = 2,
    max_area: int = 4000,
) -> List[DetectionCandidate]:
    """
    Extracción de candidatos mediante cv2.connectedComponentsWithStats:
      - Ejecución en C++ con conectividad 4 (idéntica a scipy.ndimage.label).
      - Análisis localizado exclusivamente en el bounding-box de cada mancha.
    """
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
    candidates: List[DetectionCandidate] = []

    for bid in range(1, num_labels):
        area = int(stats[bid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[bid, cv2.CC_STAT_LEFT])
        y = int(stats[bid, cv2.CC_STAT_TOP])
        w = int(stats[bid, cv2.CC_STAT_WIDTH])
        h = int(stats[bid, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[bid]

        # Recorte localizado a la sub-ventana (evita escanear la imagen completa)
        sub_labels = labels[y : y + h, x : x + w]
        blob_mask = sub_labels == bid

        sub_delta = delta[y : y + h, x : x + w]
        sub_flicker = flicker[y : y + h, x : x + w]
        sub_tc = tc[y : y + h, x : x + w]

        d_peak = float(sub_delta[blob_mask].max())
        fk = float(sub_flicker[blob_mask].max())
        t_peak = float(sub_tc[blob_mask].max())

        feats = {
            "flicker_peak_C": fk,
            "T_peak_C": t_peak,
            "delta_peak_C": d_peak,
            "sigma_peak": d_peak / sigma,
            "area_px": float(area),
        }
        candidates.append(
            DetectionCandidate(
                cx=int(round(cx)),
                cy=int(round(cy)),
                features=feats,
                flicker_peak=fk,
                bbox=(x, y, w, h),
            )
        )
    return candidates


# ============================================================================
# Lógica de Rastreo Temporal con Histéresis Asimétrica
# ============================================================================

class Track:
    """Foco seguido temporalmente con período de gracia y evidencia log-odds."""
    _next_id = 0

    def __init__(self, cx: int, cy: int, flicker_hist: int = 12):
        self.id = Track._next_id
        Track._next_id += 1
        self.cx = cx
        self.cy = cy
        self.logodds = 0.0
        self.state = "eval"  # eval | FUEGO | no
        self.missed = 0
        self.hits = 1
        self.obs = 1
        self.flick_ok = deque(maxlen=flicker_hist)
        self.last_flicker = 0.0
        self.last_T = 0.0
        self.area = 4.0
        self._matched = False

    def update(
        self,
        cx: int,
        cy: int,
        p: float,
        feats: Dict[str, float],
        cfg: Dict[str, Any],
        flicker_ok: bool,
    ) -> None:
        self.cx = int(0.6 * self.cx + 0.4 * cx)
        self.cy = int(0.6 * self.cy + 0.4 * cy)
        self.missed = 0
        self.hits += 1
        self.obs += 1
        self.last_flicker = feats.get("flicker_peak_C", 0.0)
        self.last_T = feats.get("T_peak_C", 0.0)
        self.area = feats.get("area_px", self.area)
        self.flick_ok.append(1 if flicker_ok else 0)

        # Compuerta sostenida
        sostenido = (
            len(self.flick_ok) >= cfg["flick_need"]
            and sum(self.flick_ok) >= cfg["flick_need"]
        )
        if not sostenido:
            p = min(p, 0.2)

        p = min(max(p, 1e-4), 1.0 - 1e-4)
        step = np.log(p / (1.0 - p)) * cfg["ev_gain"]
        self.logodds = float(np.clip(self.logodds * cfg["decay"] + step, -14.0, 14.0))

        # Período de gracia e histéresis asimétrica
        grace_ok = self.obs >= cfg["min_obs"]

        if self.state != "FUEGO":
            if grace_ok and sostenido and self.logodds >= cfg["on"]:
                self.state = "FUEGO"
            elif self.logodds <= cfg["off"]:
                self.state = "no"
        else:
            if self.logodds <= cfg["off"]:
                self.state = "no"

    def miss(self, cfg: Dict[str, Any]) -> None:
        self.missed += 1
        self.hits = 0
        self.flick_ok.append(0)
        self.logodds = float(np.clip(self.logodds * cfg["decay"], -14.0, 14.0))

    def confidence(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self.logodds)))

    def to_snapshot(self) -> TrackSnapshot:
        return TrackSnapshot(
            id=self.id,
            state=self.state,
            cx=self.cx,
            cy=self.cy,
            confidence=round(self.confidence(), 3),
            last_flicker=round(self.last_flicker, 2),
            last_T=round(self.last_T, 1),
            area_px=float(self.area),
            hits=self.hits,
            obs=self.obs,
        )


# ============================================================================
# Motor Central de Inferencia (VisionEngine)
# ============================================================================

class VisionEngine:
    """Motor desacoplado de percepción térmica e inferencia de flicker."""

    def __init__(
        self,
        model_path: Path,
        flicker_window: int = 10,
        k: float = 4.0,
        t_abs: float = 80.0,
        t_floor: float = 45.0,
        sigma_floor: float = 0.05,
        merge_px: int = 3,
        min_area: int = 2,
        max_area: int = 4000,
        match_dist: float = 25.0,
        keep_missed: int = 4,
        min_obs: int = 20,
        flick_need: int = 8,
        flick_hist: int = 12,
        on_thresh: float = 6.0,
        off_thresh: float = -1.5,
        ev_gain: float = 0.6,
        decay: float = 0.92,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encuentra el modelo: {self.model_path}")

        self.model = joblib.load(self.model_path)
        meta_path = self.model_path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        self.features = meta.get("features", ["flicker_peak_C"])
        self.model_floor = float(meta.get("flicker_floor", 10.0))

        self.flicker_window = flicker_window
        self.k = k
        self.t_abs = t_abs
        self.t_floor = t_floor
        self.sigma_floor = sigma_floor
        self.merge_px = merge_px
        self.min_area = min_area
        self.max_area = max_area
        self.match_dist = match_dist
        self.keep_missed = keep_missed

        self.cfg = {
            "min_obs": min_obs,
            "flick_need": flick_need,
            "flick_hist": flick_hist,
            "on": on_thresh,
            "off": off_thresh,
            "ev_gain": ev_gain,
            "decay": decay,
        }

        self.fbuf: deque[np.ndarray] = deque(maxlen=flicker_window)
        self.tracks: List[Track] = []
        self.warmup = flicker_window
        self._fps = 0.0
        self._t_last = time.perf_counter()

    def reset_warmup(self) -> None:
        """Reinicia buffer y tracks tras un NUC."""
        self.fbuf.clear()
        self.tracks.clear()
        self.warmup = self.flicker_window
        logger.info("Motor reiniciado: NUC ejecutado y warm-up reseteado.")

    def process_frame(self, frame_c: np.ndarray) -> VisionFrameResult:
        """Procesa un frame térmico en grados Celsius."""
        now = time.perf_counter()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        # 1. Buffer y mapa de flicker (std temporal)
        self.fbuf.append(frame_c)
        if len(self.fbuf) >= 3:
            flicker = np.stack(self.fbuf, axis=0).std(axis=0)
        else:
            flicker = np.zeros_like(frame_c)

        if self.warmup > 0:
            self.warmup -= 1

        # 2. Detección rápida
        mask, delta, sigma = detect_fast(
            frame_c,
            k=self.k,
            t_abs=self.t_abs,
            t_floor=self.t_floor,
            sigma_floor=self.sigma_floor,
            merge_px=self.merge_px,
        )

        # 3. Extracción de blobs con sub-window slicing
        candidates = extract_blobs_fast(
            mask, delta, sigma, frame_c, flicker, self.min_area, self.max_area
        )

        # 4. Inferencia con compuerta física
        if candidates:
            Xf = np.array(
                [[c.features.get(f, 0.0) for f in self.features] for c in candidates],
                dtype=np.float64,
            )
            if np.isnan(Xf).any() or np.isinf(Xf).any():
                logger.warning(
                    "Vector de features con NaN/Inf detectado antes de la inferencia; "
                    "descartando candidatos de este frame sin propagar al modelo."
                )
                probs = np.zeros(len(candidates), dtype=np.float64)
            else:
                try:
                    raw_probs = self.model.predict_proba(Xf)[:, 1]
                    fkv = np.array([c.flicker_peak for c in candidates])
                    probs = np.where(fkv >= self.model_floor, raw_probs, 0.0)
                except Exception as e:
                    logger.error(f"Fallo en predict_proba del modelo de visión: {e}. Frame degradado sin detener el hilo.")
                    probs = np.zeros(len(candidates), dtype=np.float64)
        else:
            probs = np.array([], dtype=np.float64)

        # 5. Asociación y actualización de tracks
        for tr in self.tracks:
            tr._matched = False

        for cand, p in zip(candidates, probs):
            best: Optional[Track] = None
            best_dist = self.match_dist
            for tr in self.tracks:
                if tr._matched:
                    continue
                d = np.hypot(tr.cx - cand.cx, tr.cy - cand.cy)
                if d < best_dist:
                    best = tr
                    best_dist = d

            if best is None:
                best = Track(cand.cx, cand.cy, self.cfg["flick_hist"])
                self.tracks.append(best)

            best._matched = True
            if self.warmup > 0:
                best.cx, best.cy = cand.cx, cand.cy
                best.missed = 0
                best.hits += 1
                best.obs += 1
                best.last_flicker = cand.flicker_peak
                best.last_T = cand.features.get("T_peak_C", 0.0)
                best.area = cand.features.get("area_px", 4.0)
            else:
                best.update(
                    cand.cx,
                    cand.cy,
                    p,
                    cand.features,
                    self.cfg,
                    cand.flicker_peak >= self.model_floor,
                )

        for tr in self.tracks:
            if not tr._matched:
                tr.miss(self.cfg)

        self.tracks = [t for t in self.tracks if t.missed <= self.keep_missed]

        # 6. Empaquetar resultado
        num_fires = sum(1 for t in self.tracks if t.state == "FUEGO" and t.missed == 0)
        status = "OBSERVANDO" if self.warmup > 0 else "ACTIVO"

        t_min, t_max, thermal_grid = downsample_thermal(frame_c)

        return VisionFrameResult(
            timestamp=time.time(),
            status=status,
            num_fires=num_fires,
            max_temp_c=t_max,
            min_temp_c=t_min,
            fps=round(self._fps, 1),
            tracks=[t.to_snapshot() for t in self.tracks],
            thermal_grid_16x12=thermal_grid
        )


# ============================================================================
# Hilo de Captura Desacoplado para la Cámara P3
# ============================================================================

class P3CaptureThread(threading.Thread):
    """Lector USB dedicado en hilo independiente con buffering de baja latencia."""

    def __init__(self, gain: str = "low"):
        super().__init__(daemon=True)
        self.gain_str = gain
        self._active = True
        self._running = False
        self._camera: Optional[Any] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._nuc_requested = threading.Event()

    def is_connected(self) -> bool:
        return self._running and self._camera is not None

    def _cleanup_camera(self) -> None:
        if self._camera:
            try:
                self._camera.stop_streaming()
            except Exception:
                pass
            dev = getattr(self._camera, "dev", None)
            if dev:
                try:
                    import usb.util
                    usb.util.release_interface(dev, 0)
                    usb.util.release_interface(dev, 1)
                    usb.util.dispose_resources(dev)
                except Exception:
                    pass
            try:
                self._camera.disconnect()
            except Exception:
                pass
            self._camera = None

    def run(self) -> None:
        if not HAS_P3_DRIVER:
            logger.error("Driver p3_camera no disponible.")
            return

        logger.info("Iniciando hilo de captura de cámara P3 (con autoreconexión)...")
        gain_mode = GainMode.HIGH if self.gain_str == "high" else GainMode.LOW

        while self._active:
            if not self._running:
                try:
                    import usb.core
                    # Comprobar presencia de la cámara en el bus USB
                    dev = usb.core.find(idVendor=0x3474, idProduct=0x45A2)
                    if dev is None:
                        time.sleep(1.0)
                        continue

                    self._camera = P3Camera()
                    self._camera.connect()
                    self._camera.init()
                    self._camera.set_gain_mode(gain_mode)
                    self._camera.start_streaming()
                    logger.info("Cámara P3 física conectada y transmitiendo.")
                    self._running = True
                except Exception as e:
                    logger.debug(f"Aguardando cámara P3: {e}")
                    self._cleanup_camera()
                    self._running = False
                    time.sleep(1.5)
                    continue

            # Modo streaming activo
            if self._nuc_requested.is_set():
                try:
                    self._camera.trigger_shutter()
                    logger.info("NUC ejecutado exitosamente en hardware.")
                except Exception as e:
                    logger.warning(f"Fallo en trigger_shutter: {e}")
                finally:
                    self._nuc_requested.clear()

            try:
                _, raw = self._camera.read_frame_both()
                if raw is not None:
                    tc = raw_to_celsius(raw)
                    with self._lock:
                        self._latest_frame = tc
                    self._new_frame_event.set()
            except Exception as e:
                err_str = str(e).lower()
                if "no such device" in err_str or "not found" in err_str or "pipe" in err_str or "entity" in err_str:
                    logger.warning("Cámara P3 desconectada físicamente del USB. Esperando reconexión...")
                    self._cleanup_camera()
                    self._running = False
                time.sleep(0.02)

        self._cleanup_camera()
        logger.info("Cámara P3 detenida y recursos liberados.")

    def stop(self) -> None:
        self._active = False
        self._running = False

    def request_nuc(self) -> None:
        self._nuc_requested.set()

    def get_latest_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        if self._new_frame_event.wait(timeout):
            self._new_frame_event.clear()
            with self._lock:
                return self._latest_frame.copy() if self._latest_frame is not None else None
        return None

    def stop(self) -> None:
        self._running = False


# ============================================================================
# Servidor IPC por Unix Domain Socket (/tmp/ignis_vision.sock)
# ============================================================================

class VisionIPCServer:
    """Emisor IPC no bloqueante para compartir telemetría con ignis_mission."""

    def __init__(self, socket_path: str = "/tmp/ignis_vision.sock"):
        self.socket_path = socket_path
        self._server_sock: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        self._server_sock.listen(4)
        self._server_sock.settimeout(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info(f"Servidor IPC activo en {self.socket_path}")

    def _accept_loop(self) -> None:
        while self._running:
            try:
                assert self._server_sock is not None
                conn, _ = self._server_sock.accept()
                with self._lock:
                    self._clients.append(conn)
            except socket.timeout:
                continue
            except Exception:
                break

    def broadcast(self, result: VisionFrameResult) -> None:
        """Transmite el veredicto en formato binario msgpack o JSON."""
        data_dict = {
            "timestamp": result.timestamp,
            "status": result.status,
            "num_fires": result.num_fires,
            "max_temp_c": result.max_temp_c,
            "fps": result.fps,
            "tracks": [asdict(t) for t in result.tracks],
        }
        payload = msgpack.packb(data_dict) if HAS_MSGPACK else json.dumps(data_dict).encode("utf-8")
        packet = len(payload).to_bytes(4, "big") + payload

        with self._lock:
            disconnected = []
            for client in self._clients:
                try:
                    client.sendall(packet)
                except Exception:
                    disconnected.append(client)
            for c in disconnected:
                self._clients.remove(c)

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


# ============================================================================
# Main / Punto de Entrada
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IgnisEdge - Motor de Visión Embebida")
    ap.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent.parent / "scripts" / "ignis_fire_classifier.joblib"),
        help="Ruta al clasificador entrenado",
    )
    ap.add_argument("--gain", choices=["high", "low"], default="low", help="Modo de ganancia")
    ap.add_argument("--headless", action="store_true", help="Ejecutar sin GUI para modo vuelo")
    ap.add_argument("--ipc", action="store_true", default=True, help="Habilitar servidor IPC")
    ap.add_argument("--t-floor", type=float, default=45.0, help="Piso de temperatura")
    ap.add_argument("--record-dataset", action="store_true", help="Grabar dataset de entrenamiento cuando se detecte fuego o posibles fuegos")
    ap.add_argument("--dataset-dir", default="data/fire_dataset", help="Directorio destino del dataset (def: data/fire_dataset)")
    return ap.parse_args()


def main() -> None:
    args = parse_arguments()
    logger.info(f"Iniciando ignis_vision | Headless: {args.headless} | Gain: {args.gain}")

    engine = VisionEngine(model_path=Path(args.model), t_floor=args.t_floor)

    ipc_server = VisionIPCServer()
    if args.ipc:
        ipc_server.start()

    camera_thread = P3CaptureThread(gain=args.gain)
    camera_thread.start()

    running = True

    def sig_handler(_signum: int, _frame: Any) -> None:
        nonlocal running
        logger.info("Señal de parada recibida. Cerrando ordenadamente...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    if not args.headless:
        win_name = "IgnisEdge - Vision Embebida"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    recorder = None
    if args.record_dataset:
        from edge_core.vision.dataset_recorder import DatasetRecorder
        recorder = DatasetRecorder(output_dir=args.dataset_dir, max_fps=5.0)
        logger.info(f"Grabador de dataset activo en: {args.dataset_dir}")

    try:
        while running:
            frame_c = camera_thread.get_latest_frame(timeout=0.05)
            if frame_c is None:
                continue

            result = engine.process_frame(frame_c)

            if recorder:
                recorder.maybe_record(frame_c, result)

            if args.ipc:
                ipc_server.broadcast(result)

            if not args.headless:
                # Renderizado interactivo para banco de pruebas
                norm = np.clip((frame_c - 15.0) / (200.0 - 15.0), 0, 1)
                u8 = (norm * 255).astype(np.uint8)
                disp = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
                disp = cv2.resize(disp, (disp.shape[1] * 3, disp.shape[0] * 3), interpolation=cv2.INTER_NEAREST)

                for tr in result.tracks:
                    x, y = tr.cx * 3, tr.cy * 3
                    r = max(12, int(np.sqrt(tr.area_px) * 3))
                    color = (0, 255, 0) if tr.state == "FUEGO" else ((0, 200, 255) if tr.state == "eval" else (255, 128, 0))
                    cv2.rectangle(disp, (x - r, y - r), (x + r, y + r), color, 2)
                    cv2.putText(disp, f"{tr.state} {tr.confidence*100:.0f}%", (x - r, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imshow(win_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("n"):
                    camera_thread.request_nuc()
                    engine.reset_warmup()

    finally:
        if recorder:
            recorder.stop()
        camera_thread.stop()
        camera_thread.join(timeout=2.0)
        if args.ipc:
            ipc_server.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("Módulo ignis_vision finalizado.")


if __name__ == "__main__":
    main()
