"""
IgnisEdge - Morphological Classifier with Thermal Tracking (v3)

Extends the original morphological classifier with:
- ThermalTracker class: Kalman-based tracking with leaky bucket for occlusions
- Frame-to-frame identity persistence across tree canopy occlusions
- Confirmation logic: tracker only alerts after N_HITS observations

Design decisions:
- Ego-motion compensation via image-space velocity estimation (NOT world-space).
  When MAVLink integration is available, migrate to world coordinates.
- Leaky bucket: tracker survives up to MAX_BLINDNESS consecutive missed frames
  before being killed. Simulates occlusion by forest canopy.
- Kalman state: [cx, cy, vx, vy] in image coordinates (pixels).
- Association: greedy nearest-neighbor by centroid distance, gated by max jump.

Physical justification of thresholds:
- MAX_BLINDNESS = 8 frames (~320ms at 25Hz). Typical canopy opening/closing cycle.
- N_HITS_CONFIRM = 5 frames (~200ms). Enough to reject transient hot spikes.
- MAX_ASSOCIATION_DIST_PX = 40 px. At P3 native res, ~10m at 60m altitude.
  Faster than any real fire could spread between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import binary_propagation, label as ndi_label


# ============================================================================
# Physical thresholds (unchanged from v2)
# ============================================================================

EXTEND_TEMP_C: float = 80.0
SEED_TEMP_C: float = 150.0
CORE_TEMP_C: float = 250.0
INTENSE_TEMP_C: float = 400.0

# Morphological classification thresholds
MIN_REGION_AREA_PX: int = 30
POINT_SOURCE_MAX_AREA: int = 200
POINT_SOURCE_MIN_CIRCULARITY: float = 0.75
EXTENDED_MIN_ASPECT: float = 4.0
EXTENDED_MIN_SOLIDITY: float = 0.85
UNIFORM_MAX_GRADIENT_C: float = 30.0
HIGH_CORE_RATIO: float = 0.7
WILDFIRE_HALO_EXTENT: float = 1.5
WILDFIRE_HIGH_HALO: float = 1.8
WILDFIRE_HIGH_GRADIENT_C: float = 80.0

# ============================================================================
# Tracker parameters
# ============================================================================

N_HITS_CONFIRM: int = 5          # Hits required to confirm a tracker
MAX_BLINDNESS: int = 8           # Max consecutive missed frames before kill
MAX_ASSOCIATION_DIST_PX: float = 40.0  # Max centroid jump per frame for association

# Kalman process/measurement noise (tuned for 25Hz, image space pixels)
KALMAN_PROCESS_NOISE_POS: float = 2.0    # std of position prediction noise
KALMAN_PROCESS_NOISE_VEL: float = 5.0    # std of velocity prediction noise
KALMAN_MEAS_NOISE: float = 3.0            # std of measurement noise (centroid uncertainty)


# ============================================================================
# Enums (unchanged)
# ============================================================================

class ThreatClass(Enum):
    WILDFIRE = "wildfire"
    EXTENDED_HOT = "extended_hot"
    POINT_SOURCE = "point_source"
    AMBIGUOUS = "ambiguous"
    NOISE = "noise"


class AlertLevel(Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    WHITE = "white"
    NONE = "none"


class TrackerState(Enum):
    TENTATIVE = "tentative"     # Not yet confirmed (hits < N_HITS_CONFIRM)
    CONFIRMED = "confirmed"     # Valid fire detection
    COASTING = "coasting"       # Predicted during occlusion (no measurement)
    DEAD = "dead"               # To be deleted


# ============================================================================
# Feature extraction (unchanged from v2)
# ============================================================================

@dataclass
class RegionFeatures:
    """Geometric + thermal features of one thermal region."""
    region_id: int
    area_px: int
    bbox: tuple[int, int, int, int]       # (x, y, w, h)
    centroid: tuple[float, float]         # (cx, cy)
    max_temp_c: float
    mean_temp_c: float
    temp_gradient_c: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    solidity: float
    core_ratio: float
    halo_extent: float
    n_subregions: int


def _compute_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _compute_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def _equivalent_radius(area_px: float) -> float:
    return float(np.sqrt(max(area_px, 0) / np.pi))


def extract_region_features(
    celsius: np.ndarray,
    region_mask: np.ndarray,
    region_id: int = 0,
) -> Optional[RegionFeatures]:
    """Extract all features for a single connected region."""
    if region_mask.sum() < MIN_REGION_AREA_PX:
        return None

    mask_u8 = region_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = float(x + w / 2), float(y + h / 2)

    temps_in_region = celsius[region_mask]
    if temps_in_region.size == 0:
        return None

    max_t = float(temps_in_region.max())
    mean_t = float(temps_in_region.mean())
    grad_t = float(temps_in_region.std())

    core_mask = region_mask & (celsius >= CORE_TEMP_C)
    core_area = int(core_mask.sum())
    core_ratio = core_area / max(area, 1)
    core_radius = _equivalent_radius(core_area) if core_area > 0 else 0.0

    halo_radius = _equivalent_radius(area)
    halo_extent = halo_radius / max(core_radius, 1.0) if core_radius > 0 else 0.0

    if core_area > 0:
        _, n_sub = ndi_label(core_mask)
    else:
        n_sub = 0

    aspect = max(w, h) / max(min(w, h), 1)
    circ = _compute_circularity(area, perimeter)
    solid = _compute_solidity(contour)

    return RegionFeatures(
        region_id=region_id,
        area_px=int(area),
        bbox=(int(x), int(y), int(w), int(h)),
        centroid=(float(cx), float(cy)),
        max_temp_c=max_t,
        mean_temp_c=mean_t,
        temp_gradient_c=grad_t,
        perimeter=float(perimeter),
        circularity=circ,
        aspect_ratio=float(aspect),
        solidity=solid,
        core_ratio=float(core_ratio),
        halo_extent=float(halo_extent),
        n_subregions=int(n_sub),
    )


# ============================================================================
# Classification rules (unchanged from v2)
# ============================================================================

def classify_region(f: RegionFeatures) -> tuple[ThreatClass, AlertLevel, str]:
    """Apply morphological rules. Returns (class, alert_level, reasoning)."""
    reasons: list[str] = []

    if f.area_px < MIN_REGION_AREA_PX:
        return ThreatClass.NOISE, AlertLevel.NONE, f"Area {f.area_px}px < {MIN_REGION_AREA_PX}"

    if f.area_px < POINT_SOURCE_MAX_AREA and f.circularity > POINT_SOURCE_MIN_CIRCULARITY:
        reasons.append(f"area={f.area_px}<{POINT_SOURCE_MAX_AREA}")
        reasons.append(f"circ={f.circularity:.2f}>{POINT_SOURCE_MIN_CIRCULARITY}")
        return ThreatClass.POINT_SOURCE, AlertLevel.NONE, "Small circular: " + ", ".join(reasons)

    if f.aspect_ratio > EXTENDED_MIN_ASPECT and f.solidity > EXTENDED_MIN_SOLIDITY:
        reasons.append(f"aspect={f.aspect_ratio:.1f}>{EXTENDED_MIN_ASPECT}")
        reasons.append(f"solidity={f.solidity:.2f}>{EXTENDED_MIN_SOLIDITY}")
        return ThreatClass.EXTENDED_HOT, AlertLevel.YELLOW, "Elongated solid: " + ", ".join(reasons)

    if f.temp_gradient_c < UNIFORM_MAX_GRADIENT_C and f.core_ratio > HIGH_CORE_RATIO:
        reasons.append(f"grad={f.temp_gradient_c:.1f}<{UNIFORM_MAX_GRADIENT_C}")
        reasons.append(f"core_ratio={f.core_ratio:.2f}>{HIGH_CORE_RATIO}")
        return ThreatClass.POINT_SOURCE, AlertLevel.NONE, "Uniform high core: " + ", ".join(reasons)

    # Rule 3.5: small flame source
    if (f.halo_extent < 0.5 and
        f.temp_gradient_c < 30.0 and
        2.0 < f.aspect_ratio < 4.0 and
        f.area_px < 1000):
        reasons.append(f"halo={f.halo_extent:.2f}<0.5")
        reasons.append(f"grad={f.temp_gradient_c:.1f}<30")
        reasons.append(f"aspect={f.aspect_ratio:.1f} in [2,4]")
        return ThreatClass.POINT_SOURCE, AlertLevel.NONE, "Small flame source: " + ", ".join(reasons)

    if f.halo_extent > WILDFIRE_HIGH_HALO and f.n_subregions >= 2:
        level = AlertLevel.RED if f.max_temp_c > INTENSE_TEMP_C else AlertLevel.ORANGE
        reasons.append(f"halo={f.halo_extent:.2f}>{WILDFIRE_HIGH_HALO}")
        reasons.append(f"n_sub={f.n_subregions}>=2")
        reasons.append(f"max={f.max_temp_c:.0f}C")
        return ThreatClass.WILDFIRE, level, "Strong wildfire: " + ", ".join(reasons)

    if f.halo_extent > WILDFIRE_HALO_EXTENT and f.temp_gradient_c > WILDFIRE_HIGH_GRADIENT_C:
        level = AlertLevel.RED if f.max_temp_c > INTENSE_TEMP_C else AlertLevel.ORANGE
        reasons.append(f"halo={f.halo_extent:.2f}>{WILDFIRE_HALO_EXTENT}")
        reasons.append(f"grad={f.temp_gradient_c:.1f}>{WILDFIRE_HIGH_GRADIENT_C}")
        return ThreatClass.WILDFIRE, level, "Wildfire: " + ", ".join(reasons)

    return ThreatClass.AMBIGUOUS, AlertLevel.WHITE, (
        f"No rule matched: area={f.area_px} circ={f.circularity:.2f} "
        f"aspect={f.aspect_ratio:.1f} grad={f.temp_gradient_c:.1f} "
        f"halo={f.halo_extent:.2f} n_sub={f.n_subregions}"
    )


# ============================================================================
# Kalman-based Thermal Tracker
# ============================================================================

class ThermalTracker:
    """
    Single-region tracker with 2D Kalman filter + leaky bucket for occlusions.

    State vector: [cx, cy, vx, vy]
    Measurement:  [cx, cy]

    Transitions:
      TENTATIVE -> CONFIRMED when hit_streak >= N_HITS_CONFIRM
      CONFIRMED -> COASTING when no measurement this frame
      COASTING  -> CONFIRMED when measurement arrives again
      ANY       -> DEAD when blindness > MAX_BLINDNESS
    """

    _next_id: int = 0

    def __init__(
        self,
        initial_features: RegionFeatures,
        initial_threat: ThreatClass,
        initial_alert: AlertLevel,
        initial_reasoning: str,
    ) -> None:
        ThermalTracker._next_id += 1
        self.track_id: int = ThermalTracker._next_id

        # Kalman filter setup (OpenCV KalmanFilter)
        self.kf: cv2.KalmanFilter = cv2.KalmanFilter(4, 2)
        # State transition: x' = x + vx, y' = y + vy, vx' = vx, vy' = vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # Measurement: we observe [cx, cy]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # Process noise covariance
        q_pos = KALMAN_PROCESS_NOISE_POS ** 2
        q_vel = KALMAN_PROCESS_NOISE_VEL ** 2
        self.kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        # Measurement noise covariance
        r = KALMAN_MEAS_NOISE ** 2
        self.kf.measurementNoiseCov = np.diag([r, r]).astype(np.float32)
        # Initial state: position from centroid, velocity 0
        cx, cy = initial_features.centroid
        self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

        # Tracker state
        self.state: TrackerState = TrackerState.TENTATIVE
        self.hit_streak: int = 1
        self.total_hits: int = 1
        self.blindness: int = 0
        self.age: int = 1

        # Most recent observation
        self.last_features: RegionFeatures = initial_features
        self.last_threat: ThreatClass = initial_threat
        self.last_alert: AlertLevel = initial_alert
        self.last_reasoning: str = initial_reasoning

        # History of threats (for stability voting)
        self.threat_history: list[ThreatClass] = [initial_threat]
        self.max_temp_history: list[float] = [initial_features.max_temp_c]

    @property
    def is_confirmed(self) -> bool:
        """True if tracker has accumulated enough hits without dying."""
        return self.state == TrackerState.CONFIRMED

    @property
    def predicted_centroid(self) -> tuple[float, float]:
        """Return the current Kalman-predicted centroid."""
        state = self.kf.statePost
        return float(state[0, 0]), float(state[1, 0])

    @property
    def predicted_velocity(self) -> tuple[float, float]:
        state = self.kf.statePost
        return float(state[2, 0]), float(state[3, 0])

    def predict(self) -> tuple[float, float]:
        """Advance Kalman prediction one frame. Returns predicted centroid."""
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update_with_measurement(
        self,
        features: RegionFeatures,
        threat: ThreatClass,
        alert: AlertLevel,
        reasoning: str,
    ) -> None:
        """Associate this tracker with a measurement and update Kalman state."""
        cx, cy = features.centroid
        meas = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.correct(meas)

        self.hit_streak += 1
        self.total_hits += 1
        self.blindness = 0
        self.age += 1

        self.last_features = features
        self.last_threat = threat
        self.last_alert = alert
        self.last_reasoning = reasoning
        self.threat_history.append(threat)
        self.max_temp_history.append(features.max_temp_c)

        # Trim history to avoid unbounded memory
        if len(self.threat_history) > 50:
            self.threat_history = self.threat_history[-50:]
            self.max_temp_history = self.max_temp_history[-50:]

        # State transition
        if self.state == TrackerState.TENTATIVE and self.total_hits >= N_HITS_CONFIRM:
            self.state = TrackerState.CONFIRMED
        elif self.state == TrackerState.COASTING:
            self.state = TrackerState.CONFIRMED

    def mark_missed(self) -> None:
        """No measurement associated this frame — leaky bucket logic."""
        self.blindness += 1
        self.hit_streak = 0
        self.age += 1

        if self.state == TrackerState.CONFIRMED:
            self.state = TrackerState.COASTING

        if self.blindness > MAX_BLINDNESS:
            self.state = TrackerState.DEAD

    def distance_to(self, features: RegionFeatures) -> float:
        """Euclidean distance (pixels) between predicted centroid and measurement."""
        px, py = self.predicted_centroid
        mx, my = features.centroid
        return float(np.hypot(px - mx, py - my))

    def stable_threat(self) -> ThreatClass:
        """Majority vote over recent threat history (last 10 frames)."""
        recent = self.threat_history[-10:]
        if not recent:
            return ThreatClass.AMBIGUOUS
        counts: dict[ThreatClass, int] = {}
        for t in recent:
            counts[t] = counts.get(t, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]


# ============================================================================
# Detection + Tracking pipeline
# ============================================================================

@dataclass
class TrackedDetection:
    """Output of detect_and_track: current-frame detections with tracker context."""
    tracker: ThermalTracker
    features: RegionFeatures
    threat_class: ThreatClass
    alert_level: AlertLevel
    reasoning: str
    is_confirmed: bool
    predicted_centroid: tuple[float, float]
    measured_this_frame: bool  # False if coasting (predicted only)


class ThermalTrackingPipeline:
    """
    Stateful pipeline: call .step(celsius) once per frame.
    Returns current-frame detections, including coasting trackers.
    """

    def __init__(self) -> None:
        self.trackers: list[ThermalTracker] = []
        self.frame_count: int = 0

    def step(self, celsius: np.ndarray) -> list[TrackedDetection]:
        """Process one frame: predict, associate, update, prune."""
        self.frame_count += 1

        # --- Step 1: Extract current-frame regions ---
        seed = celsius >= SEED_TEMP_C
        extend = celsius >= EXTEND_TEMP_C
        if seed.any():
            full_mask = binary_propagation(seed, mask=extend)
            labeled, n_regions = ndi_label(full_mask)
        else:
            labeled, n_regions = np.zeros_like(celsius, dtype=np.int32), 0

        current_features: list[tuple[RegionFeatures, ThreatClass, AlertLevel, str]] = []
        for region_id in range(1, n_regions + 1):
            region_mask = labeled == region_id
            feats = extract_region_features(celsius, region_mask, region_id)
            if feats is None:
                continue
            threat, alert, reason = classify_region(feats)
            current_features.append((feats, threat, alert, reason))

        # --- Step 2: Predict all existing trackers one step forward ---
        for tracker in self.trackers:
            tracker.predict()

        # --- Step 3: Greedy association (nearest-neighbor by Kalman distance) ---
        used_tracker_indices: set[int] = set()
        used_measurement_indices: set[int] = set()

        # Build distance matrix
        if self.trackers and current_features:
            pairs: list[tuple[float, int, int]] = []
            for t_idx, tracker in enumerate(self.trackers):
                for m_idx, (feats, _, _, _) in enumerate(current_features):
                    dist = tracker.distance_to(feats)
                    if dist <= MAX_ASSOCIATION_DIST_PX:
                        pairs.append((dist, t_idx, m_idx))

            # Sort by distance, greedy assign
            pairs.sort(key=lambda p: p[0])
            for dist, t_idx, m_idx in pairs:
                if t_idx in used_tracker_indices or m_idx in used_measurement_indices:
                    continue
                feats, threat, alert, reason = current_features[m_idx]
                self.trackers[t_idx].update_with_measurement(feats, threat, alert, reason)
                used_tracker_indices.add(t_idx)
                used_measurement_indices.add(m_idx)

        # --- Step 4: Unmatched trackers are marked missed (leaky bucket) ---
        for t_idx, tracker in enumerate(self.trackers):
            if t_idx not in used_tracker_indices:
                tracker.mark_missed()

        # --- Step 5: Unmatched measurements spawn new trackers ---
        for m_idx, (feats, threat, alert, reason) in enumerate(current_features):
            if m_idx not in used_measurement_indices:
                new_tracker = ThermalTracker(feats, threat, alert, reason)
                self.trackers.append(new_tracker)

        # --- Step 6: Prune dead trackers ---
        self.trackers = [t for t in self.trackers if t.state != TrackerState.DEAD]

        # --- Step 7: Build output ---
        detections: list[TrackedDetection] = []
        for tracker in self.trackers:
            measured = tracker.blindness == 0
            detections.append(TrackedDetection(
                tracker=tracker,
                features=tracker.last_features,
                threat_class=tracker.stable_threat(),
                alert_level=tracker.last_alert,
                reasoning=tracker.last_reasoning,
                is_confirmed=tracker.is_confirmed,
                predicted_centroid=tracker.predicted_centroid,
                measured_this_frame=measured,
            ))

        return detections

    def reset(self) -> None:
        """Clear all trackers (e.g. on camera reset)."""
        self.trackers = []
        self.frame_count = 0


# ============================================================================
# Backwards-compatible function (drop-in replacement for v2)
# ============================================================================

_default_pipeline: Optional[ThermalTrackingPipeline] = None


def detect_and_classify(celsius: np.ndarray) -> list[TrackedDetection]:
    """
    Drop-in replacement for v2's detect_and_classify, but now with tracking.

    NOTE: This uses a module-level singleton pipeline. For multi-camera or
    explicit control, instantiate ThermalTrackingPipeline directly.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ThermalTrackingPipeline()
    return _default_pipeline.step(celsius)


def reset_default_pipeline() -> None:
    """Reset the module-level default pipeline (e.g. on camera restart)."""
    global _default_pipeline
    _default_pipeline = None


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ThermalTrackingPipeline self-test")
    print("=" * 70)

    pipeline = ThermalTrackingPipeline()

    # Simulate a wildfire moving across 10 frames, with occlusion frames 4-6
    print("\nTest 1: wildfire with occlusion (frames 4-6)")
    for frame_idx in range(10):
        img = np.full((200, 200), 25.0, dtype=np.float32)

        if frame_idx not in (3, 4, 5):  # Simulated occlusion
            # Fire centroid moves diagonally
            cx = 60 + frame_idx * 5
            cy = 60 + frame_idx * 3
            # Halo
            cv2.ellipse(img, (cx, cy), (40, 30), 0, 0, 360, 120.0, -1)
            cv2.ellipse(img, (cx, cy), (25, 18), 0, 0, 360, 200.0, -1)
            # Multiple cores
            cv2.circle(img, (cx - 5, cy), 6, 380.0, -1)
            cv2.circle(img, (cx + 8, cy + 3), 5, 420.0, -1)

        detections = pipeline.step(img)
        for det in detections:
            marker = "✓" if det.measured_this_frame else "~ (coasting)"
            conf = "CONFIRMED" if det.is_confirmed else "tentative"
            print(f"  Frame {frame_idx}: tracker#{det.tracker.track_id} {marker} "
                  f"[{conf}] threat={det.threat_class.value} "
                  f"pred=({det.predicted_centroid[0]:.0f},{det.predicted_centroid[1]:.0f}) "
                  f"hits={det.tracker.total_hits} blind={det.tracker.blindness}")

    print(f"\nFinal tracker count: {len(pipeline.trackers)}")
    for t in pipeline.trackers:
        print(f"  #{t.track_id}: {t.state.value}, hits={t.total_hits}, "
              f"stable_threat={t.stable_threat().value}")
