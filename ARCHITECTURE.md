# IgnisEdge — Architecture & Design Rationale

**A Safety-Critical Edge-AI System for Early Wildfire Detection**

> Memoria de Título — Ingeniería en Informática, Universidad Técnica Federico Santa María (USM), Chile
> Equipo: Pablo Silva (Visión Embebida, FSM, MAVLink, Edge AI) · Camilo (Red LoRa y Nodos Terrestres) · Bastián (Dashboard C2 y Fusión Satelital) — Profesor guía: Jorge Portilla
> Plataforma: Holybro S500 (Pixhawk 6C / ArduCopter) · Jetson Orin Nano 8GB · Cámara radiométrica InfiRay Thermal Master P3 · Heltec WiFi LoRa 32 V3 (ESP32-S3 + SX1262) · Bosch BME688

---

## Abstract

IgnisEdge is a proof-of-concept aerial system for the early detection of forest fires in their incipient phase, designed to close the coverage gap between periodic satellite revisit (~6 h for services such as OroraTech) and continuous human patrol. The system couples a battery-powered, solar-triggered ground sensor mesh (LoRa 915 MHz, Bosch BME688 gas sensing) with an autonomous quadcopter carrying a radiometric thermal camera and an onboard Edge-AI inference stack (NVIDIA Jetson Orin Nano). Detection confirmation relies on a **temporal-variance ("flicker") discriminator** rather than static spatial signatures, exploiting the fact that combustion is a stochastically oscillating thermal process (1–15 Hz) that no static hot surface (metal roof, rock, machinery) reproduces. Flight authority is strictly layered: the flight controller (Pixhawk 6C / ArduCopter) is the *inviolable real-time authority*; the Jetson is a *suggestive, non-authoritative* co-processor that proposes GUIDED-mode waypoints but can never command the aircraft without passing through a human-in-the-loop authorization gate and a continuously re-evaluated energy-reserve lock. The system targets sub-meter (~0.5–0.8 m) hotspot resolution — five to eight times finer than current satellite-based competitors — at TRL 3–4, validated on a hardware-in-the-loop (HITL) bench with live Pixhawk, Jetson, and LoRa hardware.

This document is the authoritative architectural reference for the monorepo. It is written for a technical readership already fluent in embedded systems, control theory, and machine learning, and it privileges *why* over *what*: every major design decision is presented with its physical, statistical, or doctrinal justification, its rejected alternatives, and the empirical evidence that settled it.

---

## Table of Contents

1. [System Overview and Design Philosophy](#1-system-overview-and-design-philosophy)
2. [Repository Structure](#2-repository-structure)
3. [The Three-Pillar Architecture](#3-the-three-pillar-architecture)
4. [Flight Authority Doctrine: A Four-Level Hierarchy](#4-flight-authority-doctrine-a-four-level-hierarchy)
5. [The Mission FSM: `ignis_mission.py`](#5-the-mission-fsm-ignis_missionpy)
6. [Thermal Vision Pipeline and the Flicker Discriminator](#6-thermal-vision-pipeline-and-the-flicker-discriminator)
7. [Communications Architecture: Dual-Band RF Doctrine](#7-communications-architecture-dual-band-rf-doctrine)
8. [Binary Protocols and Framing](#8-binary-protocols-and-framing)
9. [Aero-Energetic Route Planning (ADR-11)](#9-aero-energetic-route-planning-adr-11)
10. [Backend: FastAPI C2 and Geospatial Services](#10-backend-fastapi-c2-and-geospatial-services)
11. [Frontend: React Tactical Dashboard](#11-frontend-react-tactical-dashboard)
12. [Concurrency Model](#12-concurrency-model)
13. [Failure Detection, Isolation, and Recovery (FDIR)](#13-failure-detection-isolation-and-recovery-fdir)
14. [Engineering Standard: NASA/JPL "Power of 10" Adaptation](#14-engineering-standard-nasajpl-power-of-10-adaptation)
15. [Hardware Stack](#15-hardware-stack)
16. [Verification & Validation](#16-verification--validation)
17. [Architecture Decision Records (ADR Index)](#17-architecture-decision-records-adr-index)
18. [Known Gaps, Open Risks, and Roadmap](#18-known-gaps-open-risks-and-roadmap)
19. [Competitive Positioning and Academic Context](#19-competitive-positioning-and-academic-context)
20. [Glossary](#20-glossary)
21. [Bibliography](#21-bibliography)

---

## 1. System Overview and Design Philosophy

### 1.1 Mission Statement

Detect forest fires in their **incipient phase** using a drone with embedded thermal vision and Edge AI, delivering georeferenced alerts before the fire escalates — covering the critical gap between periodic satellite sampling (~6 h revisit) and continuous surveillance.

### 1.2 Reference Scenario

Autonomous patrol at 100 m above canopy, 15:00 hrs in January (worst case for solar radiation and drought), Biobío Region, Chile — a Mediterranean climate zone with among the highest wildfire risk indices in South America.

### 1.3 Governing Philosophy: Safety First, Design for Failure

The project adopts, almost verbatim, the engineering doctrine used for safety-critical flight software at NASA/JPL (an adaptation of Holzmann's "Power of Ten" rules; see [§14](#14-engineering-standard-nasajpl-power-of-10-adaptation)). Two axioms dominate every architectural decision in this repository:

1. **When "optimal" conflicts with "safe," safe wins ("safing").** A false positive that dispatches a fire brigade to an empty field is *expensive*; a few extra seconds of confirmation before a verdict is *free*, because a real fire, once in the camera's field of view, remains observable for a long time. The system is therefore built to be **conservative by construction**, at every layer, from the ML classifier's hysteresis to the FSM's authorization gate.
2. **The system is designed for the failure case, not the happy path.** Every subsystem — vision, radio link, flight control, battery — has an explicit, tested fallback state. A component that has no defined recovery path is treated as an open risk, not a shippable feature.

### 1.4 The Central Engineering Tension

IgnisEdge sits at the intersection of two domains whose engineering cultures usually conflict:

- **Machine learning / computer vision**, which is probabilistic, iterative, and tolerant of occasional error.
- **Flight-critical embedded control**, which is deterministic, bounded, and intolerant of *any* unrecovered fault.

The architecture resolves this tension by **physically and logically separating the two domains** (see [§12](#12-concurrency-model)) rather than trying to make the ML pipeline "safe enough" to share a control loop with flight logic. The Jetson's vision thread can stall, throw exceptions, or return garbage; by design, none of that can propagate into the aircraft's stabilization loop, which lives entirely inside the Pixhawk's real-time firmware.

---

## 2. Repository Structure

```
IgnisEdge_Monorepo/
├── edge_core/                  # The drone's "brain" — runs on the Jetson Orin Nano
│   └── edge_core/
│       ├── mission/             ignis_mission.py    — Mission FSM (the strategic decision layer)
│       ├── flight/               ignis_mavlink.py    — pymavlink wrapper, heartbeat watchdog
│       ├── comms/                ignis_link.py        — LoRa serial driver (COBS + CRC16)
│       │                         ignis_proto.py       — Binary protocol structs (ALERT/TASK/ALERT_NODE)
│       ├── vision/               ignis_vision.py      — Thermal inference engine (Stage-1 + flicker classifier)
│       ├── tools/                web_stream.py        — Embedded HTTP/JSON telemetry + auth endpoints
│       │                         check_hardware.py    — Hardware diagnostics
│       ├── tests/                5 test suites (FSM, MAVLink mock, protocol/COBS, vision, web endpoints)
│       └── launcher.py           — 25 Hz orchestrator tying all threads together
├── backend/                     # FastAPI Cloud/C2 service (operator PC)
│   └── app/
│       ├── api/                  v1_routes.py, ws_routes.py
│       ├── core/                 gee_client.py, mqtt_client.py, simulation_manager.py, config.py
│       ├── services/             drone_router.py (RRT*-3D), terrain_service.py (DEM/SRTM),
│       │                         fire_propagation.py, smoke_dispersion.py, triangulation.py,
│       │                         weather_service.py, mesh_optimizer.py, gee_analyzer.py
│       └── models/                schemas.py
├── frontend/                    # React 18 + TypeScript + Vite C2 Dashboard
│   └── src/
│       ├── components/ignis/     TopBar, MapContainer, ThermalHUD, AvionicsHUD, VerdictHUD,
│       │                         FlightAuthDialog, AbortMissionDialog, SystemLogsPanel, ...
│       ├── hooks/                 useJetsonTelemetry.ts, useLiveWS.ts, useSimulationWS.ts, useTelemetryWS.ts
│       └── store/                 useIgnisStore.ts (Zustand global state)
└── ARCHITECTURE.md               — this document
```

This monorepo unifies the three architectural pillars of the thesis project (see [§3](#3-the-three-pillar-architecture)) that historically lived in separate repositories during development (`~/IgnisEdge/edge_core`, `~/ignis-edge-backend`, `~/Descargas/H1/frontend/ignis-edge-frontend`).

---

## 3. The Three-Pillar Architecture

### 3.1 `edge_core/` — The Drone's Brain

**Language:** Python 3.10 (no PyTorch, no GPU dependency in production inference — runs entirely on CPU).
**Runs on:** Jetson Orin Nano 8GB, physically mounted on the Holybro S500.
**Mission:** A finite state machine that owns the drone's autonomous decision-making. It listens to the Heltec LoRa modem over a serial port; when a ground node reports a fire signature, the FSM computes feasibility (geofence distance, energy budget), requests human authorization, and — only once authorized — commands the flight controller via MAVLink toward the target. During flight it processes the InfiRay P3 radiometric feed through a temporal-variance ("flicker") classifier to discriminate live fire from static hot surfaces.

### 3.2 `backend/` — FastAPI Cloud/C2

**Language:** Python (FastAPI).
**Mission:** Bridges satellite/geospatial data (Google Earth Engine — NDVI/NDMI, SRTM 30 m DEM, NOAA GFS wind) to the frontend; hosts the RRT\*-3D aero-energetic route planner; runs the simulation engine (fire spread via Anderson/Rothermel models, smoke dispersion via Gaussian Pasquill–Gifford plumes, TDoA triangulation for ground-node fusion) that produces the 3D digital twin consumed by the dashboard.

### 3.3 `frontend/` — React C2 Dashboard

**Language:** TypeScript, React 18, Vite, Tailwind, Mapbox GL / deck.gl.
**Mission:** The Command & Control console. Renders the 3D digital twin of the patrol area (Hualpén/Biobío terrain), live telemetry (position, attitude, battery, per-motor PWM effort), the reduced-resolution thermal feed with an Inferno colormap, and the human-authorization gate the FSM blocks on before it is allowed to fly.

### 3.4 Design Note: Why This Split, and Not a Monolith

The three pillars are deliberately isolated processes running on different machines (Jetson vs. operator PC) connected by narrow-bandwidth radio, not a shared codebase or shared memory space. This is not an accident of incremental development — it is the direct consequence of the **"Zero WiFi in the Field" doctrine** (see [§7](#7-communications-architecture-dual-band-rf-doctrine)): the drone must be able to fly a complete mission with the C2 link severed, so the C2 stack cannot be a load-bearing dependency of the drone's control loop under any circumstance.

---

## 4. Flight Authority Doctrine: A Four-Level Hierarchy

IgnisEdge's single most important architectural invariant is that **the Jetson never has final authority over the aircraft**. This is enforced structurally, not just by policy:

```
┌───────────────────────────────────────────────────────────────────┐
│  LEVEL 0 — MAXIMUM AUTHORITY                                       │
│  Human Safety Pilot (physical RC transmitter, or QGroundControl    │
│  gamepad over the 433 MHz telemetry link)                          │
│  → Instantaneous override at any time, in any FSM state.            │
├───────────────────────────────────────────────────────────────────┤
│  LEVEL 1 — REAL-TIME AUTHORITY                                     │
│  Pixhawk 6C (ArduCopter firmware)                                   │
│  → Owns stabilization, geofence, and all hard failsafes             │
│    (battery, RC loss, GCS loss). Executes RTL/LAND unconditionally  │
│    regardless of what the Jetson is doing or whether it is alive.  │
├───────────────────────────────────────────────────────────────────┤
│  LEVEL 2 — EDGE INTELLIGENCE (SUGGESTIVE ONLY)                     │
│  Jetson Orin Nano — ignis_mission.py FSM                            │
│  → Perceives, decides, and *proposes* GUIDED-mode waypoints.        │
│    Cannot command anything until WAITING_AUTH is cleared by a       │
│    human, and every proposal remains subject to Level 1 failsafes. │
├───────────────────────────────────────────────────────────────────┤
│  LEVEL 3 — PERIPHERAL / TRANSPORT                                   │
│  Heltec LoRa modem                                                  │
│  → Pure data transport (915 MHz mesh); carries no flight authority. │
└───────────────────────────────────────────────────────────────────┘
```

**Golden Rule:** if the Jetson freezes, reboots, or `ignis_vision` crashes, the Pixhawk continues flying (or falls back to `LOITER`) autonomously. The companion computer is causally downstream of flight safety, never upstream of it.

### 4.1 Concrete Failsafe Bindings (ArduCopter parameters)

| Parameter | Value | Safety function |
|---|---|---|
| `FS_GCS_ENABLE` | `1` | Ground-station link loss (433 MHz, >5 s) → automatic RTL |
| `RTL_ALT` | `3000` (30 m) | Climbs to a safe altitude above canopy before returning, to clear treetops |
| `BATT_FS_LOW_ACT` | `2` | Low-battery threshold (20%) → automatic RTL |
| `BATT_FS_CRT_ACT` | `1` | Critical-battery threshold (10%) → immediate LAND |
| `FS_THR_ENABLE` | `1` | Throttle/RC-signal-loss failsafe |
| `FENCE_ENABLE` | `1` | Perimeter geofence enforced in firmware, independent of the Jetson's software geofence |

### 4.2 Manual Override Watchdog

`ignis_mission.py` continuously monitors the Pixhawk's reported flight mode. If the safety pilot flips the RC switch to `LOITER`, `STABILIZE`, or `ALT_HOLD`, the FSM detects the mode change on its next 25 Hz tick and transitions to `MANUAL_OVERRIDE` — it stops sending guidance commands entirely rather than "fighting the stick." This is the software-side complement to the hardware truth that the Pixhawk always honors RC input over a MAVLink `SET_POSITION_TARGET_GLOBAL_INT` command.

### 4.3 Level 2 Operator Authority (C2 Dashboard)

The operator's emergency-abort control in the dashboard (`AbortMissionDialog.tsx`) is password-gated (administrator password, currently `ignis2026` — a placeholder to be replaced with a proper secrets-managed credential before any real flight) specifically to prevent an accidental tap from aborting a legitimate mission. Symmetrically, the `/api/authorize` and `/api/abort` endpoints exposed by `web_stream.py` on the Jetson require the same credential on every call, since the operator's authority must be authenticated at the point where it is exercised, not merely at the UI layer.

---

## 5. The Mission FSM: `ignis_mission.py`

### 5.1 State Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
   READY ──alert──▶ TRIAGE ──feasible──▶ WAITING_AUTH ──authorize──▶ TRANSIT ──arrival──▶ ON_STATION ──dwell──▶ VERDICT ──┐
     ▲                 │                       │                       │                      │                          │
     │            infeasible              timeout/                MANUAL_OVERRIDE      MANUAL_OVERRIDE            RTL ◀──┘
     │                 │                battery-drop                   │                      │                (or forced by
     └─────────────────┴───────────────────────┴───────────────────────┴──────────────────────┘                 battery watchdog)
```

Eight `MissionState` enum members: `READY`, `TRIAGE`, `WAITING_AUTH`, `TRANSIT`, `ON_STATION`, `VERDICT`, `RTL`, `MANUAL_OVERRIDE` (`edge_core/edge_core/mission/ignis_mission.py:31-39`).

| State | Function |
|---|---|
| `READY` | Idle; listens for LoRa `ALERT_NODE` messages from ground sensors |
| `TRIAGE` | Computes Haversine distance to target, scores candidates (confidence + multi-sensor bonus − distance penalty), evaluates energy feasibility |
| `WAITING_AUTH` | **Human-in-the-loop safety gate** (see [§5.3](#53-the-waiting_auth-gate-and-the-continuous-energy-lock)). No autonomous flight begins without explicit operator authorization. |
| `TRANSIT` | `GUIDED`-mode flight toward the target coordinate |
| `ON_STATION` | Orbits the target, forces a camera NUC (shutter calibration), accumulates thermal evidence for a fixed dwell period |
| `VERDICT` | Emits an `ALERT` (22 B) with the coordinate **as measured by the onboard camera**, not the ground node's self-reported position — the camera fix is authoritative because it corrects for GPS error in the ground sensor and for imprecision in the triage-time estimate |
| `RTL` | Return-to-Launch, either as normal mission completion or as an emergency battery/abort response |
| `MANUAL_OVERRIDE` | Entered whenever RC mode changes are detected during `TRANSIT`, `ON_STATION`, or `VERDICT`; the FSM stops issuing guidance until the pilot returns control |

### 5.2 The Original Design Flaw (and Why It Mattered)

An early audit (2026-09-07) of the FSM found that it transitioned **directly from `TRIAGE` to `TRANSIT`** — i.e., the drone would launch autonomously the instant the energy budget looked feasible, with no human ever in the loop. This was flagged as the single highest-priority defect in the entire system, because it violated the most basic tenet of the flight-authority doctrine (§4): an autonomous aircraft cannot self-authorize its own flight. The `WAITING_AUTH` state (and the `authorize_mission()` / `abort_mission()` methods that gate it) was introduced specifically to close this gap, and its correctness is covered by the FSM's unit-test suite (31 tests as of the current codebase, up from an initial 27 after the `SERVO_OUTPUT_RAW` telemetry work).

### 5.3 The `WAITING_AUTH` Gate and the Continuous Energy Lock

The most subtle failure mode this system had to design against is **temporal staleness of a safety computation**. A naive implementation checks "is there enough battery for this mission?" once, at the moment the alert arrives, and then treats that answer as valid indefinitely. But `WAITING_AUTH` can last anywhere from seconds to minutes while a human operator reviews the alert — and battery state is a physical quantity that decays continuously, including while the drone idles.

IgnisEdge therefore implements a **Triple Energy Lock**, evaluated asynchronously and independently at three distinct points in the mission lifecycle:

1. **Continuous monitoring during `WAITING_AUTH`** (5 Hz): while the FSM waits for the operator's decision, it keeps re-reading live Pixhawk telemetry. If the battery crosses the reserve threshold *during the wait*, the pending alert is automatically revoked — with no operator action required, and specifically without waiting for the operator to notice.
2. **Atomic re-check inside `authorize_mission()`**: at the exact moment the operator clicks "authorize," the FSM performs one more live read of the battery and re-validates that outbound + orbit + return + 20% reserve is still achievable before it allows the transition to `TRANSIT`. This closes the race between "the dashboard displayed a battery percentage" and "the battery percentage when the click actually lands."
3. **In-flight watchdog** (`TRANSIT`, `ON_STATION`, `VERDICT`): if the battery decays past the reserve floor at any point during active flight, an immediate forced `RTL` is issued regardless of mission progress.

This is not a redundant, belt-and-suspenders pattern for its own sake — it is a direct response to a *specific*, named failure mode ("perishable energy calculation," in the project's own vocabulary): the risk that an operator's decision latency silently converts a feasible mission into an infeasible one while the FSM's only knowledge of the world is a stale snapshot from `t₀`.

### 5.4 Dynamic Geofence

Rather than a fixed perimeter, the FSM computes the **Haversine great-circle distance** between the LoRa-reported target and the drone's live GPS position on every triage cycle. If that distance exceeds 300 m, the mission is instantly aborted as infeasible. This geofence has been empirically stress-tested by an unintentional real-world edge case: during an indoor desk test with no GPS fix, the ground node's position defaulted to an invalid value, producing a computed distance of ~8,507 km to the target — the system correctly rejected the mission rather than silently trusting a degenerate position. This is treated as evidence the fail-closed design works as intended, not as a defect to "fix" by relaxing the check.

### 5.5 On Desktop-Test Bypasses (and Why None May Ship)

During HITL bench testing without a live Pixhawk/GPS fix, the team has, at times, temporarily patched `web_stream.py` with desktop-only bypasses (fixed 50 m distance injection, battery-check skip, GPS `ARMING_CHECK` skip, or a `reserve_battery_pct = -100.0` override to defeat the continuous energy lock). Every one of these bypasses was subsequently and deliberately reverted before the session closed, and the pattern is explicitly flagged in the project's engineering log as **forbidden to ever reach a build connected to a real Pixhawk or real battery**. This project convention is treated as a standing constraint on any future contribution: a bypass introduced for desk-testing convenience is scoped to that testing session and must never survive into hardware-connected code.

---

## 6. Thermal Vision Pipeline and the Flicker Discriminator

### 6.1 The Central Insight: Why Temporal Variance, Not Spatial CNNs

The dominant failure mode in forest-fire thermal detection is the **false positive generated by static overheated surfaces** — a sun-baked metal roof, a rock face, idling machinery — which can register temperatures well within the range of active combustion in a single static frame. A purely spatial classifier (CNN over one frame) has no principled way to distinguish these from real fire.

IgnisEdge's answer is grounded in the combustion physics itself: **fire is not thermally static.** Flame fronts oscillate (flicker) at roughly 1–15 Hz due to buoyancy-driven turbulence in the combustion plume — a signature that essentially no static hot object reproduces. This is why the system's central engineered feature is a **temporal-variance channel**, not a spatial one.

### 6.2 Empirical Grounding

| Statistic | Value |
|---|---|
| Mean flicker amplitude, fire | **66 °C** |
| Mean flicker amplitude, non-fire (hot but static) | **3.5 °C** |
| Fires with flicker > 5 °C | 88% |
| Non-fires with flicker > 5 °C | 8% |
| Trivial rule `flicker > 20 °C` alone | 95% accuracy, 99% recall, 6.4% FP |

The separation between the fire and non-fire populations on this single feature is large enough that a trivial threshold rule already performs close to production quality — a strong signal that the feature, not model complexity, is doing the real discriminative work.

### 6.3 Pipeline Architecture

```
P3 raw capture (.npy, 256×192 px radiometric)
    → Stage 1: contextual hot-spot detector (mathematical, untrained)
    → 64×64 px, 3-channel crops + manifest CSV
    → Dataset assembly (session-balanced)
    → HistGradientBoostingClassifier training
    → .joblib + .json model artifact
    → Live detection (production inference loop)
```

**Stage 1 (untrained, purely mathematical):**

```
mask = (delta > k·σ) OR (T > t_abs)
```
where `delta = frame − local_background` (background estimated via large-window median), `σ = 1.4826 × MAD` (a robust, outlier-resistant scale estimator), `k = 4.0`, and `t_abs = 80 °C` for the HIGH gain range. This stage is deliberately not a learned model: it is a cheap, interpretable, self-calibrating anomaly filter that reduces a 256×192 frame to a small number of candidate 64×64 regions of interest (ROIs) before any classifier runs.

**Three channels per ROI crop:**

| Channel | Content | Role |
|---|---|---|
| `ch0` | Absolute scaled temperature | "How hot?" |
| `ch1` | Local anomaly in σ-units | "How anomalous relative to its surroundings?" — cancels domain gap across cameras/altitudes |
| `ch2` | Temporal standard deviation per pixel (**flicker**) | "Does it oscillate?" — the definitive fire/non-fire discriminator |

### 6.4 Production Classifier

**Model:** `HistGradientBoostingClassifier` (scikit-learn), trained via `train_feature_classifier.py`, artifact `ignis_fire_classifier.joblib` / `.json`.

**Feature set: `flicker_peak_C` alone.** This single-feature design was arrived at empirically, not chosen a priori — see [§6.6](#66-gain-invariance-a-hard-won-empirical-constraint) for the AUC comparison that eliminated the alternatives.

**Validated metrics (`GroupKFold` cross-validation, split by capture session — never by frame):**

| Metric | Value |
|---|---|
| Recall (fire detection) | **97.58%** |
| False-positive rate | **3.74%** |
| Precision | **86.53%** |
| Decision threshold | 0.40 |
| Flicker floor (physical gate) | 10.0 °C |

`GroupKFold`-by-session is not a stylistic preference; it is the only cross-validation scheme that cannot be gamed by a model that memorizes background scenery rather than learning fire physics — see [§6.5](#65-the-abandoned-cnn-a-cautionary-tale-in-evaluation-methodology).

### 6.5 The Abandoned CNN: A Cautionary Tale in Evaluation Methodology

The project's first classifier was `BaselineCNN`, a ~150k-parameter convolutional network (4× Conv2d+BN+ReLU+MaxPool blocks over the 64×64×3 crop, exported to ONNX for `trtexec --int8` deployment). It was abandoned (ADR-4) not because CNNs are inappropriate in principle, but because of two converging, project-specific failures:

1. **Data scarcity.** With only a handful of capture sessions, the CNN learned to memorize *backgrounds* rather than fire signatures — a textbook case of a high-capacity model overfitting to nuisance correlations when the effective sample diversity is low.
2. **A broken evaluation methodology masked the first failure.** Random (frame-level) train/validation splits let near-duplicate frames from the same session leak across the split boundary, producing misleadingly high validation metrics that collapsed (recall near 0) once session-level splitting (`GroupKFold`) was introduced and the leakage was closed.

The `HistGradientBoostingClassifier` over hand-engineered physical features was chosen as the replacement specifically because it (a) is far more sample-efficient than a CNN under session-scarce data, (b) is interpretable and defensible in an academic thesis defense, (c) runs on CPU with no TensorRT/GPU dependency, simplifying the Jetson's runtime footprint, and (d) is numerically stable rather than prone to the CNN's collapse mode.

### 6.6 Gain Invariance: A Hard-Won Empirical Constraint

The InfiRay P3 has two gain ranges — **HIGH** (−20…150 °C, used for patrol) and **LOW** (0…550 °C, used to confirm an already-hot fire without saturating the sensor). The entire training dataset was captured in LOW gain, but the drone flies patrol in HIGH gain, and re-recording the dataset was not an option. This created a hard constraint: **every feature used in production must be provably invariant to the camera's gain setting.**

| Feature | AUC | Gain-invariant? | Disposition |
|---|---|---|---|
| `flicker_peak_C` | 0.988 | ✅ Yes | **Selected** — sole production feature |
| `sigma_peak` | 0.994 | ✅ Yes | Candidate, not currently used |
| `delta_peak_C` | — | ❌ No | **Discarded** — this was the feature responsible for a live-fire candle reading only 25% confidence in early testing, because its raw scale shifts with gain |
| `T_peak_C` | — | ❌ No | **Discarded**, same reason |

This constraint was verified with a dedicated regression test, `test_gain_invariant.py`, and confirmed in live testing: the same candle flame is classified as fire consistently in both LOW and HIGH gain once `delta_peak_C` and `T_peak_C` are removed from the feature set.

A related, explicitly falsified hypothesis: shape/symmetry features (`fill_ratio`, `aspect`, `solidity`) were hypothesized to separate flame geometry from vapor/steam geometry, but empirical testing (`test_simetria.py`) showed AUC ≈ 0.5–0.57 — statistically indistinguishable from chance at this crop resolution. The hypothesis was discarded on the strength of the data, not intuition (ADR-7) — a deliberate methodological stance: features are kept or cut by measured discriminative power, not by how plausible they sound.

### 6.7 Live Detection: Defense in Depth Against False Positives

`live_classify.py` (and its hardened production analogue inside `edge_core`) layers five independent anti-false-positive mechanisms, stacked so that **no single mechanism is a single point of failure for a false alarm**:

1. **Global warm-up.** Until the temporal-flicker window (needed to compute `ch2`) is fully populated, the system only observes — it emits no verdict at all. This suppresses the inherent instability of a cold start.
2. **Fast candidate detection + blob merging.** A downsampled+median detector (~5 ms vs. ~950 ms for the full-resolution pass) finds candidate hot regions; dilation merges fragments of what is physically one fire into one blob.
3. **Tracker with a grace period.** A newly observed hot spot enters an "evaluating" state and cannot be declared fire until it has been continuously observed for `min_obs = 20` frames (~0.8 s at 25 fps) — a fire that vanishes in under a second was never a fire.
4. **Sustained-flicker gate.** The classifier requires high flicker in *several* recent frames (`flick_need = 8` of the last 12), not a single transient spike — this specifically defeats camera-motion artifacts and single-frame sensor noise that could momentarily mimic flicker.
5. **Bayesian log-odds accumulation with asymmetric hysteresis.** Evidence accumulates as log-odds over time; turning the verdict **ON** requires crossing a high threshold (`on = 6.0`), while turning it **OFF** only requires crossing a much looser one (`off = −1.5`). This asymmetry is deliberate: it is expensive to declare fire, cheap to keep declaring it once declared, and cheap to retract if evidence weakens — mirroring the operational cost asymmetry (false alarm costs a dispatch; missed confirmation of an ongoing fire costs nothing since the fire stays in view).

In live testing, this stack reliably distinguishes a candle (steady green verdict, ~100% confidence) from an oven, kettle, or a person occupying the same scene (rejected as non-fire) — including a kettle whose turbulent steam plume was briefly "suspicious" before being correctly rejected once its flicker failed to sustain.

### 6.8 Optics and Coverage

| Altitude | Ground footprint | GSD | Minimum reliable hotspot |
|---|---|---|---|
| 100 m | ~73 × 54 m (~3,900 m²) | ~28 cm/px | ~0.5–0.8 m |
| 60 m | ~44 × 32 m | ~17 cm/px | ~0.5 m |

**Sub-pixel dilution** is a measured, physically expected effect: an object smaller than one pixel's footprint has its temperature averaged against the colder background, so apparent temperature collapses with distance (a candle measured at 201 °C at 1.5 m reads only 43 °C at 6 m — below the useful signal floor). This bounds the achievable detection range and directly motivated the calibration protocol (short-range captures at 1–2 m for dataset construction).

### 6.9 Known Open Risk: Camera-Motion-Induced False Flicker

The flicker channel assumes the camera is observing from a stable position between frames. If the camera itself moves (vibration, gimbal jitter, wind-induced airframe motion), pixel-registration drift can produce *artificial* temporal variance at static hot edges — "ghost frames" that could, in principle, mimic the flicker signature of real fire. The architecturally correct fix is **image registration** in the vision pipeline before computing `ch2`, and it is **explicitly not yet implemented**. This is the single most safety-relevant open item in the vision stack and must be closed, or at minimum bounded by a quantified false-positive-rate-under-vibration test, before any untethered flight test is attempted.

### 6.10 Alternative Pipeline Branch: Morphological Classifier + YOLO Hybrid

Independently of the flicker-classifier production path described above, the codebase also contains a **physics-rules morphological classifier** (`morphological_classifier.py` v4, and a Kalman-tracked v3 variant) plus a **YOLOv8s-seg** model trained on the FLAME 3 dataset, combined via an arbitration layer (`edge_hybrid_node.py`) that fuses both pipelines' verdicts (`RED`/`ORANGE`/`YELLOW`/`WHITE`/`NONE`) with graceful degradation if either pipeline fails. This hybrid path exists as a richer, higher-recall research branch (e.g., it implements an "incipient fire" rule — small area but anomalously hot regions, Rule 0.5 — directly targeting the FLAME 3 "early ignition" scenario per Hopkins et al. 2024) but is **not** the flight-production path; `live_classify.py`'s single-feature flicker classifier remains the system-of-record for flight because of its simplicity, CPU-only footprint, and interpretability. The two pipelines' feature engineering do share code (`preprocess.py`'s hysteresis masking, `harvest_crops.py`'s three-channel crop construction), so improvements to Stage-1 detection propagate to both branches.

---

## 7. Communications Architecture: Dual-Band RF Doctrine

### 7.1 The "Zero WiFi in the Field" Doctrine

For remote forest missions, cellular or WiFi coverage is nonexistent or unreliable. IgnisEdge is architected around the assumption that **the operator PC may have internet access, but the drone is always offline**, dependent entirely on two narrow-bandwidth radio links:

```
┌──────────────────────────────────────────────────────────────┐
│                      OPERATOR PC (has internet)                │
│  React C2 Frontend ⇄ FastAPI Backend ⇄ [WiFi/HTTP monitoring]  │
└───────────────────────────────┬────────────────────────────────┘
                                 │ 433 MHz — MAVLink (bidirectional, long-range)
┌────────────────────────────────┴────────────────────────────────┐
│                            DRONE (fully offline)                 │
│  InfiRay P3 → Jetson Orin Nano (ignis_vision/mission/link) → Pixhawk 6C
│                                   ▲
│                                   │ 915 MHz — LoRa (Heltec bridge, /dev/ttyACM1)
└───────────────────────────────────┼────────────────────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │   Ground LoRa Mesh     │
                         │  Heltec + BME688 nodes │
                         └────────────────────────┘
```

| Band | Purpose | Endpoint on Jetson |
|---|---|---|
| **LoRa 915 MHz** | Ground-node-to-drone data plane: sensor mesh alerts, heartbeats, multi-hop relay | Heltec ESP32-S3 modem, `/dev/ttyACM1`, 115200 baud |
| **433 MHz + Yagi antenna** | Control plane: bidirectional MAVLink telemetry/control between the Pixhawk and the ground station, RC override | Direct Pixhawk link |
| **WiFi/HTTP (C2)** | Strictly a monitoring convenience layer for the React dashboard | `web_stream.py` on port 8080 |

**The critical architectural consequence:** if the WiFi connection to the C2 dashboard drops, or the React tab crashes, **the FSM on the Jetson keeps operating the drone autonomously**, because its actual operational inputs are LoRa (for alerts) and MAVLink over 433 MHz/UART (for flight), neither of which route through WiFi at all. This was empirically confirmed — not merely assumed — during a desk HITL session where the Jetson dropped off WiFi mid-test: the frontend correctly went dark (a *separate*, since-fixed frontend bug caused a full white-screen crash rather than a graceful "telemetry lost" state — see [§18](#18-known-gaps-open-risks-and-roadmap)), but the underlying FSM logic was unaffected, confirming the layering is real and not just a stated intention.

### 7.2 Why LoRa for the Ground Mesh (ADR-2)

Ground nodes must survive for months on battery in the forest and communicate over kilometers. LoRa 915 MHz provides long range at minimal power draw (a 20-byte transmission at spreading factor 12 costs roughly 0.03 mAh). The cost of this range/power tradeoff is bandwidth: messages must be bytes, not images — which is precisely why the protocol design (§8) is built around fixed, tightly packed binary structs rather than any richer serialization.

### 7.3 Why Not ROS2 / MAVROS / DroneKit (ADR-3)

For three cooperating processes on a single board, a full robotics middleware stack (ROS2, with MAVROS as its MAVLink bridge, or the higher-level DroneKit) was assessed as unjustified complexity and overhead. The chosen alternative is deliberately minimal: **direct `pymavlink`** for Pixhawk communication, and **Unix domain sockets + msgpack** for inter-process communication between the Jetson's own vision/mission/link processes. The underlying principle: the Pixhawk is the flight authority and the Jetson only suggests, so the link between them should be as simple and robust as possible — not a heavyweight middleware layer that adds its own failure surface to a safety-critical path.

---

## 8. Binary Protocols and Framing

### 8.1 LoRa Message Formats

**`ALERT_NODE`** (14 bytes) — ground node → drone:
```
type(1) node_id(2) seq(1) conf(1) sensors(1) lat_e7(4) lon_e7(4)
```
`sensors` is a bitfield (`NODE_GAS=0x01`, `NODE_HEAT=0x02`); lat/lon are encoded as `degrees × 1e7` in a signed 32-bit integer, giving ~1.1 cm of positional resolution at the equator without any floating-point encoding overhead.

**`ALERT`** (22 bytes) — drone → base:
```
type(1) src_id(1) seq(1) lat_e7(4) lon_e7(4) alt_m(2) level(1)
confidence(1) area_dm2(2) frp_w(2) evidence_mask(1) track_age_s(2)
```
`level`: 0=no fire, 1=possible, 2=confirmed. `evidence_mask` is a bitfield recording *which* independent signals contributed to the verdict (RAD/GEO/TMP/CNN/RGB/NODE/TRACK) — this is the field that lets a downstream operator (or thesis committee) audit *why* the system reached a given verdict without needing raw sensor logs.

**`TASK`** (18 bytes) — base → drone:
```
type(1) task_id(1) lat_e7(4) lon_e7(4) alt_m(2) radius_m(2)
dwell_s(2) action(1) pad(1)
```
`action`: `GOTO_VERIFY=0x01`, `PATROL=0x02`, `RTL=0x03`.

### 8.2 Serial Framing (Jetson ⇄ Heltec)

```
COBS(payload + CRC16-CCITT) + 0x00
```

**Consistent Overhead Byte Stuffing (COBS)** guarantees the frame delimiter byte (`0x00`) never appears inside the payload, so the receiver can always resynchronize on a corrupted or partial stream without ambiguity — a property that matters specifically because LoRa is a lossy link where partial-frame reception is the normal case, not the exception. **CRC16-CCITT** provides frame-integrity verification independent of LoRa's own (weaker, and not universally enabled) link-layer error detection.

An ASCII fallback (`ALERT,node_id,lat,lon,confidence\n` / `HEARTBEAT,node_id,lat,lon,batt,temp,gas_kohm\n`) exists for simplified/legacy nodes that do not implement the binary+COBS stack.

### 8.3 Why Byte-Level Framing Matters Here

Every protocol decision in this layer is downstream of a single hard constraint: the LoRa physical layer, at the spreading factors needed for multi-kilometer range in forest terrain, delivers on the order of tens of bytes per transmission economically. This is why the `ALERT` message — the system's most information-dense payload — is engineered to 22 bytes rather than, say, a JSON blob that would be an order of magnitude larger for equivalent content. The architecture treats bandwidth as a hard physical resource to be budgeted, not an assumed abundance.

---

## 9. Aero-Energetic Route Planning (ADR-11)

### 9.1 Why Planning Happens on the Ground, Not the Jetson

Trajectory planning could, in principle, run onboard. IgnisEdge deliberately puts it on the **operator PC** instead, for a resource-asymmetry reason: the PC has internet access to weather APIs and full-resolution SRTM 30 m elevation data, and abundant compute, while the Jetson must reserve its CPU budget for the thermal vision pipeline running at ≥20 fps. Centralizing the expensive part of planning on the resource-rich node and shipping only the *result* over the narrow 433 MHz link is a direct application of the same bandwidth-budgeting philosophy that shapes the LoRa protocol (§8.3).

### 9.2 The Planner

**Algorithm:** RRT\*-3D (`backend/app/services/drone_router.py`), operating in UTM coordinates over the SRTM 30 m digital elevation model, enforcing a minimum terrain clearance of `h_safe ≥ 80 m` above ground and avoiding modeled smoke plume volumes.

**Aero-energetic wind model:** the cost function incorporates the meteorological wind vector (`w⃗ · u⃗`), bonifying tailwind segments (up to ~35% energy savings) and heavily penalizing headwind segments to protect the Holybro S500's motors from sustained high-load operation. The planner supports **asymmetric routes** — the outbound and return legs may differ — specifically to exploit prevailing wind direction rather than retracing the same path both ways.

### 9.3 Pre-Flight Uplink and In-Flight Re-Routing

1. The simplified route (8–14 key waypoints) is transmitted over the 433 MHz MAVLink telemetry radio to the Jetson **before motors are armed**.
2. The Jetson validates total route distance against its energy triple-lock (§5.3) and internalizes the waypoints in RAM before entering `WAITING_AUTH`.
3. If wind conditions change or a smoke plume expands mid-flight, the ground station recomputes from the drone's live GPS position and pushes a `ROUTE_UPDATE` packet. The Jetson atomically validates remaining battery and, if feasible, hot-swaps the navigation queue without interrupting flight.
4. **Link-loss fail-safe:** if the 433 MHz link is lost, the Jetson **keeps flying the last fully-verified route** and never discards the active route until a replacement has been checksum-verified in its entirety. The route in memory is treated as more trustworthy than the absence of new data.

### 9.4 Status

ADR-11 is formalized at the architecture-decision level and the backend-side planner (`drone_router.py`, `terrain_service.py`) is implemented and functional. The Jetson-side consumption of `ROUTE_PACKAGE`/`ROUTE_UPDATE` over the 433 MHz link is designed but **not yet wired into `ignis_mission.py`** — see [§18](#18-known-gaps-open-risks-and-roadmap).

---

## 10. Backend: FastAPI C2 and Geospatial Services

| Module | Responsibility |
|---|---|
| `terrain_service.py` | Downloads and processes SRTM 30 m DEM from Google Earth Engine; computes slope/aspect via Horn's method for altimetric safety margins |
| `drone_router.py` | RRT\*-3D trajectory planner (§9) |
| `fire_propagation.py` | Elliptical fire-growth model (Anderson/Rothermel-derived) for simulation scenarios |
| `smoke_dispersion.py` | Gaussian Pasquill–Gifford plume dispersion model, feeding simulated gas concentrations to ground nodes in HITL/simulation mode |
| `triangulation.py` | TDoA-based multilateration of ground-node detections, wind-corrected |
| `weather_service.py` | Meteorological data ingestion feeding the aero-energetic wind cost model |
| `mesh_optimizer.py` | Ground-node mesh placement/coverage optimization |
| `gee_analyzer.py` / `gee_client.py` | Google Earth Engine integration (NDVI/NDMI vegetation-dryness indices) |
| `simulation_manager.py` | Tick-based simulation loop (2 Hz) driving the digital-twin demo mode |
| `mqtt_client.py` | MQTT bridge for telemetry, currently the least-integrated module — see [§18](#18-known-gaps-open-risks-and-roadmap) |
| `ws_routes.py` / `v1_routes.py` | WebSocket and REST surface consumed by the frontend |

The backend serves two distinct operational purposes that share infrastructure but should not be conflated: (1) a **simulation/demo engine** producing physically-plausible synthetic fire/smoke/telemetry for HITL bench testing and thesis-defense demonstration, and (2) the **real C2 service layer** (terrain, routing) that a genuine field deployment depends on. Care should be taken, when extending this backend, not to let simulation-mode conveniences leak into the code paths a live mission would exercise — mirroring the same discipline enforced for desktop-test bypasses on the edge side (§5.5).

---

## 11. Frontend: React Tactical Dashboard

### 11.1 Core Architecture

React 18 + TypeScript + Vite, styled with Tailwind in a dark C2 theme, rendering a Mapbox GL 3D terrain view overlaid with deck.gl layers (ground nodes, LoRa links, fire perimeters, smoke plumes, triangulation solutions, drone position). Global state is centralized in a Zustand store (`useIgnisStore.ts`) with types kept 1:1 with the backend's schemas to avoid drift between the two.

### 11.2 Live Telemetry Path

`useJetsonTelemetry.ts` polls the Jetson's embedded `web_stream.py` HTTP/JSON endpoint (`:8080/telemetry.json`) directly. This telemetry payload carries:

- **Reduced thermal grid:** the P3's native 256×192 radiometric frame is downsampled on the Jetson to a **16×12 `uint8` grid** (~192 bytes) via a pure-`numpy.reshape` max-pooling implementation (chosen specifically over OpenCV's `INTER_AREA`, which *averages* and would dissolve small, sharp thermal peaks that the max-pool preserves) before transmission — this is a direct, deliberate consequence of the "Zero WiFi in the Field" bandwidth doctrine (§7.1): even over the monitoring WiFi link, the system is engineered as if bandwidth were as scarce as the 433 MHz link, so the same compact representation is reused rather than maintained as two separate code paths. `ThermalHUD.tsx` reconstructs and renders this grid with a custom Inferno colormap.
- **Motor effort:** MAVLink `SERVO_OUTPUT_RAW` PWM values for all four rotors, clamped and normalized on the Jetson side (`_pwm_to_pct`, with saturation handling for anomalous ESC signals such as 0, 900, or 2100 µs) before being exposed as `motor_1_pct`…`motor_4_pct`, driving `AvionicsHUD.tsx`'s live motor bars.
- **`WAITING_AUTH` context:** target position, computed distance, and an authorization countdown timer, so the operator sees exactly what the FSM is waiting on and how much time remains before the request times out.

### 11.3 Human Authorization UI

`FlightAuthDialog.tsx` presents the mandatory pre-flight authorization prompt the FSM's `WAITING_AUTH` state blocks on; `AbortMissionDialog.tsx` implements the password-gated emergency-RTL control (§4.3). Both are the frontend-side embodiment of the Level 0/Level 2 human-authority requirements from §4 — the UI does not merely display these gates decoratively, it is the literal mechanism by which a human clears them.

### 11.4 A Debugging Postmortem Worth Recording

A white-screen-of-death (WSOD) observed during HITL testing was initially suspected to be a Jetson/network failure. Root-cause analysis instead traced it to a pure frontend bug: `useJetsonTelemetry.ts` failed to propagate `lat`/`lon` into `setPendingFlightAuth` during the FSM's transition into `WAITING_AUTH`, leaving React state incomplete and crashing the render tree. The lesson generalized into a standing engineering habit for this project: **when a robotics-adjacent UI misbehaves, rule out frontend state bugs before assuming hardware or connectivity is at fault** — a mixed embedded/web system makes it easy to reflexively blame the "exotic" hardware layer for what is, often, an ordinary rendering bug.

---

## 12. Concurrency Model

### 12.1 The Problem

Processing 16-bit radiometric matrices and running ML inference (the flicker classifier, or the heavier YOLO/morphological hybrid path) introduces non-deterministic latency (jitter) — a single frame can take anywhere from a few milliseconds to several hundred milliseconds depending on scene complexity and system load.

### 12.2 The Decision: Hard Thread Separation

MAVLink (flight telemetry) and computer-vision threads operate in **completely decoupled spaces**, synchronized via `RLock` where they must share state. Concretely: if a thermal frame takes 500 ms to process because of a transient GPU/CPU saturation spike, **this must never block the flight-control thread**. The Jetson continues reading battery and altitude at its own fixed rate regardless of what the vision pipeline is doing, so that life-or-death decisions (RTL, LAND) can never stall behind an inference bottleneck.

### 12.3 A Concrete Race Condition, Found and Fixed

A multi-agent code audit (Gemini as orchestrator, Claude Code performing the fix) identified a real race condition: without an explicit lock, concurrent web-request handler threads (serving the React dashboard's authorize/abort endpoints) could interleave with the FSM's own state-transition logic, creating a window where one thread's emergency abort could be clobbered by another thread's concurrent action. The fix was a `threading.RLock()` injected into `IgnisMissionFSM`, specifically chosen (re-entrant, not a plain `Lock`) because the FSM's own internal methods call each other recursively during a single logical transition, and a non-reentrant lock would have deadlocked the very thread trying to protect. A related, narrower race — two operators clicking "authorize" within the same millisecond — was assessed and consciously **accepted** rather than defended against with additional locking, because the Pixhawk itself de-duplicates redundant `GUIDED` target commands; the team judged the residual risk not worth the added complexity (documented as risk RA-01).

### 12.4 Concurrency Design Rule (Derived Lesson)

The project's own retrospective on this incident generalizes to a standing rule worth stating explicitly: **unifying safety-critical failsafe evaluation (battery, RC-mode watchdog) with the same execution context as ML inference is an anti-pattern.** If the ML model's latency varies, and the failsafe check shares its cycle, the drone's *reaction time to a real emergency* becomes coupled to inference jitter — which is precisely the coupling this architecture's hard thread separation exists to prevent.

---

## 13. Failure Detection, Isolation, and Recovery (FDIR)

Per the NASA/JPL-derived engineering doctrine (§14), every identified failure mode must specify **how it is detected, how it is isolated, and how it is recovered** — an unrecovered failure mode is treated as an open risk in the project's own tracking documents, not a shippable gap.

| Failure mode | Detection | Isolation | Recovery |
|---|---|---|---|
| Jetson freeze/crash | Pixhawk MAVLink heartbeat watchdog (2.0 s timeout) | Jetson has no direct actuation path — it can only *suggest* via MAVLink | Pixhawk holds current mode (`LOITER` or continues active waypoint queue) autonomously; operator commands RTL from ground |
| Vision pipeline stall (`ignis_vision` hang or exception) | Isolated by hard thread separation (§12) | Flight-control thread is architecturally decoupled and unaffected | Vision thread can be restarted independently; flight continues unaffected in the interim |
| LoRa link loss (ground mesh → drone) | Absence of expected heartbeat/`ALERT_NODE` traffic | Drone has no autonomous trigger source | Drone enters/remains in a predefined safe mode (patrol/RTL per mission policy) rather than improvising — per the explicit flight-rule doctrine (§14.2) |
| 433 MHz telemetry link loss (ground station ⇄ drone) | Pixhawk `FS_GCS_ENABLE` watchdog, >5 s silence | Route-update channel is severed | Pixhawk auto-RTL (§4.1); if mid-route-uplink, Jetson retains last fully-verified route rather than a partial one (§9.3) |
| Battery depletion during `WAITING_AUTH` wait | Continuous 5 Hz re-poll of live telemetry (§5.3) | Pending authorization is invalidated before it can be acted on | Alert automatically revoked; operator sees mission canceled, not a stale "go" button |
| RC mode change by safety pilot | Pixhawk-reported flight-mode polling in the FSM loop | FSM stops issuing guidance the instant a mode change is observed | Transition to `MANUAL_OVERRIDE`; pilot has full authority via Level 0 |
| Malformed/corrupted radiometric frame (`NaN`/`Inf`) | Explicit validation before it reaches the ML classifier | Corrupt matrix is rejected before feeding the model | Frame is dropped rather than risking an undefined classifier output or a process crash on a malformed input |
| Camera reconnect failure (`edge_hybrid_node.py` branch) | `FRAME_WATCHDOG_TIMEOUT_S` (5.0 s) with exponential backoff reconnect | `CameraManager` owns the sole open/close lifecycle point for the device | Circuit breaker after `MAX_CAMERA_RECONNECT_ATTEMPTS` (10); flight loop never exits on a transient camera fault, only on operator signal or a fatal-error ceiling |
| YOLO inference repeated failure (hybrid branch) | `YOLO_ERROR_TOLERANCE` (10 consecutive errors) | YOLO auto-disables itself | Node continues on the physics-only morphological pipeline — degraded but safe, per the `NodeHealth.DEGRADED` state |

---

## 14. Engineering Standard: NASA/JPL "Power of 10" Adaptation

IgnisEdge adopts an explicit, written engineering doctrine (adapted from Holzmann's "Power of Ten" rules for safety-critical flight software) rather than leaving code-quality expectations implicit. The rules, in full, as they govern this codebase and any agent or contributor working on it:

### 14.1 Governing Philosophy
- Safety first: when "optimal" and "safe" conflict, safe wins (safing).
- Design for the failure case, not just the happy path.

### 14.2 Flight Rules (decided in advance, not improvised under pressure)
- `EMERGENCY` has absolute priority and its path is **never** blocked by anything else.
- Every sequence (takeoff/land/RTL) evaluates its command queue non-blockingly on each iteration; if an `EMERGENCY` appears, it breaks immediately, flushes the queue, and executes safing.
- LoRa link loss ⇒ the drone enters a **predefined** safe mode; it does not improvise a response in the moment.

### 14.3 Code Standard (adaptation of the "Power of 10," Holzmann/JPL)
- Every loop has a demonstrable upper bound; no `while` without a guaranteed exit.
- Every relevant call's return value is checked, and parameter validity is verified.
- Every piece of data is declared in the smallest scope possible (avoids implicit global state).
- Unbounded recursion is forbidden on any control path.
- Functions are short and single-responsibility; if it doesn't fit on one screen, it gets split.
- Assertions at critical points validate state invariants.
- Zero warnings: static analysis (`ruff`/`pylint` + `mypy`) is mandatory before integration.

### 14.4 Failure Management (FDIR)
- Every identified failure mode must specify: how it is detected, how it is isolated, how it is recovered. No recovery path defined = an open risk that must be logged, not silently accepted.

### 14.5 Verification & Validation
- "Test as you fly, fly as you test": what is tested is what flies.
- FSM logic must be testable **without hardware** — model loading and drone connection are built as separable concerns specifically to preserve this property; see `edge_core/edge_core/tests/test_mission_fsm.py`, which exercises the full state machine with a mocked MAVLink layer.
- Any change to an FSM or to an emergency-response path requires a test covering it.

### 14.6 Configuration Management (baseline)
- The three-layer architecture, the shared-state model, and the FSMs are a **frozen baseline**: they are not altered without a recorded decision in ADR form (§17).

### 14.7 Rules for AI Coding Agents Working on This Repository
- Auditing = read-only + report. Do not edit code without an explicit request.
- Never weaken a safety path "to make it compile" or "to make it simpler."
- Always cite `file:line`. If a finding is uncertain, mark it "to verify."
- If a step is missing information, **ask** before assuming.

---

## 15. Hardware Stack

### 15.1 Edge Compute

| Component | Spec |
|---|---|
| Companion computer | Jetson Orin Nano 8GB, unified CPU+GPU RAM |
| OS | Ubuntu 22.04 LTS via JetPack 6 (L4T r36.3, kernel `5.15.136-tegra`, ARM64) |
| Runtime | Python 3.10 in venv; numpy, scipy, pandas, scikit-learn, joblib, OpenCV 4.5.4 (CUDA-enabled), pyusb, tifffile, imagecodecs |
| Production inference | **CPU-only** — no PyTorch, no GPU dependency for the flight classifier |

### 15.2 Thermal Camera

InfiRay Thermal Master P3 — 256×192 px, 4.3 mm focal length, 40°×30.2° FOV, 12 µm pixel pitch, ~40 mK NETD, manual focus. Dual gain: HIGH (−20…150 °C, patrol) / LOW (0…550 °C, confirmation). Conversion: `°C = raw/64 − 273.15` (15.6 mK quantization). **Not UVC** — InfiRay's vendor-specific protocol over libusb/pyusb (VID `0x3474`, PID `0x45a2`), via the `p3-ir-camera` driver.

### 15.3 Flight Controller

Pixhawk 6C — STM32H743 main MCU (ARM Cortex-M7, 480 MHz) + STM32F103 co-processor, redundant IMUs (ICM-42688-P + Bosch), integrated barometer/magnetometer, running ArduCopter. Navigation via `GUIDED` mode + `SET_POSITION_TARGET_GLOBAL_INT`; automatic RTL. All battery/GPS/link/geofence failsafes live in the Pixhawk, independent of the companion computer (§4).

**Jetson ⇄ Pixhawk link (ADR-10):** USB (`/dev/ttyACM0`, 115200 baud) for bench development; UART TELEM2 (`/dev/ttyTHS1`, 921600 baud, TX/RX/GND crossed) for flight, chosen specifically because UART is more robust to vibration and electrical noise in flight than USB. **Hard safety rule: 5V power is never connected between the two boards** — only TX/RX/GND.

**Terrain following:** JAXA ALOS 30/100 m elevation database (loaded via Mission Planner), `TERRAIN_ENABLE=1`, ~10 m typical accuracy (up to ~35 m on peaks), only meaningful above ~60 m AGL. This is explicitly **not obstacle avoidance** — trees and cables require proximity sensors, which this platform does not carry. A TF-Luna single-beam rangefinder has been evaluated for landing/descent use (not yet mounted); a YDLIDAR X2L was evaluated and rejected (indoor-oriented, sun-sensitive, wrong axis orientation for this use case).

### 15.4 Ground Sensor Nodes

Heltec WiFi LoRa 32 V3 (ESP32-S3 + SX1262, 915 MHz) + Bosch BME688 gas sensor. The BME688 detects combustion volatiles via a resistance drop (**not** PM2.5 particulate matter) — see §15.6 for the physical mechanism and its counterintuitive failure mode. Nodes are designed for duty-cycled, mostly-sleeping operation to survive months unattended in the field.

### 15.5 Airframe

Holybro S500 quadcopter, carbon-fiber X-frame, Pixhawk 6C integrated. Current platform is the bench/validation aircraft; a fixed-wing VTOL (DeltaQuad Evo) carrying a Workswell WIRIS Pro radiometric camera (640×512, ≤50 mK NETD) is identified as a future production-scale platform, not currently owned.

### 15.6 Hard-Won Hardware Lessons (worth preserving for future contributors)

- **BME688 gas-sensing physics is counterintuitive and safety-relevant.** Butane lighter gas (used to simulate smoke in bench tests) is expelled cold, which cools the sensor's internal 320 °C hotplate and causes its resistance reading to rise. Real combustion smoke, by contrast, reacts chemically with the hot plate and causes resistance to **fall** sharply (e.g., 150 kΩ → 30 kΩ). **Detection logic must trigger on resistance drops, not rises** — a naive implementation calibrated against lighter-gas bench tests would have the polarity of its trigger condition backwards for real fires.
- **ESP32-S3 JTAG pin conflict.** Pins 41/42 are frequently locked by JTAG debug fuses on this SoC family; I²C must be relocated to free GPIOs (SDA=4, SCL=5 in this project's wiring).
- **Adafruit_BME680 library constructor bug.** Passing the secondary I²C bus (`&Wire1`) to `begin()` is silently ignored; it must be injected in the object's constructor instead, or the library falls back to the wrong bus and fails.
- **RadioLib SX1262 half-duplex trap.** After a successful `readData()`, the radio drops to Standby and stops listening. `startReceive()` must be called again explicitly after every read, or the receiver goes permanently deaf after its first successfully received packet. Relatedly, using the blocking `radio.receive()` call in the main loop risks tripping the ESP32's watchdog timer during quiet periods, causing spurious reboots — the correct pattern is `startReceive()` once, then non-blocking `readData()` polling in the loop.
- **Heltec V3 power gating.** The OLED display, the LoRa radio, and any externally powered sensor share a power rail gated by the `Vext` pin (GPIO 36); it must be driven `LOW` early in `setup()` or the entire peripheral set stays dark despite correct wiring.
- **ESP32-S3 native USB CDC reconnect-on-flash.** After flashing, the board's hard reset causes the USB device to re-enumerate, silently reverting any `chmod`-granted serial permissions — `chmod 666` must be reapplied after flashing completes, not only before (or, durably, the user should be added to the `dialout` group / a udev rule installed).

---

## 16. Verification & Validation

### 16.1 Test Suites (edge_core)

`edge_core/edge_core/tests/` — five suites: FSM logic (`test_mission_fsm.py`), MAVLink layer with a mocked link (`test_mavlink_mock.py`), the COBS/CRC16 binary protocol (`test_proto_cobs.py`), the vision engine (`test_vision_engine.py`), and the embedded HTTP/JSON web endpoints (`test_web_endpoints.py`). The FSM suite specifically exists to satisfy §14.5's requirement that mission logic be testable without physical hardware — it exercises all eight `MissionState` transitions, including the `WAITING_AUTH` gate, the triple energy lock, and the manual-override watchdog, against a mocked MAVLink connection.

### 16.2 Vision Pipeline Validation

- `test_gain_invariant.py` — confirms `flicker_peak_C`'s AUC (0.988) and gain-invariance claim empirically, rather than by inspection (§6.6).
- `test_simetria.py` — the test that falsified the shape/symmetry feature hypothesis (§6.6), demonstrating the project's discipline of testing hypotheses to destruction rather than merely to confirmation.
- Session-grouped `GroupKFold` cross-validation is the standard evaluation protocol project-wide, specifically because frame-level random splits were shown to produce misleadingly optimistic metrics on the abandoned CNN (§6.5).

### 16.3 System-Level Targets (Quantified Success Criteria)

| Criterion | Target | Status |
|---|---|---|
| Fire-detection recall (`GroupKFold`) | > 0.97 | **Met** — 97.58% |
| False-positive rate | < 0.04 | **Met** — 3.74% |
| Onboard inference latency, p99 | < 80 ms at ≥ 20 fps | Bench-validated |
| Memory footprint | RSS < 1.5 GB | Bench-validated |
| SoC thermal | GPU/CPU < 80 °C sustained | Bench-validated |
| End-to-end HITL flight validation | Full sensor-to-verdict-to-RTL chain on physical hardware | **Pending** — blocked on 433 MHz telemetry antenna hardware arrival |

### 16.4 What Remains Outstanding for V&V

- **Real flight data** as a gold-standard validation set — all current metrics are bench/candle validated, not flight-validated.
- **Image registration** for flicker computation under camera motion (§6.9) — currently an open risk, not a solved problem.
- **Dilution augmentation** to extend reliable detection to warm/distant fire (60–130 °C at 100 m) — the current dataset's short-range calibration (§6.8) does not yet cover this regime.
- **Full HITL bench integration** of Jetson + Pixhawk + Heltec simultaneously, gated on the return of 433 MHz telemetry hardware.

---

## 17. Architecture Decision Records (ADR Index)

The full ADR text lives in the project's Obsidian knowledge vault (`00_Gobernanza_y_Reglas/01_ADR_Decisiones_Arquitectonicas.md`); this index summarizes each decision and its one-line justification for readers of this document.

| ADR | Decision | Core justification |
|---|---|---|
| **ADR-1** | AI runs on the edge (Jetson), not on the ground | LoRa cannot carry video (22-byte messages); perception and verdict must happen onboard, with only the compressed result transmitted |
| **ADR-2** | LoRa 915 MHz for the ground data plane | Multi-kilometer range at ~0.03 mAh/transmission — the only way ground nodes survive months unattended |
| **ADR-3** | No ROS2/MAVROS/DroneKit; pure `pymavlink` + Unix sockets + msgpack | Unjustified complexity for 3 processes on one board; keep the link to flight authority as simple and robust as possible |
| **ADR-4** | Physical-feature classifier (`HistGradientBoostingClassifier`), not a CNN | CNN memorized backgrounds under data scarcity; the alternative is sample-efficient, interpretable, CPU-only, and numerically stable |
| **ADR-5** | Flicker as the central discriminator | No static hot object reproduces fire's 1–15 Hz thermal oscillation; empirically the strongest single feature by a wide margin |
| **ADR-6** | Gain-invariant features only (`flicker_peak_C`); never re-record the dataset | Dataset is LOW-gain, flight is HIGH-gain; `T_peak_C`/`delta_peak_C` are gain-dependent and were empirically shown to break live inference |
| **ADR-7** | Discard shape/symmetry features | Hypothesis tested and falsified by data (AUC ≈ 0.5–0.57), not retained on intuition |
| **ADR-8** | Multi-layer conservative gating; never verdict from a single frame | A missed-response cost asymmetry (false positive dispatches resources; confirmation delay costs nothing) justifies a system that is slow to alarm and fast to stand down |
| **ADR-9** | Bench platform is a candle rig + Holybro S500, explicitly **not** a Tello | Production platform commitment; bench tests must generalize to the real airframe, not a toy proxy |
| **ADR-10** | USB for development, UART for flight (Jetson ⇄ Pixhawk) | UART is more robust to in-flight vibration/electrical noise than USB |
| **ADR-11** | Centralized aero-energetic route planning (C2) + dynamic in-flight re-routing over 433 MHz | Exploits the PC's compute/weather-API access vs. the Jetson's need to reserve compute for vision; link-loss fail-safe keeps the last verified route active |

---

## 18. Known Gaps, Open Risks, and Roadmap

This section is deliberately explicit about what is **not** yet true of the system, in keeping with the project's own doctrine (§14.4) that an unrecovered failure mode or an unclosed gap is a tracked risk, not a hidden one.

### 18.1 Structural / Cross-Cutting

- **Backend fragmentation risk (historical):** at various points in development, up to three separate ad-hoc HTTP servers have competed for port 8000 (`ignis-edge-backend`, a `bridge.py` prototype, a lightweight `dashboard.py`). The architectural intent is that `backend/` is the single unified service; ad-hoc bridge scripts should not accumulate as permanent parallel services.
- **Direct-to-Jetson-IP HTTP polling from the browser** (rather than routing all telemetry through the backend's WebSocket layer) has been flagged as an anti-pattern: it bypasses the C2 backend entirely and fails silently if the Jetson isn't running a compatible ad-hoc server, or if the operator's browser can't route to `192.168.0.9` in the field. The reduced-thermal-grid and motor-telemetry work (§11.2) partially addresses this by keeping payloads small, but the underlying "browser talks directly to the drone's IP" pattern remains a topology worth revisiting for a field-hardened deployment.
- **433 MHz Jetson-side software** for consuming `ROUTE_PACKAGE`/`ROUTE_UPDATE` (§9.4) is designed at the ADR level but not yet implemented in `ignis_mission.py`.
- **MQTT bridge (`mqtt_client.py`)** is the least-integrated backend module — not yet connected to the live Jetson/Heltec telemetry path.

### 18.2 Vision

- **Camera-motion-induced false flicker (§6.9)** — no image registration yet; the single highest-priority open item before untethered flight.
- **Warm/distant fire regime** (60–130 °C at 100 m) is under-represented in the current dataset relative to the short-range calibration captures.

### 18.3 Hardware Validation

- **Full physical HITL bench test** (Jetson + Pixhawk + Heltec + live LoRa-triggered mission, end-to-end to RTL) is **on hold pending the arrival of a 433 MHz telemetry antenna** — this is the single hardware blocker standing between the current software-complete state and a first live-hardware validation run.
- All flight-authority logic, FSM transitions, and telemetry plumbing described in this document are validated in software (unit tests, desk-based HITL with partial hardware) but await this full-stack physical confirmation.

### 18.4 Security / Credential Hygiene

- The Level-2 administrator password gating emergency abort/authorize endpoints is currently a hardcoded placeholder (`ignis2026`). This is acceptable for a bench-test PoC but must be replaced with a properly managed credential (environment-injected secret, rotated, not committed to source) before any deployment handling real flight authority outside a controlled lab.

### 18.5 Desktop-Only Bypass Discipline

As documented in §5.5, temporary safety-check bypasses (fixed-distance geofence injection, battery-check skip, GPS arming-check skip) have been used for desk testing without live GPS/battery hardware, and have each time been deliberately reverted. This pattern is workable but fragile — it depends on developer discipline rather than a structural guard. A more robust future direction would be an explicit, clearly-named `--desktop-test` flag that the production entrypoint refuses to accept, rather than inline code edits that must be remembered and undone by hand every session.

---

## 19. Competitive Positioning and Academic Context

### 19.1 Differentiator vs. Satellite-Based Detection

| | Detection resolution | Revisit cadence |
|---|---|---|
| Satellite-based (e.g., OroraTech, used in Chile by CONAF-adjacent services) | ~4×4 m per hotspot | ~6 hours |
| **IgnisEdge** | **~0.5–0.8 m** (5–8× finer) | Continuous active patrol |

This differential is the project's central competitive and scientific claim: IgnisEdge does not compete on coverage area (satellites win there trivially) but on **catching a fire while it is still small enough that a single aircraft's near-field radiometric resolution can distinguish it from the terrain around it** — precisely the phase of a fire's lifecycle where intervention is cheapest and most likely to succeed.

### 19.2 Academic and Programmatic Context

- **Program:** Ingeniería en Informática (Engineering degree), Universidad Técnica Federico Santa María (USM), Chile.
- **Format:** Undergraduate thesis (*Memoria de Título*), three-student team, each owning one architectural pillar (Pablo Silva — embedded vision/FSM/MAVLink/Edge AI; Camilo — LoRa mesh and ground nodes; Bastián — C2 dashboard and satellite fusion).
- **Technology Readiness Level:** TRL 3–4 — validated in laboratory and hardware-in-the-loop bench conditions with real hardware, pending full physical flight validation (§18.3).
- **External validation track:** the project competed in Fundación Copec-UC's "Aplica tu Idea" competition, whose jury includes Copec executives — Copec being ARAUCO's parent company and an OroraTech commercial partner, which is precisely why the sub-meter-vs-4-meter differentiator (§19.1) carries particular strategic weight in that context, beyond its purely technical merit.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **flicker (parpadeo)** | Temporal thermal oscillation of fire (1–15 Hz); channel `ch2` = per-pixel temporal standard deviation; feature `flicker_peak_C`. The system's central discriminator. |
| **physical gate** | Hard rule: no sustained flicker above `flicker_floor` ⇒ never classified as fire, regardless of temperature. |
| **ROI** | Region of Interest — the 64×64 crop around a detected hot spot. (Not "Return on Investment.") |
| **evidence_mask** | Bitfield in the `ALERT` message recording which independent signals (RAD/GEO/TMP/CNN/RGB/NODE/TRACK) contributed to a verdict. |
| **NUC** | Non-Uniformity Correction / shutter calibration of the thermal camera; forced on entering `ON_STATION`. |
| **log-odds** | Bayesian accumulation of per-focus evidence over time, with asymmetric hysteresis, producing a stable verdict. |
| **warm-up / grace period** | Initial frames during which the system only observes and emits no verdict, suppressing cold-start instability. |
| **sub-pixel dilution** | The effect by which a heat source smaller than one pixel's ground footprint reads a temperature averaged with the cooler background. |
| **SharedState** | Not a literal construct in this codebase — actual inter-process communication is Unix domain sockets + msgpack. |
| **Terrain Following** | Pixhawk functionality that maintains height-above-ground using a DEM; explicitly **not** obstacle avoidance. |
| **GUIDED** | ArduCopter flight mode in which the Jetson sends target GPS coordinates via MAVLink. |
| **RTL** | Return to Launch — automatic return to the takeoff point. |
| **GroupKFold** | Cross-validation scheme that never splits a single capture session across train/validation, preventing scene memorization from inflating metrics. |
| **FRP** | Fire Radiative Power — a field carried in the `ALERT` message, following the remote-sensing convention of Wooster et al. (2005). |
| **NADIR** | Straight-down viewing geometry; the drone patrols in nadir view. |
| **COBS** | Consistent Overhead Byte Stuffing — framing scheme ensuring the delimiter byte never appears inside a binary payload. |
| **CRC16** | 16-bit Cyclic Redundancy Check — integrity verification for LoRa messages. |
| **FLAME 3** | Public high-altitude thermal fire imagery dataset (Hopkins et al. 2024); used as high-altitude positive examples in the hybrid vision branch. |
| **BME688** | Bosch combustion-volatile gas sensor used in ground nodes; measures VOCs via resistance change, **not** PM2.5 particulate matter. |

---

## 21. Bibliography

- Giglio, L., Schroeder, W., & Justice, C. O. (2016). *The Collection 6 MODIS active fire detection algorithm and fire products.* Remote Sensing of Environment, 178. doi:10.1016/j.rse.2016.02.054
- Wooster, M. J., Roberts, G., Perry, G. L. W., & Kaufman, Y. J. (2005). *Retrieval of biomass combustion rates and totals from fire radiative power observations.* Journal of Geophysical Research: Atmospheres, 110, D24311. doi:10.1029/2005JD006318
- Hopkins, B., et al. (2024). *FLAME 3 Dataset: Unleashing the Power of Radiometric and Thermal Infrared Imagery for Wildfire Management.* arXiv:2412.02831
- Babrauskas, V. (2003). *Ignition Handbook.* — cellulosic combustion temperature thresholds used in the morphological classifier's physical constants.
- NWCG (2024). *Glossary of Wildland Fire Terminology* — terminology reference for spot fire, incipient stage, and canopy pre-heating.
- Holzmann, G. J. *The Power of Ten: Rules for Developing Safety-Critical Code* — basis for this project's adapted engineering standard (§14).

---

*This document supersedes the previous, condensed version of `ARCHITECTURE.md`. It is derived from and cross-referenced against the project's full Obsidian engineering vault (governance/ADRs, session logs, vision/ML knowledge base, hardware/comms references, and safety protocols) as well as the current state of the source tree, as of the vault's last recorded update (2026-09-24).*
