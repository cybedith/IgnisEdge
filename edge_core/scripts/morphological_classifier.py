"""
IgnisEdge - Morphological Thermal Threat Classifier (v4.0)
===========================================================

Pipeline morfológico del sistema híbrido de detección de incendios forestales.
Extrae características geométricas y radiométricas de regiones térmicamente
conectadas y las clasifica mediante reglas físicas explícitas, complementando
al detector neuronal (YOLOv8-seg) en el árbitro final.

Cambios clave respecto a v3.1
-----------------------------
1.  `CORE_TEMP_C` reducido de 250 °C a 180 °C.
    Justificación: la combustión celulósica superficial (papel, hojarasca,
    pasto seco) arde en ~200–260 °C (Babrauskas 2003, *Ignition Handbook*).
    Con el umbral anterior, un fuego incipiente de 220 °C tenía `core_ratio = 0`
    y era indistinguible morfológicamente de un charco cálido.

2.  Nueva **Rule 0.5 — INCIPIENT FIRE** (prioritaria, tras el filtro de ruido).
    Detecta regiones pequeñas (30–400 px) pero anómalamente calientes
    (> 200 °C) y las eleva a WILDFIRE/ORANGE (o RED si > 350 °C), *salvo* que
    la firma morfológica sea claramente artificial (uniforme + geométrica +
    sin halo). Esto cubre el caso FLAME 3 "early ignition" descrito por
    Hopkins et al. 2024 y el protocolo de *spot fire* del NWCG.

3.  **Rule 1** (small + circular → POINT_SOURCE) reescrita con *gate* de
    intensidad: solo descarta como fuente puntual si la temperatura máxima
    es moderada (< 200 °C). Sobre ese umbral, el control pasa a Rule 0.5.

4.  Añadidos umbrales auxiliares para el discriminador artificial/natural y
    documentación con referencias físicas en cada bloque.

Referencias
-----------
- Hopkins, B. et al. (2024). *FLAME 3: Aerial and Ground-Based Multi-Spectral
  Imagery Dataset for Wildfire Research*. Dataset doc.
- NWCG (2024). *Glossary of Wildland Fire, PMS 205*. Spot fire, incipient stage.
- Wooster, M. J. et al. (2005). *Retrieval of biomass combustion rates and
  totals from fire radiative power observations*. JGR Atmospheres 110, D24311.
- Babrauskas, V. (2003). *Ignition Handbook*. Fire Science Publishers.
- Giglio, L., Schroeder, W., Justice, C. O. (2016). *The collection 6 MODIS
  active fire detection algorithm and product*. RSE 178: 31–41.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import label as ndi_label, binary_propagation


# ---------------------------------------------------------------------------
# Physical thresholds (°C)
# ---------------------------------------------------------------------------
# Semilla y extensión para la histéresis radiométrica (ya aplicada aguas
# arriba; se replican aquí por autoconsistencia).
EXTEND_TEMP_C = 80.0      # Frontera convectiva mínima (aire caliente alrededor)
SEED_TEMP_C = 150.0       # Núcleo candidato a combustión activa

# CORE_TEMP_C: temperatura a partir de la cual un pixel se considera parte del
# "core" incandescente. Bajado a 180 °C para capturar combustión celulósica
# superficial (papel, pasto) — ver Babrauskas 2003, Tabla 6.3.
CORE_TEMP_C = 180.0

# INTENSE_TEMP_C: frontera superior típica de combustión sostenida de
# biomasa leñosa en llama abierta (Wooster et al. 2005).
INTENSE_TEMP_C = 400.0


# ---------------------------------------------------------------------------
# Morphological thresholds
# ---------------------------------------------------------------------------
MIN_REGION_AREA_PX = 30

# Fuegos incipientes: regiones pequeñas pero radiométricamente activas.
# Rango 30–400 px corresponde a ~0.25–3 m² a 40 m AGL con Thermal Master P3.
INCIPIENT_MIN_TEMP_C = 200.0      # Umbral de ignición celulósica mínima
INCIPIENT_RED_TEMP_C = 350.0      # Sobre este umbral: alarma RED
INCIPIENT_MAX_AREA_PX = 400

# Discriminador "fuente artificial": una región *parece* fabricada cuando
# combina (a) bajo gradiente térmico (superficie uniforme controlada),
# (b) alta regularidad geométrica (diseño ingenieril), (c) halo térmico
# despreciable (aislamiento térmico efectivo).
ARTIFICIAL_MAX_GRADIENT_C = 22.0
ARTIFICIAL_MIN_CIRCULARITY = 0.80
ARTIFICIAL_MAX_HALO_EXTENT = 0.35
ARTIFICIAL_MAX_TEMP_C = 260.0     # Por encima de esto es muy raro en electrodomésticos

# Point source (fuente puntual no-incipiente, ej. foco, brasero pequeño frío).
POINT_SOURCE_MAX_AREA = 200
POINT_SOURCE_MIN_CIRCULARITY = 0.75
POINT_SOURCE_MAX_TEMP_C = 200.0   # Gate de intensidad: sobre esto lo maneja Rule 0.5

# Extended hot (industrial: chimeneas, techos metálicos soleados).
EXTENDED_MIN_ASPECT = 4.0
EXTENDED_MIN_SOLIDITY = 0.85

# Heurísticas de wildfire establecido (llama confirmada + halo grande).
UNIFORM_MAX_GRADIENT_C = 30.0
HIGH_CORE_RATIO = 0.7
WILDFIRE_HALO_EXTENT = 1.5
WILDFIRE_HIGH_HALO = 1.8
WILDFIRE_HIGH_GRADIENT_C = 80.0


# ---------------------------------------------------------------------------
# Enums y dataclasses (API estable — no cambiar sin actualizar árbitro)
# ---------------------------------------------------------------------------
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


@dataclass
class RegionFeatures:
    region_id: int
    area_px: int
    bbox: tuple
    centroid: tuple
    max_temp_c: float
    mean_temp_c: float
    temp_gradient_c: float     # std de temperaturas en la región (proxy de turbulencia)
    perimeter: float
    circularity: float         # 4πA/P² ∈ [0,1], 1 = círculo perfecto
    aspect_ratio: float
    solidity: float            # A / A_convex_hull ∈ [0,1]
    core_ratio: float          # A_core / A_total
    halo_extent: float         # r_eq(total) / r_eq(core); >1 indica envolvente caliente
    n_subregions: int          # núcleos desconectados (wildfires multi-foco)


@dataclass
class Detection:
    features: RegionFeatures
    threat_class: ThreatClass
    alert_level: AlertLevel
    reasoning: str


# ---------------------------------------------------------------------------
# Helpers geométricos
# ---------------------------------------------------------------------------
def _compute_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _compute_solidity(contour) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def _equivalent_radius(area_px: float) -> float:
    """Radio del círculo con misma área (proxy de escala espacial)."""
    return float(np.sqrt(area_px / np.pi))


def _looks_artificial(f: RegionFeatures) -> bool:
    """
    Discriminador multi-criterio: ¿la firma térmica sugiere origen fabricado?

    Fuentes artificiales típicas (hornos, placas de inducción, luminarias)
    exhiben simultáneamente:
      - Superficie térmicamente uniforme (baja varianza, regulación activa)
      - Geometría regular (diseño ingenieril → alta circularidad)
      - Ausencia de gradiente núcleo/halo (ver nota abajo)
      - Temperatura máxima acotada (<260 °C en carcasas/ventanas típicas)

    Nota sobre "halo":
        Una placa de cocina uniformemente a 230 °C tiene `core_ratio ≈ 1.0`
        (toda la región está por encima de CORE_TEMP_C) y por tanto
        `halo_extent ≈ 1.0` — no hay anillo convectivo externo porque la
        regulación térmica del aparato impide transferencia difusiva al aire.
        Un fuego real, en cambio, genera una envolvente convectiva visible
        (halo_extent > 1.2) *porque* el combustible cede calor al entorno.
        Entonces "no tiene halo" se traduce en dos morfologías:
            (a) halo_extent pequeño (<0.35) — core pequeño relativo a región, o
            (b) core_ratio muy alto (>0.85) — región entera uniforme = placa.
        El OR entre ambas condiciones captura el caso (b) que rompía v3.1.

    La combustión natural viola *al menos una* de estas condiciones:
      - Turbulencia de llama → gradiente alto
      - Combustible irregular → geometría caótica
      - Transferencia convectiva/radiativa al ambiente → halo o core_ratio
        intermedio (ni sin halo, ni toda caliente)
    """
    uniform_temp = f.temp_gradient_c < ARTIFICIAL_MAX_GRADIENT_C
    regular_shape = f.circularity > ARTIFICIAL_MIN_CIRCULARITY
    bounded_temp = f.max_temp_c < ARTIFICIAL_MAX_TEMP_C
    # Caso (a) núcleo pequeño aislado  OR  caso (b) superficie toda-caliente.
    no_convective_halo = (
        f.halo_extent < ARTIFICIAL_MAX_HALO_EXTENT
        or f.core_ratio > 0.85
    )
    return uniform_temp and regular_shape and bounded_temp and no_convective_halo


# ---------------------------------------------------------------------------
# Extracción de features
# ---------------------------------------------------------------------------
def extract_region_features(
    celsius: np.ndarray,
    region_mask: np.ndarray,
    region_id: int = 0,
) -> Optional[RegionFeatures]:
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

    n_sub = 0
    if core_area > 0:
        _, n_sub = ndi_label(core_mask)

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


# ---------------------------------------------------------------------------
# Clasificador — cascada de reglas
# ---------------------------------------------------------------------------
def classify(f: RegionFeatures) -> tuple[ThreatClass, AlertLevel, str]:
    """
    Cascada ordenada por *prioridad operativa*:
      0.   Filtro de ruido (área mínima)
      0.5. Fuego incipiente (pequeño + caliente, con discriminador artificial)
      1.   Fuente puntual fría (pequeña + circular + tibia)
      2.   Caliente extendido (industrial, alargado sólido)
      3.   Fuente puntual uniforme (radiación homogénea)
      3.5. Llama pequeña contenida (estufa, vela, soldadura)
      4.   Wildfire multi-foco con halo grande
      5.   Wildfire con halo + gradiente alto
      *.   Ambiguo (requiere desempate por YOLO en el árbitro)
    """
    reasons = []

    # ---- Rule 0: noise filter --------------------------------------------
    if f.area_px < MIN_REGION_AREA_PX:
        return (ThreatClass.NOISE, AlertLevel.NONE,
                f"Area {f.area_px}px < {MIN_REGION_AREA_PX}")

    # ---- Rule 0.5: INCIPIENT FIRE (nueva, prioritaria) -------------------
    # Detecta el escenario crítico: región pequeña (30–400 px) pero con
    # temperatura inequívocamente pirolítica (>200 °C). Ningún electrodoméstico
    # doméstico opera establemente por encima de ~260 °C en superficie expuesta
    # (Babrauskas 2003), así que temperaturas mayores son casi siempre
    # combustión. Aún así, aplicamos el discriminador `_looks_artificial` para
    # blindar contra hornos industriales, placas halógenas, etc.
    if (f.max_temp_c > INCIPIENT_MIN_TEMP_C
            and MIN_REGION_AREA_PX <= f.area_px <= INCIPIENT_MAX_AREA_PX):

        if _looks_artificial(f):
            # Firma artificial confirmada — dejamos que las reglas posteriores
            # la etiqueten (normalmente como POINT_SOURCE).
            pass
        else:
            level = (AlertLevel.RED
                     if f.max_temp_c > INCIPIENT_RED_TEMP_C
                     else AlertLevel.ORANGE)
            return (ThreatClass.WILDFIRE, level,
                    f"Incipient fire: area={f.area_px}px "
                    f"max_t={f.max_temp_c:.0f}°C "
                    f"grad={f.temp_gradient_c:.1f}°C "
                    f"circ={f.circularity:.2f} "
                    f"halo={f.halo_extent:.2f} "
                    f"core_ratio={f.core_ratio:.2f}")

    # ---- Rule 1: small + circular + NOT hot = point source ---------------
    # Gate de intensidad añadido: si la temperatura máxima supera
    # POINT_SOURCE_MAX_TEMP_C (200 °C) no la clasificamos aquí; ese caso
    # ya lo habría capturado Rule 0.5 salvo que `_looks_artificial` fuera
    # True, en cuyo caso queremos que caiga como POINT_SOURCE abajo sin
    # recrear el chequeo.
    if (f.area_px < POINT_SOURCE_MAX_AREA
            and f.circularity > POINT_SOURCE_MIN_CIRCULARITY
            and f.max_temp_c < POINT_SOURCE_MAX_TEMP_C):
        reasons.append(f"area={f.area_px}<{POINT_SOURCE_MAX_AREA}")
        reasons.append(f"circ={f.circularity:.2f}>{POINT_SOURCE_MIN_CIRCULARITY}")
        reasons.append(f"max_t={f.max_temp_c:.0f}<{POINT_SOURCE_MAX_TEMP_C}")
        return (ThreatClass.POINT_SOURCE, AlertLevel.NONE,
                "Small circular cool: " + ", ".join(reasons))

    # Variante de Rule 1 para casos donde Rule 0.5 detectó artificial-looking:
    # región pequeña, caliente pero con firma de electrodoméstico.
    if (f.area_px < POINT_SOURCE_MAX_AREA
            and f.circularity > POINT_SOURCE_MIN_CIRCULARITY
            and _looks_artificial(f)):
        reasons.append(f"area={f.area_px}<{POINT_SOURCE_MAX_AREA}")
        reasons.append(f"circ={f.circularity:.2f}")
        reasons.append(f"grad={f.temp_gradient_c:.1f} (uniform)")
        reasons.append(f"halo={f.halo_extent:.2f} (none)")
        return (ThreatClass.POINT_SOURCE, AlertLevel.NONE,
                "Small circular artificial: " + ", ".join(reasons))

    # ---- Rule 2: elongated + solid = extended hot (industrial) -----------
    if f.aspect_ratio > EXTENDED_MIN_ASPECT and f.solidity > EXTENDED_MIN_SOLIDITY:
        reasons.append(f"aspect={f.aspect_ratio:.1f}>{EXTENDED_MIN_ASPECT}")
        reasons.append(f"solidity={f.solidity:.2f}>{EXTENDED_MIN_SOLIDITY}")
        return (ThreatClass.EXTENDED_HOT, AlertLevel.YELLOW,
                "Elongated solid: " + ", ".join(reasons))

    # ---- Rule 3: uniform temp + high core ratio = artificial -------------
    if f.temp_gradient_c < UNIFORM_MAX_GRADIENT_C and f.core_ratio > HIGH_CORE_RATIO:
        reasons.append(f"grad={f.temp_gradient_c:.1f}<{UNIFORM_MAX_GRADIENT_C}")
        reasons.append(f"core_ratio={f.core_ratio:.2f}>{HIGH_CORE_RATIO}")
        return (ThreatClass.POINT_SOURCE, AlertLevel.NONE,
                "Uniform high core: " + ", ".join(reasons))

    # ---- Rule 3.5: elongated flame with no halo + uniform temp -----------
    # Típico: llama de estufa de gas, vela, soldadura — alargada pero sin
    # halo convectivo (contenida por el quemador/mecha).
    if (f.halo_extent < 0.5
            and f.temp_gradient_c < 30.0
            and 2.0 < f.aspect_ratio < 4.0
            and f.area_px < 1000):
        reasons.append(f"halo_ext={f.halo_extent:.2f}<0.5")
        reasons.append(f"grad={f.temp_gradient_c:.1f}<30")
        reasons.append(f"aspect={f.aspect_ratio:.1f} in [2,4]")
        return (ThreatClass.POINT_SOURCE, AlertLevel.NONE,
                "Small flame source: " + ", ".join(reasons))

    # ---- Rule 4: large halo + multiple sub-cores = wildfire (strong) -----
    if f.halo_extent > WILDFIRE_HIGH_HALO and f.n_subregions >= 2:
        level = AlertLevel.RED if f.max_temp_c > INTENSE_TEMP_C else AlertLevel.ORANGE
        reasons.append(f"halo_ext={f.halo_extent:.2f}>{WILDFIRE_HIGH_HALO}")
        reasons.append(f"n_sub={f.n_subregions}>=2")
        reasons.append(f"max_t={f.max_temp_c:.0f}C")
        return (ThreatClass.WILDFIRE, level,
                "Strong wildfire pattern: " + ", ".join(reasons))

    # ---- Rule 5: moderate halo + high gradient = wildfire (moderate) -----
    if (f.halo_extent > WILDFIRE_HALO_EXTENT
            and f.temp_gradient_c > WILDFIRE_HIGH_GRADIENT_C):
        level = AlertLevel.RED if f.max_temp_c > INTENSE_TEMP_C else AlertLevel.ORANGE
        reasons.append(f"halo_ext={f.halo_extent:.2f}>{WILDFIRE_HALO_EXTENT}")
        reasons.append(f"grad={f.temp_gradient_c:.1f}>{WILDFIRE_HIGH_GRADIENT_C}")
        return (ThreatClass.WILDFIRE, level,
                "Wildfire pattern: " + ", ".join(reasons))

    # ---- Default: ambiguous — delegamos el desempate al árbitro ----------
    return (ThreatClass.AMBIGUOUS, AlertLevel.WHITE,
            f"No rule matched: area={f.area_px} "
            f"max_t={f.max_temp_c:.0f} "
            f"circ={f.circularity:.2f} "
            f"aspect={f.aspect_ratio:.1f} "
            f"grad={f.temp_gradient_c:.1f} "
            f"halo_ext={f.halo_extent:.2f} "
            f"n_sub={f.n_subregions}")


# ---------------------------------------------------------------------------
# Pipeline de alto nivel
# ---------------------------------------------------------------------------
def detect_and_classify(celsius: np.ndarray) -> list[Detection]:
    """
    Segmenta por histéresis (semilla 150 °C, extensión 80 °C) y clasifica
    cada región conectada. Devuelve lista de Detection compatible con el
    árbitro YOLO+morph.
    """
    seed = celsius >= SEED_TEMP_C
    extend = celsius >= EXTEND_TEMP_C
    if not seed.any():
        return []
    full_mask = binary_propagation(seed, mask=extend)

    labeled, n_regions = ndi_label(full_mask)
    if n_regions == 0:
        return []

    detections = []
    for region_id in range(1, n_regions + 1):
        region_mask = labeled == region_id
        feats = extract_region_features(celsius, region_mask, region_id)
        if feats is None:
            continue
        threat, alert, reason = classify(feats)
        detections.append(Detection(
            features=feats,
            threat_class=threat,
            alert_level=alert,
            reasoning=reason,
        ))
    return detections


# ---------------------------------------------------------------------------
# Self-test: fuego incipiente vs horno caliente (papel quemándose vs placa)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _scene_incipient_paper():
        """Papel de ~45 px, pico 226 °C, llama turbulenta."""
        img = np.full((128, 128), 25.0, dtype=np.float32)
        ys, xs = np.ogrid[:128, :128]
        # Núcleo irregular (papel arrugado ardiendo)
        cy, cx = 64, 64
        r2 = (ys - cy) ** 2 + (xs - cx) ** 2
        core = r2 < 25
        img[core] = 226.0 + np.random.RandomState(0).normal(0, 30, core.sum())
        # Halo convectivo tenue
        halo = (r2 >= 25) & (r2 < 80)
        img[halo] = 95.0 + np.random.RandomState(1).normal(0, 10, halo.sum())
        return img

    def _scene_oven_burner():
        """Placa de cocina: círculo perfecto, ~230 °C, uniforme, sin halo."""
        img = np.full((128, 128), 25.0, dtype=np.float32)
        ys, xs = np.ogrid[:128, :128]
        r2 = (ys - 64) ** 2 + (xs - 64) ** 2
        disk = r2 < 120
        img[disk] = 230.0 + np.random.RandomState(2).normal(0, 5, disk.sum())
        return img

    def _scene_large_wildfire():
        """Wildfire establecido: halo grande, múltiples núcleos, gradiente alto."""
        img = np.full((256, 256), 30.0, dtype=np.float32)
        ys, xs = np.ogrid[:256, :256]
        # Halo convectivo amplio
        r2 = (ys - 128) ** 2 + (xs - 128) ** 2
        halo = r2 < 6000
        img[halo] = 120.0 + np.random.RandomState(3).normal(0, 40, halo.sum())
        # Dos núcleos
        for (cy, cx) in [(118, 122), (140, 136)]:
            core = (ys - cy) ** 2 + (xs - cx) ** 2 < 200
            img[core] = 450.0 + np.random.RandomState(cy).normal(0, 50, core.sum())
        return img

    scenes = {
        "incipient paper (expected: WILDFIRE/ORANGE)": _scene_incipient_paper(),
        "oven burner (expected: POINT_SOURCE/NONE)":   _scene_oven_burner(),
        "large wildfire (expected: WILDFIRE/RED)":      _scene_large_wildfire(),
    }

    for name, img in scenes.items():
        print(f"\n=== {name} ===")
        dets = detect_and_classify(img)
        if not dets:
            print("  (no detections)")
        for d in dets:
            print(f"  {d.threat_class.value:14s} | {d.alert_level.value:6s} "
                  f"| area={d.features.area_px:4d}px "
                  f"max={d.features.max_temp_c:5.0f}°C "
                  f"grad={d.features.temp_gradient_c:5.1f} "
                  f"circ={d.features.circularity:.2f} "
                  f"halo={d.features.halo_extent:.2f}")
            print(f"    -> {d.reasoning}")