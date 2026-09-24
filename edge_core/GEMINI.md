# IgnisEdge — Reglas activas del proyecto

Estás trabajando en **IgnisEdge**: sistema de detección temprana de incendios forestales basado en dron con visión térmica embebida e IA en el borde. Tesis de Ingeniería en Informática, USM Chile. Autor del subsistema de visión: Pablo Silva.

Para contexto técnico completo activa el skill `ignis-edge`.

## Reglas de oro (NUNCA violar)

1. El **Pixhawk es la autoridad de vuelo**. La Jetson solo sugiere. Failsafes del Pixhawk son inviolables.
2. La lógica de misión NUNCA vive en el proceso de visión. Si `ignis_vision` se cuelga, no arrastra el control de vuelo.
3. El clasificador de producción usa **SOLO `flicker_peak_C`**. NO usar `delta_peak_C` ni `T_peak_C` (dependientes del gain → falsos negativos en vuelo).
4. **Sin PyTorch en producción.** `HistGradientBoostingClassifier` en CPU.
5. **Sin ROS2, sin MAVROS, sin DroneKit.** pymavlink puro + Unix sockets + msgpack.
6. Validación **siempre GroupKFold por sesión.** Nunca mezclar sesiones entre train y val.
7. Sistema **conservador**: un falso positivo (mandar brigadas al vado) es más caro que tardar.
8. **NUNCA conectar 5V entre Jetson y Pixhawk.** Solo TX/RX/GND cruzados.
9. **Registro Obligatorio en Obsidian:** Al concluir cada sesión de trabajo, el agente DEBE actualizar obligatoriamente `Bitacora_de_Avances.md` y `Handover_y_Estado_de_Sesion.md` en `BOVEDA IGNIS EDGE`.

## Stack tecnológico

- Python en venv. Dependencias: numpy, scipy, pandas, scikit-learn, joblib, opencv, pyusb, tifffile, imagecodecs.
- Cámara: Thermal Master P3 (InfiRay, 256×192, VID 0x3474 / PID 0x45a2). Driver: `p3-ir-camera`. API: `P3Camera`, `raw_to_celsius`, `GainMode`.
- Conversión raw→°C: `°C = raw/64 - 273.15`.
- Jetson: Ubuntu via JetPack. Procesos: `ignis_vision` (encarnado en `live_classify.py`), `ignis_mission` (DISEÑO), `ignis_link` (DISEÑO).
- Pixhawk 6C (ArduCopter). Comunicación: UART `/dev/ttyTHS1`, pymavlink, 921600 baud.

## Archivos del núcleo

- `live_classify.py` — Detector en vivo de producción (CONSTRUIDO)
- `train_feature_classifier.py` — Entrenador del clasificador (CONSTRUIDO)
- `harvest_crops.py` — Etapa 1 + extracción de features (CONSTRUIDO)
- `build_dataset.py` — Dataset maestro (CONSTRUIDO)
- `capture_p3.py` — Captura del P3 (CONSTRUIDO)
- `ignis_fire_classifier.joblib` + `.json` — Modelo de producción

## Bug crítico abierto

Mover la cámara genera flicker falso → cuadros fantasma. Solución de raíz: **registro de imagen** en `ignis_vision`. NO construido. El sistema HOY asume cámara estable (banco).
