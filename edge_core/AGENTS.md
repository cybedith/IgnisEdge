# IgnisEdge — Agente Experto y Reglas del Workspace

Este workspace aloja el proyecto **IgnisEdge**: sistema de detección temprana de incendios forestales para la región del Biobío (Chile) basado en dron con visión térmica embebida e IA en el borde (Edge AI). Memoria para optar al título de Ingeniero en Informática en la Universidad Técnica Federico Santa María (USM).
- **Autor de visión embebida e IA:** Pablo Silva
- **Colaboradores:** Camilo (red LoRa multi-salto BME688), Bastián (dashboard y fusión satelital)
- **Profesor guía:** Jorge Portilla

---

## 1. Reglas de Oro Inviolables (NUNCA Romper)

1. **El Pixhawk 6C es la autoridad de vuelo:** La Jetson Orin Nano solo sugiere coordenadas y waypoints en modo GUIDED. Los failsafes del autopiloto (batería crítica, pérdida de enlace MAVLink, pérdida de GPS, geofence y override de RC) son estrictamente prioritarios y no negociables.
2. **Desacoplamiento de procesos:** La lógica de misión (`ignis_mission`) NUNCA vive en el proceso de visión (`ignis_vision`). Si el motor de visión o la cámara se reinician o caen, el control de vuelo no se ve afectado.
3. **Clasificador de producción dependiente de features invariantes:** Se utiliza ÚNICAMENTE `flicker_peak_C`. NUNCA introducir `delta_peak_C` ni `T_peak_C` en el clasificador de producción, ya que varían fuertemente con la ganancia de la cámara (LOW vs HIGH) e inducen falsos negativos en vuelo.
4. **Cero PyTorch en producción:** El modelo en vuelo es `HistGradientBoostingClassifier` de scikit-learn ejecutado en CPU, liviano y con baja huella de memoria.
5. **Cero ROS2, CERO MAVROS, CERO DroneKit:** La IPC entre procesos se realiza vía Unix domain sockets con serialización msgpack. El control de Pixhawk se efectúa con `pymavlink` puro.
6. **Validación estadística rigurosa:** La validación se hace SIEMPRE mediante `GroupKFold` agrupado por sesión de captura real. Nunca mezclar frames de una misma sesión en train y validación.
7. **Diseño conservador:** Un falso positivo (despachar brigadas innecesariamente) es inaceptable. El sistema prioriza estabilidad y cero falsos verdes sobre velocidad extrema.
8. **Seguridad eléctrica:** NUNCA conectar el pin de 5V entre Jetson y Pixhawk. La UART solo conecta TX, RX y GND cruzados.
9. **Registro Obligatorio en Bitácora y Handover de Obsidian:** Al concluir cualquier sesión de trabajo, hito técnico o resolución de errores, el agente DEBE actualizar obligatoriamente:
   - `/home/pablo-silva/Documentos/Obsidian_n1/BOVEDA IGNIS EDGE/IgnisEdge/01_Gestion_Estado_y_Roadmap/Bitacora_de_Avances.md`: agregar la entrada de la sesión con fecha, objetivo, lo realizado, aprendizajes y estado de cierre.
   - `/home/pablo-silva/Documentos/Obsidian_n1/BOVEDA IGNIS EDGE/IgnisEdge/01_Gestion_Estado_y_Roadmap/Handover_y_Estado_de_Sesion.md`: refrescar el estado del hardware y el próximo paso inmediato.

---

## 2. Jerarquía de Hardware y Procesos

| Nivel | Dispositivo | Función Principal | Comunicación / Protocolo |
|---|---|---|---|
| **Nivel 1** | Pixhawk 6C (ArduCopter) | Autoridad de vuelo, control de motores, estabilización y failsafes. | UART `/dev/ttyTHS1` a 921600 baud con Jetson (pymavlink). |
| **Nivel 2** | Jetson Orin Nano 8GB | Inteligencia a bordo, ejecución de los 3 procesos Python. | Unix domain sockets + msgpack en memoria local. |
| **Nivel 3** | Heltec LoRa 32 (915 MHz) | Módem de telemetría de largo alcance hacia red en tierra. | UART `/dev/ttyUSB0` o serial a 115200 baud (COBS + CRC16). |

### Los Tres Procesos en la Jetson
1. **`ignis_vision`**: Captura de cámara P3, detector contextual Etapa 1, tracker con período de gracia, compuerta de flicker y clasificación. Emite eventos de detección por socket.
2. **`ignis_mission`**: FSM de misión autónoma (`READY` -> `TRIAGE` -> `TRANSIT` -> `ON_STATION` -> `VERDICT` -> `RTL`). Evalúa presupuesto energético, comanda waypoints GUIDED y arma la alerta final.
3. **`ignis_link`**: Gestiona el framing LoRa (ALERT v1 de 22 bytes con `evidence_mask`, TASK de 18 bytes) hacia el módem Heltec.

---

## 3. Pipeline de Visión Térmica e IA

- **Cámara:** Thermal Master P3 (InfiRay, 256×192 px, 12 µm, FOV 40°×30.2°, focal 4.3 mm, NETD ~40 mK). Gain HIGH (-20..150 °C) para patrulla; Gain LOW (0..550 °C) para confirmación de fuego caliente. Conversión: `°C = raw/64 - 273.15`.
- **Discriminador físico:** El flicker temporal térmico (1–15 Hz) generado por la combustión y turbulencia de la llama.
- **Etapa 1:** Detección contextual mediante mediana espacial 11×11 sobre imagen reducida (~5 ms). Condición: `(delta > k*sigma) OR (T > t_abs)`. Dilatación binaria para consolidación de manchas.
- **5 Capas anti-falso-verde en vivo:**
  1. *Warm-up global:* Se descartan veredictos hasta llenar el búfer de flicker.
  2. *Detector rápido + dilatación:* Une manchas contiguas.
  3. *Tracker con período de gracia:* Exige al menos 20 frames de observación continua (`min_obs=20`).
  4. *Compuerta sostenida:* Exige flicker alto en 8 de los últimos 12 frames.
  5. *Log-odds e histéresis:* Encendido exigente (`on=6.0`), apagado protegido (`off=-1.5`).
- **Bug abierto conocido:** El desplazamiento angular de la cámara induce flicker aparente en bordes de alto contraste. La solución de raíz planificada es registro/estabilización de imagen en software.

---

## 4. Estructura del Código

- `edge_core/`: Paquete modular de producción integrado con tests automatizados:
  - `vision/ignis_vision.py`: Motor de visión headless con IPC.
  - `flight/ignis_mavlink.py`: Manejador de vuelo MAVLink puro con watchdog de 2.0s.
  - `comms/ignis_proto.py` e `ignis_link.py`: Protocolo binario LoRa con COBS/CRC16.
  - `mission/ignis_mission.py`: Máquina de estados finitos y triage.
  - `launcher.py`: Orquestador configurable (`--sim`, `--bench`, `--flight`).
  - `tests/`: 16 pruebas unitarias (`python3 -m unittest discover -s edge_core/tests -p "test_*.py"`).
- `scripts/`: Scripts originales de investigación y calibración (`live_classify.py`, `harvest_crops.py`, `build_dataset.py`, `train_feature_classifier.py`, `capture_p3.py`).
- `venv/`: Entorno virtual Python con dependencias instaladas.
