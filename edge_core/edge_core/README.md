# IgnisEdge Core (`edge_core`)

Módulo de producción para la integración de hardware a bordo: **Jetson Orin Nano**, **Pixhawk 6C** y **Heltec LoRa 915 MHz**.

> **Nota de seguridad:** La carpeta original `scripts/` se mantiene 100% intacta como respaldo. Todo el código de producción para las placas vive en esta subcarpeta aislada `edge_core/`.

---

## 1. Estructura de Paquetes

```
edge_core/
├── vision/
│   └── ignis_vision.py      # Motor de visión optimizado (OpenCV NEON, captura en hilo, modo headless, IPC)
├── flight/
│   └── ignis_mavlink.py     # Wrapper pymavlink puro para Pixhawk (GUIDED, telemetría, watchdog 2.0s)
├── comms/
│   ├── ignis_proto.py       # Serialización binaria: ALERT v1 (22B), TASK (18B), COBS + CRC16
│   └── ignis_link.py        # Proceso de enlace serial UART con módem Heltec (915 MHz)
├── mission/
│   └── ignis_mission.py     # FSM de misión autónoma (READY -> TRIAGE -> TRANSIT -> ON_STATION -> VERDICT -> RTL)
├── tests/
│   ├── test_proto_cobs.py      # 5 tests unitarios de protocolo LoRa y framing COBS/CRC16
│   ├── test_vision_engine.py   # 4 tests unitarios de VisionEngine y detección térmica
│   ├── test_mavlink_mock.py    # 4 tests unitarios de comandos MAVLink y watchdog
│   └── test_mission_fsm.py     # 3 tests unitarios del ciclo FSM completo (triage, GUIDED, RTL)
└── launcher.py              # Orquestador del sistema de vuelo (soporta --sim, --bench y --flight)
```

---

## 2. Validación Automatizada (100% Tests Pass)

Para verificar que todos los componentes funcionen a nivel de producción sin necesidad de tener el hardware físico conectado:

```bash
# Desde la raíz de IgnisEdge:
source venv/bin/activate
python3 -m unittest discover -s edge_core/tests -p "test_*.py"
```

Resultado verificado:
`Ran 16 tests in 0.719s, OK` (16 pruebas unitarias exitosas, cero fallos).

---

## 3. Conexión de Hardware y Cableado

### Jetson Orin Nano ↔ Pixhawk 6C (UART TELEM2)
* **Puerto Jetson:** `/dev/ttyTHS1`
* **Puerto Pixhawk:** `TELEM2` (baudrate: `921600`, `SERIAL2_PROTOCOL=2`)
* **Cableado:**
  * Jetson Pin 8 (TX) ──> Pixhawk Pin 3 (RX)
  * Jetson Pin 10 (RX) ──> Pixhawk Pin 2 (TX)
  * Jetson Pin 6 (GND) ──> Pixhawk Pin 6 (GND)
  * ⚠️ **REGLA DE SEGURIDAD CRÍTICA:** NUNCA conectar el pin de 5V entre ambas placas.

### Jetson Orin Nano ↔ Heltec LoRa (UART / USB)
* **Puerto:** `/dev/ttyUSB0` o UART dedicado (`115200` baud).

### Jetson Orin Nano ↔ Cámara Térmica P3
* **Conexión:** Cable USB directo.
* Regla udev activa (`99-p3-camera.rules`) para acceso sin `sudo`.

---

## 4. Modos de Ejecución

```bash
# 1. Modo Simulación (Software-in-the-loop sin placas):
python3 edge_core/launcher.py --sim

# 2. Modo Banco de Pruebas (con cámara P3 conectada y visualización en pantalla):
python3 edge_core/launcher.py --bench

# 3. Modo Vuelo Autónomo (100% headless para operación a bordo del dron):
python3 edge_core/launcher.py --flight
```
