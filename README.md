# IgnisEdge 🌲🚁🔥

Sistema de detección temprana de incendios forestales basado en Edge AI (Jetson Orin Nano + Cámara Térmica P3) a bordo de un dron (Pixhawk), validado por un enjambre terrestre de nodos LoRa (BME688).

Este monorepositorio unifica los 3 pilares arquitectónicos del proyecto de tesis.

## 🏗️ Arquitectura del Proyecto

El código está dividido en 3 grandes directorios:

1. **`edge_core/` (El Cerebro del Dron)**
   - **Lenguaje:** Python (Jetson Orin Nano).
   - **Misión:** Máquina de Estados Finita (FSM) que controla el dron. Escucha el módem Heltec (LoRa) por el puerto serial. Cuando un nodo terrestre detecta humo, la Jetson valida la distancia (FDIR Nivel 2), pide autorización al operador humano, y comanda el vuelo vía MAVLink hacia el incendio.
   - **Visión Térmica:** Durante el vuelo, procesa el feed de la cámara radiométrica P3 utilizando un modelo HistGradientBoostingClassifier entrenado con el dataset FLAME 3. Evalúa la varianza temporal de los pixeles calientes (`flicker_peak_C`) para discriminar entre fuego real y superficies calientes estáticas.

2. **`backend/` (FastAPI Cloud/C2)**
   - **Lenguaje:** Python (FastAPI).
   - **Misión:** Sirve de puente entre la nube (Google Earth Engine) y el frontend. Genera la topografía y orografía de la región de patrullaje (ej. Hualpén) en 3D para el gemelo digital.

3. **`frontend/` (React C2 Dashboard)**
   - **Lenguaje:** TypeScript, React, Vite, Mapbox / DeckGL.
   - **Misión:** Consola de Comando y Control (C2). Muestra el gemelo digital en 3D, la telemetría en tiempo real, el feed térmico de la cámara y permite al operador autorizar vuelos cuando la máquina de estados lo solicita.

## 🚀 Cómo Ejecutar el Entorno (Pruebas Locales)

Tus compañeros pueden replicar el entorno de desarrollo siguiendo estos pasos en sus PCs:

### 1. Iniciar el Backend (Topografía 3D)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

### 2. Iniciar el Frontend (Dashboard C2)
```bash
cd frontend
npm install
npm run dev
```

### 3. Iniciar el Simulador Edge (Opcional - Pruebas HITL)
Si desean correr la lógica del dron sin tener el dron físico:
```bash
cd edge_core
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Requiere conectar un Pixhawk por USB o correr SITL ArduPilot
python tools/web_stream.py --pixhawk-port /dev/ttyACM0
```

## 🔒 Notas de Seguridad de Vuelo
- La FSM (`ignis_mission.py`) posee un "Candado Energético Continuo" y una geocerca dinámica (300 metros por defecto).
- Si en una prueba de escritorio (sin señal GPS) el dron rechaza las alertas de los nodos, **es un comportamiento deseado**, ya que la distancia calculada a Null Island (0,0) superará el radio seguro.
- Para pruebas HITL en escritorio, se debe inyectar temporalmente un `GPS_GLOBAL_ORIGIN` falso en el Pixhawk, o salir a cielo abierto para tomar fix 3D real.

---
*Desarrollado para la validación de concepto (PoC) de combate de incendios forestales de próxima generación.*
