# IgnisEdge: Decisiones de Arquitectura y Diseño (ADR)

Este documento detalla las justificaciones técnicas y académicas detrás de las decisiones de arquitectura del sistema IgnisEdge, diseñado para operar en entornos de misión crítica.

## 1. Visión Artificial: ¿Por qué Análisis de Varianza Temporal (Flicker) en lugar de CNNs Estáticas?
Uno de los mayores desafíos en la visión térmica forestal son los "falsos positivos" generados por superficies estáticas sobrecalentadas (rocas, techos metálicos en verano, etc.).
* **Decisión:** En lugar de depender exclusivamente de Redes Neuronales Convolucionales (CNN) evaluando firmas espaciales en *frames* aislados, IgnisEdge incorpora un análisis temporal.
* **Justificación Matemática:** El fuego posee un comportamiento termodinámico errático (parpadeo). El sistema calcula la Transformada Rápida de Fourier (FFT) y la varianza temporal (`flicker_peak_C`) a lo largo de un historial de *frames* (memoria de estado temporal en `ignis_vision.py`). Si un objeto está muy caliente pero su firma térmica es estática, el clasificador HistGradientBoosting lo descarta por falta de *flicker*, logrando inmunidad frente al ruido térmico estacional.

## 2. Autoridad de Vuelo y Seguridad (Doctrina FDIR Nivel 2)
El dron opera bajo una estricta jerarquía de autoridad donde el Edge Computer (Jetson) es un "cerebro pasivo/sugestivo", mientras que el Autopiloto (Pixhawk) es el "tronco encefálico dictatorial".
* **Geocerca Dinámica:** La Máquina de Estados Finitos (FSM) calcula la distancia de Haversine entre el objetivo reportado por el nodo LoRa y la posición actual del GPS. Si supera los 300 metros, la misión se aborta instantáneamente (Failsafe).
* **Candado Energético Continuo:** La viabilidad energética no se evalúa una sola vez al recibir la alerta. Se evalúa de forma asíncrona a 5Hz durante todo el ciclo de `WAITING_AUTH` (espera del humano). Si durante esos segundos la batería física cruza el umbral de reserva, el sistema revoca la alerta automáticamente sin intervención del operador.

## 3. Concurrencia Asíncrona (Separación de Hilos Críticos)
Procesar matrices de 16-bits radiométricos y modelos de Machine Learning (YOLO/HGB) introduce un delay no determinista (jitter).
* **Decisión:** Los hilos de MAVLink (Telemetría de Vuelo) y Visión Artificial operan en espacios completamente desacoplados mediante Locks (`RLock`).
* **Justificación:** Si un *frame* térmico tarda 500ms en procesarse debido a saturación en la GPU/CPU, esto **no bloquea** el hilo de control de vuelo. La Jetson sigue leyendo la batería y la altitud a 10Hz, asegurando que las decisiones de vida o muerte (RTL, Land) nunca se congelen por un cuello de botella de inferencia de IA.

## 4. Comunicaciones: Doctrina "Cero WiFi en Campo"
Para misiones forestales remotas, la cobertura celular o WiFi es inexistente o poco confiable.
* **Decisión:** Toda la red de sensores (Swarm Terrestre) y el dron están vinculados mediante telemetría LoRa (915MHz / 433MHz) conectada directamente al puerto serial (`/dev/ttyACM1` vía módem Heltec). El protocolo MAVLink y los `AlertNodeMessage` transitan íntegramente por este enlace físico de ultra-largo alcance.
* **El Rol del WiFi/C2:** La conexión HTTP al servidor FastAPI / React (Dashboard) es estrictamente una capa de monitoreo *Comando y Control (C2)*. Si se pierde la conexión WiFi o la pantalla de React colapsa, la FSM en el Edge continúa operando el dron autónomamente basándose en sus lecturas LoRa locales.
