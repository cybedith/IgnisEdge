# Configuración de Mando PC y Emergencias en QGroundControl
## Control Manual de Emergencia y Return-to-Home (RTL) para IgnisEdge

Este documento detalla la configuración del mando de PC (joystick/gamepad Xbox o PlayStation) conectado a QGroundControl vía la antena de telemetría de **433 MHz** para pilotaje manual de emergencia y cancelación inmediata de misiones con **Return-to-Home (RTL)**.

---

## 1. Arquitectura de Control de Emergencia

```text
[Piloto con Mando PC (USB)]
           │
     (Ejes + Botones)
           ▼
[Laptop con QGroundControl]
           │
(MAVLink MANUAL_CONTROL vía Antena 433 MHz)
           ▼
[Pixhawk 6C en el Dron]
   ├── Botón RTL: Cancela misión autónoma -> Retorna y aterriza en el punto de despegue
   ├── Botón LOITER: Frena el dron inmediatamente y lo deja flotando en el aire
   └── Sticks manuales: Control total de cabeceo, alabeo, giro y aceleración
```

---

## 2. Preparación del Sistema en el PC (Linux)

### A. Permisos para la Antena de 433 MHz
Para que Ubuntu permita que QGroundControl use la antena de radio sin errores:
```bash
# Añadir usuario al grupo dialout (acceso a puertos seriales USB)
sudo usermod -a -G dialout $USER

# Eliminar modemmanager (evita que el sistema confunda la antena 433 con un módem celular)
sudo apt-get remove modemmanager -y
```

### B. Descargar QGroundControl (si no lo tienes instalado)
```bash
cd ~
wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage
chmod +x QGroundControl.AppImage
./QGroundControl.AppImage
```

---

## 3. Configuración del Mando en QGroundControl

1. Conecta el mando (Xbox / PS4 / PS5 / Logitech F310) por cable USB o Bluetooth a la laptop.
2. Abre **QGroundControl**.
3. Haz clic en el ícono de **Engranaje (Vehicle Setup)** arriba a la izquierda.
4. Selecciona la pestaña **Joystick**.
5. Haz clic en **Calibrate** y sigue los pasos en pantalla moviendo las palancas en cruz y círculos.

### Mapeo de Palancas (Modo 2 Estándar):
* **Palanca Izquierda:**
  * Arriba / Abajo: **Throttle** (Subir / Bajar altitud).
  * Izquierda / Derecha: **Yaw** (Girar orientación del dron).
* **Palanca Derecha:**
  * Arriba / Abajo: **Pitch** (Avanzar / Retroceder).
  * Izquierda / Derecha: **Roll** (Desplazamiento lateral Izquierda / Derecha).

---

## 4. Mapeo de Botones de Emergencia (Asignación Recomendada)

En la sección **Button Assignment** de la pestaña Joystick en QGroundControl, asigna las siguientes funciones críticas a los botones físicos:

| Botón | Función en QGC | Descripción Operativa |
|---|---|---|
| **Botón B (o Círculo)** | **Flight Mode: RTL** | **CANCELAR MISIÓN Y RETORNO A CASA.** El dron sube a la altitud de seguridad, vuela en línea recta al punto de despegue y aterriza solo. |
| **Botón A (o Cruz)** | **Flight Mode: LOITER (o PosHold)** | **FRENO DE EMERGENCIA.** El dron frena en seco, mantiene posición GPS y altitud fija contra el viento. |
| **Botón Y (o Triángulo)** | **Flight Mode: GUIDED** | **DEVOLVER CONTROL A LA JETSON.** Reactiva el modo autónomo para que `ignis_mission` continúe con el patrullaje o verificación de fuego. |
| **Botón RB (o R1)** | **Flight Mode: LAND** | **ATERRIZAJE IN SITU.** Aterriza verticalmente donde está ahora mismo en caso de falla grave. |

---

## 5. Parámetros de Seguridad en el Pixhawk (ArduCopter)

En QGroundControl ve a **Parameters** y asegúrate de tener estos valores configurados:

### A. Failsafe por Pérdida de Enlace de Radio 433 MHz (`FS_GCS_ENABLE`)
Si la laptop se apaga, la antena se desconecta o el dron sale del rango de la antena de 433 MHz:
* **`FS_GCS_ENABLE = 1`** -> El Pixhawk activa **RTL automático** tras 5 segundos sin señal.

### B. Altitud de Seguridad de Retorno (`RTL_ALT`)
Para evitar que al volver a casa el dron choque contra árboles o postes:
* **`RTL_ALT = 3000`** (o 4000) -> 30 a 40 metros sobre el punto de despegue. Si está por debajo, sube primero antes de avanzar.

### C. Failsafe de Batería Crítica
* **`BATT_FS_LOW_ACT = 2`** (RTL al alcanzar batería baja).
* **`BATT_FS_CRT_ACT = 1`** (Aterrizaje inmediato si la batería llega al nivel crítico de seguridad).
