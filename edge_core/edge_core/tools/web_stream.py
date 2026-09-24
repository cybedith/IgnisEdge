#!/usr/bin/env python3
"""
web_stream.py -- Servidor Web de Monitoreo Térmico y Control de Misión Autónoma por Coordenadas GPS.
================================================================================================

Levanta un centro de operaciones en la Jetson Orin Nano accesible por WiFi:
  1. Streaming de video térmico calibrado en paleta Inferno (15-25 fps).
  2. Telemetría completa del Pixhawk (Batería, Modo, Altitud, Coordenadas GPS).
  3. Misión Autónoma basada en DOS Coordenadas GPS:
     - Coordenada A: Posición del Dron (Origen / Base)
     - Coordenada B: Objetivo / Foco a Inspeccionar
  4. Monitoreo del cerebro FSM y veredicto de fuego en tiempo real.
  5. Difusión de mensajes STATUSTEXT vía MAVLink hacia la radio 433 MHz (Mission Planner).
  6. Grabación selectiva y asíncrona de datasets (.npy, .jpg, .json).

Acceso desde cualquier navegador en la red local (WiFi/LAN):
  http://<IP_LAN_JETSON>:8080   (ej. http://192.168.0.9:8080)

NOTA -- "Modo WiFi" de validacion HITL en banco:
  Este servidor SIEMPRE ha sido un dashboard WiFi/LAN (bind 0.0.0.0 + CORS
  abierto); es independiente del enlace de radio 433 MHz MAVLink, que solo
  transporta STATUSTEXT y ROUTE_PACKAGE/ROUTE_UPDATE entre la Jetson y la
  Central C2 (ver broadcast_message() y edge_core.flight.ignis_mavlink).
  Usar WiFi aqui para pruebas de banco sin antena 433 MHz NO requiere
  revertir nada en este archivo: basta con reconectar la radio de
  telemetria 433 MHz cuando este disponible, ya que ese enlace vive fuera
  de este servidor HTTP. Ver doctrina: BOVEDA IGNIS EDGE ->
  04_Seguridad_Aeronautica_y_Operaciones/Simulacion_HITL_y_Telemetria_Ligera.md
  (Seccion 2: "Cero WiFi en Campo" aplica al despliegue forestal real, no
  al banco HITL en laboratorio).
"""

from __future__ import annotations
from pymavlink import mavutil

import argparse
import io
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_core.comms.ignis_link import IgnisLink
from edge_core.flight.ignis_mavlink import IgnisMavlink
from edge_core.mission.ignis_mission import IgnisMissionFSM, MissionState
from edge_core.vision.dataset_recorder import DatasetRecorder
from edge_core.vision.ignis_vision import P3CaptureThread, VisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [web_stream] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_stream")

# Frame inicial placeholder
_init_canvas = np.zeros((384, 512, 3), dtype=np.uint8)
cv2.putText(_init_canvas, "IGNIS-EDGE STREAMING...", (90, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 140, 0), 2)
_, _init_encoded = cv2.imencode(".jpg", _init_canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])

# Variables globales de streaming y control
latest_jpeg_frame: bytes = _init_encoded.tobytes()
latest_telemetry_json: str = "{}"
recent_radio_messages: deque = deque(maxlen=10)
frame_lock = threading.Lock()
fsm_instance: Optional[IgnisMissionFSM] = None
mavlink_instance: Optional[IgnisMavlink] = None
recorder_instance: Optional[DatasetRecorder] = None
latest_raw_frame: Optional[np.ndarray] = None
latest_verdict: Optional[Any] = None


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia en metros entre dos coordenadas geográficas."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula el rumbo / azimut inicial en grados (0..360) de (lat1, lon1) hacia (lat2, lon2)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def broadcast_message(text: str, severity: int = 6) -> None:
    """Registra mensaje para la web y lo envía por radio 433 MHz vía MAVLink."""
    ts = time.strftime("%H:%M:%S")
    recent_radio_messages.append(f"[{ts}] {text}")
    if mavlink_instance:
        mavlink_instance.send_statustext(text, severity=severity)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IgnisEdge -- Misión de Dos Coordenadas y Visión Térmica</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #151d30;
      --border: #23314f;
      --accent: #f97316;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --green: #22c55e;
      --red: #ef4444;
      --blue: #38bdf8;
    }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    header {
      width: 100%;
      max-width: 1100px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
      margin-bottom: 16px;
    }
    h1 { margin: 0; font-size: 1.3rem; color: var(--accent); display: flex; align-items: center; gap: 8px; }
    .badge {
      background-color: var(--green);
      color: #000;
      padding: 4px 12px;
      border-radius: 9999px;
      font-weight: bold;
      font-size: 0.8rem;
    }
    .layout-grid {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
      width: 100%;
      max-width: 1100px;
    }
    @media (max-width: 890px) {
      .layout-grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .video-container {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      border: 1px solid var(--border);
      aspect-ratio: 4 / 3;
    }
    .video-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .hud-overlay {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      padding: 6px 12px;
      border-radius: 6px;
      font-family: monospace;
      font-size: 0.85rem;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .stat-box {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .stat-label { font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 2px; }
    .stat-val { font-size: 1.35rem; font-weight: bold; }
    .fire-alert {
      background: var(--red) !important;
      color: #fff !important;
      animation: pulse 1s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.8; }
    }
    /* Control Panel */
    .control-panel { display: flex; flex-direction: column; gap: 14px; }
    .section-title { font-size: 0.88rem; font-weight: bold; color: var(--accent); margin-bottom: 8px; border-bottom: 1px solid var(--border); padding-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }
    button {
      background: #2563eb;
      color: white;
      border: none;
      padding: 9px 12px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      transition: background 0.15s, transform 0.05s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    button:hover { background: #1d4ed8; }
    button:active { transform: scale(0.98); }
    button.btn-success { background: #16a34a; }
    button.btn-success:hover { background: #15803d; }
    button.btn-warning { background: #d97706; }
    button.btn-warning:hover { background: #b45309; }
    button.btn-danger { background: #dc2626; }
    button.btn-danger:hover { background: #b91c1c; }
    button.btn-muted { background: #334155; }
    button.btn-muted:hover { background: #475569; }
    .coord-block {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 8px;
    }
    .coord-title {
      font-size: 0.78rem;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .input-grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    input {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border);
      color: #fff;
      padding: 7px 9px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-family: monospace;
    }
    input:focus { outline: none; border-color: var(--accent); }
    .calc-box {
      background: #060911;
      border: 1px solid #1e3a8a;
      border-radius: 6px;
      padding: 8px 10px;
      font-family: monospace;
      font-size: 0.8rem;
      color: #7dd3fc;
      margin-bottom: 10px;
      line-height: 1.4;
    }
    .radio-log-box {
      background: #060911;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      height: 100px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 0.74rem;
      color: #38bdf8;
      line-height: 1.35;
    }
    .fsm-state-badge {
      padding: 4px 8px;
      border-radius: 6px;
      font-weight: bold;
      display: inline-block;
    }
  </style>
</head>
<body>
  <header>
    <h1>🔥 IgnisEdge — Misión por Coordenadas & Visión Térmica</h1>
    <span class="badge" id="conn-badge">CONECTADO</span>
  </header>

  <div class="layout-grid">
    <!-- Columna Izquierda: Video Térmico + Telemetría -->
    <div style="display: flex; flex-direction: column; gap: 14px;">
      <div class="video-container">
        <img id="thermal-feed" src="/snapshot.jpg" alt="Video Térmico en Vivo (P3)">
        <div class="hud-overlay" id="hud-overlay">CÁMARA: INICIANDO...</div>
      </div>

      <div style="display: flex; gap: 10px; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border);">
        <button class="btn-muted" style="font-size: 0.78rem; padding: 6px 12px;" onclick="captureManualDataset()">
          📸 Capturar Dataset Manual
        </button>
        <span style="font-size: 0.85rem; color: var(--text-muted);">
          📁 Muestras Dataset: <strong id="dataset-count" style="color: var(--accent); font-size: 1.1rem;">0</strong>
        </span>
      </div>

      <div class="stat-grid">
        <div class="stat-box" id="fire-box">
          <div class="stat-label">Detección de Fuego</div>
          <div class="stat-val" id="fire-val">SIN FUEGO</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Temperatura Máxima / FPS</div>
          <div class="stat-val"><span id="temp-val">--</span> °C <span style="font-size: 0.9rem; color: var(--text-muted);">(<span id="fps-val">--</span> fps)</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Batería Dron / Modo Pixhawk</div>
          <div class="stat-val"><span id="bat-val">--</span>% <span style="font-size: 0.95rem; color: var(--blue);">[<span id="mode-val">--</span>]</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Altitud Relativa / Satélites</div>
          <div class="stat-val"><span id="alt-val">--</span> m <span style="font-size: 0.85rem; color: var(--text-muted);">(Sats: <span id="sats-val">--</span>)</span></div>
        </div>
      </div>
    </div>

    <!-- Columna Derecha: Centro de Mando por Dos Coordenadas -->
    <div class="card control-panel">
      <div>
        <div class="section-title">
          <span>ESTADO DE MISIÓN (FSM)</span>
          <span class="fsm-state-badge" id="fsm-badge" style="background: #334155; color: #fff;">READY</span>
        </div>
      </div>

      <!-- SECCIÓN: DEFINICIÓN DE DOS COORDENADAS GPS -->
      <div>
        <div class="section-title">
          <span>📍 MISIÓN DE 2 COORDENADAS</span>
        </div>

        <!-- Coordenada A: Posición del Dron -->
        <div class="coord-block">
          <div class="coord-title" style="color: var(--blue);">
            <span>1. PUNTO A (Dron / Origen):</span>
            <button class="btn-muted" style="padding: 2px 6px; font-size: 0.7rem;" onclick="copyPixhawkGpsToA()">
              📡 Copiar GPS
            </button>
          </div>
          <div class="input-grid-2">
            <input type="number" step="0.000001" id="lat-a" placeholder="Latitud A" value="-36.820100" oninput="recalcGeodesic()">
            <input type="number" step="0.000001" id="lon-a" placeholder="Longitud A" value="-73.044100" oninput="recalcGeodesic()">
          </div>
        </div>

        <!-- Coordenada B: Objetivo a Inspeccionar -->
        <div class="coord-block">
          <div class="coord-title" style="color: var(--accent);">
            <span>2. PUNTO B (Objetivo / Foco):</span>
          </div>
          <div class="input-grid-2">
            <input type="number" step="0.000001" id="lat-b" placeholder="Latitud B" value="-36.820220" oninput="recalcGeodesic()">
            <input type="number" step="0.000001" id="lon-b" placeholder="Longitud B" value="-73.044150" oninput="recalcGeodesic()">
          </div>
        </div>

        <!-- Radio de Seguridad -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <span style="font-size: 0.78rem; color: var(--text-muted);">Radio máx de seguridad:</span>
          <div style="display: flex; align-items: center; gap: 4px;">
            <input type="number" step="20" id="max-radius" value="300" style="width: 65px; text-align: center;" oninput="recalcGeodesic()">
            <span style="font-size: 0.78rem;">m</span>
          </div>
        </div>

        <!-- Altitud de Inspección -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <span style="font-size: 0.78rem; color: var(--text-muted);">Altitud de inspección:</span>
          <div style="display: flex; align-items: center; gap: 4px;">
            <input type="number" step="1" id="alt-target" value="8" style="width: 55px; text-align: center;">
            <span style="font-size: 0.78rem;">m</span>
          </div>
        </div>

        <!-- Caja de cálculo en tiempo real -->
        <div class="calc-box" id="calc-info">
          Distancia A ➔ B: -- m | Rumbo: --°
        </div>

        <!-- Botón de Envío -->
        <button class="btn-success" style="width: 100%; font-size: 0.95rem; font-weight: bold; margin-bottom: 8px;" onclick="dispatchMissionAB()">
          🚀 Despachar Dron de A hacia B
        </button>

        <!-- Botones de Interrupción y Seguridad -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <button class="btn-warning" onclick="sendAction('loiter')">🛑 LOITER (Frenar)</button>
          <button class="btn-danger" onclick="sendAction('rtl')">🏠 RTL (Volver a A)</button>
        </div>
      </div>

      <!-- SECCIÓN: COMPUERTA DE AUTORIZACIÓN HUMANA (WAITING_AUTH) -->
      <div id="auth-panel" style="display: none;">
        <div class="section-title">
          <span>🔐 AUTORIZACIÓN DE VUELO REQUERIDA</span>
          <span id="auth-countdown" style="font-size: 0.78rem; color: var(--red); font-weight: bold;">--s</span>
        </div>
        <div class="coord-block" style="border-color: var(--accent);">
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 6px;">
            Objetivo pendiente de autorización:
          </div>
          <div style="font-family: monospace; font-size: 0.82rem; color: var(--blue); margin-bottom: 4px;">
            📍 <span id="auth-target-coords">--</span>
          </div>
          <div style="font-family: monospace; font-size: 0.82rem; color: var(--text-muted); margin-bottom: 4px;">
            📏 Distancia: <span id="auth-target-dist" style="color: var(--accent);">--</span> m
          </div>
          <div style="font-family: monospace; font-size: 0.82rem; color: var(--text-muted);">
            🔋 Batería: <span id="auth-battery" style="color: var(--green);">--</span>%
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <button class="btn-success" style="font-size: 1rem; font-weight: bold; padding: 12px;" onclick="authorizeFlight()">
            ✅ AUTORIZAR VUELO
          </button>
          <button class="btn-danger" style="font-size: 1rem; font-weight: bold; padding: 12px;" onclick="abortFlight()">
            🚫 CANCELAR
          </button>
        </div>
      </div>

      <!-- SECCIÓN: MENSAJES DE RADIO 433 MHz -->
      <div>
        <div class="section-title">RADIO 433 MHz (STATUSTEXT PIXHAWK)</div>
        <div class="radio-log-box" id="radio-logs">
          <div>[--:--:--] Esperando enlace MAVLink...</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function haversineDistM(lat1, lon1, lat2, lon2) {
      const R = 6371000.0;
      const dLat = (lat2 - lat1) * Math.PI / 180.0;
      const dLon = (lon2 - lon1) * Math.PI / 180.0;
      const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2);
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      return R * c;
    }

    function bearingDeg(lat1, lon1, lat2, lon2) {
      const phi1 = lat1 * Math.PI / 180.0;
      const phi2 = lat2 * Math.PI / 180.0;
      const dLon = (lon2 - lon1) * Math.PI / 180.0;
      const y = Math.sin(dLon) * Math.cos(phi2);
      const x = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLon);
      const deg = Math.atan2(y, x) * 180.0 / Math.PI;
      return (deg + 360.0) % 360.0;
    }

    function recalcGeodesic() {
      const latA = parseFloat(document.getElementById('lat-a').value);
      const lonA = parseFloat(document.getElementById('lon-a').value);
      const latB = parseFloat(document.getElementById('lat-b').value);
      const lonB = parseFloat(document.getElementById('lon-b').value);
      const infoBox = document.getElementById('calc-info');

      if (isNaN(latA) || isNaN(lonA) || isNaN(latB) || isNaN(lonB)) {
        infoBox.innerText = 'Coordenadas incompletas.';
        infoBox.style.borderColor = '#334155';
        return;
      }

      const dist = haversineDistM(latA, lonA, latB, lonB);
      const bear = bearingDeg(latA, lonA, latB, lonB);

      let cardinal = 'N';
      if (bear >= 22.5 && bear < 67.5) cardinal = 'NE';
      else if (bear >= 67.5 && bear < 112.5) cardinal = 'E';
      else if (bear >= 112.5 && bear < 157.5) cardinal = 'SE';
      else if (bear >= 157.5 && bear < 202.5) cardinal = 'S';
      else if (bear >= 202.5 && bear < 247.5) cardinal = 'SO';
      else if (bear >= 247.5 && bear < 292.5) cardinal = 'O';
      else if (bear >= 292.5 && bear < 337.5) cardinal = 'NO';

      const maxR = parseFloat(document.getElementById('max-radius').value) || 300.0;
      infoBox.innerText = `Distancia A ➔ B: ${dist.toFixed(1)} m | Rumbo: ${bear.toFixed(0)}° (${cardinal})`;
      if (dist > maxR) {
        infoBox.style.borderColor = '#ef4444';
        infoBox.innerHTML += `<div style="color: #fca5a5; font-size: 0.75rem; margin-top: 4px; font-weight: bold;">⚠️ Supera el radio de seguridad (${maxR}m). Aumenta el radio arriba para autorizar.</div>`;
      } else {
        infoBox.style.borderColor = '#16a34a';
        infoBox.innerHTML += `<div style="color: #86efac; font-size: 0.72rem; margin-top: 2px;">✅ Distancia segura dentro del perímetro de ${maxR}m.</div>`;
      }
    }

    let liveLat = null, liveLon = null;

    function copyPixhawkGpsToA() {
      if (liveLat !== null && liveLon !== null && liveLat !== 0.0) {
        document.getElementById('lat-a').value = liveLat.toFixed(6);
        document.getElementById('lon-a').value = liveLon.toFixed(6);
        recalcGeodesic();
      } else {
        alert('Aún no hay coordenadas GPS válidas del Pixhawk.');
      }
    }

    async function dispatchMissionAB() {
      const latA = parseFloat(document.getElementById('lat-a').value);
      const lonA = parseFloat(document.getElementById('lon-a').value);
      const latB = parseFloat(document.getElementById('lat-b').value);
      const lonB = parseFloat(document.getElementById('lon-b').value);
      const alt = parseFloat(document.getElementById('alt-target').value) || 8.0;
      const maxR = parseFloat(document.getElementById('max-radius').value) || 300.0;

      if (isNaN(latA) || isNaN(lonA) || isNaN(latB) || isNaN(lonB)) {
        alert('Por favor ingresa las dos coordenadas completas (Punto A y Punto B).');
        return;
      }

      const dist = haversineDistM(latA, lonA, latB, lonB);
      const bear = bearingDeg(latA, lonA, latB, lonB);

      if (dist > maxR) {
        if (!confirm(`La distancia al objetivo (${dist.toFixed(1)}m) supera el radio actual de seguridad (${maxR}m).\\n\\n¿Deseas aumentar el radio de seguridad a ${Math.ceil(dist + 25)}m y despachar la misión?`)) {
          return;
        }
        document.getElementById('max-radius').value = Math.ceil(dist + 25);
      }

      if (!confirm(`¿Confirmas ordenar al dron despegar de Punto A hacia Punto B?\\n\\n- Origen A: (${latA.toFixed(6)}, ${lonA.toFixed(6)})\\n- Objetivo B: (${latB.toFixed(6)}, ${lonB.toFixed(6)})\\n- Distancia: ${dist.toFixed(1)}m (Rumbo: ${bear.toFixed(0)}°)\\n- Altitud: ${alt}m\\n- Radio máx: ${document.getElementById('max-radius').value}m`)) {
        return;
      }

      try {
        const res = await fetch('/api/mission_ab', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            drone_lat: latA,
            drone_lon: lonA,
            target_lat: latB,
            target_lon: lonB,
            alt_m: alt,
            max_radius_m: parseFloat(document.getElementById('max-radius').value) || 300.0
          })
        });
        const ans = await res.json();
        alert((ans.status === 'ok' ? '🚀 ' : '⚠️ ') + (ans.message || ans.status));
      } catch (e) {
        alert('Error despachando misión: ' + e);
      }
    }

    async function captureManualDataset() {
      try {
        const res = await fetch('/api/record_snapshot', { method: 'POST' });
        const ans = await res.json();
        alert((ans.status === 'ok' ? '📸 ' : '⚠️ ') + (ans.message || ans.status));
      } catch (e) {
        alert('Error capturando muestra: ' + e);
      }
    }

    async function sendAction(action) {
      try {
        const res = await fetch('/api/' + action, { method: 'POST' });
        const ans = await res.json();
        alert(ans.message || ans.status);
      } catch (e) { alert('Error: ' + e); }
    }

    async function authorizeFlight() {
      if (!confirm('⚠️ ¿Confirmas AUTORIZAR el despegue autónomo del dron hacia el objetivo?')) return;
      try {
        const res = await fetch('/api/authorize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}'
        });
        const ans = await res.json();
        if (ans.status === 'ok') {
          alert('✅ ' + (ans.message || 'Vuelo AUTORIZADO'));
        } else {
          alert('⚠️ ' + (ans.message || 'Autorización rechazada'));
        }
      } catch (e) { alert('Error autorizando vuelo: ' + e); }
    }

    async function abortFlight() {
      try {
        const res = await fetch('/api/abort', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'Operador C2 (web)' })
        });
        const ans = await res.json();
        alert('🚫 ' + (ans.message || 'Aborto ejecutado'));
      } catch (e) { alert('Error abortando: ' + e); }
    }

    async function updateDashboard() {
      try {
        const res = await fetch('/telemetry.json');
        const data = await res.json();
        
        liveLat = data.lat;
        liveLon = data.lon;

        document.getElementById('temp-val').innerText = data.max_temp_c !== undefined ? data.max_temp_c.toFixed(1) : '--';
        document.getElementById('fps-val').innerText = data.fps !== undefined ? data.fps.toFixed(1) : '--';
        document.getElementById('bat-val').innerText = data.battery_pct !== undefined ? data.battery_pct : '--';
        document.getElementById('mode-val').innerText = data.mode || '--';
        document.getElementById('alt-val').innerText = data.alt_m !== undefined ? data.alt_m.toFixed(1) : '--';
        document.getElementById('sats-val').innerText = data.sats !== undefined ? data.sats : '--';

        if (data.dataset_count !== undefined) {
          const dsEl = document.getElementById('dataset-count');
          if (dsEl) dsEl.innerText = data.dataset_count;
        }

        const hud = document.getElementById('hud-overlay');
        hud.innerText = `MODO: ${data.mode} | BAT: ${data.battery_pct}% | ALT: ${data.alt_m ? data.alt_m.toFixed(1) : 0}m | FSM: ${data.fsm_state}`;

        // Alerta de fuego
        const fireBox = document.getElementById('fire-box');
        const fireVal = document.getElementById('fire-val');
        if (data.num_fires > 0) {
          fireBox.classList.add('fire-alert');
          fireVal.innerText = `¡FUEGO DETECTADO! (${data.num_fires})`;
        } else {
          fireBox.classList.remove('fire-alert');
          fireVal.innerText = 'SIN FUEGO';
        }

        // FSM Badge
        const fsmBadge = document.getElementById('fsm-badge');
        fsmBadge.innerText = data.fsm_state || 'READY';
        if (data.fsm_state === 'READY') fsmBadge.style.background = '#22c55e';
        else if (data.fsm_state === 'TRIAGE') fsmBadge.style.background = '#f59e0b';
        else if (data.fsm_state === 'WAITING_AUTH') fsmBadge.style.background = '#dc2626';
        else if (data.fsm_state === 'TRANSIT') fsmBadge.style.background = '#3b82f6';
        else if (data.fsm_state === 'ON_STATION') fsmBadge.style.background = '#a855f7';
        else if (data.fsm_state === 'VERDICT') fsmBadge.style.background = '#ef4444';
        else if (data.fsm_state === 'RTL') fsmBadge.style.background = '#64748b';
        else if (data.fsm_state === 'MANUAL_OVERRIDE') fsmBadge.style.background = '#b91c1c';

        // Panel de autorización: visible solo en WAITING_AUTH
        const authPanel = document.getElementById('auth-panel');
        if (data.fsm_state === 'WAITING_AUTH') {
          authPanel.style.display = 'block';
          if (data.auth_target_lat !== undefined) {
            document.getElementById('auth-target-coords').innerText =
              `(${data.auth_target_lat.toFixed(6)}, ${data.auth_target_lon.toFixed(6)})`;
            document.getElementById('auth-target-dist').innerText =
              data.auth_target_dist_m !== undefined ? data.auth_target_dist_m.toFixed(1) : '--';
          }
          document.getElementById('auth-battery').innerText = data.battery_pct !== undefined ? data.battery_pct : '--';
          if (data.auth_time_remaining_s !== undefined) {
            const secs = Math.max(0, Math.round(data.auth_time_remaining_s));
            document.getElementById('auth-countdown').innerText = secs + 's';
            document.getElementById('auth-countdown').style.color = secs < 30 ? '#ef4444' : '#f59e0b';
          }
        } else {
          authPanel.style.display = 'none';
        }

        // Logs de radio 433 MHz
        if (data.radio_logs && data.radio_logs.length > 0) {
          const logBox = document.getElementById('radio-logs');
          logBox.innerHTML = data.radio_logs.map(msg => `<div>${msg}</div>`).join('');
        }
      } catch (e) {}
    }

    // Streaming de video térmico continuo y robusto (25 FPS)
    const feedImg = document.getElementById('thermal-feed');
    let videoStreamActive = true;

    function refreshFrame() {
      if (!videoStreamActive) return;
      const next = new Image();
      next.onload = () => {
        if (feedImg) feedImg.src = next.src;
        setTimeout(refreshFrame, 40);
      };
      next.onerror = () => {
        setTimeout(refreshFrame, 250);
      };
      next.src = '/snapshot.jpg?t=' + Date.now();
    }
    refreshFrame();

    recalcGeodesic();
    setInterval(updateDashboard, 500);
  </script>
</body>
</html>
"""


class WebMissionHandler(BaseHTTPRequestHandler):
    """Manejador HTTP para la interfaz web y la API de control de vuelo."""

    def do_GET(self) -> None:
        global latest_jpeg_frame, latest_telemetry_json

        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif self.path == "/telemetry.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with frame_lock:
                self.wfile.write(latest_telemetry_json.encode("utf-8"))

        elif self.path == "/snapshot.jpg":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with frame_lock:
                frame = latest_jpeg_frame
            if frame:
                self.wfile.write(frame)

        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_jpeg_frame
                    if frame:
                        header = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n"
                        )
                        self.wfile.write(header + frame + b"\r\n")
                    time.sleep(0.04)  # ~25 fps
            except Exception:
                pass
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        global fsm_instance, mavlink_instance, recorder_instance, latest_raw_frame, latest_verdict

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        resp: Dict[str, Any] = {"status": "error", "message": "Comando no reconocido"}

        if self.path == "/api/record_snapshot":
            if recorder_instance and latest_raw_frame is not None and latest_verdict is not None:
                telem = mavlink_instance.get_telemetry() if mavlink_instance else None
                rec_ok = recorder_instance.force_record(latest_raw_frame, latest_verdict, telemetry=telem)
                cnt = recorder_instance.frame_count
                broadcast_message(f"📸 Muestra manual #{cnt} guardada en dataset")
                resp = {"status": "ok" if rec_ok else "busy", "message": f"Muestra #{cnt} guardada exitosamente en disco."}
            else:
                resp = {"status": "error", "message": "Grabador no inicializado o sin datos de cámara todavía."}

        elif self.path == "/api/mission_ab":
            password = data.get("password", "")
            if password != "ignis2026":
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "FDIR Nivel 2: Contraseña de Despacho Inválida."}).encode("utf-8"))
                return

            d_lat = float(data.get("drone_lat", 0.0))
            d_lon = float(data.get("drone_lon", 0.0))
            t_lat = float(data.get("target_lat", 0.0))
            t_lon = float(data.get("target_lon", 0.0))
            alt = float(data.get("alt_m", 8.0))
            max_radius = float(data.get("max_radius_m", 0.0))

            if max_radius > 0 and fsm_instance:
                fsm_instance.max_allowed_dist_m = max_radius

            if d_lat != 0.0 and d_lon != 0.0 and mavlink_instance:
                # Si el Pixhawk no tiene GPS fix real (laboratorio/banco), actualizar posición drone
                telem = mavlink_instance.get_telemetry()
                if telem.gps_fix_type < 3 or telem.lat == 0.0:
                    mavlink_instance.telemetry.lat = d_lat
                    mavlink_instance.telemetry.lon = d_lon
                    mavlink_instance.telemetry.alt_rel_m = alt

            dist_m = haversine_distance_m(d_lat, d_lon, t_lat, t_lon)
            bearing_deg = calculate_bearing_deg(d_lat, d_lon, t_lat, t_lon)

            if fsm_instance:
                if dist_m > fsm_instance.max_allowed_dist_m:
                    msg = (
                        f"RECHAZADO POR SEGURIDAD: La distancia ({dist_m:.1f} m) supera el límite máximo "
                        f"permitido ({fsm_instance.max_allowed_dist_m:.1f} m). Aumenta el radio de seguridad en el panel."
                    )
                    broadcast_message(f"⚠️ Rechazado: d={dist_m:.0f}m > max={fsm_instance.max_allowed_dist_m:.0f}m")
                    resp = {
                        "status": "rejected",
                        "dist_m": dist_m,
                        "bearing_deg": bearing_deg,
                        "message": msg,
                    }
                else:
                    success = fsm_instance.command_direct_target(t_lat, t_lon, alt)
                    broadcast_message(f"🚀 Misión A->B: d={dist_m:.1f}m rumbo={bearing_deg:.0f}° alt={alt}m")
                    resp = {
                        "status": "ok" if success else "rejected",
                        "dist_m": dist_m,
                        "bearing_deg": bearing_deg,
                        "message": f"Misión DESPACHADA hacia Punto B ({t_lat:.6f}, {t_lon:.6f}). Distancia: {dist_m:.1f} m, Rumbo: {bearing_deg:.0f}°. Modo autónomo ordenado.",
                    }

        elif self.path == "/api/target":
            lat = data.get("lat")
            lon = data.get("lon")
            alt = data.get("alt_m", 8.0)
            if lat is not None and lon is not None and fsm_instance:
                success = fsm_instance.command_direct_target(float(lat), float(lon), float(alt))
                broadcast_message(f"Comando misión: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt}m")
                resp = {
                    "status": "ok" if success else "rejected",
                    "message": f"Misión enviada a ({lat:.6f}, {lon:.6f}). FSM en {fsm_instance.state.name}",
                }

        elif self.path == "/api/loiter":
            if fsm_instance:
                fsm_instance.transition_to(MissionState.READY)
                fsm_instance.active_target = None
            if mavlink_instance:
                mavlink_instance.set_mode("LOITER")
                broadcast_message("Solicitado modo LOITER (Frenado y FSM en READY)")
                resp = {"status": "ok", "message": "Modo LOITER solicitado al Pixhawk (FSM en READY)"}

        elif self.path == "/api/rtl":
            if fsm_instance:
                fsm_instance.transition_to(MissionState.RTL)
            if mavlink_instance:
                mavlink_instance.request_rtl()
                broadcast_message("Solicitado modo RTL (Retorno a Base)")
                resp = {"status": "ok", "message": "Modo RTL solicitado al Pixhawk"}

        elif self.path == "/api/authorize":
            password = data.get("password", "")
            if password != "ignis2026":
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "FDIR Nivel 2: Contraseña de Autorización Inválida."}).encode("utf-8"))
                return

            if fsm_instance:
                success = fsm_instance.authorize_mission()
                if success:
                    telem = mavlink_instance.get_telemetry() if mavlink_instance else None
                    bat = telem.battery_pct if telem else -1
                    broadcast_message("✅ Misión AUTORIZADA por operador C2")
                    resp = {
                        "status": "ok",
                        "message": "Misión AUTORIZADA. Dron en modo GUIDED.",
                        "fsm_state": fsm_instance.state.name,
                        "battery_pct": bat
                    }
                else:
                    resp = {
                        "status": "rejected",
                        "message": "Autorización rechazada (verifica estado WAITING_AUTH y batería).",
                        "fsm_state": fsm_instance.state.name
                    }
            else:
                resp = {"status": "error", "message": "FSM no inicializada."}

        elif self.path == "/api/abort":
            password = data.get("password", "")
            if password != "ignis2026":
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "FDIR Nivel 2: Contraseña de Aborto Inválida."}).encode("utf-8"))
                return

            reason = data.get("reason", "Operador C2 (web)")
            if fsm_instance:
                fsm_instance.abort_mission(reason=reason)
                resp = {
                    "status": "ok",
                    "message": f"ABORTO ejecutado: {reason}. Dron en RTL.",
                    "fsm_state": fsm_instance.state.name
                }
            else:
                resp = {"status": "error", "message": "FSM no inicializada."}
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        """CORS Preflight para permitir peticiones cross-origin (React Frontend)."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="IgnisEdge Live Web Video & Two-Coordinate Mission Monitor")
    parser.add_argument("--port", type=int, default=8080, help="Puerto del servidor web (def: 8080)")
    parser.add_argument("--pixhawk-port", default=None, help="Puerto serial Pixhawk (def: auto-detect)")
    parser.add_argument("--heltec-port", default="/dev/ttyACM1", help="Puerto Heltec (def: /dev/ttyACM1)")
    parser.add_argument("--max-radius", type=float, default=300.0, help="Radio máximo de seguridad en metros (def: 300.0m)")
    parser.add_argument("--patrol-alt", type=float, default=8.0, help="Altitud relativa de patrulla en metros (def: 8.0m)")
    parser.add_argument("--record-dataset", action="store_true", help="Grabar automáticamente frames de fuego en disco")
    parser.add_argument("--dataset-dir", default="data/fire_dataset", help="Directorio para dataset")
    parser.add_argument("--sim", action="store_true", help="Forzar modo simulación sin hardware")
    args = parser.parse_args()

    global latest_jpeg_frame, latest_telemetry_json, fsm_instance, mavlink_instance, recorder_instance, latest_raw_frame, latest_verdict

    logger.info("==========================================================")
    logger.info("   Iniciando IgnisEdge Live Web & Two-Coordinate Mission")
    logger.info("==========================================================")

    # 1. Pixhawk Connection
    mavlink = None
    if not args.sim:
        candidates = [args.pixhawk_port] if args.pixhawk_port else ["/dev/ttyACM0", "/dev/ttyTHS1"]
        for p in candidates:
            if p and os.path.exists(p):
                baud = 921600 if "THS" in p else 115200
                candidate_mav = IgnisMavlink(connection_string=p, baud=baud)
                if candidate_mav.connect(timeout_s=3.0):
                    mavlink = candidate_mav
                    logger.info(f"Pixhawk conectado exitosamente en {p} ({baud} baud)!")
                    break
                candidate_mav.disconnect()

    if mavlink is None:
        logger.warning("Pixhawk físico no conectado. Iniciando emulador de telemetría segura.")
        mavlink = IgnisMavlink(connection_string="/dev/null")
        mavlink.telemetry.lat = -36.820100
        mavlink.telemetry.lon = -73.044100
        mavlink.telemetry.alt_rel_m = 8.0
        mavlink.telemetry.battery_pct = 95
        mavlink.telemetry.mode = "LOITER"
        mavlink.telemetry.armed = True
        mavlink.telemetry.heartbeat_healthy = True

    mavlink_instance = mavlink

    # 2. Heltec Link (si existe)
    heltec_avail = os.path.exists(args.heltec_port) and not args.sim
    link = IgnisLink(port=args.heltec_port, sim_mode=not heltec_avail)
    link.connect()

    # 3. Cerebro FSM
    fsm = IgnisMissionFSM(
        mavlink=mavlink,
        link=link,
        max_allowed_dist_m=args.max_radius,
        nominal_patrol_alt_m=args.patrol_alt,
    )
    fsm_instance = fsm
    broadcast_message("Sistema inicializado. FSM lista en READY.")

    # 4. Motor de Visión & Cámara Térmica P3
    model_path = REPO_ROOT / "scripts" / "ignis_fire_classifier.joblib"
    engine = VisionEngine(model_path=model_path)
    camera_thread: Optional[P3CaptureThread] = None
    sim_thermal = args.sim

    if not args.sim:
        try:
            camera_thread = P3CaptureThread(gain="low")
            camera_thread.start()
            logger.info("Cámara térmica P3 conectada y transmitiendo a bajo nivel.")
        except Exception as e:
            logger.warning(f"Cámara P3 física no disponible: {e}. Activando generador térmico de prueba.")
            sim_thermal = True

    # 5. Dataset Recorder
    recorder: Optional[DatasetRecorder] = None
    if args.record_dataset:
        recorder = DatasetRecorder(output_dir=args.dataset_dir, max_fps=5.0)
        recorder_instance = recorder
        logger.info(f"Grabador de dataset activo en: {args.dataset_dir}")

    # 6. Servidor Web en hilo dedicado con ThreadingHTTPServer (evita bloqueos de streaming)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), WebMissionHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info(f"Servidor Web activo y listo en: http://0.0.0.0:{args.port}")

    running = True

    def sig_handler(_sig: int, _frame: Any) -> None:
        nonlocal running
        logger.info("Interrupción recibida. Apagando servidor y liberando hardware...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    synthetic_tick = 0

    try:
        while running:
            # Paso de FSM
            fsm.step()

            # Obtención de frame térmico (físico o sintético si desconectada)
            frame_c = None
            if camera_thread and camera_thread.is_connected():
                frame_c = camera_thread.get_latest_frame(timeout=0.04)

            if frame_c is None:
                synthetic_tick += 1
                base = 21.0 + np.random.normal(0, 0.4, (192, 256)).astype(np.float32)
                if fsm.state in (MissionState.ON_STATION, MissionState.VERDICT):
                    flicker = 180.0 + 15.0 * math.sin(synthetic_tick * 0.4)
                    base[85:105, 115:135] = flicker
                frame_c = base
                sim_thermal = True
                time.sleep(0.04)
            else:
                sim_thermal = False

            if frame_c is not None:
                verdict = engine.process_frame(frame_c)
                fsm.on_vision_verdict_received({
                    "status": verdict.status,
                    "num_fires": verdict.num_fires,
                    "tracks": [asdict(t) for t in verdict.tracks],
                })

                telem = mavlink.get_telemetry()
                with frame_lock:
                    latest_raw_frame = frame_c
                    latest_verdict = verdict

                if recorder:
                    if fsm.state in (MissionState.ON_STATION, MissionState.VERDICT):
                        recorded = recorder.force_record(frame_c, verdict, telemetry=telem)
                    else:
                        recorded = recorder.maybe_record(frame_c, verdict, telemetry=telem)
                    if recorded:
                        broadcast_message(f"💾 Guardada muestra #{recorder.frame_count} en dataset")

                # Renderizar frame térmico con paleta Inferno
                norm = np.clip((frame_c - 15.0) / (200.0 - 15.0), 0, 1)
                u8 = (norm * 255).astype(np.uint8)
                color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
                disp = cv2.resize(color, (512, 384), interpolation=cv2.INTER_NEAREST)

                # Bounding boxes de fuegos detectados
                for tr in verdict.tracks:
                    x = int(tr.cx * (512 / 256))
                    y = int(tr.cy * (384 / 192))
                    r = max(18, int(math.sqrt(tr.area_px) * 2.5))
                    col = (0, 255, 0) if tr.state == "FUEGO" else ((0, 200, 255) if tr.state == "eval" else (255, 128, 0))
                    cv2.rectangle(disp, (x - r, y - r), (x + r, y + r), col, 2)
                    cv2.putText(disp, f"{tr.state} {tr.confidence*100:.0f}%", (x - r, y - r - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                # Marca de agua si es simulación térmica
                if sim_thermal:
                    cv2.putText(disp, "[SIMULADOR TERMICO - P3 NO DETECTADA]", (12, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                _, jpeg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 75])

                telem_dict = {
                    "max_temp_c": verdict.max_temp_c,
                    "fps": verdict.fps,
                    "num_fires": verdict.num_fires,
                    "status": verdict.status,
                    "battery_pct": telem.battery_pct,
                    "mode": telem.mode,
                    "lat": telem.lat,
                    "lon": telem.lon,
                    "alt_m": telem.alt_rel_m,
                    "sats": telem.satellites_visible,
                    "fsm_state": fsm.state.name,
                    "max_radius_m": fsm.max_allowed_dist_m,
                    "dataset_count": recorder.frame_count if recorder else 0,
                    "radio_logs": list(recent_radio_messages),
                    "motor_1_pct": getattr(telem, "motor_1_pct", 0),
                    "motor_2_pct": getattr(telem, "motor_2_pct", 0),
                    "motor_3_pct": getattr(telem, "motor_3_pct", 0),
                    "motor_4_pct": getattr(telem, "motor_4_pct", 0),
                    "min_temp_c": getattr(verdict, "min_temp_c", 0.0),
                    "thermal_grid": getattr(verdict, "thermal_grid_16x12", []),
                }

                if fsm.state == MissionState.WAITING_AUTH and fsm.active_target:
                    c_lat = telem.lat if telem.lat != 0.0 else fsm.active_target.lat
                    c_lon = telem.lon if telem.lon != 0.0 else fsm.active_target.lon
                    d_m = haversine_distance_m(c_lat, c_lon, fsm.active_target.lat, fsm.active_target.lon)
                    telem_dict.update({
                        "auth_pending": True,
                        "auth_target_lat": fsm.active_target.lat,
                        "auth_target_lon": fsm.active_target.lon,
                        "auth_target_dist_m": d_m,
                        "auth_time_remaining_s": max(0.0, fsm.auth_timeout_s - (time.time() - fsm.t_state_entered))
                    })

                with frame_lock:
                    latest_jpeg_frame = jpeg.tobytes()
                    latest_telemetry_json = json.dumps(telem_dict)

    finally:
        if recorder:
            recorder.stop()
        if camera_thread:
            camera_thread.stop()
        link.disconnect()
        mavlink.disconnect()
        server.shutdown()
        logger.info("Servidor detenido y recursos liberados exitosamente.")


if __name__ == "__main__":
    main()
