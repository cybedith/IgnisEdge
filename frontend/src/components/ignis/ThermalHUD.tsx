import React, { useEffect, useState, useRef } from "react";
import { useIgnisStore } from "@/store/useIgnisStore";

// Paleta científica Inferno
const INFERNO_STOPS = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
];

function infernoRgb(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const x = v * (INFERNO_STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = INFERNO_STOPS[i];
  const b = INFERNO_STOPS[Math.min(i + 1, INFERNO_STOPS.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export function ThermalHUD() {
  const isTelemetryDown = useIgnisStore((s) => s.isTelemetryDown);
  const verdict = useIgnisStore((s) => s.latestVerdict as any);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [maxT, setMaxT] = useState<number>(22.0);
  const [minT, setMinT] = useState<number>(20.0);
  const [fps, setFps] = useState<number>(24);
  const [activeFoco, setActiveFoco] = useState<{ cx: number; cy: number; temp: number } | null>(null);

  // Sincroniza labels térmicos explícitamente cuando cambia el veredicto
  useEffect(() => {
    if (verdict) {
      if (verdict.fps) setFps(Math.round(verdict.fps));
      if (verdict.max_T) setMaxT(Math.round(verdict.max_T * 10) / 10);
      if (verdict.min_T) setMinT(Math.round(verdict.min_T * 10) / 10);
      
      if (verdict.focos && verdict.focos.length > 0) {
        const f = verdict.focos[0];
        setActiveFoco({ cx: f.cx, cy: f.cy, temp: f.temp });
      } else {
        setActiveFoco(null);
      }
    }
  }, [verdict]);

  // Renderiza el frame térmico real sobre el canvas HTML5
  useEffect(() => {
    const grid: number[] | undefined = verdict?.thermal_grid;
    const cv = canvasRef.current;
    if (!cv) return;
    
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const cols = 16;
    const rows = 12;

    // Fix: Si el link cae o el grid es inválido, limpia el canvas (no más frames congelados falsos)
    if (isTelemetryDown || !grid || grid.length !== 192) {
      ctx.clearRect(0, 0, cols, rows);
      return;
    }

    // Fix: imageSmoothingEnabled = false para que los píxeles térmicos no se vean borrosos al estirarse
    ctx.imageSmoothingEnabled = false;

    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const valUint8 = grid[r * cols + c];
        const norm = valUint8 / 255.0; // Ya viene normalizado 0-255 desde la Jetson
        const [red, green, blue] = infernoRgb(norm);
        
        const idx = (r * cols + c) * 4;
        img.data[idx] = red;
        img.data[idx + 1] = green;
        img.data[idx + 2] = blue;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    // Dibuja retícula sobre el foco si hay uno detectado
    if (activeFoco) {
      ctx.strokeStyle = "#00ff66";
      ctx.lineWidth = 1;
      const x = Math.round((activeFoco.cx / 256) * cols);
      const y = Math.round((activeFoco.cy / 192) * rows);
      ctx.strokeRect(x - 2, y - 2, 4, 4);
    }
  }, [verdict?.thermal_grid, isTelemetryDown, activeFoco]);

  return (
    <div className="absolute bottom-6 right-6 z-10 w-64 rounded-md border border-border/60 bg-card/90 p-2 backdrop-blur-md shadow-2xl overflow-hidden font-mono">
      <div className="flex justify-between items-center mb-1 text-[10px]">
        <span className="text-primary font-bold tracking-wider">CÁMARA P3 RADIOMÉTRICA</span>
        <span className="flex items-center gap-1 text-[9px] font-semibold">
          {isTelemetryDown ? (
            <span className="text-red-500 font-bold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>
              STALE DATA
            </span>
          ) : (
            <span className="text-emerald-400 font-bold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              EN VIVO
            </span>
          )}
        </span>
      </div>
      <div className={`relative border rounded bg-black ${isTelemetryDown ? 'border-red-500/50 opacity-60' : 'border-border/40'}`}>
        <canvas
          ref={canvasRef}
          width={16}
          height={12}
          className="w-full h-auto image-rendering-pixelated rounded-sm"
          style={{ height: "140px", display: "block" }}
        />
        {isTelemetryDown && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-10 backdrop-blur-sm">
            <span className="text-red-500 font-bold text-xs tracking-widest bg-black/80 px-2 py-1 border border-red-900 rounded">
              LINK CAÍDO
            </span>
          </div>
        )}
        <div className="absolute top-1 left-1 text-[9px] text-white bg-black/60 px-1 rounded z-0">
          {minT}°C &mdash; {maxT}°C
        </div>
        <div className="absolute bottom-1 right-1 text-[9px] text-white bg-black/60 px-1 rounded z-0">
          {fps} FPS
        </div>
        {activeFoco && !isTelemetryDown && (
          <div className="absolute bottom-1 left-1 text-[9px] text-emerald-400 bg-black/60 px-1 rounded font-bold z-0">
            FOCO: {activeFoco.temp.toFixed(1)}°C
          </div>
        )}
      </div>
    </div>
  );
}
