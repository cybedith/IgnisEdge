import React from "react";
import { useIgnisStore } from "@/store/useIgnisStore";

export function AvionicsHUD() {
  const liveTick = useIgnisStore((s) => s.liveTick);
  const drone = liveTick?.drones?.[0];

  if (!drone) return null;

  const motors = (drone as any).motors || [50, 50, 50, 50];
  const attitude = (drone as any).attitude || { pitch: 0, roll: 0, yaw: drone.heading_deg || 0 };
  const flightMode = (drone as any).flight_mode || drone.status;
  const altMsl = (drone as any).alt_amsl_m ?? drone.altitude_m;
  const altAgl = (drone as any).alt_agl_m ?? 80.0;
  const terrainElev = (drone as any).terrain_elev_m ?? (altMsl - altAgl);

  return (
    <div className="absolute left-16 bottom-6 z-10 w-80 rounded-md border border-border/60 bg-card/90 p-3 backdrop-blur-md shadow-2xl text-xs font-mono">
      {/* Encabezado */}
      <div className="flex justify-between items-center border-b border-border/40 pb-1.5 mb-2.5">
        <span className="text-primary font-bold tracking-wider">PIXHAWK AVIONICS</span>
        <span className="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded font-bold">
          {flightMode}
        </span>
      </div>

      {/* Grid de Cotas y Actitud */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Cotas */}
        <div className="space-y-1 bg-black/40 p-2 rounded border border-border/30">
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">ALT MSL</span>
            <span className="text-foreground font-semibold">{altMsl.toFixed(1)} m</span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">ALT AGL</span>
            <span className="text-emerald-400 font-semibold">{altAgl.toFixed(1)} m</span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">RELIEVE</span>
            <span className="text-muted-foreground font-semibold">{terrainElev.toFixed(1)} m</span>
          </div>
        </div>

        {/* Actitud */}
        <div className="space-y-1 bg-black/40 p-2 rounded border border-border/30">
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">PITCH</span>
            <span className="text-foreground font-semibold">{attitude.pitch?.toFixed(1) || 0.0}°</span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">ROLL</span>
            <span className="text-foreground font-semibold">{attitude.roll?.toFixed(1) || 0.0}°</span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">YAW</span>
            <span className="text-foreground font-semibold">{attitude.yaw?.toFixed(1) || 0.0}°</span>
          </div>
        </div>
      </div>

      {/* Potencia de los 4 Motores (PWM %) */}
      <div>
        <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
          <span>MOTORES PWM (M1 - M4)</span>
          <span className="text-primary font-bold">
            {Math.round(motors.reduce((a: number, b: number) => a + b, 0) / 4)}% PROM
          </span>
        </div>
        <div className="grid grid-cols-4 gap-1.5">
          {motors.slice(0, 4).map((m: number, idx: number) => (
            <div key={idx} className="bg-black/50 p-1.5 rounded border border-border/30 text-center">
              <span className="text-[9px] text-muted-foreground block">M{idx + 1}</span>
              <div className="w-full bg-secondary/50 h-1.5 rounded-full overflow-hidden my-1">
                <div
                  className="h-full bg-primary transition-all duration-300 rounded-full"
                  style={{ width: `${Math.max(0, Math.min(100, m))}%` }}
                />
              </div>
              <span className="text-[10px] font-bold text-foreground">{Math.round(m)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
