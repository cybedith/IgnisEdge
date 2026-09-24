/**
 * Floating HUD (top-right of map) with live weather and mission stats.
 * Compass SVG rotates with wind_direction_deg.
 */
import { useIgnisStore } from "@/store/useIgnisStore";

export function FloatingHUD() {
  const perimeter = useIgnisStore((s) => s.perimeter);
  const nodes = useIgnisStore((s) => s.nodes);
  const liveTick = useIgnisStore((s) => s.liveTick);
  const session = useIgnisStore((s) => s.session);
  const triangulation = useIgnisStore((s) => s.triangulation);

  const w = liveTick?.weather;
  const detecting = (liveTick?.node_status ?? []).filter((n) => n.detected).length;
  const simT = liveTick?.sim_time_s ?? session?.sim_time_s ?? 0;
  const mm = Math.floor(simT / 60).toString().padStart(2, "0");
  const ss = Math.floor(simT % 60).toString().padStart(2, "0");

  return (
    <div className="pointer-events-none absolute right-3 top-3 z-10 w-[210px] rounded-md border border-border/60 bg-card/70 p-3 backdrop-blur-md">
      <div className="mb-2 flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Sector · Hualpén
        </div>
        {session?.running && (
          <span className="flex items-center gap-1 text-[9px] font-semibold text-destructive">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-destructive/70" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-destructive" />
            </span>
            LIVE SIM
          </span>
        )}
      </div>

      {w ? (
        <div className="mb-2 flex items-center gap-2">
          <Compass deg={w.wind_direction_deg} />
          <div className="flex-1 space-y-0.5 font-mono text-[10px]">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Viento</span>
              <span>{w.wind_speed_ms.toFixed(1)} m/s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Dir</span>
              <span>{w.wind_direction_deg.toFixed(0)}°</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Temp</span>
              <span>{w.temperature_c.toFixed(1)}°C</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">HR</span>
              <span>{w.relative_humidity_pct.toFixed(0)}%</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-2 rounded border border-dashed border-border/50 bg-background/30 px-2 py-1.5 text-center text-[10px] text-muted-foreground">
          Sin meteo en vivo
        </div>
      )}

      <div className="space-y-0.5 border-t border-border/40 pt-2 text-[10px]">
        <Row label="Sim" value={`${mm}:${ss}`} />
        <Row label="Perímetro" value={`${perimeter.length} pts`} />
        <Row label="Nodos" value={`${detecting} / ${nodes.length}`} valueClass="text-blue-400" />
        <Row label="Focos" value={String(liveTick?.fires.length ?? 0)} valueClass="text-destructive" />
        <Row label="Triang." value={triangulation ? "✓" : "—"} valueClass={triangulation ? "text-success" : ""} />
      </div>
    </div>
  );
}

function Row({ label, value, valueClass = "text-foreground" }: {
  label: string; value: string; valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between text-muted-foreground">
      <span>{label}</span>
      <span className={`font-mono ${valueClass}`}>{value}</span>
    </div>
  );
}

function Compass({ deg }: { deg: number }) {
  return (
    <svg viewBox="0 0 60 60" className="h-12 w-12 shrink-0">
      <circle cx="30" cy="30" r="26" fill="hsl(var(--background) / 0.6)" stroke="hsl(var(--border))" strokeWidth="1" />
      {["N", "E", "S", "W"].map((label, i) => {
        const angle = (i * 90 - 90) * (Math.PI / 180);
        const r = 21;
        const x = 30 + Math.cos(angle) * r;
        const y = 30 + Math.sin(angle) * r;
        return (
          <text
            key={label}
            x={x}
            y={y + 3}
            textAnchor="middle"
            fontSize="7"
            fill={label === "N" ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))"}
            fontFamily="monospace"
          >
            {label}
          </text>
        );
      })}
      <g transform={`rotate(${deg} 30 30)`}>
        <polygon points="30,12 26,30 30,28 34,30" fill="hsl(var(--primary))" />
        <polygon points="30,48 26,30 30,32 34,30" fill="hsl(var(--muted-foreground))" />
        <circle cx="30" cy="30" r="2" fill="hsl(var(--foreground))" />
      </g>
    </svg>
  );
}
