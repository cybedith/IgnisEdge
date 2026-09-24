/**
 * TimelinePanel — collapsible bottom panel with charts.
 * Tabs: Detecciones (smoke ppm per node) · Cobertura (burned ha) · Eventos.
 */
import { useState, useMemo } from "react";
import { ChevronUp, Activity, Flame, ListOrdered } from "lucide-react";
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid,
} from "recharts";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useIgnisStore } from "@/store/useIgnisStore";
import { cn } from "@/lib/utils";

const DETECT_THRESHOLD = 50;
const NODE_COLORS = [
  "hsl(217, 91%, 60%)",  // blue
  "hsl(189, 94%, 55%)",  // cyan
  "hsl(142, 70%, 45%)",  // success
  "hsl(42, 95%, 58%)",   // warning
  "hsl(280, 85%, 65%)",  // purple
  "hsl(340, 85%, 60%)",  // pink
];

export function TimelinePanel() {
  const [open, setOpen] = useState(false);
  const tickBuffer = useIgnisStore((s) => s.tickBuffer);
  const nodes = useIgnisStore((s) => s.nodes);
  const session = useIgnisStore((s) => s.session);
  const triangulation = useIgnisStore((s) => s.triangulation);

  const ppmData = useMemo(
    () =>
      tickBuffer.map((s) => {
        const row: Record<string, number> = { t: Math.round(s.t) };
        for (const n of nodes) row[n.node_id] = s.byNode[n.node_id] ?? 0;
        return row;
      }),
    [tickBuffer, nodes],
  );

  const areaData = useMemo(
    () => tickBuffer.map((s) => ({ t: Math.round(s.t), ha: +s.area_ha.toFixed(2) })),
    [tickBuffer],
  );

  const events = useMemo(() => {
    const out: { t: number; label: string; tone: string }[] = [];
    if (session) out.push({ t: 0, label: "Sesión iniciada", tone: "text-foreground" });
    let firstDetectionAt: number | null = null;
    for (const s of tickBuffer) {
      if (firstDetectionAt == null) {
        const anyDetect = Object.values(s.byNode).some((v) => v > DETECT_THRESHOLD);
        if (anyDetect) {
          firstDetectionAt = s.t;
          out.push({ t: s.t, label: "Primera detección de humo", tone: "text-warning" });
        }
      }
    }
    if (triangulation) out.push({ t: triangulation.estimated_t0_s, label: "Origen triangulado", tone: "text-primary" });
    return out;
  }, [session, tickBuffer, triangulation]);

  return (
    <div
      className={cn(
        "pointer-events-auto absolute inset-x-3 bottom-3 z-10 overflow-hidden rounded-md border border-border/60 bg-card/80 backdrop-blur-md transition-[height]",
        open ? "h-[280px]" : "h-8",
      )}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex h-8 w-full items-center justify-between px-3 text-[10px] font-medium uppercase tracking-widest text-muted-foreground hover:text-foreground"
      >
        <span className="flex items-center gap-2">
          <Activity className="h-3 w-3" />
          Timeline {tickBuffer.length > 0 && <span className="font-mono text-foreground/60">· {tickBuffer.length} ticks</span>}
        </span>
        <ChevronUp className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <Tabs defaultValue="ppm" className="h-[248px] px-3 pb-2">
          <TabsList className="h-7 w-fit rounded-md bg-background/40">
            <TabsTrigger value="ppm" className="h-6 px-2 text-[10px]">
              <Activity className="mr-1 h-3 w-3" /> Detecciones
            </TabsTrigger>
            <TabsTrigger value="ha" className="h-6 px-2 text-[10px]">
              <Flame className="mr-1 h-3 w-3" /> Cobertura
            </TabsTrigger>
            <TabsTrigger value="events" className="h-6 px-2 text-[10px]">
              <ListOrdered className="mr-1 h-3 w-3" /> Eventos
            </TabsTrigger>
          </TabsList>

          <TabsContent value="ppm" className="h-[200px] pt-1">
            {ppmData.length === 0 ? (
              <Empty msg="Sin lecturas. Inicia una simulación." />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ppmData} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="hsl(var(--border))" />
                  <XAxis dataKey="t" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                  <YAxis tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={{ fontSize: 10 }} />
                  <ReferenceLine
                    y={DETECT_THRESHOLD}
                    stroke="hsl(var(--destructive))"
                    strokeDasharray="3 3"
                    label={{ value: "thr", fontSize: 9, fill: "hsl(var(--destructive))" }}
                  />
                  {nodes.map((n, i) => (
                    <Line
                      key={n.node_id}
                      dataKey={n.node_id}
                      type="monotone"
                      dot={false}
                      stroke={NODE_COLORS[i % NODE_COLORS.length]}
                      strokeWidth={1.5}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}
          </TabsContent>

          <TabsContent value="ha" className="h-[200px] pt-1">
            {areaData.length === 0 ? (
              <Empty msg="Sin focos activos." />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={areaData} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="hsl(var(--border))" />
                  <XAxis dataKey="t" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                  <YAxis tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Area
                    dataKey="ha"
                    type="monotone"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary) / 0.25)"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </TabsContent>

          <TabsContent value="events" className="h-[200px] overflow-y-auto pt-1">
            {events.length === 0 ? (
              <Empty msg="Sin eventos registrados." />
            ) : (
              <ul className="space-y-1 pr-2">
                {events.map((e, i) => (
                  <li
                    key={i}
                    className="flex items-center justify-between rounded-md border border-border/40 bg-background/40 px-2 py-1 text-[11px]"
                  >
                    <span className={e.tone}>{e.label}</span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      t = {e.t.toFixed(0)}s
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}

const tooltipStyle = {
  background: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  fontSize: 11,
};

function Empty({ msg }: { msg: string }) {
  return (
    <div className="grid h-full place-items-center text-[11px] text-muted-foreground">
      {msg}
    </div>
  );
}
