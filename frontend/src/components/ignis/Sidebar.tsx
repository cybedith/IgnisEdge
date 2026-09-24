/**
 * Mission wizard sidebar — 4 stages: Perímetro → Análisis → Mesh → Sim/Respuesta.
 * Vertical tabs; stages "iluminate" when their prerequisites are met.
 */
import { useState, useEffect } from "react";
import { MapPinned, Activity, Radio, Satellite, Check, Lock, Cpu, Wrench } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useIgnisStore } from "@/store/useIgnisStore";
import { StageWorkbench } from "./stages/StageWorkbench";
import { StagePerimeter } from "./stages/StagePerimeter";
import { StageMesh } from "./stages/StageMesh";
import { StageSimulation } from "./stages/StageSimulation";
import { StageResponse } from "./stages/StageResponse";
import { cn } from "@/lib/utils";

type StageKey = "workbench" | "perimeter" | "mesh" | "sim" | "response";

const STAGE_META: { key: StageKey; icon: typeof Cpu; label: string }[] = [
  { key: "workbench", icon: Cpu, label: "Pruebas" },
  { key: "perimeter", icon: MapPinned, label: "Predio" },
  { key: "mesh", icon: Radio, label: "Mesh" },
  { key: "sim", icon: Activity, label: "Fuego" },
  { key: "response", icon: Satellite, label: "Misión" },
];

export function Sidebar() {
  const fuelRisk = useIgnisStore((s) => s.fuelRisk);
  const meshResult = useIgnisStore((s) => s.meshResult);
  const session = useIgnisStore((s) => s.session);

  const completed: Record<StageKey, boolean> = {
    workbench: !!session,
    perimeter: !!fuelRisk,
    mesh: !!meshResult,
    sim: !!session,
    response: false,
  };

  // En banco de pruebas de ingeniería, todas las pestañas están desbloqueadas
  const enabled: Record<StageKey, boolean> = {
    workbench: true,
    perimeter: true,
    mesh: true,
    sim: true,
    response: true,
  };

  const [active, setActive] = useState<StageKey>("workbench");

  return (
    <aside className="flex h-full w-[380px] shrink-0 flex-col border-r border-border/60 bg-sidebar/70 backdrop-blur-xl">
      <Tabs value={active} onValueChange={(v) => setActive(v as StageKey)} className="flex h-full flex-col">
        <TabsList className="h-auto w-full shrink-0 grid-cols-5 gap-1 rounded-none border-b border-border/60 bg-transparent p-1.5">
          {STAGE_META.map((s) => {
            const isEnabled = enabled[s.key];
            const isDone = completed[s.key];
            const Icon = s.icon;
            return (
              <TabsTrigger
                key={s.key}
                value={s.key}
                disabled={!isEnabled}
                className={cn(
                  "flex h-auto flex-col items-center gap-1 rounded-md border border-transparent px-1 py-1.5 text-[9px] uppercase tracking-wider data-[state=active]:border-primary/40 data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:shadow-none",
                  !isEnabled && "opacity-40",
                )}
              >
                <div className="relative">
                  <Icon className="h-3.5 w-3.5" />
                  {isDone && (
                    <span className="absolute -right-1.5 -top-1.5 flex h-2.5 w-2.5 items-center justify-center rounded-full bg-success text-[7px] text-background">
                      <Check className="h-1.5 w-1.5" strokeWidth={4} />
                    </span>
                  )}
                </div>
                <span>{s.label}</span>
              </TabsTrigger>
            );
          })}
        </TabsList>

        <ScrollArea className="flex-1">
          <div className="p-3">
            <TabsContent value="workbench" className="mt-0">
              <StageHeader n="HITL" title="Módulo de Pruebas & Vuelo" />
              <StageWorkbench />
            </TabsContent>
            <TabsContent value="perimeter" className="mt-0">
              <StageHeader n="1" title="Definir predio" />
              <StagePerimeter onContinue={() => setActive("mesh")} />
            </TabsContent>
            <TabsContent value="mesh" className="mt-0">
              <StageHeader n="2" title="Optimizar red" />
              <StageMesh onContinue={() => setActive("sim")} />
            </TabsContent>
            <TabsContent value="sim" className="mt-0">
              <StageHeader n="3" title="Simular incendio" />
              <StageSimulation onContinue={() => setActive("response")} />
            </TabsContent>
            <TabsContent value="response" className="mt-0">
              <StageHeader n="4" title="Detección y respuesta" />
              <StageResponse />
            </TabsContent>
          </div>
        </ScrollArea>
      </Tabs>

      <footer className="shrink-0 border-t border-border/60 px-4 py-2 text-[10px] uppercase tracking-widest text-muted-foreground">
        v1.0 · Hualpén C2
      </footer>
    </aside>
  );
}

function StageHeader({ n, title }: { n: string | number; title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2 border-b border-border/40 pb-2">
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/15 font-mono text-[11px] text-primary ring-1 ring-primary/40">
        {n}
      </span>
      <h2 className="text-xs font-semibold uppercase tracking-wider text-foreground">{title}</h2>
    </div>
  );
}
