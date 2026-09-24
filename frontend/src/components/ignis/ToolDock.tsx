/**
 * Floating tool dock (bottom-right). Switches the active toolMode.
 * 🚧 Bloque 3 cablea PLACE_HOTSPOT y PLACE_DRONE_TARGET al click del mapa.
 */
import { Pencil, Flame, Plane, Eye, Radio } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import { useIgnisStore } from "@/store/useIgnisStore";
import { cn } from "@/lib/utils";
import type { ToolMode } from "@/lib/types";

interface DockBtn {
  mode: ToolMode;
  icon: typeof Pencil;
  label: string;
  enabled: (s: ReturnType<typeof useIgnisStore.getState>) => boolean;
}

const BUTTONS: DockBtn[] = [
  { mode: "DRAW_PERIMETER", icon: Pencil, label: "Dibujar perímetro", enabled: () => true },
  {
    mode: "PLACE_NODE",
    icon: Radio,
    label: "Colocar nodo LoRa en mapa",
    enabled: () => true,
  },
  {
    mode: "PLACE_HOTSPOT",
    icon: Flame,
    label: "Colocar foco de calor",
    enabled: (s) => !!s.session,
  },
  {
    mode: "PLACE_DRONE_TARGET",
    icon: Plane,
    label: "Lanzar dron a coordenada",
    enabled: () => true,
  },
];

export function ToolDock() {
  const toolMode = useIgnisStore((s) => s.toolMode);
  const setToolMode = useIgnisStore((s) => s.setToolMode);
  const layerVisibility = useIgnisStore((s) => s.layerVisibility);
  const toggleLayer = useIgnisStore((s) => s.toggleLayer);
  const state = useIgnisStore.getState();

  return (
    <TooltipProvider delayDuration={200}>
      <div className="pointer-events-auto absolute bottom-14 right-3 z-10 flex flex-col gap-1.5">
        {BUTTONS.map((b) => {
          const active = toolMode === b.mode;
          const enabled = b.enabled(state);
          const Icon = b.icon;
          return (
            <Tooltip key={b.mode}>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={!enabled}
                  onClick={() => setToolMode(active ? "IDLE" : b.mode)}
                  className={cn(
                    "h-12 w-12 rounded-md border border-border/60 bg-card/70 backdrop-blur-md",
                    active && "border-primary/60 bg-primary/15 text-primary",
                  )}
                >
                  <Icon className="h-5 w-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">{b.label}</TooltipContent>
            </Tooltip>
          );
        })}

        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-12 w-12 rounded-md border border-border/60 bg-card/70 backdrop-blur-md"
            >
              <Eye className="h-5 w-5" />
            </Button>
          </PopoverTrigger>
          <PopoverContent side="left" className="w-52 space-y-2">
            <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              Capas visibles
            </div>
            {(
              [
                ["nodes", "Nodos"],
                ["links", "Enlaces LoS"],
                ["fires", "Focos"],
                ["smoke", "Humo"],
                ["drones", "Drones"],
              ] as const
            ).map(([key, label]) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span>{label}</span>
                <Switch
                  checked={layerVisibility[key]}
                  onCheckedChange={() => toggleLayer(key)}
                />
              </div>
            ))}
          </PopoverContent>
        </Popover>
      </div>
    </TooltipProvider>
  );
}
