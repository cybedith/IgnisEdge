/**
 * Modal to confirm a hotspot placement (temperature + smoke emission).
 * Campo `smoke_emission_g_per_s` alineado con el backend.
 */
import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Flame } from "lucide-react";

export interface HotspotDraft {
  longitude: number;
  latitude: number;
}

interface Props {
  draft: HotspotDraft | null;
  onCancel: () => void;
  onConfirm: (h: {
    longitude: number;
    latitude: number;
    temperature_c: number;
    smoke_emission_g_per_s: number;
  }) => void;
}

export function HotspotDialog({ draft, onCancel, onConfirm }: Props) {
  const [temp, setTemp] = useState(600);
  const [smoke, setSmoke] = useState(800);

  useEffect(() => {
    if (draft) { setTemp(600); setSmoke(800); }
  }, [draft]);

  return (
    <Dialog open={!!draft} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base">
            <Flame className="h-4 w-4 text-primary" /> Foco de calor
          </DialogTitle>
        </DialogHeader>

        {draft && (
          <div className="space-y-4 text-xs">
            <div className="rounded-md border border-border/50 bg-background/40 px-2.5 py-1.5 font-mono text-[11px]">
              {draft.longitude.toFixed(5)}, {draft.latitude.toFixed(5)}
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Temperatura
                </Label>
                <span className="font-mono">{temp} °C</span>
              </div>
              <Slider value={[temp]} min={300} max={1200} step={10} onValueChange={([v]) => setTemp(v)} />
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Emisión humo
                </Label>
                <span className="font-mono">{smoke} g/s</span>
              </div>
              <Slider value={[smoke]} min={100} max={3000} step={50} onValueChange={([v]) => setSmoke(v)} />
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} size="sm">Cancelar</Button>
          <Button
            size="sm"
            onClick={() =>
              draft &&
              onConfirm({
                longitude: draft.longitude,
                latitude: draft.latitude,
                temperature_c: temp,
                smoke_emission_g_per_s: smoke,
              })
            }
          >
            Encender
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
