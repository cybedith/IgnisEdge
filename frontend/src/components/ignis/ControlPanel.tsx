import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { useIgnisStore } from "@/store/useIgnisStore";
import { Wind, Compass, Thermometer } from "lucide-react";

/**
 * Environmental telemetry control panel.
 * In SIMULATION: sliders drive the store directly.
 * In LIVE_HARDWARE: these readings should be pushed from sensor WebSocket
 * via setEnvironmentParams (see useIgnisStore).
 */
export function ControlPanel() {
  const { windSpeed, windDirection, temperature } = useIgnisStore(
    (s) => s.environmentParams
  );
  const setWindSpeed = useIgnisStore((s) => s.setWindSpeed);
  const setWindDirection = useIgnisStore((s) => s.setWindDirection);
  const setTemperature = useIgnisStore((s) => s.setTemperature);

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
            <Wind className="h-3.5 w-3.5" /> Velocidad Viento
          </Label>
          <span className="font-mono text-sm text-foreground">
            {windSpeed.toFixed(0)} km/h
          </span>
        </div>
        <Slider
          value={[windSpeed]}
          min={0}
          max={120}
          step={1}
          onValueChange={([v]) => setWindSpeed(v)}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
            <Compass className="h-3.5 w-3.5" /> Dirección Viento
          </Label>
          <span className="font-mono text-sm text-foreground">{windDirection}°</span>
        </div>
        <Slider
          value={[windDirection]}
          min={0}
          max={360}
          step={1}
          onValueChange={([v]) => setWindDirection(v)}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
            <Thermometer className="h-3.5 w-3.5" /> Temperatura
          </Label>
          <span className="font-mono text-sm text-foreground">
            {temperature.toFixed(0)} °C
          </span>
        </div>
        <Slider
          value={[temperature]}
          min={-10}
          max={50}
          step={1}
          onValueChange={([v]) => setTemperature(v)}
        />
      </div>
    </div>
  );
}
