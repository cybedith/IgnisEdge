/**
 * Global keyboard shortcuts for the C2:
 *   Esc       → exit any tool mode (back to IDLE)
 *   Space     → pause/resume simulation
 *   Cmd/Ctrl+K → (placeholder) command palette toggle
 */
import { useEffect } from "react";
import { useIgnisStore } from "@/store/useIgnisStore";
import { pauseSimulation, startSimulation } from "@/lib/api";

export function useKeyboardShortcuts() {
  useEffect(() => {
    const onKey = async (e: KeyboardEvent) => {
      // Don't fire while typing in inputs
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;

      const store = useIgnisStore.getState();

      if (e.key === "Escape") {
        if (store.toolMode !== "IDLE") {
          e.preventDefault();
          store.setToolMode("IDLE");
        }
        return;
      }

      if (e.code === "Space") {
        const session = store.session;
        if (!session) return;
        e.preventDefault();
        try {
          const updated = session.running
            ? await pauseSimulation(session.session_id)
            : await startSimulation(session.session_id);
          store.setSession(updated);
        } catch {
          /* ignore */
        }
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
