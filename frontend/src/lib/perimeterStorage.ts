/**
 * Persistencia local de perímetros guardados (localStorage).
 * Permite guardar, listar y eliminar polígonos definidos por el usuario.
 */
export interface SavedPerimeter {
  id: string;
  name: string;
  coords: [number, number][];
  createdAt: number;
}

const KEY = "ignis.savedPerimeters.v1";

export function listSavedPerimeters(): SavedPerimeter[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const data = JSON.parse(raw);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function writeAll(items: SavedPerimeter[]) {
  localStorage.setItem(KEY, JSON.stringify(items));
}

export function savePerimeter(name: string, coords: [number, number][]): SavedPerimeter {
  const item: SavedPerimeter = {
    id: `prm_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
    name: name.trim() || `Perímetro ${new Date().toLocaleString()}`,
    coords,
    createdAt: Date.now(),
  };
  writeAll([item, ...listSavedPerimeters()]);
  return item;
}

export function deleteSavedPerimeter(id: string): void {
  writeAll(listSavedPerimeters().filter((p) => p.id !== id));
}

export function clearSavedPerimeters(): void {
  writeAll([]);
}
