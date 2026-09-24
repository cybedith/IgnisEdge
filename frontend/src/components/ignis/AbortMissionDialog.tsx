import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { ShieldAlert } from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AbortMissionDialog({ open, onOpenChange }: Props) {
  const [password, setPassword] = useState("");

  const handleAbort = async () => {
    if (password !== "ignis2026") {
      toast.error("Contraseña de Administrador incorrecta. Acceso denegado.");
      return;
    }

    try {
      // Disparamos la instrucción de aborto al cerebro de la Jetson AI
      const res = await fetch("http://192.168.0.9:8080/api/abort", { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }) 
      });
      
      if (!res.ok) {
        throw new Error("Contraseña rechazada por la Jetson (Nivel 2).");
      }

      toast.success("Misión Abortada. Forzando RTL (Return to Launch).");
      setPassword("");
      onOpenChange(false);
    } catch (e) {
      toast.error(`Error de red al intentar abortar: ${(e as Error).message}`);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md border-red-900 bg-black/95 text-white shadow-2xl shadow-red-900/50">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-500 tracking-wider">
            <ShieldAlert className="h-5 w-5 animate-pulse" />
            ABORTAR MISIÓN (EMERGENCIA)
          </DialogTitle>
          <DialogDescription className="text-gray-400 text-xs">
            Esta acción cancelará cualquier objetivo actual e instruirá a la Pixhawk a ejecutar un <strong className="text-white">RTL (Return to Launch)</strong> de inmediato. Requiere autorización aeronáutica Nivel 2.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col space-y-4 py-4">
          <Input
            type="password"
            placeholder="Contraseña de Administrador C2"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="bg-black/50 border-red-900/50 text-red-400 font-mono tracking-widest text-center"
            autoComplete="off"
          />
        </div>
        <DialogFooter className="sm:justify-between">
          <Button type="button" variant="ghost" className="text-gray-400 hover:text-white" onClick={() => onOpenChange(false)}>
            Cancelar
          </Button>
          <Button type="button" variant="destructive" className="font-bold tracking-widest" onClick={handleAbort}>
            CONFIRMAR RTL
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
