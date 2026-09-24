import { useEffect, useState } from 'react';
import { ReconnectingWS } from '../lib/ws';
import { useIgnisStore } from '../store/useIgnisStore';

export function useLiveWS() {
  const [connected, setConnected] = useState(false);
  // Asumimos que los metodos setLiveState y setWsConnected se agregaran a useIgnisStore
  const setLiveState = useIgnisStore((state: any) => state.setLiveState);
  const setWsConnected = useIgnisStore((state: any) => state.setWsConnected);

  useEffect(() => {
    const wsUrl = 'ws://192.168.0.9:8000/ws/live';
    const ws = new ReconnectingWS(wsUrl);

    ws.onOpen = () => {
      console.log('🔗 Conectado a Live Bridge WS');
      setConnected(true);
      if(setWsConnected) setWsConnected(true);
    };

    ws.onClose = () => {
      console.log('❌ Desconectado de Live Bridge WS');
      setConnected(false);
      if(setWsConnected) setWsConnected(false);
    };

    ws.onMessage = (data) => {
      if (data.type === 'tick' && setLiveState) {
        setLiveState(data);
      }
    };

    ws.connect();
    return () => ws.close();
  }, [setLiveState, setWsConnected]);

  return { connected };
}
