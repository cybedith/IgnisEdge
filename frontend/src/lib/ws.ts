/**
 * ReconnectingWS — small WebSocket wrapper with exponential backoff.
 *
 * 🔌 FUTURE: telemetry topics from LoRa gateway will use this same client.
 */

export interface WSHandlers {
  onMessage?: (data: unknown) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
}

export class ReconnectingWS {
  private ws: WebSocket | null = null;
  private closedByUser = false;
  private retry = 0;
  private timer: number | null = null;

  constructor(
    private readonly url: string,
    private readonly handlers: WSHandlers = {},
  ) {
    this.connect();
  }

  private connect() {
    try {
      this.ws = new WebSocket(this.url);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("[ws] connect failed", e);
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.retry = 0;
      this.handlers.onOpen?.();
    };
    this.ws.onmessage = (ev) => {
      try {
        this.handlers.onMessage?.(JSON.parse(ev.data));
      } catch {
        this.handlers.onMessage?.(ev.data);
      }
    };
    this.ws.onerror = (ev) => this.handlers.onError?.(ev);
    this.ws.onclose = () => {
      this.handlers.onClose?.();
      if (!this.closedByUser) this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    const delay = Math.min(30_000, 1_000 * Math.pow(2, this.retry++));
    this.timer = window.setTimeout(() => this.connect(), delay);
  }

  sendJson(obj: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  close() {
    this.closedByUser = true;
    if (this.timer) window.clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
  }

  get isOpen() {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
