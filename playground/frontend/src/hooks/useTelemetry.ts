import { useState, useEffect, useRef, useCallback } from "react";
import { API_URL } from "@/lib/api-client";

export interface TelemetryEvent {
  type: string;
  event_type: string;
  agent_id: string;
  timestamp: string;
  content: any;
  metadata: Record<string, any>;
}

export function useTelemetry(executionId: string | null) {
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!executionId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = API_URL.replace(/^https?:\/\//, "");
    
    const wsUrl = `${protocol}//${host}/api/telemetry/ws/${executionId}`;
    
    console.log(`Connecting to telemetry WebSocket: ${wsUrl}`);
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("Telemetry WebSocket connected");
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "telemetry") {
          setEvents((prev) => [...prev, data as TelemetryEvent]);
        }
      } catch (e) {
        console.error("Failed to parse telemetry event", e);
      }
    };

    socket.onclose = () => {
      console.log("Telemetry WebSocket disconnected");
      setIsConnected(false);
      // Reconnect after 3 seconds if executionId still exists
      if (executionId) {
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      }
    };

    socket.onerror = (error) => {
      console.error("Telemetry WebSocket error", error);
    };
  }, [executionId]);

  useEffect(() => {
    if (executionId) {
      setEvents([]);
      connect();
    } else {
      if (socketRef.current) {
        socketRef.current.close();
      }
      setEvents([]);
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [executionId, connect]);

  return { events, isConnected };
}
