/**
 * useSessionManager Hook
 *
 * Manages chat sessions with MongoDB backend
 */

import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";

export interface ChatMessage {
  role: string;
  content: string;
  timestamp: string;
  model?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
  metadata?: Record<string, any>;
}

export interface ChatSession {
  _id?: string;
  session_id: string;
  title: string;
  feature_mode: string;
  model: string;
  messages: ChatMessage[];
  feature_options?: Record<string, any>;
  created_at: string;
  updated_at: string;
  total_tokens: number;
  message_count: number;
}

export interface UseSessionManagerOptions {
  apiUrl?: string;
  autoSave?: boolean;
}

export function useSessionManager(options: UseSessionManagerOptions = {}) {
  const { apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000", autoSave = true } = options;

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch sessions from API
  const fetchSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${apiUrl}/api/chat/sessions?limit=50`);
      if (!response.ok) {
        throw new Error(`Failed to fetch sessions: ${response.statusText}`);
      }

      const data = await response.json();
      setSessions(data.sessions || []);
    } catch (err: any) {
      setError(err.message);
      console.error("Failed to fetch sessions:", err);
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Load sessions on mount
  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Create new session
  const createSession = useCallback(
    async (title: string, featureMode: string, model: string, featureOptions?: Record<string, any>) => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${apiUrl}/api/chat/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title,
            feature_mode: featureMode,
            model,
            feature_options: featureOptions || {},
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to create session: ${response.statusText}`);
        }

        const data = await response.json();
        const newSession = data.session;

        setSessions((prev) => [newSession, ...prev]);
        setCurrentSession(newSession);

        toast.success(`Session "${title}" created`);
        return newSession;
      } catch (err: any) {
        setError(err.message);
        toast.error(`Failed to create session: ${err.message}`);
        console.error("Failed to create session:", err);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [apiUrl]
  );

  // Load specific session
  const loadSession = useCallback(
    async (sessionId: string) => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${apiUrl}/api/chat/sessions/${sessionId}`);
        if (!response.ok) {
          throw new Error(`Failed to load session: ${response.statusText}`);
        }

        const data = await response.json();
        setCurrentSession(data.session);

        return data.session;
      } catch (err: any) {
        setError(err.message);
        toast.error(`Failed to load session: ${err.message}`);
        console.error("Failed to load session:", err);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [apiUrl]
  );

  // Add message to current session
  const addMessage = useCallback(
    async (
      sessionId: string,
      role: string,
      content: string,
      model?: string,
      usage?: { total_tokens?: number },
      metadata?: Record<string, any>
    ) => {
      if (!autoSave) {
        // If autoSave is disabled, just update local state
        if (currentSession && currentSession.session_id === sessionId) {
          const newMessage: ChatMessage = {
            role,
            content,
            timestamp: new Date().toISOString(),
            model,
            usage,
            metadata,
          };
          setCurrentSession({
            ...currentSession,
            messages: [...currentSession.messages, newMessage],
            message_count: currentSession.message_count + 1,
            total_tokens: currentSession.total_tokens + (usage?.total_tokens || 0),
          });
        }
        return;
      }

      try {
        const response = await fetch(`${apiUrl}/api/chat/sessions/${sessionId}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role,
            content,
            model,
            usage,
            metadata,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to add message: ${response.statusText}`);
        }

        const data = await response.json();
        setCurrentSession(data.session);

        // Update session in list
        setSessions((prev) =>
          prev.map((s) => (s.session_id === sessionId ? data.session : s))
        );
      } catch (err: any) {
        console.error("Failed to add message:", err);
        // Don't show error toast for every message failure
      }
    },
    [apiUrl, autoSave, currentSession]
  );

  // Update session title
  const updateTitle = useCallback(
    async (sessionId: string, title: string) => {
      try {
        const response = await fetch(`${apiUrl}/api/chat/sessions/${sessionId}/title?title=${encodeURIComponent(title)}`, {
          method: "PATCH",
        });

        if (!response.ok) {
          throw new Error(`Failed to update title: ${response.statusText}`);
        }

        // Update local state
        setSessions((prev) =>
          prev.map((s) => (s.session_id === sessionId ? { ...s, title } : s))
        );

        if (currentSession && currentSession.session_id === sessionId) {
          setCurrentSession({ ...currentSession, title });
        }

        toast.success("Title updated");
      } catch (err: any) {
        toast.error(`Failed to update title: ${err.message}`);
        console.error("Failed to update title:", err);
      }
    },
    [apiUrl, currentSession]
  );

  // Delete session
  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        const response = await fetch(`${apiUrl}/api/chat/sessions/${sessionId}`, {
          method: "DELETE",
        });

        if (!response.ok) {
          throw new Error(`Failed to delete session: ${response.statusText}`);
        }

        // Update local state
        setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));

        if (currentSession && currentSession.session_id === sessionId) {
          setCurrentSession(null);
        }

        toast.success("Session deleted");
      } catch (err: any) {
        toast.error(`Failed to delete session: ${err.message}`);
        console.error("Failed to delete session:", err);
      }
    },
    [apiUrl, currentSession]
  );

  // Start new session (local only, not saved until first message)
  const startNewSession = useCallback(
    (title: string, featureMode: string, model: string) => {
      const newSession: ChatSession = {
        session_id: `temp_${Date.now()}`,
        title,
        feature_mode: featureMode,
        model,
        messages: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        total_tokens: 0,
        message_count: 0,
      };

      setCurrentSession(newSession);
      return newSession;
    },
    []
  );

  return {
    sessions,
    currentSession,
    loading,
    error,
    fetchSessions,
    createSession,
    loadSession,
    addMessage,
    updateTitle,
    deleteSession,
    startNewSession,
    setCurrentSession,
  };
}
