/**
 * Chat Page with Session Management
 *
 * Complete integration of SessionList and auto-save functionality.
 * This is a new version - replace the original page.tsx with this after testing.
 */

"use client";

import { useState, useRef, useEffect } from "react";
import { SessionList } from "@/components/SessionList";
import { useSessionManager } from "@/hooks/useSessionManager";
// ... (keep all other imports from original page.tsx)

// Copy Message interface and ModelParameters from original

export default function ChatPageWithSessions() {
  // Session management
  const {
    sessions,
    currentSession,
    loading: sessionLoading,
    createSession,
    loadSession,
    addMessage,
    updateTitle,
    deleteSession,
    startNewSession,
    setCurrentSession,
  } = useSessionManager();

  // Copy all state from original ChatPage
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("qwen2.5:0.5b");
  const [selectedFeature, setSelectedFeature] = useState<FeatureMode>("chat");
  // ... (copy all other state)

  // Load session messages when currentSession changes
  useEffect(() => {
    if (currentSession) {
      // Convert session messages to local format
      const sessionMessages = currentSession.messages.map((msg: any) => ({
        id: `${msg.timestamp}`,
        role: msg.role,
        content: msg.content,
        timestamp: new Date(msg.timestamp),
        model: msg.model,
        usage: msg.usage,
      }));
      setMessages(sessionMessages);
      setModel(currentSession.model);
      setSelectedFeature(currentSession.feature_mode as FeatureMode);
    }
  }, [currentSession]);

  // Handle session selection
  const handleSelectSession = async (sessionId: string) => {
    await loadSession(sessionId);
  };

  // Handle new session creation
  const handleCreateSession = async () => {
    const title = `New Chat ${new Date().toLocaleString()}`;
    await createSession(title, selectedFeature, model);
  };

  // Modified handleSubmit with auto-save
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() && attachedImages.length === 0 && attachedFiles.length === 0) {
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
      images: attachedImages,
      files: attachedFiles,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setAttachedImages([]);
    setAttachedFiles([]);
    setLoading(true);

    try {
      // Create session if not exists
      let sessionId = currentSession?.session_id;
      if (!sessionId) {
        const newSession = await createSession(
          input.substring(0, 50) + "...",
          selectedFeature,
          model
        );
        sessionId = newSession?.session_id;
      }

      // Save user message
      if (sessionId) {
        await addMessage(sessionId, "user", input, model);
      }

      // ... (copy chat logic from original handleSubmit)
      // After getting assistant response:

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.content,
        timestamp: new Date(),
        model,
        usage: response.usage,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Save assistant message
      if (sessionId) {
        await addMessage(
          sessionId,
          "assistant",
          response.content,
          model,
          response.usage
        );
      }

    } catch (error: any) {
      toast.error(getUserFriendlyError(error));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen">
      {/* Session Sidebar */}
      <SessionList
        sessions={sessions}
        currentSession={currentSession}
        onSelectSession={handleSelectSession}
        onCreateSession={handleCreateSession}
        onDeleteSession={deleteSession}
        onUpdateTitle={updateTitle}
        loading={sessionLoading}
      />

      {/* Main Chat Area - Copy from original page.tsx */}
      <PageLayout
        title="Unified Chat"
        description="All features in one interface"
        className="flex-1"
      >
        {/* Copy entire chat UI from original page.tsx */}
        {/* Just replace the handleSubmit reference */}
      </PageLayout>
    </div>
  );
}
