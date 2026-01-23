# Frontend Development Patterns

**자동 활성화**: React, Next.js, TypeScript 작업 시
**모델**: sonnet

## Skill Description

React, Next.js, TypeScript를 사용한 프론트엔드 개발 패턴을 제공합니다. State 관리, Custom Hooks, Server Components 등 모던 프론트엔드 베스트 프랙티스를 포함합니다.

## When to Use

이 스킬은 다음 키워드 감지 시 자동 활성화됩니다:
- "React", "Next.js", "component"
- "useState", "useEffect", "hook"
- "TypeScript", "interface", "type"
- "Tailwind", "CSS", "스타일"
- "fetch", "API", "client"

## React Component Patterns

### Functional Components

```typescript
// ✅ Good: Type-safe functional component
interface ChatMessageProps {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export function ChatMessage({
  role,
  content,
  timestamp,
  isStreaming = false,
}: ChatMessageProps) {
  return (
    <div className={`message message-${role}`}>
      <div className="message-header">
        <span className="role">{role}</span>
        <time className="timestamp">
          {timestamp.toLocaleTimeString()}
        </time>
      </div>
      <div className="message-content">
        {content}
        {isStreaming && <span className="cursor">▋</span>}
      </div>
    </div>
  );
}
```

### Custom Hooks

```typescript
// ✅ Good: Reusable custom hook
import { useState, useCallback } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export function useChatStream(apiUrl: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      setIsLoading(true);
      setError(null);

      // Add user message
      const userMessage: Message = { role: "user", content };
      setMessages((prev) => [...prev, userMessage]);

      try {
        const response = await fetch(`${apiUrl}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [...messages, userMessage],
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        // Parse SSE stream
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let assistantMessage = "";

        while (reader) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = JSON.parse(line.slice(6));
              if (data.content) {
                assistantMessage += data.content;
                // Update streaming message
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage?.role === "assistant") {
                    newMessages[newMessages.length - 1] = {
                      role: "assistant",
                      content: assistantMessage,
                    };
                  } else {
                    newMessages.push({
                      role: "assistant",
                      content: assistantMessage,
                    });
                  }
                  return newMessages;
                });
              }
            }
          }
        }
      } catch (err) {
        setError(err as Error);
      } finally {
        setIsLoading(false);
      }
    },
    [apiUrl, messages]
  );

  return { messages, isLoading, error, sendMessage };
}
```

### Context Pattern

```typescript
// ✅ Good: Type-safe context
import { createContext, useContext, useState, ReactNode } from "react";

interface ModelContextType {
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  availableModels: string[];
}

const ModelContext = createContext<ModelContextType | undefined>(undefined);

export function ModelProvider({ children }: { children: ReactNode }) {
  const [selectedModel, setSelectedModel] = useState("gpt-4o");
  const [availableModels] = useState([
    "gpt-4o",
    "claude-sonnet-4",
    "gemini-2.5-pro",
  ]);

  return (
    <ModelContext.Provider
      value={{ selectedModel, setSelectedModel, availableModels }}
    >
      {children}
    </ModelContext.Provider>
  );
}

export function useModel() {
  const context = useContext(ModelContext);
  if (!context) {
    throw new Error("useModel must be used within ModelProvider");
  }
  return context;
}

// Usage
function ChatPage() {
  const { selectedModel, setSelectedModel } = useModel();

  return (
    <div>
      <ModelSelector
        value={selectedModel}
        onChange={setSelectedModel}
      />
    </div>
  );
}
```

## Next.js Patterns

### Server Components (App Router)

```typescript
// ✅ Good: Server Component (default in app/)
// app/chat/page.tsx
import { ChatInterface } from "@/components/ChatInterface";

// Fetch data on server
async function getModels() {
  const res = await fetch("http://localhost:8000/api/models", {
    cache: "no-store", // Dynamic data
  });
  return res.json();
}

export default async function ChatPage() {
  const models = await getModels();

  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold">Chat</h1>
      <ChatInterface models={models} />
    </main>
  );
}
```

### Client Components

```typescript
// ✅ Good: Client Component (interactive)
"use client";

import { useState } from "react";
import { useChatStream } from "@/hooks/useChatStream";

export function ChatInterface({ models }: { models: string[] }) {
  const [input, setInput] = useState("");
  const { messages, isLoading, sendMessage } = useChatStream(
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    await sendMessage(input);
    setInput("");
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} {...msg} />
        ))}
      </div>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          placeholder="Type a message..."
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
```

### API Routes

```typescript
// ✅ Good: API route with error handling
// app/api/chat/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate request
    if (!body.messages || !Array.isArray(body.messages)) {
      return NextResponse.json(
        { error: "Invalid request: messages is required" },
        { status: 400 }
      );
    }

    // Forward to backend
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Chat API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
```

## State Management

### useState + useReducer

```typescript
// ✅ Good: Complex state with useReducer
import { useReducer } from "react";

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: Error | null;
  settings: {
    temperature: number;
    maxTokens: number;
  };
}

type ChatAction =
  | { type: "ADD_MESSAGE"; payload: Message }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: Error | null }
  | { type: "UPDATE_SETTINGS"; payload: Partial<ChatState["settings"]> }
  | { type: "RESET" };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "ADD_MESSAGE":
      return {
        ...state,
        messages: [...state.messages, action.payload],
      };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "UPDATE_SETTINGS":
      return {
        ...state,
        settings: { ...state.settings, ...action.payload },
      };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

const initialState: ChatState = {
  messages: [],
  isLoading: false,
  error: null,
  settings: {
    temperature: 0.7,
    maxTokens: 2000,
  },
};

export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  const sendMessage = async (content: string) => {
    dispatch({ type: "SET_LOADING", payload: true });
    try {
      // ... send message
      dispatch({
        type: "ADD_MESSAGE",
        payload: { role: "user", content },
      });
    } catch (error) {
      dispatch({ type: "SET_ERROR", payload: error as Error });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  };

  return { state, dispatch, sendMessage };
}
```

## Form Handling

### Controlled Components

```typescript
// ✅ Good: Type-safe form handling
interface ChatSettings {
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
}

export function ChatSettingsForm({
  onSave,
}: {
  onSave: (settings: ChatSettings) => void;
}) {
  const [settings, setSettings] = useState<ChatSettings>({
    model: "gpt-4o",
    temperature: 0.7,
    maxTokens: 2000,
    systemPrompt: "",
  });

  const handleChange = <K extends keyof ChatSettings>(
    field: K,
    value: ChatSettings[K]
  ) => {
    setSettings((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(settings);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="model">Model</label>
        <select
          id="model"
          value={settings.model}
          onChange={(e) => handleChange("model", e.target.value)}
        >
          <option value="gpt-4o">GPT-4o</option>
          <option value="claude-sonnet-4">Claude Sonnet 4</option>
        </select>
      </div>

      <div>
        <label htmlFor="temperature">
          Temperature: {settings.temperature}
        </label>
        <input
          type="range"
          id="temperature"
          min="0"
          max="2"
          step="0.1"
          value={settings.temperature}
          onChange={(e) =>
            handleChange("temperature", parseFloat(e.target.value))
          }
        />
      </div>

      <button type="submit">Save Settings</button>
    </form>
  );
}
```

## Error Boundaries

```typescript
// ✅ Good: Error boundary for graceful error handling
"use client";

import { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error("Error boundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="error-container">
            <h2>Something went wrong</h2>
            <details>
              <summary>Error details</summary>
              <pre>{this.state.error?.message}</pre>
            </details>
            <button onClick={() => this.setState({ hasError: false })}>
              Try again
            </button>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <ChatInterface />
</ErrorBoundary>;
```

## Tailwind CSS Patterns

```typescript
// ✅ Good: Reusable Tailwind classes with cn() utility
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Component with variant props
interface ButtonProps {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  children: ReactNode;
  onClick?: () => void;
}

export function Button({
  variant = "primary",
  size = "md",
  children,
  onClick,
}: ButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        // Base styles
        "font-medium rounded-lg transition-colors",
        // Variant styles
        {
          "bg-blue-600 text-white hover:bg-blue-700": variant === "primary",
          "bg-gray-200 text-gray-900 hover:bg-gray-300":
            variant === "secondary",
          "bg-transparent hover:bg-gray-100": variant === "ghost",
        },
        // Size styles
        {
          "px-3 py-1.5 text-sm": size === "sm",
          "px-4 py-2 text-base": size === "md",
          "px-6 py-3 text-lg": size === "lg",
        }
      )}
    >
      {children}
    </button>
  );
}
```

## Related Documents

- `.claude/rules/coding-standards.md` - 코딩 스타일
- `playground/frontend/` - Next.js 예시
- `CLAUDE.md` - 프로젝트 컨텍스트
