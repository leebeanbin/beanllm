import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useCallback,
  useRef,
} from "react";
import { useQueryState } from "nuqs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { ArrowRight } from "lucide-react";
import { createBeanLLMClient } from "@/lib/beanllm-client";
import { toast } from "sonner";
import { BeanIcon } from "@/components/icons/BeanIcon";
import type { Message } from "@langchain/langgraph-sdk";

/**
 * StreamState compatible with LangGraph SDK
 */
export interface StreamState {
  messages: Message[];
  ui?: any[];
}

/**
 * Submit options (compatible with LangGraph SDK)
 */
export interface SubmitOptions {
  command?: {
    update?: any[];
    resume?: any;
    [key: string]: any;
  };
  streamMode?: string | string[];
  streamSubgraphs?: boolean;
  streamResumable?: boolean;
  optimisticValues?: (prev: StreamState) => StreamState;
  checkpoint?: any;
  [key: string]: any;
}

/**
 * Stream Context Type (compatible with LangGraph SDK)
 */
interface StreamContextType {
  // State
  values: StreamState;
  isLoading: boolean;

  // Methods
  submit: (input: any, options?: SubmitOptions) => Promise<void>;
  stop: () => void;
  next: () => Promise<void>;

  // Additional
  assistantId: string;
  setAssistantId: (id: string) => void;
  model: string;
  setModel: (model: string) => void;
  clearMessages: () => void;

  // Compatibility shortcuts
  messages: Message[];
  error?: string;
  interrupt?: any;

  // Additional LangGraph SDK compatibility methods (stubs)
  getMessagesMetadata: (message: Message) => any;
  setBranch: (branch: string) => void;
}

const StreamContext = createContext<StreamContextType | undefined>(undefined);

/**
 * Convert response to LangGraph SDK Message format
 */
function toLangGraphMessage(content: string, role: "user" | "assistant" = "assistant"): Message {
  return {
    id: `msg-${Date.now()}-${Math.random()}`,
    type: role === "user" ? "human" : "ai",
    content,
    additional_kwargs: {},
    response_metadata: {},
  } as Message;
}

/**
 * Stream Session Component
 */
const StreamSession = ({
  children,
  apiUrl,
  assistantId: initialAssistantId,
}: {
  children: ReactNode;
  apiUrl: string;
  assistantId: string;
}) => {
  const [values, setValues] = useState<StreamState>({
    messages: [],
    ui: [],
  });
  const [isLoading, setIsLoading] = useState(false);
  const [assistantId, setAssistantId] = useState(initialAssistantId);
  const [model, setModel] = useState("gpt-4o-mini");
  const abortControllerRef = useRef<AbortController | null>(null);

  const client = createBeanLLMClient(apiUrl);

  /**
   * Submit message (compatible with LangGraph SDK)
   */
  const submit = useCallback(
    async (input: any, options?: SubmitOptions) => {
      setIsLoading(true);

      // Extract message content
      let messageContent = "";
      if (typeof input === "string") {
        messageContent = input;
      } else if (input && typeof input === "object") {
        if ("messages" in input && Array.isArray(input.messages)) {
          const lastMsg = input.messages[input.messages.length - 1];
          messageContent = typeof lastMsg === "string" ? lastMsg : lastMsg?.content || "";
        } else if ("content" in input) {
          messageContent = input.content;
        }
      }

      if (!messageContent) {
        setIsLoading(false);
        return;
      }

      // Add user message
      const userMessage = toLangGraphMessage(messageContent, "user");
      setValues((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      try {
        // Create abort controller
        abortControllerRef.current = new AbortController();

        // Convert to beanllm format
        const messagesForApi = [
          ...values.messages.map((m) => ({
            role: m.type === "human" ? "user" : "assistant",
            content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
          })),
          { role: "user", content: messageContent },
        ];

        // Call backend
        const response = await client.chat({
          messages: messagesForApi,
          assistant_id: assistantId,
          model: model,
          stream: false,
        });

        // Add assistant message
        const assistantMessage = toLangGraphMessage(response.content, "assistant");

        setValues((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
        }));
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          // Cancelled by user
          return;
        }

        const errorMessage = error instanceof Error ? error.message : "Unknown error";
        toast.error("Failed to send message", {
          description: errorMessage,
          duration: 5000,
        });
      } finally {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    },
    [client, values.messages, assistantId, model]
  );

  /**
   * Stop current request
   */
  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  }, []);

  /**
   * Next (placeholder for compatibility)
   */
  const next = useCallback(async () => {
    // Placeholder - not used in beanllm
  }, []);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setValues({
      messages: [],
      ui: [],
    });
  }, []);

  /**
   * Get messages metadata (stub for compatibility)
   */
  const getMessagesMetadata = useCallback((message: Message) => {
    // Stub - return undefined
    return undefined;
  }, []);

  /**
   * Set branch (stub for compatibility)
   */
  const setBranch = useCallback((branch: string) => {
    // Stub - branch switching not implemented in beanllm
  }, []);

  return (
    <StreamContext.Provider
      value={{
        values,
        isLoading,
        submit,
        stop,
        next,
        assistantId,
        setAssistantId,
        model,
        setModel,
        clearMessages,
        // Compatibility shortcuts
        messages: values.messages,
        error: undefined,
        interrupt: undefined,
        getMessagesMetadata,
        setBranch,
      }}
    >
      {children}
    </StreamContext.Provider>
  );
};

// Default values for the form
const DEFAULT_API_URL = "http://localhost:8000";
const DEFAULT_ASSISTANT_ID = "chat";

/**
 * Stream Provider Component
 */
export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  // Get environment variables
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;

  // Use URL params with env var fallbacks
  const [apiUrl, setApiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || DEFAULT_API_URL,
  });
  const [assistantId, setAssistantId] = useQueryState("assistantId", {
    defaultValue: envAssistantId || DEFAULT_ASSISTANT_ID,
  });

  // Show the form if we don't have required values
  if (!apiUrl || !assistantId) {
    return (
      <div className="flex min-h-screen w-full items-center justify-center p-4">
        <div className="mx-auto flex w-full max-w-md flex-col gap-8">
          {/* Header */}
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="flex items-center gap-3">
              <BeanIcon className="h-12 w-12 text-primary" />
              <h1 className="text-4xl font-bold">BeanLLM</h1>
            </div>
            <p className="text-muted-foreground">
              Unified LLM Framework Playground
            </p>
          </div>

          {/* Form */}
          <form
            onSubmit={(e) => {
              e.preventDefault();

              const form = e.target as HTMLFormElement;
              const formData = new FormData(form);
              const apiUrl = formData.get("apiUrl") as string;
              const assistantId = formData.get("assistantId") as string;

              setApiUrl(apiUrl);
              setAssistantId(assistantId);

              form.reset();
            }}
            className="flex flex-col gap-6 rounded-lg border bg-card p-6"
          >
            <div className="flex flex-col gap-2">
              <Label htmlFor="apiUrl">Backend URL</Label>
              <Input
                id="apiUrl"
                name="apiUrl"
                defaultValue={apiUrl || DEFAULT_API_URL}
                placeholder="http://localhost:8000"
                required
              />
              <p className="text-xs text-muted-foreground">
                FastAPI backend server address
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="assistantId">Starting Assistant</Label>
              <select
                id="assistantId"
                name="assistantId"
                defaultValue={assistantId || DEFAULT_ASSISTANT_ID}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                required
              >
                <option value="chat">Chat - General conversation</option>
                <option value="kg">Knowledge Graph - Entity extraction</option>
                <option value="rag">RAG - Document Q&A</option>
                <option value="agent">Agent - Autonomous tasks</option>
                <option value="web">Web Search - Search the web</option>
                <option value="rag_debug">RAG Debug - Debug pipeline</option>
                <option value="optimizer">Optimizer - Optimize performance</option>
                <option value="multi_agent">Multi-Agent - Agent collaboration</option>
                <option value="orchestrator">Orchestrator - Workflow management</option>
              </select>
              <p className="text-xs text-muted-foreground">
                You can switch anytime after launch
              </p>
            </div>

            <Button type="submit" className="mt-2">
              Launch Playground
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <StreamSession apiUrl={apiUrl} assistantId={assistantId}>
      {children}
    </StreamSession>
  );
};

/**
 * Hook to use Stream Context
 */
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

// Alias for convenience
export const useStream = useStreamContext;

export default StreamContext;
