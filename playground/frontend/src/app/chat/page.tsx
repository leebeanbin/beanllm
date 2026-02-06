"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { PageLayout } from "@/components/PageLayout";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ModelSelectorSimple } from "@/components/ModelSelectorSimple";
import { createBeanLLMClient } from "@/lib/beanllm-client";
import {
  LoaderCircle,
  Send,
  X,
  User,
  Bot,
  Brain,
  MessageSquare,
  Download,
  Upload,
  Trash2,
  Edit2,
  Image as ImageIcon,
  Paperclip,
  MoreHorizontal,
  Settings,
  Sparkles,
  Lightbulb,
  FileText,
  Search,
  Code,
  Zap,
  ArrowRight,
  SidebarOpen,
  SidebarClose,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ThinkMode } from "@/components/ThinkMode";
import { StreamingText } from "@/components/StreamingText";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Switch } from "@/components/ui/switch";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import { getUserFriendlyError } from "@/lib/error-messages";
import { streamMCPChat } from "@/lib/mcp-client";
import { ToolCallDisplay, ToolCallProgress } from "@/components/ToolCallDisplay";
import { TypingIndicator } from "@/components/TypingIndicator";
import { ReasoningStepper, DEFAULT_REASONING_STEPS } from "@/components/ReasoningStepper";
import { ReasoningNarrative } from "@/components/ReasoningNarrative";
import { FeatureSelector } from "@/components/FeatureSelector";
import { FeatureBadge } from "@/components/FeatureBadge";
import { GoogleServiceSelector } from "@/components/GoogleServiceSelector";
import { InfoPanel } from "@/components/InfoPanel";
import { ProviderWarning } from "@/components/ProviderWarning";
import { ApiKeyModal } from "@/components/ApiKeyModal";
import { GoogleConnectModal } from "@/components/GoogleConnectModal";
import { FeatureMode, GoogleService, ProviderConfig, AvailableModels } from "@/types/chat";

interface Message {
  id?: string;
  role: string;
  content: string;
  timestamp?: Date;
  thinking?: string;
  images?: string[];
  files?: Array<{ name: string; type: string; data: string }>;
  toolCalls?: ToolCallProgress[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
  requestId?: string;
  responseTime?: number;
  model?: string;
  provider?: string;
  /** Placeholder row shown while waiting for first stream chunk (smooth loading → chat transition) */
  isPlaceholder?: boolean;
}

interface ModelParameters {
  supports: {
    temperature: boolean;
    max_tokens: boolean;
    top_p: boolean;
    frequency_penalty: boolean;
    presence_penalty: boolean;
  };
  max_tokens: number;
  default_temperature: number;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(""); // 빈 문자열로 시작 - API에서 가져온 모델로 설정
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);
  const [topP, setTopP] = useState(1.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [loading, setLoading] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [activeToolCalls, setActiveToolCalls] = useState<ToolCallProgress[]>([]);
  const [reasoningPhaseIndex, setReasoningPhaseIndex] = useState<number>(0);
  /** 의사결정 UI: intent / tool_select 이벤트로 채워짐 */
  const [decisionBlock, setDecisionBlock] = useState<{
    intent?: string;
    confidence?: number;
    tools?: string[];
  } | null>(null);
  /** 제안 단계: proposal 이벤트 (챗 말풍선으로 표시) */
  const [lastProposal, setLastProposal] = useState<{
    nodes: number;
    pipeline: string[];
    reason: string;
  } | null>(null);
  /** Human-in-the-loop: human_approval/stream_paused 시 [Run][Cancel][Change tool], run_id로 재개 */
  const [humanApproval, setHumanApproval] = useState<{
    tool: string;
    query_snippet: string;
    actions: string[];
    run_id?: string;
  } | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const [modelParams, setModelParams] = useState<ModelParameters | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<FeatureMode>("agentic");
  const [selectedGoogleServices, setSelectedGoogleServices] = useState<GoogleService[]>([]);
  const [enableThinking, setEnableThinking] = useState(false);
  const [customInstruction, setCustomInstruction] = useState("");
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const [attachedFiles, setAttachedFiles] = useState<Array<{ name: string; type: string; data: string }>>([]);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState("");
  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const [isDragging, setIsDragging] = useState(false);
  const [modelSupportsThinking, setModelSupportsThinking] = useState(false);
  const [showGuide, setShowGuide] = useState(false);
  const [showInfoPanel, setShowInfoPanel] = useState(true); // Always visible by default
  const [infoPanelTab, setInfoPanelTab] = useState<"quickstart" | "models" | "session" | "settings">("quickstart");
  const [isInfoPanelCollapsed, setIsInfoPanelCollapsed] = useState(false);
  const [documentPreviewContent, setDocumentPreviewContent] = useState<string | undefined>();
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [googleModalOpen, setGoogleModalOpen] = useState(false);
  // Session state
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionSummary, setSessionSummary] = useState<string>("");
  const [sessionSummaryLoading, setSessionSummaryLoading] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch provider config and set default model on mount
  useEffect(() => {
    const fetchProviderConfig = async () => {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      try {
        await Promise.all([
          fetch(`${apiUrl}/api/config/providers`).catch(() => null),
        ]);

        // Fetch available models and set default to first installed Ollama model
        const modelsResponse = await fetch(`${apiUrl}/api/config/models`).catch(() => null);
        if (modelsResponse && modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          
          // Ollama 모델이 있으면 작은 모델을 우선 선택 (메모리 부족 방지)
          if (modelsData.ollama && Array.isArray(modelsData.ollama) && modelsData.ollama.length > 0) {
            // 작은 모델 우선순위: 0.5b, 1b, 1.3b, mini, 2b, 3b, 7b, 13b, 70b
            const smallModelPatterns = [
              /0\.5b|0\.5/i,
              /1b|1\.3b|mini/i,
              /2b/i,
              /3b/i,
              /7b/i,
              /13b/i,
              /70b/i,
            ];
            
            let selectedModel = modelsData.ollama[0]; // 기본값: 첫 번째 모델
            
            // 작은 모델 찾기
            for (const pattern of smallModelPatterns) {
              const smallModel = modelsData.ollama.find((m: string) => pattern.test(m));
              if (smallModel) {
                selectedModel = smallModel;
                break;
              }
            }
            
            // 모델이 아직 설정되지 않았거나, 현재 모델이 설치되지 않은 모델이면 변경
            if (!model || !modelsData.ollama.includes(model)) {
              setModel(selectedModel);
              console.log(`Default model set to: ${selectedModel} (memory-optimized selection from ${modelsData.ollama.length} available models)`);
            }
          } else if (!model) {
            // Ollama 모델이 없고 모델도 설정되지 않았으면 다른 Provider 모델 시도
            const allModels = Object.values(modelsData).flat() as string[];
            if (allModels.length > 0) {
              setModel(allModels[0]);
              console.log(`Default model set to: ${allModels[0]} (first available model)`);
            }
          }
        }
      } catch (error) {
        // Silently fail - backend might not be running
      }
    };
    fetchProviderConfig();
  }, []); // 빈 의존성 배열 - 마운트 시 한 번만 실행

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, activeToolCalls]);

  // Check if model supports thinking
  useEffect(() => {
    setModelSupportsThinking(model.includes("deepseek") || model.includes("qwen"));
  }, [model]);

  // Show guide on first visit (client-side only)
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const hasVisited = localStorage.getItem("beanllm_visited");
      if (!hasVisited && messages.length === 0) {
        setShowGuide(true);
        localStorage.setItem("beanllm_visited", "true");
      }
    } catch (error) {
      // localStorage not available, skip
    }
  }, [messages.length]);

  // Generate session ID on mount
  useEffect(() => {
    if (!sessionId) {
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      setSessionId(newSessionId);
    }
  }, [sessionId]);

  // Load session summary when session has enough messages
  const loadSessionSummary = async () => {
    if (!sessionId || messages.length < 10) return;

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    setSessionSummaryLoading(true);

    try {
      const response = await fetch(`${apiUrl}/api/chat/sessions/${sessionId}/summary`, {
        method: "GET",
      });

      if (response.ok) {
        const data = await response.json();
        if (data.summary) {
          setSessionSummary(data.summary);
        }
      }
    } catch (error) {
      console.debug("Failed to load session summary:", error);
    } finally {
      setSessionSummaryLoading(false);
    }
  };

  // Auto-load summary when messages reach threshold
  useEffect(() => {
    if (messages.length === 10 || messages.length === 20) {
      loadSessionSummary();
    }
  }, [messages.length]);

  // Load a past session from chat history (panel)
  const loadSession = useCallback(async (sid: string) => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    try {
      const response = await fetch(`${apiUrl}/api/chat/sessions/${sid}`);
      if (!response.ok) return;
      const data = await response.json();
      const session = data.session;
      if (!session) return;
      const rawMessages = session.messages || [];
      const mapped: Message[] = rawMessages.map((m: { role?: string; content?: string; content_preview?: string; timestamp?: string; model?: string; usage?: Record<string, number> }, idx: number) => ({
        id: (m as { message_id?: string }).message_id || `msg-${idx}`,
        role: m.role || "user",
        content: typeof m.content === "string" ? m.content : (m.content_preview || ""),
        timestamp: m.timestamp ? new Date(m.timestamp) : undefined,
        model: m.model,
        usage: m.usage,
      }));
      setMessages(mapped);
      setSessionId(session.session_id);
      setSessionSummary(session.summary ?? undefined);
    } catch (e) {
      console.debug("Failed to load session:", e);
    }
  }, []);

  // New chat — 10개 미만이면 "저장할까요?" 물어보고, 저장 시 백엔드에 세션+메시지 저장 후 비우기
  const handleNewChat = useCallback(async () => {
    const apiUrlForNew = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const effectiveFeature = mode === "auto" ? "agentic" : selectedFeature;

    if (messages.length > 0 && messages.length < 10) {
      const save = window.confirm("이 대화를 저장한 뒤 새 채팅을 시작할까요?");
      if (save) {
        try {
          const firstContent = messages.find((m) => m.role === "user")?.content ?? "";
          const createRes = await fetch(`${apiUrlForNew}/api/chat/sessions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: firstContent.slice(0, 50).trim() || "New chat",
              feature_mode: effectiveFeature,
              model,
              feature_options: {},
            }),
          });
          if (createRes.ok) {
            const data = await createRes.json();
            const sid = data.session?.session_id;
            if (sid) {
              for (const m of messages) {
                if (m.role !== "user" && m.role !== "assistant") continue;
                await fetch(`${apiUrlForNew}/api/chat/sessions/${sid}/messages`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    role: m.role,
                    content: m.content,
                    model: m.model ?? model,
                    usage: m.usage ? { total_tokens: m.usage.total_tokens } : undefined,
                  }),
                });
              }
              toast.success("대화가 저장되었습니다. 새 채팅을 시작합니다.");
            }
          }
        } catch (e) {
          console.debug("Save before new chat failed:", e);
        }
      }
    }

    setMessages([]);
    setSessionSummary("");
    setSessionId(`session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`);
  }, [messages, mode, selectedFeature, model]);

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const result = e.target?.result as string;
          setAttachedImages((prev) => [...prev, result]);
        };
        reader.readAsDataURL(file);
      }
    });

    if (e.target) {
      e.target.value = "";
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // Add files to UI state (for display in chat)
    for (const file of fileArray) {
      const reader = new FileReader();
      reader.onload = (readerEvent) => {
        const result = readerEvent.target?.result as string;
        setAttachedFiles((prev) => [
          ...prev,
          { name: file.name, type: file.type, data: result },
        ]);
      };
      reader.readAsDataURL(file);
    }

    // Upload to session RAG for auto-indexing (if session exists)
    if (sessionId) {
      const formData = new FormData();
      for (const file of fileArray) {
        formData.append("files", file);
      }

      try {
        const response = await fetch(
          `${apiUrl}/api/rag/session/${sessionId}/upload`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.ok) {
          const data = await response.json();
          if (data.added_documents > 0) {
            toast.success(
              `${data.added_documents} document(s) indexed for RAG`,
              { description: `Total: ${data.total_documents} documents in session` }
            );
          }
          if (data.failed_files?.length > 0) {
            toast.warning(
              `${data.failed_files.length} file(s) failed to process`,
              { description: data.failed_files.map((f: { filename: string }) => f.filename).join(", ") }
            );
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "Upload failed");
        }
      } catch (error) {
        console.error("RAG upload error:", error);
        toast.error("Failed to index documents for RAG", {
          description: error instanceof Error ? error.message : "Unknown error",
        });
      }
    }

    if (e.target) {
      e.target.value = "";
    }
  };

  const handleRemoveImage = (index: number) => {
    setAttachedImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRemoveFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const processFiles = async (files: FileList) => {
    const imageFiles: string[] = [];
    const otherFiles: Array<{ name: string; type: string; data: string }> = [];
    const ragFiles: File[] = [];

    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        const promise = new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
        });
        reader.readAsDataURL(file);
        imageFiles.push(await promise);
      } else {
        const reader = new FileReader();
        const promise = new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
        });
        reader.readAsDataURL(file);
        const data = await promise;
        otherFiles.push({
          name: file.name,
          type: file.type,
          data,
        });
        ragFiles.push(file);
      }
    }

    if (imageFiles.length > 0) {
      setAttachedImages((prev) => [...prev, ...imageFiles]);
    }
    if (otherFiles.length > 0) {
      setAttachedFiles((prev) => [...prev, ...otherFiles]);
    }

    // Upload non-image files to session RAG for auto-indexing
    if (ragFiles.length > 0 && sessionId) {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const formData = new FormData();
      for (const file of ragFiles) {
        formData.append("files", file);
      }

      try {
        const response = await fetch(
          `${apiUrl}/api/rag/session/${sessionId}/upload`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.ok) {
          const data = await response.json();
          if (data.added_documents > 0) {
            toast.success(
              `${data.added_documents} document(s) indexed for RAG`,
              { description: `Total: ${data.total_documents} documents in session` }
            );
          }
          if (data.failed_files?.length > 0) {
            toast.warning(
              `${data.failed_files.length} file(s) failed to process`
            );
          }
        }
      } catch (error) {
        console.error("RAG upload error (drag-drop):", error);
        toast.error("Failed to index documents for RAG");
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await processFiles(files);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading || (!input.trim() && attachedImages.length === 0 && attachedFiles.length === 0)) return;
    
    // 모델이 설정되지 않았으면 에러 표시
    if (!model || !model.trim()) {
      toast.error("Select a model. Loading available models...");
      return;
    }

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
      images: attachedImages.length > 0 ? [...attachedImages] : undefined,
      files: attachedFiles.length > 0 ? [...attachedFiles] : undefined,
    };

    const placeholderId = `gen-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      userMessage,
      { role: "assistant", content: "", id: placeholderId, isPlaceholder: true },
    ]);
    setInput("");
    setAttachedImages([]);
    setAttachedFiles([]);
    setLoading(true);
    setActiveToolCalls([]);

    const effectiveFeature = mode === "auto" ? "agentic" : selectedFeature;

    // FeatureMode를 IntentType으로 매핑
    // "agentic"은 자동 분류이므로 force_intent를 전달하지 않음
    const featureToIntentMap: Record<string, string | undefined> = {
      "agentic": undefined,  // 자동 분류
      "chat": "chat",
      "rag": "rag",
      "multi-agent": "multi_agent",
      "knowledge-graph": "kg",
      "audio": "audio",
      "ocr": "ocr",
      "web-search": "web_search",
      "google": undefined,  // Google은 여러 서비스가 있어 자동 분류
    };
    
    const forceIntent = featureToIntentMap[effectiveFeature];

    // 첫 메시지면 세션 생성 → 히스토리에 제목으로 바로 표시
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    let sidForSave: string | null = messages.length === 0 ? null : sessionId;
    if (messages.length === 0) {
      try {
        const createRes = await fetch(`${apiUrl}/api/chat/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: (userMessage.content || "New chat").slice(0, 50).trim() || "New chat",
            feature_mode: effectiveFeature,
            model,
            feature_options: {},
          }),
        });
        if (createRes.ok) {
          const data = await createRes.json();
          sidForSave = data.session?.session_id ?? null;
          if (sidForSave) {
            setSessionId(sidForSave);
            toast.success("새 채팅이 저장되었습니다. 히스토리에서 확인할 수 있습니다.");
          }
        }
      } catch (e) {
        console.debug("Create session failed:", e);
      }
    }

    try {
      setReasoningPhaseIndex(0);
      setDecisionBlock(null);
      setLastProposal(null);
      setHumanApproval(null);
      streamAbortRef.current = new AbortController();
      const startTime = Date.now();
      let assistantContent = "";
      let thinking: string | undefined;
      let toolCalls: ToolCallProgress[] = [];
      let usage: Message["usage"];
      let responseModel: string | undefined;
      let responseProvider: string | undefined;

      console.log("Starting MCP chat stream:", {
        model,
        temperature,
        max_tokens: maxTokens,
        effectiveFeature,
        force_intent: forceIntent || "auto",
        message_count: messages.length + 1,
      });

      await streamMCPChat(
        {
          messages: [
            ...messages.map((m) => ({
              role: m.role as "user" | "assistant" | "system",
              content: m.content,
            })),
            {
              role: "user",
              content: userMessage.content,
            },
          ],
          model,
          temperature,
          max_tokens: maxTokens,
          force_intent: forceIntent,  // undefined면 자동 분류
        },
        (event) => {
          console.log("MCP event received:", event.type, event.data);
          if (event.type === "intent") {
            setDecisionBlock((prev) => ({
              ...prev,
              intent: event.data.primary_intent as string,
              confidence: event.data.confidence as number,
            }));
          } else if (event.type === "tool_select") {
            setDecisionBlock((prev) => ({
              ...prev,
              tools: (event.data.tools as string[]) || [],
            }));
          } else if (event.type === "proposal") {
            const nodes = (event.data.nodes as number) ?? 0;
            const pipeline = (event.data.pipeline as string[]) ?? [];
            const reason = (event.data.reason as string) ?? "";
            setLastProposal({ nodes, pipeline, reason });
            // 챗 말풍선에 제안 텍스트 반영: assistant 메시지가 있으면 앞에 붙임
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === "assistant") {
                const prefix = last.content ? last.content + "\n\n" : "";
                last.content = prefix + reason;
              } else {
                next.push({
                  role: "assistant",
                  content: reason,
                  timestamp: new Date(),
                  id: `msg-${Date.now()}`,
                  isPlaceholder: true,
                });
                setStreamingMessageId(`msg-${Date.now()}`);
              }
              return next;
            });
          } else if (event.type === "human_approval") {
            setHumanApproval({
              tool: (event.data.tool as string) ?? "",
              query_snippet: (event.data.query_snippet as string) ?? "",
              actions: (event.data.actions as string[]) ?? ["run", "cancel", "change_tool"],
              run_id: (event.data.run_id as string) ?? undefined,
            });
          } else if (event.type === "stream_paused") {
            setHumanApproval((prev) =>
              prev
                ? { ...prev, run_id: (event.data.run_id as string) ?? prev.run_id }
                : {
                    tool: (event.data.tool as string) ?? "",
                    query_snippet: "",
                    actions: ["run", "cancel", "change_tool"],
                    run_id: (event.data.run_id as string) ?? undefined,
                  }
            );
          } else if (event.type === "text") {
            setReasoningPhaseIndex(2);
            const content = event.data.content as string;
            if (content) {
              assistantContent += content;
              setMessages((prev) => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage && lastMessage.role === "assistant") {
                  lastMessage.content = assistantContent;
                  lastMessage.isPlaceholder = false;
                  setStreamingMessageId(lastMessage.id ?? null);
                } else {
                  const newMsg: Message = {
                    role: "assistant",
                    content: assistantContent,
                    timestamp: new Date(),
                    id: `msg-${Date.now()}`,
                    isPlaceholder: false,
                  };
                  newMessages.push(newMsg);
                  setStreamingMessageId(newMsg.id ?? null);
                }
                return newMessages;
              });
            }
          } else if (event.type === "tool_call" || event.type === "tool_start") {
            setReasoningPhaseIndex(1);
            const toolCall: ToolCallProgress = {
              tool: (event.data.tool as string) || "unknown",
              status: "running",
              step: event.data.step as string,
              message: event.data.message as string,
              arguments: event.data.arguments as Record<string, any>,
            };
            toolCalls.push(toolCall);
            setActiveToolCalls([...toolCalls]);
          } else if (event.type === "tool_result") {
            const toolName = event.data.tool as string;
            const index = toolCalls.findIndex((tc) => tc.tool === toolName);
            if (index !== -1) {
              toolCalls[index].result = event.data.result as Record<string, any>;
              toolCalls[index].status = "completed";
              setActiveToolCalls([...toolCalls]);
            }
          } else if (event.type === "tool_progress") {
            setReasoningPhaseIndex(1);
            const toolName = event.data.tool as string;
            const index = toolCalls.findIndex((tc) => tc.tool === toolName);
            if (index !== -1) {
              toolCalls[index].progress = event.data.progress as number;
              toolCalls[index].step = event.data.step as string;
              toolCalls[index].message = event.data.message as string;
              setActiveToolCalls([...toolCalls]);
            }
          } else if (event.type === "error") {
            console.error("MCP error event:", event.data);
            const errorMessage = (event.data.message as string) || "An unknown error occurred";
            const originalError = (event.data.original_error as string) || errorMessage;

            let userMessage = errorMessage;
            if (errorMessage.includes("메모리") || errorMessage.includes("memory")) {
              userMessage = "Out of memory: choose a smaller model or try another.";
            } else if (errorMessage.includes("not found") || errorMessage.includes("404")) {
              userMessage = "Model not found. Select a valid model from the model selector.";
            }

            toast.error(userMessage, {
              description: originalError !== errorMessage ? originalError : undefined,
              duration: 5000,
            });

            setMessages((prev) => {
              const newMessages = [...prev];
              const lastMessage = newMessages[newMessages.length - 1];
              if (lastMessage && lastMessage.role === "assistant") {
                lastMessage.content = `Error: ${userMessage}`;
              } else {
                newMessages.push({
                  role: "assistant",
                  content: `Error: ${userMessage}`,
                  timestamp: new Date(),
                });
              }
              return newMessages;
            });
            throw new Error(errorMessage);
          } else if (event.type === "done") {
            console.log("MCP stream completed:", event.data);
            setStreamingMessageId(null);
            setReasoningPhaseIndex(0);
            setDecisionBlock(null);
            setLastProposal(null);
            setHumanApproval(null);
          } else {
            console.log("Unhandled MCP event type:", event.type, event.data);
          }
        },
        streamAbortRef.current?.signal
      );

      const responseTime = Date.now() - startTime;

      setMessages((prev) => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          lastMessage.thinking = thinking;
          lastMessage.usage = usage;
          lastMessage.responseTime = responseTime;
          lastMessage.model = responseModel;
          lastMessage.provider = responseProvider;
        }
        return newMessages;
      });

      // 세션에 user/assistant 메시지 저장 → 히스토리에서 불러올 수 있게
      if (sidForSave && userMessage.content) {
        try {
          await fetch(`${apiUrl}/api/chat/sessions/${sidForSave}/messages`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ role: "user", content: userMessage.content, model }),
          });
          await fetch(`${apiUrl}/api/chat/sessions/${sidForSave}/messages`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              role: "assistant",
              content: assistantContent,
              model: responseModel ?? model,
              usage: usage ? { total_tokens: usage.total_tokens } : undefined,
            }),
          });
        } catch (e) {
          console.debug("Save messages to session failed:", e);
        }
      }

      setActiveToolCalls([]);
      setReasoningPhaseIndex(0);
    } catch (error: any) {
      setReasoningPhaseIndex(0);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.isPlaceholder) next.pop();
        return next;
      });
      const errorInfo = getUserFriendlyError(error);
      const errorMessage = typeof errorInfo === "string" ? errorInfo : errorInfo.message;
      
      let userMessage = errorMessage;
      if (errorMessage.includes("메모리") || errorMessage.includes("memory") || errorMessage.includes("requires more")) {
        userMessage = "Out of memory: choose a smaller model or try another.";
      } else if (errorMessage.includes("not found") || errorMessage.includes("404")) {
        userMessage = "Model not found. Select a valid model from the model selector.";
      }

      toast.error(userMessage, {
        description: errorMessage !== userMessage ? errorMessage : undefined,
        duration: 5000,
      });

      setMessages((prev) => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          lastMessage.content = `Error: ${userMessage}`;
        } else {
          newMessages.push({
            role: "assistant",
            content: `Error: ${userMessage}`,
            timestamp: new Date(),
          });
        }
        return newMessages;
      });
    } finally {
      setLoading(false);
      setActiveToolCalls([]);
      setStreamingMessageId(null);
    }
  };

  const handleStartEdit = (messageId: string, content: string) => {
    setEditingMessageId(messageId);
    setEditingContent(content);
  };

  const handleSaveEdit = (messageId: string) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === messageId ? { ...msg, content: editingContent } : msg))
    );
    setEditingMessageId(null);
    setEditingContent("");
  };

  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setEditingContent("");
  };

  const handleDeleteMessage = (messageId: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(messages, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `chat-${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const imported = JSON.parse(e.target?.result as string);
            setMessages(imported);
            toast.success("Chat history imported");
          } catch (error) {
            toast.error("Cannot read file");
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const handleClear = () => {
    if (confirm("Are you sure you want to delete all messages?")) {
      setMessages([]);
      toast.success("Chat history cleared");
    }
  };

  return (
    <PageLayout
      title="Chat"
      description="Unified LLM Interface"
      headerTrailing={
        showInfoPanel ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsInfoPanelCollapsed((c) => !c)}
            className="min-h-[44px] min-w-[44px] h-9 w-9 sm:h-8 sm:w-8 p-0 text-muted-foreground hover:text-foreground hover:bg-muted/50 touch-manipulation"
            aria-label={isInfoPanelCollapsed ? "Expand panel" : "Collapse panel"}
          >
            {isInfoPanelCollapsed ? (
              <ChevronLeft className="h-4 w-4" strokeWidth={1.5} />
            ) : (
              <ChevronRight className="h-4 w-4" strokeWidth={1.5} />
            )}
          </Button>
        ) : null
      }
    >
      {/* Provider SDK 경고 */}
      <ProviderWarning />
      
      {/* 2-Column Layout: Chat (left) + Info Panel (right) */}
      <div className="flex h-full min-h-0 bg-background">
        {/* Left: Chat Area - Uses flex-1 to take remaining space */}
        <div className="flex flex-col flex-1 min-w-0">

          {/* Messages Area */}
          <div
            className={cn(
              "flex-1 min-h-0",
              messages.length === 0 ? "overflow-hidden" : "overflow-y-auto",
              "py-4 sm:py-6"
            )}
          >
          {/* Container - Same max-width as input area */}
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Empty State - Functional minimalism: 카드 없음, 타이포 중심 */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center w-full h-full py-8">
                <div className="w-full max-w-2xl space-y-6">
                  <div className="text-center space-y-3">
                    <div className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                      <Sparkles className="h-6 w-6 text-muted-foreground" strokeWidth={2} />
                    </div>
                    <h2 className="text-lg font-semibold text-foreground">
                      How can I help you today?
                    </h2>
                    <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                      Open panel (→) → History → Start new chat. Model & settings in the panel.
                    </p>
                  </div>

                  {/* Quick Actions - border만, 그라데이션·shadow 없음 */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {[
                      { icon: Lightbulb, title: "Generate Ideas", prompt: "Suggest creative project ideas" },
                      { icon: Code, title: "Write Code", prompt: "Create a React component" },
                      { icon: FileText, title: "Analyze Document", prompt: "Summarize the main content of this document" },
                      { icon: Search, title: "Search Information", prompt: "Tell me about the latest AI trends" },
                    ].map((example, idx) => {
                      const Icon = example.icon;
                      return (
                        <button
                          key={idx}
                          onClick={() => {
                            setInput(example.prompt);
                            textareaRef.current?.focus();
                          }}
                          className="group flex items-start gap-3 p-3 rounded-lg border border-border/50 bg-background hover:bg-muted/30 text-left focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                          <Icon className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" strokeWidth={1.5} />
                          <div className="flex-1 min-w-0 space-y-0.5">
                            <span className="text-sm font-medium text-foreground block">{example.title}</span>
                            <span className="text-xs text-muted-foreground line-clamp-2 block">{example.prompt}</span>
                          </div>
                          <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 mt-1" strokeWidth={1.5} />
                        </button>
                      );
                    })}
                  </div>

                  {/* Progressive hints - Collapsible */}
                  {showGuide && (
                    <div className="pt-4 border-t border-border/30">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-medium text-foreground">Quick Tips</h3>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setShowGuide(false)}
                          className="h-7 w-7 p-0"
                        >
                          <X className="h-4 w-4" strokeWidth={1.5} />
                        </Button>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/30 border border-border/30">
                          <Zap className="h-4 w-4 shrink-0 mt-0.5 text-primary" strokeWidth={1.5} />
                          <div className="space-y-1">
                            <div className="text-sm font-medium text-foreground">Auto Mode</div>
                            <div className="text-xs text-muted-foreground leading-relaxed">AI automatically selects the appropriate feature</div>
                          </div>
                        </div>
                        <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/30 border border-border/30">
                          <Paperclip className="h-4 w-4 shrink-0 mt-0.5 text-primary" strokeWidth={1.5} />
                          <div className="space-y-1">
                            <div className="text-sm font-medium text-foreground">Attach Files</div>
                            <div className="text-xs text-muted-foreground leading-relaxed">Drag and drop files to attach them</div>
                          </div>
                        </div>
                        <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/30 border border-border/30">
                          <Settings className="h-4 w-4 shrink-0 mt-0.5 text-primary" strokeWidth={1.5} />
                          <div className="space-y-1">
                            <div className="text-sm font-medium text-foreground">Manual Mode</div>
                            <div className="text-xs text-muted-foreground leading-relaxed">Select the feature you want directly</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Messages - ChatGPT/Gemini style: clean and simple */}
            <div className="space-y-6">
              {messages.map((msg) => {
                const isStreaming = streamingMessageId === msg.id && loading;
                return (
                  <div key={msg.id || `msg-${msg.timestamp?.getTime()}`} className="group/message">
                  <div
                    className={cn(
                      "flex gap-3 items-start",
                      msg.role === "user" ? "flex-row-reverse" : "flex-row"
                    )}
                  >
                    {/* Avatar - Simple and clean */}
                    {msg.role === "assistant" && (
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                        <Bot className="h-4 w-4 text-primary" strokeWidth={2} />
                      </div>
                    )}
                    {msg.role === "user" && (
                      <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0">
                        <User className="h-4 w-4 text-foreground" strokeWidth={2} />
                      </div>
                    )}

                    {/* Message Content - ChatGPT style */}
                    <div className="flex-1 min-w-0">
                      <div
                        className={cn(
                          "max-w-[85%] sm:max-w-[80%] rounded-2xl px-4 py-3 relative group",
                          msg.role === "user"
                            ? "bg-primary text-primary-foreground ml-auto"
                            : "bg-muted/50"
                        )}
                      >
                        {/* Actions menu - ChatGPT style (top right, always visible on hover) */}
                        {msg.role === "user" && !editingMessageId && (
                          <div className="absolute -top-8 right-0 flex gap-1 opacity-0 group-hover/message:opacity-100 transition-opacity z-10">
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => msg.id && handleStartEdit(msg.id, msg.content)}
                                  className="h-7 w-7 p-0 hover:bg-muted rounded-md"
                                >
                                  <Edit2 className="h-3.5 w-3.5" strokeWidth={1.5} />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>Edit</p>
                              </TooltipContent>
                            </Tooltip>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => msg.id && handleDeleteMessage(msg.id)}
                                  className="h-7 w-7 p-0 hover:bg-destructive/10 hover:text-destructive rounded-md"
                                >
                                  <Trash2 className="h-3.5 w-3.5" strokeWidth={1.5} />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>Delete</p>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                        )}

                      {/* Edit mode */}
                      {editingMessageId === msg.id ? (
                        <div className="space-y-3">
                          <Textarea
                            value={editingContent}
                            onChange={(e) => setEditingContent(e.target.value)}
                            className="min-h-[80px] text-sm sm:text-base bg-background/50 border border-border/30 rounded-lg"
                            autoFocus
                          />
                          <div className="flex gap-2 justify-end">
                            <Button variant="ghost" size="sm" onClick={handleCancelEdit} className="h-8 text-sm">
                              Cancel
                            </Button>
                            <Button size="sm" onClick={() => msg.id && handleSaveEdit(msg.id)} className="h-8 text-sm">
                              Save
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <>
                        {/* Attached images */}
                        {msg.images && msg.images.length > 0 && (
                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3">
                            {msg.images.map((img, idx) => (
                              <img
                                key={idx}
                                src={img}
                                alt={`Image ${idx + 1}`}
                                className="w-full aspect-square object-cover rounded-lg border border-border/50"
                              />
                            ))}
                          </div>
                        )}
                        {/* Attached files */}
                        {msg.files && msg.files.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-3">
                            {msg.files.map((file, idx) => (
                              <div
                                key={idx}
                                className="inline-flex items-center gap-2 px-2.5 py-1.5 bg-muted/50 rounded-md border border-border/30"
                              >
                                <Paperclip className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                                <span className="text-xs truncate max-w-[120px] sm:max-w-[200px]">{file.name}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {/* Content - 말풍선 안만: placeholder "Generating" + cursor → 스트리밍은 빠르게 이어짐 */}
                        {msg.role === "assistant" ? (
                          <div
                            className={cn(
                              "text-[15px] sm:text-base break-words leading-[1.75] prose prose-sm dark:prose-invert max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-headings:my-3 prose-headings:font-semibold prose-code:bg-muted/50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm",
                              !msg.isPlaceholder && "transition-opacity duration-100"
                            )}
                          >
                            {msg.isPlaceholder ? (
                              <span className="text-muted-foreground inline-flex items-center gap-0.5" role="status" aria-label="Generating response">
                                Generating
                                <span className="text-muted-foreground animate-[cursor-blink_0.8s_step-end_infinite]" aria-hidden>|</span>
                              </span>
                            ) : isStreaming ? (
                              <>
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  components={{
                                    code({ node, inline, className, children, ...props }: any) {
                                      const match = /language-(\w+)/.exec(className || "");
                                      return !inline && match ? (
                                        <div className="my-3 rounded-lg overflow-hidden border border-border/50 shadow-sm">
                                          <SyntaxHighlighter
                                            style={vscDarkPlus}
                                            language={match[1]}
                                            PreTag="div"
                                            className="text-sm !m-0"
                                            {...props}
                                          >
                                            {String(children).replace(/\n$/, "")}
                                          </SyntaxHighlighter>
                                        </div>
                                      ) : (
                                        <code className={cn("bg-muted/50 px-1.5 py-0.5 rounded text-sm font-mono", className)} {...props}>
                                          {children}
                                        </code>
                                      );
                                    },
                                  }}
                                >
                                  {msg.content}
                                </ReactMarkdown>
                                <span className="inline-block w-0.5 h-[1.2em] bg-primary ml-0.5 align-baseline animate-[cursor-blink_0.8s_step-end_infinite]" aria-hidden />
                              </>
                            ) : (
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                  code({ node, inline, className, children, ...props }: any) {
                                    const match = /language-(\w+)/.exec(className || "");
                                    return !inline && match ? (
                                      <div className="my-3 rounded-lg overflow-hidden border border-border/50 shadow-sm">
                                        <SyntaxHighlighter
                                          style={vscDarkPlus}
                                          language={match[1]}
                                          PreTag="div"
                                          className="text-sm !m-0"
                                          {...props}
                                        >
                                          {String(children).replace(/\n$/, "")}
                                        </SyntaxHighlighter>
                                      </div>
                                    ) : (
                                      <code className={cn("bg-muted/50 px-1.5 py-0.5 rounded text-sm font-mono", className)} {...props}>
                                        {children}
                                      </code>
                                    );
                                  },
                                }}
                              >
                                {msg.content}
                              </ReactMarkdown>
                            )}
                          </div>
                        ) : (
                          <p className="text-[15px] sm:text-base whitespace-pre-wrap break-words leading-[1.75]">
                            {msg.content}
                          </p>
                        )}
                        {/* Usage info - Simple and minimal (ChatGPT style) */}
                        {msg.role === "assistant" && !isStreaming && !msg.isPlaceholder && (msg.usage || msg.responseTime || msg.model) && (
                          <div className="mt-3 pt-2 border-t border-border/30">
                            <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                              {msg.model && (
                                <span>{msg.model}</span>
                              )}
                              {msg.usage?.total_tokens && (
                                <span>{msg.usage.total_tokens.toLocaleString()} tokens</span>
                              )}
                              {msg.responseTime && (
                                <span>{msg.responseTime}ms</span>
                              )}
                            </div>
                          </div>
                        )}
                        </>
                      )}
                      </div>
                    </div>
                    
                    {/* Thinking mode - Below message */}
                    {msg.role === "assistant" && msg.thinking && (
                      <div className="mt-2 ml-11">
                        <ThinkMode thoughts={[msg.thinking]} />
                      </div>
                    )}
                  </div>
                </div>
                );
              })}
            </div>

            {/* 의사결정 UI: intent / confidence / 선택 도구 (Agentic) */}
            {decisionBlock && (decisionBlock.intent || (decisionBlock.tools?.length ?? 0) > 0) && (
              <div className="mt-4 max-w-2xl rounded-lg border border-border/50 bg-muted/20 px-4 py-3 space-y-1.5">
                <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Decision</div>
                <div className="flex flex-wrap gap-3 text-sm">
                  {decisionBlock.intent != null && (
                    <span className="inline-flex items-center gap-1.5">
                      <span className="text-muted-foreground">Intent:</span>
                      <span className="font-medium text-foreground">{decisionBlock.intent}</span>
                    </span>
                  )}
                  {decisionBlock.confidence != null && (
                    <span className="inline-flex items-center gap-1.5">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium text-foreground">{(decisionBlock.confidence * 100).toFixed(0)}%</span>
                    </span>
                  )}
                  {decisionBlock.tools && decisionBlock.tools.length > 0 && (
                    <span className="inline-flex items-center gap-1.5">
                      <span className="text-muted-foreground">Tools:</span>
                      <span className="font-medium text-foreground">{decisionBlock.tools.join(" → ")}</span>
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Human-in-the-loop — 심플: 한 줄 문구 + Run / Cancel / Change */}
            {humanApproval && loading && (
              <div className="mt-4 max-w-2xl rounded-lg border border-border/50 bg-muted/5 px-3 py-2.5 flex flex-wrap items-center gap-2">
                <span className="text-xs text-muted-foreground">
                  Run <span className="font-medium text-foreground">{humanApproval.tool}</span>
                  {humanApproval.query_snippet ? ` · "${humanApproval.query_snippet}"` : ""}
                </span>
                <div className="flex gap-1.5 ml-auto">
                  <Button
                    size="sm"
                    variant="default"
                    className="h-7 text-xs rounded-md"
                    onClick={async () => {
                      if (!humanApproval.run_id) {
                        setHumanApproval(null);
                        return;
                      }
                      const runId = humanApproval.run_id;
                      const effectiveFeature = mode === "auto" ? "agentic" : selectedFeature;
                      const featureToIntentMap: Record<string, string | undefined> = {
                        agentic: undefined,
                        chat: "chat",
                        rag: "rag",
                        "multi-agent": "multi_agent",
                        "knowledge-graph": "kg",
                        audio: "audio",
                        ocr: "ocr",
                        "web-search": "web_search",
                        google: undefined,
                      };
                      const forceIntent = featureToIntentMap[effectiveFeature];
                      setHumanApproval(null);
                      streamAbortRef.current = new AbortController();
                      setLoading(true);
                      try {
                        await streamMCPChat(
                          {
                            messages: messages.map((m) => ({ role: m.role, content: m.content })),
                            model,
                            temperature,
                            max_tokens: maxTokens,
                            force_intent: forceIntent,
                            approval_response: { run_id: runId, action: "run" },
                          },
                          (event) => {
                            if (event.type === "text") {
                              setMessages((prev) => {
                                const next = [...prev];
                                const last = next[next.length - 1];
                                const content = (event.data.content as string) || "";
                                if (last?.role === "assistant") {
                                  last.content = (last.content || "") + content;
                                  last.isPlaceholder = false;
                                } else {
                                  next.push({
                                    role: "assistant",
                                    content,
                                    timestamp: new Date(),
                                    id: `msg-${Date.now()}`,
                                    isPlaceholder: false,
                                  });
                                }
                                return next;
                              });
                            } else if (event.type === "done") {
                              setStreamingMessageId(null);
                              setDecisionBlock(null);
                              setLastProposal(null);
                            } else if (event.type === "error") {
                              toast.error((event.data.message as string) || "Error");
                            }
                          },
                          streamAbortRef.current?.signal
                        );
                      } finally {
                        setLoading(false);
                      }
                    }}
                  >
                    Run
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs rounded-md border-border/40"
                    onClick={() => {
                      streamAbortRef.current?.abort();
                      setHumanApproval(null);
                      setLoading(false);
                      toast.info("Cancelled");
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 text-xs rounded-md text-muted-foreground hover:text-foreground"
                    onClick={() => {
                      setHumanApproval(null);
                      toast.info("Send a new message to pick another tool.");
                    }}
                  >
                    Change
                  </Button>
                </div>
              </div>
            )}

            {/* Think mode: 사고 단계 표현 (Understanding → Reasoning → Responding) */}
            {enableThinking && loading && (
              <div className="mt-4 max-w-2xl space-y-2">
                <ReasoningStepper
                  steps={DEFAULT_REASONING_STEPS}
                  currentIndex={reasoningPhaseIndex}
                  variant="horizontal"
                />
                {reasoningPhaseIndex === 1 && activeToolCalls.length > 0 && (() => {
                  const last = activeToolCalls[activeToolCalls.length - 1];
                  const msg = last?.message;
                  return msg ? (
                    <ReasoningNarrative text={msg} state="reasoning" />
                  ) : null;
                })()}
              </div>
            )}

            {/* Tool Call Progress - 채팅 본문이 생기기 전까지만 표시. 본문 있으면 말풍선 안 내용+커서만 */}
            {activeToolCalls.length > 0 && (() => {
              const last = messages[messages.length - 1];
              const assistantHasContent = last?.role === "assistant" && (last?.content?.length ?? 0) > 0;
              return !assistantHasContent;
            })() && (
              <div className="mt-4 space-y-2 max-w-2xl">
                {activeToolCalls.map((toolCall, idx) => (
                  <ToolCallDisplay key={`${toolCall.tool}-${idx}`} toolCall={toolCall} />
                ))}
              </div>
            )}

            {/* Loading - only when no placeholder row (placeholder = inline "Generating response" in assistant bubble) */}
            {loading && activeToolCalls.length === 0 && !messages[messages.length - 1]?.isPlaceholder && (
              <div className="flex gap-3 items-start">
                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0">
                  <Bot className="h-4 w-4 text-muted-foreground" strokeWidth={2} />
                </div>
                <TypingIndicator />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

          {/* Input Area */}
          <form
            onSubmit={handleSubmit}
            className="flex-shrink-0 bg-background sticky bottom-0 z-10"
          >
          {/* Container - Consistent max-width with messages area */}
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-3 sm:py-4">
            {/* Attachments Preview - Better visibility */}
            {(attachedImages.length > 0 || attachedFiles.length > 0) && (
              <div className="mb-3 flex flex-wrap gap-2">
                {attachedImages.map((img, idx) => (
                  <div key={idx} className="relative w-16 h-16 sm:w-20 sm:h-20">
                    <img
                      src={img}
                      alt={`Attached ${idx + 1}`}
                      className="w-full h-full object-cover rounded-lg border border-border/50 shadow-sm"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRemoveImage(idx)}
                      className="absolute -top-1.5 -right-1.5 h-6 w-6 p-0 bg-destructive hover:bg-destructive/90 text-destructive-foreground rounded-full shadow-sm"
                    >
                      <X className="h-4 w-4" strokeWidth={1.5} />
                    </Button>
                  </div>
                ))}
                {attachedFiles.map((file, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg border border-border/50 text-xs shadow-sm"
                  >
                    <Paperclip className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="truncate max-w-[120px] sm:max-w-[180px]">{file.name}</span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRemoveFile(idx)}
                      className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive shrink-0 ml-1"
                    >
                      <X className="h-4 w-4" strokeWidth={1.5} />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            {/* Input Container - items-center, 일관 높이 (Design: System-first) */}
            <div
              className={cn(
                "flex items-center gap-2",
                "bg-background border border-border/50 rounded-2xl",
                "px-3 py-2 min-h-[52px]",
                "transition-all shadow-sm",
                "focus-within:border-primary/50 focus-within:shadow-md",
                isDragging && "border-primary/50 bg-primary/5"
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {/* Hidden file inputs */}
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageUpload}
                className="hidden"
              />
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileUpload}
                className="hidden"
              />

              {/* Mode chip — 모델/모드 패널 열기 */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => {
                      setShowInfoPanel(true);
                      setInfoPanelTab("models");
                    }}
                    className="h-8 px-3 gap-1.5 shrink-0 inline-flex items-center rounded-full border border-border/40 bg-muted/10 hover:bg-muted/20 transition-colors text-[12px] font-medium text-foreground tracking-tight"
                  >
                    {mode === "auto" ? (
                      <Sparkles className="h-3.5 w-3.5 text-muted-foreground shrink-0" strokeWidth={1.5} />
                    ) : (
                      <Settings className="h-3.5 w-3.5 text-muted-foreground shrink-0" strokeWidth={1.5} />
                    )}
                    <span className="hidden sm:inline">{mode === "auto" ? "Auto" : "Manual"}</span>
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Model & mode</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {mode === "auto" ? "AI picks the feature" : "You pick the feature"}
                  </p>
                </TooltipContent>
              </Tooltip>

              {/* Text Input - ChatGPT style */}
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  adjustTextareaHeight();
                }}
                onKeyDown={handleKeyDown}
                placeholder="Message beanllm..."
                className="flex-1 min-h-[36px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 text-[15px] sm:text-base placeholder:text-muted-foreground/50 py-2 px-2 leading-[1.6]"
                rows={1}
                disabled={loading}
              />

              {/* Action buttons - h-8 통일, items-center 정렬 */}
              <div className="flex items-center gap-1 shrink-0">
                <DropdownMenu>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <DropdownMenuTrigger asChild>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50"
                        >
                          <Settings className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                        </Button>
                      </DropdownMenuTrigger>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Settings</p>
                      <p className="text-xs text-muted-foreground mt-1">API keys, Google</p>
                    </TooltipContent>
                  </Tooltip>
                  <DropdownMenuContent align="end" className="min-w-[140px]">
                    <DropdownMenuItem onClick={() => setApiKeyModalOpen(true)}>API Keys</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setGoogleModalOpen(true)}>Google</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => imageInputRef.current?.click()}
                      className="h-8 w-8 p-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    >
                      <ImageIcon className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Attach images</p>
                    <p className="text-xs text-muted-foreground mt-1">Supports JPG, PNG, GIF</p>
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                      className="h-8 w-8 p-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    >
                      <Paperclip className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Attach files</p>
                    <p className="text-xs text-muted-foreground mt-1">PDF, TXT, DOCX, etc.</p>
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="submit"
                      disabled={loading || (!input.trim() && attachedImages.length === 0 && attachedFiles.length === 0)}
                      className="h-8 w-8 p-0 shrink-0 rounded-lg bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {loading ? (
                        <LoaderCircle className="h-4 w-4 animate-spin text-primary-foreground shrink-0" strokeWidth={2} />
                      ) : (
                        <Send className="h-4 w-4 text-primary-foreground shrink-0" strokeWidth={2} />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Send message</p>
                    <p className="text-xs text-muted-foreground mt-1">Press Enter to send</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </form>
        </div>

        {/* Right: Info Panel (30-40%) */}
        <InfoPanel
          isOpen={showInfoPanel}
          onClose={() => setShowInfoPanel(false)}
          defaultTab={infoPanelTab}
          isCollapsed={isInfoPanelCollapsed}
          onCollapseChange={setIsInfoPanelCollapsed}
          model={model}
          onModelChange={setModel}
          mode={mode}
          onModeChange={setMode}
          selectedFeature={selectedFeature}
          onFeatureChange={setSelectedFeature}
          selectedGoogleServices={selectedGoogleServices}
          onGoogleServicesChange={setSelectedGoogleServices}
          modelParams={modelParams}
          temperature={temperature}
          setTemperature={setTemperature}
          maxTokens={maxTokens}
          setMaxTokens={setMaxTokens}
          topP={topP}
          setTopP={setTopP}
          frequencyPenalty={frequencyPenalty}
          setFrequencyPenalty={setFrequencyPenalty}
          presencePenalty={presencePenalty}
          setPresencePenalty={setPresencePenalty}
          customInstruction={customInstruction}
          setCustomInstruction={setCustomInstruction}
          enableThinking={enableThinking}
          setEnableThinking={setEnableThinking}
          modelSupportsThinking={modelSupportsThinking}
          messages={messages.map((m) => ({
            role: m.role,
            content: m.content,
            timestamp: m.timestamp,
          }))}
          documentPreviewContent={documentPreviewContent}
          documentPreviewTitle="Document Preview"
          sessionId={sessionId}
          sessionSummary={sessionSummary}
          sessionSummaryLoading={sessionSummaryLoading}
          onRefreshSummary={loadSessionSummary}
          onLoadSession={loadSession}
          onNewChat={handleNewChat}
          onExport={handleExport}
          onImport={handleImport}
          onClear={handleClear}
        />
      </div>

      <ApiKeyModal open={apiKeyModalOpen} onOpenChange={setApiKeyModalOpen} />
      <GoogleConnectModal open={googleModalOpen} onOpenChange={setGoogleModalOpen} />
    </PageLayout>
  );
}
