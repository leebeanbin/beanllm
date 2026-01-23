"use client";

import { useState, useRef, useEffect } from "react";
import { PageLayout } from "@/components/PageLayout";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { ModelSelectorSimple } from "@/components/ModelSelectorSimple";
import { createBeanLLMClient } from "@/lib/beanllm-client";
import { LoaderCircle, Settings, Send, X, User, Bot, Brain, MessageSquare, Download, Upload, Trash2, Edit2, Check, X as XIcon, HelpCircle, Image as ImageIcon, Paperclip } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ChatIcon } from "@/components/icons/ChatIcon";
import { ThinkMode } from "@/components/ThinkMode";
import { ParameterTooltip } from "@/components/ParameterTooltip";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import { getUserFriendlyError } from "@/lib/error-messages";
import { streamMCPChat } from "@/lib/mcp-client";
import { ToolCallDisplay, ToolCallProgress } from "@/components/ToolCallDisplay";
import { FeatureSelector } from "@/components/FeatureSelector";
import { GoogleServiceSelector } from "@/components/GoogleServiceSelector";
import { FeatureMode, GoogleService, ProviderConfig, AvailableModels } from "@/types/chat";

interface Message {
  id?: string;
  role: string;
  content: string;
  timestamp?: Date;
  thinking?: string;
  isEditing?: boolean;
  images?: string[]; // Base64 encoded images
  files?: Array<{ name: string; type: string; data: string }>; // File attachments
  toolCalls?: ToolCallProgress[]; // MCP tool calls associated with this message
  // 모니터링 정보
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
  requestId?: string;
  responseTime?: number; // ms
  model?: string;
  provider?: string;
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
  const [model, setModel] = useState("qwen2.5:0.5b");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);
  const [topP, setTopP] = useState(1.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeToolCalls, setActiveToolCalls] = useState<ToolCallProgress[]>([]);
  const [modelParams, setModelParams] = useState<ModelParameters | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<FeatureMode>("chat");
  const [selectedGoogleServices, setSelectedGoogleServices] = useState<GoogleService[]>([]);
  const [providerConfig, setProviderConfig] = useState<ProviderConfig | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModels | null>(null);
  const [enableThinking, setEnableThinking] = useState(false);
  const [customInstruction, setCustomInstruction] = useState("");
  const [showCustomInstruction, setShowCustomInstruction] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState("");
  const [attachedImages, setAttachedImages] = useState<string[]>([]); // Base64 encoded images
  const [attachedFiles, setAttachedFiles] = useState<Array<{ name: string; type: string; data: string }>>([]);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const client = createBeanLLMClient();

  // ✅ 활성화된 Provider 및 모델 목록 가져오기
  useEffect(() => {
    const fetchProviderConfig = async () => {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      try {
        const [providersRes, modelsRes] = await Promise.all([
          fetch(`${apiUrl}/api/config/providers`),
          fetch(`${apiUrl}/api/config/models`),
        ]);

        if (providersRes.ok) {
          const data = await providersRes.json();
          setProviderConfig(data);
        }

        if (modelsRes.ok) {
          const data = await modelsRes.json();
          setAvailableModels(data);
        }
      } catch (error) {
        console.error("Failed to fetch provider config:", error);
      }
    };

    fetchProviderConfig();
  }, []);

  // localStorage에서 대화 히스토리 불러오기
  useEffect(() => {
    const savedMessages = localStorage.getItem("chat-history");
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        const restoredMessages = parsed.map((msg: any) => ({
          ...msg,
          timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
        }));
        setMessages(restoredMessages);
      } catch (error) {
        console.error("Failed to load chat history:", error);
      }
    }
  }, []);

  // 메시지가 변경될 때마다 localStorage에 저장
  useEffect(() => {
    if (messages.length > 0) {
      const messagesToSave = messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp?.toISOString(),
      }));
      localStorage.setItem("chat-history", JSON.stringify(messagesToSave));
    }
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 모델 변경 시 파라미터 지원 정보 가져오기
  useEffect(() => {
    const fetchModelParameters = async () => {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      try {
        const response = await fetch(`${apiUrl}/api/models/${encodeURIComponent(model)}/parameters`);
        if (response.ok) {
          const data = await response.json();
          setModelParams(data);
          // 기본값 업데이트
          if (data.default_temperature !== undefined) {
            setTemperature(data.default_temperature);
          }
          if (data.max_tokens) {
            setMaxTokens((prev) => Math.min(prev, data.max_tokens));
          }
        }
      } catch (error) {
        // 실패 시 기본값 사용
        setModelParams({
          supports: {
            temperature: true,
            max_tokens: true,
            top_p: true,
            frequency_penalty: true,
            presence_penalty: true,
          },
          max_tokens: 4000,
          default_temperature: 0.7,
        });
      }
    };
    fetchModelParameters();
  }, [model]);

  // 대화 내보내기
  const handleExport = () => {
    const exportData = {
      messages: messages.map(msg => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp?.toISOString(),
        thinking: msg.thinking,
      })),
      model,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success("대화가 내보내졌습니다");
  };

  // 대화 가져오기
  const handleImport = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target?.result as string);
          if (data.messages && Array.isArray(data.messages)) {
            const importedMessages = data.messages.map((msg: any) => ({
              ...msg,
              timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
            }));
            setMessages(importedMessages);
            if (data.model) setModel(data.model);
            toast.success("대화가 가져와졌습니다");
          } else {
            toast.error("잘못된 파일 형식입니다");
          }
        } catch (error) {
          toast.error("파일을 읽을 수 없습니다");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  // 대화 초기화
  const handleClear = () => {
    if (confirm("모든 대화를 삭제하시겠습니까?")) {
      setMessages([]);
      localStorage.removeItem("chat-history");
      toast.success("대화가 삭제되었습니다");
    }
  };

  // 메시지 삭제
  const handleDeleteMessage = (messageId: string) => {
    setMessages(messages.filter(msg => msg.id !== messageId));
    toast.success("메시지가 삭제되었습니다");
  };

  // 메시지 편집 시작
  const handleStartEdit = (messageId: string, content: string) => {
    setEditingMessageId(messageId);
    setEditingContent(content);
  };

  // 메시지 편집 저장
  const handleSaveEdit = (messageId: string) => {
    setMessages(messages.map(msg => 
      msg.id === messageId 
        ? { ...msg, content: editingContent, isEditing: false }
        : msg
    ));
    setEditingMessageId(null);
    setEditingContent("");
    toast.success("메시지가 수정되었습니다");
  };

  // 메시지 편집 취소
  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setEditingContent("");
  };

  // 이미지 업로드 핸들러
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) {
        toast.error(`${file.name}은(는) 이미지 파일이 아닙니다.`);
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        setAttachedImages((prev) => [...prev, base64]);
      };
      reader.readAsDataURL(file);
    });

    // Reset input
    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  };

  // 파일 업로드 핸들러
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        setAttachedFiles((prev) => [
          ...prev,
          { name: file.name, type: file.type, data: base64 },
        ]);
      };
      reader.readAsDataURL(file);
    });

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // 이미지 제거
  const handleRemoveImage = (index: number) => {
    setAttachedImages((prev) => prev.filter((_, i) => i !== index));
  };

  // 파일 제거
  const handleRemoveFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && attachedImages.length === 0 && attachedFiles.length === 0) || loading) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}-${Math.random()}`,
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
      images: attachedImages.length > 0 ? [...attachedImages] : undefined,
      files: attachedFiles.length > 0 ? [...attachedFiles] : undefined,
    };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    const userQuery = input.trim();
    setInput("");
    setAttachedImages([]);
    setAttachedFiles([]);
    setLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    // ✅ Feature 선택에 따라 MCP Streaming 사용 여부 결정
    const useMCPStreaming = selectedFeature !== "chat" || shouldUseMCPStreaming(userQuery);

    try {
      if (useMCPStreaming) {
        // ✅ MCP Streaming 사용 (Tool Call Progress 표시)
        const chatMessages = newMessages.map(m => ({ role: m.role, content: m.content }));
        if (customInstruction.trim()) {
          chatMessages.unshift({ role: "system", content: customInstruction.trim() });
        }

        // Tool call progress 초기화
        setActiveToolCalls([]);

        // SSE Streaming 시작
        await streamMCPChat(
          {
            messages: chatMessages,
            model,
            temperature,
            max_tokens: maxTokens,
          },
          (event) => {
            if (event.type === "tool_call") {
              // Tool call 시작
              const newToolCall: ToolCallProgress = {
                tool: event.data.tool,
                status: "started",
                arguments: event.data.arguments,
              };
              setActiveToolCalls(prev => [...prev, newToolCall]);
            } else if (event.type === "tool_progress") {
              // Tool 진행 상황 업데이트
              setActiveToolCalls(prev =>
                prev.map(tc =>
                  tc.tool === event.data.tool
                    ? { ...tc, status: "running", progress: event.data.progress, step: event.data.step, message: event.data.message }
                    : tc
                )
              );
            } else if (event.type === "tool_result") {
              // Tool 완료
              setActiveToolCalls(prev =>
                prev.map(tc =>
                  tc.tool === event.data.tool
                    ? { ...tc, status: event.data.status === "failed" ? "failed" : "completed", result: event.data.result, error: event.data.error }
                    : tc
                )
              );
            } else if (event.type === "text") {
              // 일반 텍스트 응답 (Tool call 없음)
              const assistantMessage: Message = {
                id: `msg-${Date.now()}-${Math.random()}`,
                role: "assistant",
                content: event.data.content,
                timestamp: new Date(),
              };
              setMessages(prev => [...prev, assistantMessage]);
            } else if (event.type === "done") {
              // 완료 - Tool call 결과를 메시지로 추가
              const toolCallResults = activeToolCalls.map(tc => {
                if (tc.status === "completed" && tc.result) {
                  return `**${tc.tool}** 완료:\n\`\`\`json\n${JSON.stringify(tc.result, null, 2)}\n\`\`\``;
                } else if (tc.status === "failed") {
                  return `**${tc.tool}** 실패: ${tc.error}`;
                }
                return "";
              }).filter(Boolean).join("\n\n");

              if (toolCallResults) {
                const assistantMessage: Message = {
                  id: `msg-${Date.now()}-${Math.random()}`,
                  role: "assistant",
                  content: toolCallResults,
                  timestamp: new Date(),
                  toolCalls: [...activeToolCalls],
                };
                setMessages(prev => [...prev, assistantMessage]);
              }

              setActiveToolCalls([]);
            }
          }
        );
      } else {
        // ✅ 기존 Chat API 사용 (일반 대화)
        const chatMessages = newMessages.map(m => ({ role: m.role, content: m.content }));
        if (customInstruction.trim()) {
          chatMessages.unshift({
            role: "system",
            content: customInstruction.trim(),
          });
        }

        const requestStartTime = Date.now();
        const responseData = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: chatMessages,
            model,
            temperature,
            max_tokens: maxTokens,
            top_p: topP,
            frequency_penalty: frequencyPenalty,
            presence_penalty: presencePenalty,
            enable_thinking: enableThinking,
          }),
        });
        
        if (!responseData.ok) {
          const error = await responseData.json();
          throw new Error(error.detail || 'Chat request failed');
        }
        
        const response = await responseData.json();
        const responseTime = Date.now() - requestStartTime;
        const requestId = responseData.headers.get('X-Request-ID') || undefined;

      const assistantMessage: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: "assistant",
        content: response.content,
        timestamp: new Date(),
        usage: response.usage ? {
          input_tokens: response.usage.input_tokens,
          output_tokens: response.usage.output_tokens,
          total_tokens: response.usage.total_tokens,
        } : undefined,
        model: response.model,
        provider: response.provider,
        responseTime: responseTime,
        requestId: requestId,
      };
      
      // Check if response contains thinking process (for reasoning models)
      // Use string pattern to avoid JSX parsing issues with </think>
      const thinkStart = "<think>";
      const thinkEnd = "</think>";
      const thinkStartIdx = response.content?.indexOf(thinkStart);
      const thinkEndIdx = response.content?.indexOf(thinkEnd);
      
      if (thinkStartIdx !== undefined && thinkStartIdx !== -1 && thinkEndIdx !== undefined && thinkEndIdx !== -1 && thinkEndIdx > thinkStartIdx) {
        const thinkingContent = response.content.substring(thinkStartIdx + thinkStart.length, thinkEndIdx).trim();
        assistantMessage.thinking = thinkingContent;
        assistantMessage.content = (
          response.content.substring(0, thinkStartIdx) + 
          response.content.substring(thinkEndIdx + thinkEnd.length)
        ).trim();
      }
      
      setMessages([...newMessages, assistantMessage]);
      }
    } catch (error: any) {
      const errorInfo = getUserFriendlyError(error);
      toast.error(errorInfo.title, {
        description: errorInfo.message + (errorInfo.suggestion ? `\n\n💡 ${errorInfo.suggestion}` : ""),
      });
      setMessages(messages);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [input]);

  // 모델이 think mode를 지원하는지 확인
  const supportsThinking = (modelName: string): boolean => {
    const modelLower = modelName.toLowerCase();
    // Native thinking 지원 모델
    if (
      modelLower.startsWith("claude") || // Claude models
      modelLower.startsWith("o1") || // OpenAI O1 series
      modelLower.startsWith("o3") || // OpenAI O3 series
      modelLower.startsWith("gpt-5") || // OpenAI GPT-5 series
      modelLower.includes("deepseek-reasoner") || // DeepSeek Reasoner
      modelLower.includes("deepseek-r1") // DeepSeek R1
    ) {
      return true;
    }
    // Prompt-based thinking 지원 (모든 오픈소스 모델)
    // Qwen, Llama, Phi 등은 prompt-based thinking 지원
    return true; // 모든 모델이 prompt-based thinking을 지원
  };

  const modelSupportsThinking = supportsThinking(model);

  // Tool Call 감지 함수
  const shouldUseMCPStreaming = (query: string): boolean => {
    const queryLower = query.toLowerCase();
    // RAG, Multi-Agent, KG, Audio, OCR 키워드 감지
    const mcpKeywords = [
      "rag", "검색", "문서", "pdf",
      "에이전트", "토론", "협업", "agent", "debate",
      "지식 그래프", "knowledge graph", "kg",
      "음성", "전사", "audio", "transcribe",
      "ocr", "텍스트 추출", "이미지 인식"
    ];
    return mcpKeywords.some(keyword => queryLower.includes(keyword));
  };

  // 지원하는 파라미터만 필터링
  const supportedParams = modelParams ? {
    temperature: modelParams.supports.temperature,
    max_tokens: modelParams.supports.max_tokens,
    top_p: modelParams.supports.top_p,
    frequency_penalty: modelParams.supports.frequency_penalty,
    presence_penalty: modelParams.supports.presence_penalty,
  } : {
    temperature: true,
    max_tokens: true,
    top_p: true,
    frequency_penalty: true,
    presence_penalty: true,
  };

  const paramCount = Object.values(supportedParams).filter(Boolean).length;
  const gridCols = paramCount <= 2 ? "grid-cols-1" : paramCount <= 4 ? "grid-cols-2" : "grid-cols-3";

  return (
    <PageLayout
      title="Chat"
      description="Simple chat interface with LLM models"
    >
      <div className="flex flex-col h-[calc(100vh-12rem)] min-h-0">
        {/* Header - Standard Layout */}
        <div className="flex items-center justify-between gap-4 mb-4 flex-shrink-0">
          <div className="flex items-center gap-3 flex-1">
            <div className="w-48">
              <ModelSelectorSimple value={model} onChange={setModel} />
            </div>
            <div className="w-56">
              <FeatureSelector value={selectedFeature} onChange={setSelectedFeature} />
            </div>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleExport}
                      className="h-10 flex-shrink-0 border-border/50 bg-card/50 hover:bg-accent/50"
                      aria-label="대화 내보내기"
                    >
                      <Download className="h-4 w-4" aria-hidden="true" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>대화 내보내기</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleImport}
                      className="h-10 flex-shrink-0 border-border/50 bg-card/50 hover:bg-accent/50"
                      aria-label="대화 가져오기"
                    >
                      <Upload className="h-4 w-4" aria-hidden="true" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>대화 가져오기</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleClear}
                      className="h-10 flex-shrink-0 border-border/50 bg-card/50 hover:bg-accent/50 text-destructive hover:text-destructive"
                      aria-label="대화 초기화"
                    >
                      <Trash2 className="h-4 w-4" aria-hidden="true" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>대화 초기화</TooltipContent>
                </Tooltip>
              </>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                localStorage.removeItem("chat-onboarding-completed");
                setShowOnboarding(true);
              }}
              className="h-10 flex-shrink-0 border-border/50 bg-card/50 hover:bg-accent/50"
              aria-label="가이드 열기"
            >
              <HelpCircle className="h-4 w-4 mr-2" aria-hidden="true" />
              Guide
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSettings(!showSettings)}
              className="h-10 flex-shrink-0 border-border/50 bg-card/50 hover:bg-accent/50"
              aria-label="설정 열기"
            >
              <Settings className="h-4 w-4 mr-2" aria-hidden="true" />
              Settings
            </Button>
          </div>
        </div>

        {/* Settings Panel - 파스텔 테마에 맞게 재디자인 */}
        {showSettings && (
          <Card className="mb-4 flex-shrink-0 border-border/50 bg-card/80 backdrop-blur-sm shadow-sm">
            <CardContent className="pt-5 pb-5">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">Settings</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Adjust parameters and instructions for {model}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSettings(false)}
                  className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-accent/50"
                  aria-label="설정 닫기"
                >
                  <X className="h-3.5 w-3.5" aria-hidden="true" />
                </Button>
              </div>

              {/* Custom Instruction Section */}
              <div className="mb-6 pb-6 border-b border-border/30">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-muted-foreground" />
                    <Label htmlFor="custom-instruction" className="text-sm font-medium text-foreground">
                      Custom Instruction (System Prompt)
                    </Label>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowCustomInstruction(!showCustomInstruction)}
                    className="h-7 text-xs"
                  >
                    {showCustomInstruction ? "Hide" : "Show"}
                  </Button>
                </div>
                {showCustomInstruction && (
                  <div className="space-y-2">
                    <Textarea
                      id="custom-instruction"
                      value={customInstruction}
                      onChange={(e) => setCustomInstruction(e.target.value)}
                      placeholder="e.g., You are a helpful assistant that always responds in Korean. Be concise and friendly."
                      rows={3}
                      className="bg-background/50 border-border/50 text-sm"
                    />
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      This instruction will be added as a system message to guide the model's behavior.
                    </p>
                  </div>
                )}
              </div>
              
              {!modelParams ? (
                <div className="flex items-center justify-center py-8">
                  <LoaderCircle className="h-5 w-5 animate-spin text-muted-foreground" />
                  <span className="ml-2 text-sm text-muted-foreground">Loading parameters...</span>
                </div>
              ) : (
                <div className={cn("grid gap-5", gridCols)}>
                    {supportedParams.temperature && (
                      <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                        <ParameterTooltip parameter="temperature">
                          <Label htmlFor="temperature" className="text-sm font-medium text-foreground">
                            Temperature
                          </Label>
                        </ParameterTooltip>
                        <div className="flex items-center justify-end">
                          <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                            {temperature.toFixed(1)}
                          </span>
                        </div>
                        <Input
                          id="temperature"
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={temperature}
                          onChange={(e) => setTemperature(parseFloat(e.target.value))}
                          className="h-1.5 accent-primary/60"
                        />
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Controls randomness • 0=deterministic, 2=creative
                        </p>
                      </div>
                    )}

                    {supportedParams.max_tokens && (
                      <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                        <ParameterTooltip parameter="max_tokens">
                          <Label htmlFor="max-tokens" className="text-sm font-medium text-foreground">
                            Max Tokens
                          </Label>
                        </ParameterTooltip>
                        <div className="flex items-center justify-between">
                          {modelParams.max_tokens && (
                            <span className="text-xs text-muted-foreground">
                              Max: {modelParams.max_tokens.toLocaleString()}
                            </span>
                          )}
                        </div>
                        <Input
                          id="max-tokens"
                          type="number"
                          min="1"
                          max={modelParams.max_tokens || 4000}
                          value={maxTokens}
                          onChange={(e) => setMaxTokens(parseInt(e.target.value) || 1000)}
                          className="h-9 bg-background/50 border-border/50 focus:border-primary/50"
                        />
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Maximum tokens in response
                        </p>
                      </div>
                    )}

                    {supportedParams.top_p && (
                      <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                        <ParameterTooltip parameter="top_p">
                          <Label htmlFor="top-p" className="text-sm font-medium text-foreground">
                            Top P
                          </Label>
                        </ParameterTooltip>
                        <div className="flex items-center justify-end">
                          <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                            {topP.toFixed(2)}
                          </span>
                        </div>
                        <Input
                          id="top-p"
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={topP}
                          onChange={(e) => setTopP(parseFloat(e.target.value))}
                          className="h-1.5 accent-primary/60"
                        />
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Nucleus sampling threshold
                        </p>
                      </div>
                    )}

                    {supportedParams.frequency_penalty && (
                      <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                        <ParameterTooltip parameter="frequency_penalty">
                          <Label htmlFor="frequency-penalty" className="text-sm font-medium text-foreground">
                            Frequency Penalty
                          </Label>
                        </ParameterTooltip>
                        <div className="flex items-center justify-end">
                          <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                            {frequencyPenalty.toFixed(1)}
                          </span>
                        </div>
                        <Input
                          id="frequency-penalty"
                          type="range"
                          min="-2"
                          max="2"
                          step="0.1"
                          value={frequencyPenalty}
                          onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                          className="h-1.5 accent-primary/60"
                        />
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Reduce repetition in responses
                        </p>
                      </div>
                    )}

                    {supportedParams.presence_penalty && (
                      <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                        <ParameterTooltip parameter="presence_penalty">
                          <Label htmlFor="presence-penalty" className="text-sm font-medium text-foreground">
                            Presence Penalty
                          </Label>
                        </ParameterTooltip>
                        <div className="flex items-center justify-end">
                          <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                            {presencePenalty.toFixed(1)}
                          </span>
                        </div>
                        <Input
                          id="presence-penalty"
                          type="range"
                          min="-2"
                          max="2"
                          step="0.1"
                          value={presencePenalty}
                          onChange={(e) => setPresencePenalty(parseFloat(e.target.value))}
                          className="h-1.5 accent-primary/60"
                        />
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Encourage new topics
                        </p>
                      </div>
                    )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Google Service Selector - Google 기능 선택 시만 표시 */}
        {selectedFeature === "google" && (
          <GoogleServiceSelector
            selectedServices={selectedGoogleServices}
            onChange={setSelectedGoogleServices}
            className="mb-4"
          />
        )}

        {/* Messages Area */}
        <Card className={cn(
          "flex flex-col mb-4 overflow-hidden",
          messages.length === 0 
            ? "h-[400px] overflow-hidden" 
            : "flex-1 min-h-0"
        )}>
          <CardContent className={cn(
            "p-6",
            messages.length === 0 
              ? "flex items-center justify-center h-full overflow-hidden" 
              : "flex-1 overflow-y-auto min-h-0 space-y-4"
          )}>
            {messages.length === 0 && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-6 max-w-md">
                  <div className="flex justify-center">
                    <div className="relative group">
                      {/* Subtle gradient background */}
                      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent rounded-2xl blur-2xl group-hover:from-primary/10 transition-all duration-500" />
                      {/* Clean minimalist icon */}
                      <div className="relative flex items-center justify-center w-24 h-24">
                        <MessageSquare 
                          className="w-12 h-12 text-muted-foreground/20 group-hover:text-muted-foreground/30 transition-colors duration-300" 
                          strokeWidth={1}
                        />
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <h3 className="text-foreground/70 text-base font-medium">Start a conversation</h3>
                    <p className="text-muted-foreground/50 text-sm">Type a message to begin</p>
                  </div>
                </div>
              </div>
            )}
            {messages.map((msg) => (
              <div key={msg.id || `msg-${msg.timestamp?.getTime()}`} className="space-y-3 group/message">
                <div
                  className={cn(
                    "flex gap-3 group",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {msg.role === "assistant" && (
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  )}
                  <div
                    className={cn(
                      "max-w-[75%] rounded-2xl px-4 py-3 space-y-1 relative",
                      msg.role === "user"
                        ? "bg-primary/90 text-primary-foreground shadow-sm"
                        : "bg-muted/80 shadow-sm"
                    )}
                  >
                    {/* 편집/삭제 버튼 */}
                    {msg.role === "user" && !editingMessageId && (
                      <div className="absolute -top-2 -right-2 flex gap-1 opacity-0 group-hover/message:opacity-100 transition-opacity">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => msg.id && handleStartEdit(msg.id, msg.content)}
                              className="h-6 w-6 p-0 bg-background/90 hover:bg-background border border-border/50"
                              aria-label="메시지 편집"
                            >
                              <Edit2 className="h-3 w-3" aria-hidden="true" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>편집</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => msg.id && handleDeleteMessage(msg.id)}
                              className="h-6 w-6 p-0 bg-background/90 hover:bg-destructive/10 hover:text-destructive border border-border/50"
                              aria-label="메시지 삭제"
                            >
                              <Trash2 className="h-3 w-3" aria-hidden="true" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>삭제</TooltipContent>
                        </Tooltip>
                      </div>
                    )}

                    {/* 편집 모드 */}
                    {editingMessageId === msg.id ? (
                      <div className="space-y-2">
                        <Textarea
                          value={editingContent}
                          onChange={(e) => setEditingContent(e.target.value)}
                          className="min-h-[60px] text-sm bg-background/50 border-border/50"
                          autoFocus
                        />
                        <div className="flex gap-2 justify-end">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleCancelEdit}
                            className="h-7 text-xs"
                          >
                            취소
                          </Button>
                          <Button
                            variant="default"
                            size="sm"
                            onClick={() => msg.id && handleSaveEdit(msg.id)}
                            className="h-7 text-xs"
                          >
                            저장
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <>
                        {/* 첨부된 이미지 표시 */}
                        {msg.images && msg.images.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-2">
                            {msg.images.map((img, idx) => (
                              <img
                                key={idx}
                                src={img}
                                alt={`Image ${idx + 1}`}
                                className="max-w-[200px] max-h-[200px] object-cover rounded-lg border border-border/50"
                              />
                            ))}
                          </div>
                        )}
                        {/* 첨부된 파일 표시 */}
                        {msg.files && msg.files.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-2">
                            {msg.files.map((file, idx) => (
                              <div
                                key={idx}
                                className="inline-flex items-center gap-2 px-3 py-1.5 bg-background/50 rounded-lg border border-border/50"
                              >
                                <Paperclip className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                                <span className="text-xs text-foreground">{file.name}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {msg.role === "assistant" ? (
                          <div className="text-sm break-words leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                code({ node, inline, className, children, ...props }: any) {
                                  const match = /language-(\w+)/.exec(className || "");
                                  return !inline && match ? (
                                    <SyntaxHighlighter
                                      style={vscDarkPlus}
                                      language={match[1]}
                                      PreTag="div"
                                      className="rounded-lg my-2"
                                      {...props}
                                    >
                                      {String(children).replace(/\n$/, "")}
                                    </SyntaxHighlighter>
                                  ) : (
                                    <code className={cn("bg-muted px-1.5 py-0.5 rounded text-xs", className)} {...props}>
                                      {children}
                                    </code>
                                  );
                                },
                              }}
                            >
                              {msg.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">{msg.content}</p>
                        )}
                        {/* 모니터링 정보 표시 (assistant 메시지에만) */}
                        {msg.role === "assistant" && (msg.usage || msg.model || msg.responseTime) && (
                          <div className="mt-2 pt-2 border-t border-border/30 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                            {msg.model && (
                              <div className="flex items-center gap-1">
                                <Bot className="h-3 w-3" />
                                <span className="font-medium">{msg.model}</span>
                                {msg.provider && (
                                  <span className="text-muted-foreground/60">({msg.provider})</span>
                                )}
                              </div>
                            )}
                            {msg.usage && (
                              <>
                                {msg.usage.input_tokens !== undefined && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-muted-foreground/60">입력:</span>
                                    <span className="font-mono">{msg.usage.input_tokens.toLocaleString()}</span>
                                  </div>
                                )}
                                {msg.usage.output_tokens !== undefined && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-muted-foreground/60">출력:</span>
                                    <span className="font-mono">{msg.usage.output_tokens.toLocaleString()}</span>
                                  </div>
                                )}
                                {msg.usage.total_tokens !== undefined && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-muted-foreground/60">총:</span>
                                    <span className="font-mono font-medium">{msg.usage.total_tokens.toLocaleString()}</span>
                                  </div>
                                )}
                              </>
                            )}
                            {msg.responseTime !== undefined && (
                              <div className="flex items-center gap-1">
                                <span className="text-muted-foreground/60">응답 시간:</span>
                                <span className="font-mono">{msg.responseTime.toLocaleString()}ms</span>
                              </div>
                            )}
                            {msg.requestId && (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div className="flex items-center gap-1 cursor-help">
                                    <span className="text-muted-foreground/40 font-mono text-[10px]">
                                      ID: {msg.requestId.slice(0, 8)}...
                                    </span>
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="font-mono text-xs">{msg.requestId}</p>
                                </TooltipContent>
                              </Tooltip>
                            )}
                          </div>
                        )}
                        {msg.timestamp && (
                          <p className={cn(
                            "text-xs mt-1.5",
                            msg.role === "user" ? "text-primary-foreground/70" : "text-muted-foreground"
                          )}>
                            {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </p>
                        )}
                      </>
                    )}
                  </div>
                  {msg.role === "user" && (
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                      <User className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                </div>
                {msg.role === "assistant" && msg.thinking && (
                  <div className="ml-11">
                    <ThinkMode thoughts={[msg.thinking]} />
                  </div>
                )}
              </div>
            ))}
            {/* Tool Call Progress 표시 */}
            {activeToolCalls.length > 0 && (
              <div className="space-y-3">
                {activeToolCalls.map((toolCall, idx) => (
                  <ToolCallDisplay key={`${toolCall.tool}-${idx}`} toolCall={toolCall} />
                ))}
              </div>
            )}
            {loading && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="bg-muted/80 rounded-2xl px-4 py-3 shadow-sm">
                  <div className="flex items-center gap-2">
                    <LoaderCircle className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </CardContent>
        </Card>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="relative flex-shrink-0" aria-label="채팅 메시지 입력">
          {/* 첨부된 이미지/파일 미리보기 */}
          {(attachedImages.length > 0 || attachedFiles.length > 0) && (
            <div className="mb-2 p-2 border border-border/50 rounded-lg bg-muted/30 space-y-2">
              {attachedImages.map((img, idx) => (
                <div key={idx} className="relative inline-block mr-2 mb-2">
                  <img 
                    src={img} 
                    alt={`Attached ${idx + 1}`}
                    className="h-20 w-20 object-cover rounded-lg border border-border/50"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveImage(idx)}
                    className="absolute -top-2 -right-2 h-6 w-6 p-0 bg-destructive/90 hover:bg-destructive text-destructive-foreground rounded-full"
                    aria-label="이미지 제거"
                  >
                    <X className="h-3 w-3" aria-hidden="true" />
                  </Button>
                </div>
              ))}
              {attachedFiles.map((file, idx) => (
                <div key={idx} className="inline-flex items-center gap-2 px-3 py-1.5 bg-background/50 rounded-lg border border-border/50 mr-2 mb-2">
                  <Paperclip className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                  <span className="text-xs text-foreground max-w-[150px] truncate">{file.name}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveFile(idx)}
                    className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                    aria-label="파일 제거"
                  >
                    <X className="h-3 w-3" aria-hidden="true" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          <div className="flex items-end gap-2 p-3 border border-border/50 rounded-xl bg-card/50 backdrop-blur-sm shadow-sm">
            {/* Think Mode Toggle - Only show if model supports thinking */}
            {modelSupportsThinking && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setEnableThinking(!enableThinking)}
                    aria-label={enableThinking ? "추론 모드 끄기" : "추론 모드 켜기"}
                    aria-pressed={enableThinking}
                    className={cn(
                      "h-[44px] w-[44px] p-0 shrink-0 rounded-lg transition-all border",
                      enableThinking 
                        ? "bg-primary/10 text-primary border-primary/50 hover:border-primary" 
                        : "border-border/30 hover:border-border/50 hover:bg-accent/50 text-muted-foreground"
                    )}
                  >
                    <Brain className={cn(
                      "h-5 w-5 transition-all",
                      enableThinking 
                        ? "fill-primary stroke-primary/80 stroke-[2]" 
                        : "stroke-foreground/70 stroke-[1.5]"
                    )} aria-hidden="true" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top" className="bg-popover text-popover-foreground border-border/50">
                  <div className="text-xs">
                    {enableThinking ? "Thinking Mode: ON" : "Thinking Mode: OFF"}
                    <div className="text-muted-foreground mt-1">
                      {enableThinking 
                        ? "LLM의 추론 과정이 표시됩니다" 
                        : "클릭하여 추론 과정 표시 활성화"}
                    </div>
                  </div>
                </TooltipContent>
              </Tooltip>
            )}

            {/* 이미지 업로드 버튼 */}
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageUpload}
              className="hidden"
              aria-label="이미지 업로드"
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => imageInputRef.current?.click()}
                  className="h-[44px] w-[44px] p-0 shrink-0 rounded-lg border border-border/30 hover:border-border/50 hover:bg-accent/50 text-muted-foreground"
                  aria-label="이미지 첨부"
                >
                  <ImageIcon className="h-5 w-5" aria-hidden="true" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>이미지 첨부</TooltipContent>
            </Tooltip>

            {/* 파일 업로드 버튼 */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              aria-label="파일 업로드"
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="h-[44px] w-[44px] p-0 shrink-0 rounded-lg border border-border/30 hover:border-border/50 hover:bg-accent/50 text-muted-foreground"
                  aria-label="파일 첨부"
                >
                  <Paperclip className="h-5 w-5" aria-hidden="true" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>파일 첨부</TooltipContent>
            </Tooltip>
            
            <Textarea
              ref={textareaRef}
              id="chat-input"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
              className="flex-1 min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/60"
              rows={1}
              disabled={loading}
              aria-label="메시지 입력"
              aria-describedby="input-hint"
            />
            <Button 
              type="submit" 
              disabled={loading || (!input.trim() && attachedImages.length === 0 && attachedFiles.length === 0)}
              size="lg"
              aria-label={loading ? "전송 중..." : "메시지 전송"}
              className="h-[44px] w-[44px] p-0 shrink-0 bg-primary/90 hover:bg-primary shadow-sm disabled:opacity-50"
            >
              {loading ? (
                <LoaderCircle className="h-5 w-5 animate-spin" aria-hidden="true" />
              ) : (
                <Send className="h-5 w-5" aria-hidden="true" />
              )}
            </Button>
          </div>
          {input.length > 0 && (
            <p id="input-hint" className="text-xs text-muted-foreground mt-1.5 px-1" role="status" aria-live="polite">
              {input.length} characters • Press Enter to send
            </p>
          )}
        </form>
      </div>
    </PageLayout>
  );
}
