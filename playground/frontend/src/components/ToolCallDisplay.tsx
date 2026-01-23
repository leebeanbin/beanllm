/**
 * Tool Call Display - MCP Tool 실행 진행 상황 표시
 */
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  Wrench,
  CheckCircle2,
  XCircle,
  LoaderCircle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export interface ToolCallProgress {
  tool: string;
  status: "started" | "running" | "completed" | "failed";
  progress?: number; // 0.0 - 1.0
  step?: string;
  message?: string;
  result?: Record<string, any>;
  error?: string;
  arguments?: Record<string, any>;
}

interface ToolCallDisplayProps {
  toolCall: ToolCallProgress;
  className?: string;
}

export function ToolCallDisplay({ toolCall, className }: ToolCallDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Tool 이름을 사용자 친화적으로 변환
  const getToolDisplayName = (toolName: string): string => {
    const nameMap: Record<string, string> = {
      build_rag_system: "RAG 시스템 구축",
      query_rag_system: "RAG 질의",
      create_multiagent_system: "다중 에이전트 시스템 생성",
      run_multiagent_task: "다중 에이전트 작업 실행",
      build_knowledge_graph: "지식 그래프 구축",
      query_knowledge_graph: "지식 그래프 질의",
      transcribe_audio: "음성 전사",
      recognize_text_ocr: "OCR 텍스트 인식",
      evaluate_model: "모델 평가",
      export_to_google_docs: "Google Docs 내보내기",
    };
    return nameMap[toolName] || toolName;
  };

  // Status에 따른 아이콘
  const getStatusIcon = () => {
    switch (toolCall.status) {
      case "started":
      case "running":
        return <LoaderCircle className="h-4 w-4 animate-spin text-blue-500" />;
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Wrench className="h-4 w-4 text-muted-foreground" />;
    }
  };

  // Status에 따른 배경색
  const getStatusBgClass = () => {
    switch (toolCall.status) {
      case "started":
      case "running":
        return "bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800";
      case "completed":
        return "bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800";
      case "failed":
        return "bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800";
      default:
        return "bg-muted/50 border-border";
    }
  };

  // Progress bar 색상
  const getProgressBarClass = () => {
    switch (toolCall.status) {
      case "completed":
        return "bg-green-500";
      case "failed":
        return "bg-red-500";
      default:
        return "bg-blue-500";
    }
  };

  return (
    <Card className={cn("transition-all", getStatusBgClass(), className)}>
      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="text-sm font-medium text-foreground">
              {getToolDisplayName(toolCall.tool)}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="h-6 w-6 p-0"
            aria-label={isExpanded ? "축소" : "확장"}
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Progress Bar */}
        {toolCall.progress !== undefined && (toolCall.status === "started" || toolCall.status === "running") && (
          <div className="mb-3">
            <div className="h-1.5 bg-background/50 rounded-full overflow-hidden">
              <div
                className={cn("h-full transition-all duration-300", getProgressBarClass())}
                style={{ width: `${toolCall.progress * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Current Step Message */}
        {toolCall.message && (toolCall.status === "started" || toolCall.status === "running") && (
          <p className="text-xs text-muted-foreground mb-3">
            {toolCall.message}
          </p>
        )}

        {/* Expanded Details */}
        {isExpanded && (
          <div className="space-y-3 mt-3 pt-3 border-t border-border/50">
            {/* Arguments (if available) */}
            {toolCall.arguments && Object.keys(toolCall.arguments).length > 0 && (
              <div>
                <p className="text-xs font-medium text-foreground mb-1">Arguments:</p>
                <div className="bg-background/50 rounded-lg p-2 text-xs font-mono text-muted-foreground">
                  <pre>{JSON.stringify(toolCall.arguments, null, 2)}</pre>
                </div>
              </div>
            )}

            {/* Result (if completed) */}
            {toolCall.result && toolCall.status === "completed" && (
              <div>
                <p className="text-xs font-medium text-foreground mb-2">Result:</p>
                <div className="bg-background/50 rounded-lg p-3 text-sm">
                  {renderResult(toolCall.tool, toolCall.result)}
                </div>
              </div>
            )}

            {/* Error (if failed) */}
            {toolCall.error && toolCall.status === "failed" && (
              <div>
                <p className="text-xs font-medium text-red-600 dark:text-red-400 mb-1">
                  Error:
                </p>
                <p className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 rounded-lg p-2">
                  {toolCall.error}
                </p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Tool별 Result Formatting
 */
function renderResult(toolName: string, result: Record<string, any>) {
  // RAG 시스템 구축 결과
  if (toolName === "build_rag_system") {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Collection:</span>
          <span className="text-sm font-medium">{result.collection_name}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Documents:</span>
          <span className="text-sm font-medium">{result.document_count}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Chunks:</span>
          <span className="text-sm font-medium">{result.chunk_count}</span>
        </div>
      </div>
    );
  }

  // RAG 질의 결과
  if (toolName === "query_rag_system") {
    return (
      <div className="space-y-3">
        <div>
          <p className="text-xs font-medium text-foreground mb-1">Answer:</p>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.answer}
            </ReactMarkdown>
          </div>
        </div>

        {result.sources && result.sources.length > 0 && (
          <div>
            <p className="text-xs font-medium text-foreground mb-2">Sources:</p>
            <div className="space-y-2">
              {result.sources.slice(0, 3).map((source: any, idx: number) => (
                <div
                  key={idx}
                  className="bg-muted/50 rounded-lg p-2 text-xs"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-foreground">
                      #{source.rank}
                    </span>
                    <span className="text-muted-foreground">
                      Score: {(source.similarity_score || source.score).toFixed(2)}
                    </span>
                  </div>
                  <p className="text-muted-foreground line-clamp-2">
                    {source.content}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Multi-Agent 결과
  if (toolName === "run_multiagent_task") {
    return (
      <div className="space-y-3">
        {result.agent_responses && result.agent_responses.length > 0 && (
          <div>
            <p className="text-xs font-medium text-foreground mb-2">Agent Responses:</p>
            <div className="space-y-2">
              {result.agent_responses.map((response: any, idx: number) => (
                <div
                  key={idx}
                  className="bg-muted/50 rounded-lg p-2 text-xs"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-foreground">
                      {response.agent}
                    </span>
                    <span className="text-muted-foreground text-[10px]">
                      Round {response.round}
                    </span>
                  </div>
                  <p className="text-muted-foreground">{response.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.final_result && (
          <div>
            <p className="text-xs font-medium text-foreground mb-1">Final Result:</p>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.final_result}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    );
  }

  // 기본: JSON 형식으로 표시
  return (
    <div className="bg-background/50 rounded-lg p-2 text-xs font-mono text-muted-foreground overflow-auto">
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}
