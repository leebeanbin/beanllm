/**
 * Tool Call Display — MCP tool progress (Thinking Indicator / Streaming Cursor style)
 * Processing: text + blinking cursor (writing metaphor). No card, no spinner.
 * Completed/Failed: minimal summary + optional expand for details.
 */
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Wrench,
  CheckCircle2,
  XCircle,
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
  const [isExpanded, setIsExpanded] = useState(false);

  // Tool 이름을 사용자 친화적으로 변환
  const getToolDisplayName = (toolName: string): string => {
    const nameMap: Record<string, string> = {
      // Chat & RAG
      chat: "LLM Chat",
      rag: "RAG Document Search",
      build_rag_system: "Building RAG System",
      query_rag_system: "Querying RAG System",
      // Agent
      agent: "Agent",
      multi_agent: "Multi-Agent",
      create_multiagent_system: "Creating Multi-Agent System",
      run_multiagent_task: "Running Multi-Agent Task",
      // Knowledge Graph
      knowledge_graph: "Knowledge Graph",
      build_knowledge_graph: "Building Knowledge Graph",
      query_knowledge_graph: "Querying Knowledge Graph",
      // Audio & Vision
      audio_transcribe: "Audio Transcription",
      transcribe_audio: "Transcribing Audio",
      vision: "Image Analysis",
      ocr: "OCR Text Recognition",
      recognize_text_ocr: "OCR Text Recognition",
      // Web Search
      web_search: "Web Search",
      // Code
      code: "Code Generation",
      // Evaluation
      evaluation: "Model Evaluation",
      evaluate_model: "Evaluating Model",
      // Google Services
      google_drive: "Google Drive",
      google_docs: "Google Docs",
      google_gmail: "Gmail",
      google_calendar: "Google Calendar",
      google_sheets: "Google Sheets",
      export_to_google_docs: "Exporting to Google Docs",
      save_to_google_drive: "Saving to Google Drive",
      share_via_gmail: "Sharing via Gmail",
      list_google_drive_files: "Listing Drive Files",
    };
    return nameMap[toolName] || toolName;
  };

  const getStatusIcon = () => {
    switch (toolCall.status) {
      case "completed":
        return <CheckCircle2 className="h-3.5 w-3.5 text-muted-foreground" />;
      case "failed":
        return <XCircle className="h-3.5 w-3.5 text-destructive" />;
      default:
        return <Wrench className="h-3.5 w-3.5 text-muted-foreground" />;
    }
  };

  const isRunning = toolCall.status === "started" || toolCall.status === "running";
  let displayText =
    (isRunning && toolCall.message ? toolCall.message : null) ?? getToolDisplayName(toolCall.tool);
  // Normalize Korean progress messages to English (defensive)
  if (typeof displayText === "string" && /응답\s*생성|답변\s*생성|생성\s*중|문서\s*검색|초기화\s*중/.test(displayText)) {
    displayText = displayText.includes("검색") ? "Searching documents..." : "Generating response...";
  }
  const hasDetails =
    (toolCall.arguments && Object.keys(toolCall.arguments).length > 0) ||
    (toolCall.result && toolCall.status === "completed") ||
    (toolCall.error && toolCall.status === "failed");

  // Processing: Streaming Cursor style — text + blinking caret (writing metaphor). No card, no spinner.
  if (isRunning) {
    return (
      <div
        className={cn("flex items-center min-h-[28px] gap-1 py-1", className)}
        role="status"
        aria-label={displayText}
      >
        <span className="text-sm text-muted-foreground">{displayText}</span>
        <span
          className="text-muted-foreground animate-[cursor-blink_1s_step-end_infinite]"
          aria-hidden
        >
          |
        </span>
      </div>
    );
  }

  // Completed/Failed: minimal summary line + optional expand for details (no heavy card)
  return (
    <div className={cn("transition-all py-1", className)}>
      <div className="flex items-center gap-2 min-h-[28px]">
        {getStatusIcon()}
        <span className="text-sm text-muted-foreground flex-1 truncate">{displayText}</span>
        {hasDetails && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="h-6 w-6 p-0 shrink-0 opacity-60 hover:opacity-100"
            aria-label={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </Button>
        )}
      </div>
      {isExpanded && hasDetails && (
        <div className="space-y-2 mt-3 pt-3 border-t border-border/30 text-xs">
          {toolCall.arguments && Object.keys(toolCall.arguments).length > 0 && (
            <div>
              <p className="text-muted-foreground mb-1">Arguments</p>
              <pre className="bg-background/50 rounded p-2 font-mono text-muted-foreground overflow-auto max-h-24">
                {JSON.stringify(toolCall.arguments, null, 2)}
              </pre>
            </div>
          )}
          {toolCall.result && toolCall.status === "completed" && (
            <div>
              <p className="text-muted-foreground mb-1">Result</p>
              <div className="bg-background/50 rounded p-2">{renderResult(toolCall.tool, toolCall.result)}</div>
            </div>
          )}
          {toolCall.error && toolCall.status === "failed" && (
            <p className="text-destructive bg-destructive/10 rounded p-2">{toolCall.error}</p>
          )}
        </div>
      )}
    </div>
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

  // Google Drive 결과
  if (toolName === "google_drive" || toolName === "save_to_google_drive") {
    if (result.success) {
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-green-600 dark:text-green-500">
            <CheckCircle2 className="h-4 w-4" />
            <span className="text-sm font-medium">Saved successfully</span>
          </div>
          {result.filename && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Filename:</span>
              <span className="text-sm">{result.filename}</span>
            </div>
          )}
          {result.file_url && (
            <a
              href={result.file_url as string}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-primary hover:underline"
            >
              Open in Drive →
            </a>
          )}
          {result.message_count && (
            <div className="text-xs text-muted-foreground">
              {result.message_count} message{result.message_count !== 1 ? "s" : ""} saved
            </div>
          )}
        </div>
      );
    } else {
      return (
        <div className="text-red-600 text-sm">{result.error as string}</div>
      );
    }
  }

  // Google Docs 결과
  if (toolName === "google_docs" || toolName === "export_to_google_docs") {
    if (result.success) {
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-green-600 dark:text-green-500">
            <CheckCircle2 className="h-4 w-4" />
            <span className="text-sm font-medium">Document created</span>
          </div>
          {result.title && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Title:</span>
              <span className="text-sm">{result.title as string}</span>
            </div>
          )}
          {result.doc_url && (
            <a
              href={result.doc_url as string}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-primary hover:underline"
            >
              Open in Docs →
            </a>
          )}
        </div>
      );
    } else {
      return (
        <div className="text-red-600 text-sm">{result.error as string}</div>
      );
    }
  }

  // Gmail 결과
  if (toolName === "google_gmail" || toolName === "share_via_gmail") {
    if (result.success) {
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle2 className="h-4 w-4" />
            <span className="text-sm font-medium">Email sent</span>
          </div>
          {result.recipient && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">To:</span>
              <span className="text-sm">{result.recipient as string}</span>
            </div>
          )}
          {result.subject && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Subject:</span>
              <span className="text-sm">{result.subject as string}</span>
            </div>
          )}
        </div>
      );
    } else {
      return (
        <div className="text-red-600 text-sm">{result.error as string}</div>
      );
    }
  }

  // Drive 파일 목록 결과
  if (toolName === "list_google_drive_files") {
    if (result.success && result.files) {
      const files = result.files as Array<{
        id: string;
        name: string;
        type: string;
        modified: string;
        url: string;
      }>;
      return (
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">
            {result.file_count} file{result.file_count !== 1 ? "s" : ""}
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-1.5 rounded bg-muted/50 text-xs"
              >
                <span className="truncate flex-1">{file.name}</span>
                <a
                  href={file.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline ml-2"
                >
                  Open
                </a>
              </div>
            ))}
          </div>
        </div>
      );
    } else {
      return (
        <div className="text-red-600 text-sm">{result.error as string}</div>
      );
    }
  }

  // 기본: JSON 형식으로 표시
  return (
    <div className="bg-background/50 rounded-lg p-2 text-xs font-mono text-muted-foreground overflow-auto">
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}
