"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { LayoutGrid, Box, PanelTop, FileText, SlidersHorizontal, ChevronRight, Sparkles, Settings, Download, Upload, Trash2, RefreshCw, CheckCircle2, Brain, KeyRound, Github, MessageSquare, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { ModelSelectorSimple } from "./ModelSelectorSimple";
import { FeatureSelector } from "./FeatureSelector";
import { FeatureBadge } from "./FeatureBadge";
import { GoogleServiceSelector } from "./GoogleServiceSelector";
import { FeatureMode, GoogleService } from "@/types/chat";

interface InfoPanelProps {
  isOpen: boolean;
  onClose: () => void;
  /** "documents" | "monitor" are mapped to "session" for backward compat */
  defaultTab?: "quickstart" | "models" | "session" | "settings" | "documents" | "monitor";
  /** When set, panel uses this (controlled). Toggle lives in top tab bar. */
  isCollapsed?: boolean;
  onCollapseChange?: (collapsed: boolean) => void;
  // Model settings
  model: string;
  onModelChange: (model: string) => void;
  // Mode & Feature
  mode: "auto" | "manual";
  onModeChange: (mode: "auto" | "manual") => void;
  selectedFeature: FeatureMode;
  onFeatureChange: (feature: FeatureMode) => void;
  // Google services
  selectedGoogleServices: GoogleService[];
  onGoogleServicesChange: (services: GoogleService[]) => void;
  // Settings
  modelParams: any;
  temperature: number;
  setTemperature: (t: number) => void;
  maxTokens: number;
  setMaxTokens: (t: number) => void;
  topP: number;
  setTopP: (p: number) => void;
  frequencyPenalty: number;
  setFrequencyPenalty: (p: number) => void;
  presencePenalty: number;
  setPresencePenalty: (p: number) => void;
  customInstruction: string;
  setCustomInstruction: (i: string) => void;
  // Think mode
  enableThinking: boolean;
  setEnableThinking: (enabled: boolean) => void;
  modelSupportsThinking: boolean;
  // Messages
  messages: Array<{ role: string; content: string; timestamp?: Date }>;
  // Document preview
  documentPreviewContent?: string;
  documentPreviewTitle?: string;
  // Session summary
  sessionId?: string;
  sessionSummary?: string;
  sessionSummaryLoading?: boolean;
  onRefreshSummary?: () => void;
  // Chat history (load past session / new chat)
  onLoadSession?: (sessionId: string) => void;
  onNewChat?: () => void;
  // Actions
  onExport?: () => void;
  onImport?: () => void;
  onClear?: () => void;
}

export function InfoPanel({
  isOpen,
  onClose,
  defaultTab,
  isCollapsed: isCollapsedProp,
  onCollapseChange,
  model,
  onModelChange,
  mode,
  onModeChange,
  selectedFeature,
  onFeatureChange,
  selectedGoogleServices,
  onGoogleServicesChange,
  modelParams,
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
  topP,
  setTopP,
  frequencyPenalty,
  setFrequencyPenalty,
  presencePenalty,
  setPresencePenalty,
  customInstruction,
  setCustomInstruction,
  enableThinking,
  setEnableThinking,
  modelSupportsThinking,
  messages,
  documentPreviewContent,
  documentPreviewTitle,
  sessionId,
  sessionSummary,
  sessionSummaryLoading,
  onRefreshSummary,
  onLoadSession,
  onNewChat,
  onExport,
  onImport,
  onClear,
}: InfoPanelProps) {
  const resolvedTab =
    defaultTab && (defaultTab === "documents" || defaultTab === "monitor") ? "session" : defaultTab;
  const [activeTab, setActiveTab] = useState<"quickstart" | "models" | "session" | "settings">(
    resolvedTab || "quickstart"
  );
  const [isCollapsedLocal, setIsCollapsedLocal] = useState(false);
  const isCollapsed = isCollapsedProp !== undefined ? isCollapsedProp : isCollapsedLocal;

  // Chat history (past sessions) — fetch when session tab is active
  const [historySessions, setHistorySessions] = useState<Array<{ session_id: string; title: string; updated_at: string; message_count: number }>>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    if (resolvedTab) setActiveTab(resolvedTab);
  }, [resolvedTab]);

  useEffect(() => {
    if (activeTab !== "session" && !sessionId) return;
    let cancelled = false;
    setHistoryLoading(true);
    fetch(`${apiUrl}/api/chat/sessions?limit=20&sort_by=updated_at`)
      .then((res) => (res.ok ? res.json() : { sessions: [] }))
      .then((data) => {
        if (!cancelled) setHistorySessions(data.sessions || []);
      })
      .catch(() => { if (!cancelled) setHistorySessions([]); })
      .finally(() => { if (!cancelled) setHistoryLoading(false); });
    return () => { cancelled = true; };
  }, [activeTab, apiUrl, onLoadSession, sessionId]);

  // Always render, but control visibility with width

  if (!isOpen) {
    return <div className="w-0" />; // Hidden but takes no space
  }

  return (
    <>
      {isOpen && !isCollapsed && (
        <div
          className="fixed inset-0 z-[18] bg-black/30 md:hidden"
          aria-hidden
          onClick={onClose}
        />
      )}
      <div className={cn(
        "bg-background flex flex-col flex-shrink-0 transition-all overflow-hidden",
        isCollapsed ? "w-0" : "w-64",
        isOpen && !isCollapsed && "max-md:fixed max-md:right-0 max-md:top-14 max-md:bottom-0 max-md:z-[19] max-md:shadow-xl"
      )}>
        {!isCollapsed && (
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)} className="flex-1 flex flex-col min-h-0">
            <TabsList className="mx-2 mt-2 grid w-auto grid-cols-4 gap-0.5 p-0.5 h-auto bg-muted/20 border border-border/40 rounded-lg">
              <TabsTrigger value="quickstart" className="size-8 min-h-[44px] min-w-[44px] md:min-h-0 md:min-w-0 p-0 data-[state=active]:bg-background data-[state=active]:border data-[state=active]:border-border/40 rounded-md touch-manipulation" title="Start">
              <LayoutGrid className="h-4 w-4" strokeWidth={1.5} />
            </TabsTrigger>
            <TabsTrigger value="models" className="size-8 min-h-[44px] min-w-[44px] md:min-h-0 md:min-w-0 p-0 data-[state=active]:bg-background data-[state=active]:border data-[state=active]:border-border/40 rounded-md touch-manipulation" title="Model">
              <Box className="h-4 w-4" strokeWidth={1.5} />
            </TabsTrigger>
            <TabsTrigger value="session" className="size-8 min-h-[44px] min-w-[44px] md:min-h-0 md:min-w-0 p-0 data-[state=active]:bg-background data-[state=active]:border data-[state=active]:border-border/40 rounded-md touch-manipulation" title="History">
              <PanelTop className="h-4 w-4" strokeWidth={1.5} />
            </TabsTrigger>
            <TabsTrigger value="settings" className="size-8 min-h-[44px] min-w-[44px] md:min-h-0 md:min-w-0 p-0 data-[state=active]:bg-background data-[state=active]:border data-[state=active]:border-border/40 rounded-md touch-manipulation" title="Settings">
              <SlidersHorizontal className="h-4 w-4" strokeWidth={1.5} />
            </TabsTrigger>
          </TabsList>

          {/* Start — single entry layer: header = New chat, panel = guide */}
          <TabsContent value="quickstart" className="flex-1 flex flex-col min-h-0 mt-0 overflow-y-auto antialiased">
            <div className="p-4 space-y-4">
              <div>
                <h4 className="text-[13px] font-semibold text-foreground tracking-tight mb-0.5">Start</h4>
                <p className="text-[12px] text-muted-foreground leading-relaxed tracking-tight">
                  Type below to chat. New chat: History tab → Start new chat.
                </p>
              </div>

              <div className="space-y-1.5">
                {[
                  { step: 1, text: "Pick model & mode", tab: "models" as const },
                  { step: 2, text: "Type a message in the input below" },
                  { step: 3, text: "Answers & history here", tab: "session" as const },
                ].map(({ step, text, tab }) => (
                  <button
                    key={step}
                    type="button"
                    onClick={() => tab && setActiveTab(tab)}
                    className={cn(
                      "w-full flex items-center gap-3 p-2.5 rounded-lg border border-border/40 bg-muted/5 text-left transition-colors",
                      tab ? "hover:bg-muted/15 hover:border-border/50 cursor-pointer" : "cursor-default"
                    )}
                  >
                    <span className="flex-shrink-0 w-6 h-6 rounded-md border border-border/40 bg-background flex items-center justify-center text-[11px] font-semibold text-foreground tabular-nums tracking-tight">
                      {step}
                    </span>
                    <span className="text-[12px] font-medium text-foreground tracking-tight flex-1">{text}</span>
                    {tab && <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/80 shrink-0" strokeWidth={1.5} />}
                  </button>
                ))}
              </div>

              <div className="rounded-lg border border-border/40 bg-muted/5 p-3 space-y-2">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest">What you can do</p>
                <ul className="text-[12px] text-muted-foreground leading-relaxed space-y-1.5 tracking-tight">
                  <li>Chat — LLM dialogue</li>
                  <li>RAG — search your docs & answer</li>
                  <li>Agent — tool-use & reasoning</li>
                  <li>Google — Drive, Docs, Gmail, Calendar</li>
                  <li>Code — generate & run code</li>
                </ul>
              </div>

              <div className="rounded-lg border border-border/40 bg-muted/5 p-3 space-y-1.5">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest">Layer</p>
                <p className="text-[11px] text-muted-foreground leading-relaxed tracking-tight">
                  Top: Chat | Monitoring | Settings. Here: Start | Model | History | Settings.
                </p>
              </div>
            </div>
          </TabsContent>

          {/* Model — Mode 세그먼트형 + Model+Think → Feature → Google */}
          <TabsContent value="models" className="flex-1 flex flex-col min-h-0 mt-0 overflow-y-auto antialiased">
            <div className="p-4 space-y-4">
              {/* 1) Mode — 세그먼트 컨트롤 (pill 스타일) */}
              <div className="space-y-2">
                <h4 className="text-[13px] font-semibold text-foreground tracking-tight">Mode</h4>
                <div className="inline-flex p-0.5 rounded-lg border border-border/40 bg-muted/10 gap-0">
                  <button
                    type="button"
                    onClick={() => onModeChange("auto")}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-2 rounded-md text-[12px] font-medium tracking-tight transition-colors",
                      mode === "auto"
                        ? "bg-background text-foreground border border-border/40 shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/10"
                    )}
                  >
                    <Sparkles className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                    Auto
                  </button>
                  <button
                    type="button"
                    onClick={() => onModeChange("manual")}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-2 rounded-md text-[12px] font-medium tracking-tight transition-colors",
                      mode === "manual"
                        ? "bg-background text-foreground border border-border/40 shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/10"
                    )}
                  >
                    <Settings className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                    Manual
                  </button>
                </div>
                <p className="text-[11px] text-muted-foreground tracking-tight">
                  {mode === "auto" ? "AI picks the feature from your query." : "You pick the feature."}
                </p>
              </div>

              {/* 2) Model + Think */}
              <div className="space-y-2 pt-3 border-t border-border/40">
                <h4 className="text-[13px] font-semibold text-foreground tracking-tight">Model</h4>
                <ModelSelectorSimple value={model} onChange={onModelChange} />
                {modelSupportsThinking && (
                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center gap-2">
                      <Brain className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.5} />
                      <span className="text-[12px] font-medium text-foreground tracking-tight">Think mode</span>
                    </div>
                    <Switch checked={enableThinking} onCheckedChange={setEnableThinking} />
                  </div>
                )}
              </div>

              {/* 3) Feature (Manual only) */}
              {mode === "manual" && (
                <div className="space-y-2 pt-3 border-t border-border/40">
                  <h4 className="text-[13px] font-semibold text-foreground tracking-tight">Feature</h4>
                  <FeatureSelector value={selectedFeature} onChange={onFeatureChange} />
                  <FeatureBadge feature={selectedFeature} />
                </div>
              )}

              {/* 4) Google (when feature = google) */}
              {selectedFeature === "google" && (
                <div className="space-y-2 pt-3 border-t border-border/40">
                  <h4 className="text-[13px] font-semibold text-foreground tracking-tight">Google</h4>
                  <GoogleServiceSelector
                    selectedServices={selectedGoogleServices}
                    onChange={onGoogleServicesChange}
                  />
                  <div className="rounded-md border border-border/40 bg-muted/10 p-2 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Sync</span>
                      <span className="inline-flex items-center gap-1 text-foreground">
                        <CheckCircle2 className="h-3 w-3 text-green-600 dark:text-green-500" strokeWidth={1.5} />
                        Connected
                      </span>
                    </div>
                    <Button variant="outline" size="sm" className="w-full h-7 text-xs">
                      <RefreshCw className="h-3 w-3 mr-1.5" strokeWidth={1.5} />
                      Sync Now
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Session — chat history + stats + summary + doc preview */}
          <TabsContent value="session" className="flex-1 flex flex-col min-h-0 mt-0 overflow-y-auto">
            <div className="p-4 space-y-4 flex-1 flex flex-col min-h-0">
              {/* Chat history — list past sessions, load or new chat */}
              {(onLoadSession || onNewChat) && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="text-[13px] font-semibold text-foreground tracking-tight">Chat history</h4>
                    {onNewChat && (
                      <button
                        type="button"
                        onClick={onNewChat}
                        className="text-[11px] font-medium text-primary hover:underline touch-manipulation"
                      >
                        Start new chat
                      </button>
                    )}
                  </div>
                  {historyLoading ? (
                    <div className="rounded-lg border border-border/40 bg-muted/5 p-3 text-center text-xs text-muted-foreground">
                      Loading…
                    </div>
                  ) : historySessions.length === 0 ? (
                    <div className="rounded-lg border border-border/40 bg-muted/5 p-3 text-center text-xs text-muted-foreground">
                      <MessageSquare className="h-6 w-6 mx-auto mb-1 opacity-50" strokeWidth={1.5} />
                      No past chats
                    </div>
                  ) : (
                    <div className="max-h-40 overflow-y-auto space-y-1 rounded-lg border border-border/40 bg-muted/5 p-1.5">
                      {historySessions.map((s) => {
                        const updated = s.updated_at ? new Date(s.updated_at) : null;
                        const dateLabel = updated
                          ? updated.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
                          : "";
                        return (
                          <button
                            key={s.session_id}
                            type="button"
                            onClick={() => onLoadSession?.(s.session_id)}
                            className={cn(
                              "w-full text-left rounded-md px-2.5 py-2 border border-transparent hover:bg-muted/30 hover:border-border/40 transition-colors",
                              sessionId === s.session_id && "bg-muted/30 border-border/40"
                            )}
                          >
                            <div className="flex items-center gap-2 truncate">
                              <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground" strokeWidth={1.5} />
                              <span className="text-xs font-medium text-foreground truncate flex-1">
                                {s.title?.trim() || "New chat"}
                              </span>
                            </div>
                            <div className="flex items-center gap-2 mt-0.5 text-[11px] text-muted-foreground">
                              {dateLabel && <span className="flex items-center gap-0.5"><Clock className="h-3 w-3" strokeWidth={1.5} />{dateLabel}</span>}
                              {typeof s.message_count === "number" && <span>{s.message_count} msgs</span>}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              <div>
                <h4 className="text-sm font-medium text-foreground mb-1">Session</h4>
                <p className="text-xs text-muted-foreground">
                  {sessionId ? `ID: ${sessionId.slice(0, 8)}...` : "This chat's stats and sources."}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2.5 rounded-md border border-border/40 bg-muted/10">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Messages</div>
                  <div className="text-lg font-medium text-foreground mt-0.5">{messages.length}</div>
                </div>
                <div className="p-2.5 rounded-md border border-border/40 bg-muted/10">
                  <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Status</div>
                  <div className="text-sm font-medium text-foreground mt-0.5">
                    {messages.length > 0 ? "Active" : "Empty"}
                  </div>
                </div>
                {messages.length > 0 && (
                  <>
                    <div className="p-2.5 rounded-md border border-border/40 bg-muted/10">
                      <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">User</div>
                      <div className="text-lg font-medium text-foreground mt-0.5">
                        {messages.filter(m => m.role === "user").length}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-md border border-border/40 bg-muted/10">
                      <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Assistant</div>
                      <div className="text-lg font-medium text-foreground mt-0.5">
                        {messages.filter(m => m.role === "assistant").length}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Session Summary */}
              <div className="pt-2 border-t border-border/40">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Summary</p>
                  {sessionSummary && onRefreshSummary && (
                    <button
                      onClick={onRefreshSummary}
                      disabled={sessionSummaryLoading}
                      className="p-1 rounded hover:bg-muted/20 transition-colors disabled:opacity-50"
                      title="Refresh summary"
                    >
                      <RefreshCw className={cn("h-3 w-3 text-muted-foreground", sessionSummaryLoading && "animate-spin")} strokeWidth={1.5} />
                    </button>
                  )}
                </div>
                {sessionSummaryLoading ? (
                  <div className="rounded-md border border-border/40 bg-muted/10 p-2.5">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <RefreshCw className="h-3 w-3 animate-spin" strokeWidth={1.5} />
                      Generating summary...
                    </div>
                  </div>
                ) : sessionSummary ? (
                  <div className="rounded-md border border-border/40 bg-gradient-to-br from-muted/10 to-muted/5 p-2.5">
                    <p className="text-xs text-foreground leading-relaxed">{sessionSummary}</p>
                  </div>
                ) : messages.length >= 10 && onRefreshSummary ? (
                  <div className="rounded-md border border-border/40 bg-muted/10 p-2.5 space-y-2">
                    <p className="text-xs text-muted-foreground">
                      Generate an AI summary of this conversation.
                    </p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={onRefreshSummary}
                      className="w-full h-7 text-xs"
                    >
                      <Brain className="h-3 w-3 mr-1.5" strokeWidth={1.5} />
                      Generate Summary
                    </Button>
                  </div>
                ) : (
                  <div className="rounded-md border border-border/40 bg-muted/10 p-2.5">
                    <p className="text-xs text-muted-foreground">
                      Summary available after 10 messages ({messages.length}/10)
                    </p>
                  </div>
                )}
              </div>

              <div className="flex-1 min-h-0 flex flex-col pt-2 border-t border-border/40">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">Sources</p>
                {documentPreviewContent ? (
                  <div className="flex-1 min-h-0 overflow-y-auto rounded-md border border-border/40 bg-muted/10 p-2.5">
                    <p className="text-xs text-foreground leading-relaxed">{documentPreviewContent}</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-6 text-center rounded-md border border-border/40 bg-muted/10">
                    <FileText className="h-6 w-6 text-muted-foreground/50 mb-1.5" strokeWidth={1.5} />
                    <p className="text-xs text-muted-foreground max-w-[10rem]">
                      Related docs appear here when using RAG or document features.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          {/* Settings Tab — token: surface, border, typography; accent only for controls */}
          <TabsContent value="settings" className="flex-1 flex flex-col min-h-0 mt-0 overflow-y-auto">
            <div className="p-4 space-y-5">
              <div className="rounded-md border border-border/40 bg-muted/10 p-2.5 space-y-2">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Links</p>
                <div className="flex flex-col gap-1">
                  <Link
                    href="/settings"
                    className="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-foreground hover:bg-muted/30 transition-colors"
                  >
                    <KeyRound className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.5} />
                    API Keys · Google
                  </Link>
                  <a
                    href="https://github.com/leebeanbin/beanllm"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-foreground hover:bg-muted/30 transition-colors"
                  >
                    <Github className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.5} />
                    GitHub
                  </a>
                </div>
              </div>

              <div className="space-y-3 pt-1 border-t border-border/40">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium text-foreground">Parameters</h4>
                  <span className="text-[10px] text-muted-foreground truncate max-w-[8rem]">{model}</span>
                </div>
                
                {modelParams?.supports.temperature && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-foreground">Temperature</label>
                      <span className="text-xs text-muted-foreground tabular-nums">{temperature.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full h-2 bg-muted/50 rounded-md appearance-none cursor-pointer accent-foreground"
                    />
                  </div>
                )}

                {modelParams?.supports.max_tokens && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-foreground">Max Tokens</label>
                      <span className="text-xs text-muted-foreground">max {(modelParams?.max_tokens ?? 4000).toLocaleString()}</span>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max={modelParams?.max_tokens ?? 4000}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value) || 1000)}
                      className="w-full h-8 px-2.5 text-xs rounded-md border border-border/40 bg-background"
                    />
                  </div>
                )}

                {modelParams?.supports.top_p && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-foreground">Top P</label>
                      <span className="text-xs text-muted-foreground tabular-nums">{topP.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={topP}
                      onChange={(e) => setTopP(parseFloat(e.target.value))}
                      className="w-full h-2 bg-muted/50 rounded-md appearance-none cursor-pointer accent-foreground"
                    />
                  </div>
                )}

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-foreground">System prompt</label>
                  <textarea
                    value={customInstruction}
                    onChange={(e) => setCustomInstruction(e.target.value)}
                    placeholder="You are a helpful assistant..."
                    rows={3}
                    className="w-full px-2.5 py-2 text-xs rounded-md border border-border/40 bg-background resize-none text-foreground placeholder:text-muted-foreground"
                  />
                </div>
              </div>

              <div className="space-y-1.5 pt-3 border-t border-border/40">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Actions</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onExport}
                  disabled={messages.length === 0}
                  className="w-full h-8 text-xs justify-start rounded-md border-border/40"
                >
                  <Download className="h-3.5 w-3.5 mr-2" strokeWidth={1.5} />
                  Export JSON
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onImport}
                  className="w-full h-8 text-xs justify-start rounded-md border-border/40"
                >
                  <Upload className="h-3.5 w-3.5 mr-2" strokeWidth={1.5} />
                  Import Chat
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onClear}
                  disabled={messages.length === 0}
                  className="w-full h-8 text-xs justify-start rounded-md border-border/40 text-destructive hover:text-destructive hover:bg-destructive/5"
                >
                  <Trash2 className="h-3.5 w-3.5 mr-2" strokeWidth={1.5} />
                  Clear Chat
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      )}
      </div>
    </>
  );
}
