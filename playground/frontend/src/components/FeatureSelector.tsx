"use client";

import { FeatureMode } from "@/types/chat";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  MessageSquare,
  FileText,
  Users,
  Network,
  Mic,
  ScanText,
  FolderOpen,
  Search,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface FeatureSelectorProps {
  value: FeatureMode;
  onChange: (value: FeatureMode) => void;
  className?: string;
}

// 일반 기능 (자동 감지 가능)
const GENERAL_FEATURES: Array<{
  value: FeatureMode;
  label: string;
  icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
  description: string;
}> = [
  {
    value: "agentic",
    label: "Agentic",
    icon: Sparkles,
    description: "Auto feature selection (default)",
  },
  {
    value: "chat",
    label: "Chat",
    icon: MessageSquare,
    description: "General chat",
  },
  {
    value: "rag",
    label: "RAG",
    icon: FileText,
    description: "Document search & Q&A",
  },
  {
    value: "web-search",
    label: "Web Search",
    icon: Search,
    description: "Web search",
  },
];

// 특화 기능 (수동 선택 필요)
const SPECIAL_FEATURES: Array<{
  value: FeatureMode;
  label: string;
  icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
  description: string;
}> = [
  {
    value: "multi-agent",
    label: "Multi-Agent",
    icon: Users,
    description: "Agent collaboration",
  },
  {
    value: "knowledge-graph",
    label: "Knowledge Graph",
    icon: Network,
    description: "Knowledge graph construction & search",
  },
  {
    value: "audio",
    label: "Audio",
    icon: Mic,
    description: "Speech recognition & transcription",
  },
  {
    value: "ocr",
    label: "OCR",
    icon: ScanText,
    description: "Image text recognition",
  },
  {
    value: "google",
    label: "Google Workspace",
    icon: FolderOpen,
    description: "Docs, Drive, Gmail integration",
  },
];

// 수동 모드에서만 특화 기능 표시
const FEATURE_OPTIONS = [...GENERAL_FEATURES, ...SPECIAL_FEATURES];

export function FeatureSelector({
  value,
  onChange,
  className,
}: FeatureSelectorProps) {
  const selectedFeature = FEATURE_OPTIONS.find((f) => f.value === value);

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className={cn("w-full h-9", className)}>
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {selectedFeature?.icon && (
            <selectedFeature.icon className="h-4 w-4 shrink-0 text-muted-foreground flex-shrink-0" strokeWidth={1.5} />
          )}
          <SelectValue placeholder="Select feature" className="flex-1 min-w-0 text-left">
            {selectedFeature ? (
              <span className="truncate">{selectedFeature.label}</span>
            ) : (
              <span className="text-muted-foreground">Select feature</span>
            )}
          </SelectValue>
        </div>
      </SelectTrigger>
      <SelectContent>
        {/* General Features Section */}
        <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
          General Features
        </div>
        {GENERAL_FEATURES.map((feature) => {
          const Icon = feature.icon;
          return (
            <SelectItem key={feature.value} value={feature.value}>
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
                <div>
                  <div className="font-medium">{feature.label}</div>
                  <div className="text-xs text-muted-foreground">
                    {feature.description}
                  </div>
                </div>
              </div>
            </SelectItem>
          );
        })}
        {/* Special Features Section */}
        <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground border-t border-border/50 mt-1">
          Special Features
        </div>
        {SPECIAL_FEATURES.map((feature) => {
          const Icon = feature.icon;
          return (
            <SelectItem key={feature.value} value={feature.value}>
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
                <div>
                  <div className="font-medium">{feature.label}</div>
                  <div className="text-xs text-muted-foreground">
                    {feature.description}
                  </div>
                </div>
              </div>
            </SelectItem>
          );
        })}
      </SelectContent>
    </Select>
  );
}
