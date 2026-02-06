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
  Users,
  Network,
  Mic,
  ScanText,
  FolderOpen,
  Sparkles,
  Wand2,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface FeatureSelectorProps {
  value: FeatureMode;
  onChange: (value: FeatureMode) => void;
  className?: string;
}

/**
 * 특화 기능 목록
 *
 * - Manual 모드에서만 표시됨
 * - Auto(Agentic) 모드에서는 AI가 자동으로 chat, rag, web-search 등을 선택
 * - 특화 기능은 명시적으로 선택해야 함
 */
const SPECIAL_FEATURES: Array<{
  value: FeatureMode;
  label: string;
  icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
  description: string;
  badge?: string;
}> = [
  {
    value: "agentic",
    label: "Auto (Agentic)",
    icon: Sparkles,
    description: "AI selects chat, RAG, search automatically",
    badge: "Recommended",
  },
  {
    value: "multi-agent",
    label: "Multi-Agent",
    icon: Users,
    description: "Collaborate with multiple AI agents",
  },
  {
    value: "knowledge-graph",
    label: "Knowledge Graph",
    icon: Network,
    description: "Build & query knowledge graphs",
  },
  {
    value: "audio",
    label: "Audio",
    icon: Mic,
    description: "Speech-to-text & audio processing",
  },
  {
    value: "ocr",
    label: "OCR",
    icon: ScanText,
    description: "Extract text from images",
  },
  {
    value: "google",
    label: "Google Workspace",
    icon: FolderOpen,
    description: "Drive, Docs, Sheets, Gmail, Calendar",
  },
];

// All available features for lookup
const FEATURE_OPTIONS = SPECIAL_FEATURES;

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
        <div className="px-2 py-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
          Select Feature Mode
        </div>
        {SPECIAL_FEATURES.map((feature) => {
          const Icon = feature.icon;
          const isRecommended = feature.badge === "Recommended";
          return (
            <SelectItem key={feature.value} value={feature.value}>
              <div className="flex items-center gap-2">
                <Icon
                  className={cn(
                    "h-4 w-4",
                    isRecommended ? "text-primary" : "text-muted-foreground"
                  )}
                  strokeWidth={1.5}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className={cn("font-medium", isRecommended && "text-primary")}>
                      {feature.label}
                    </span>
                    {feature.badge && (
                      <span className="px-1.5 py-0.5 text-[9px] font-medium bg-primary/10 text-primary rounded">
                        {feature.badge}
                      </span>
                    )}
                  </div>
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
