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
} from "lucide-react";

interface FeatureSelectorProps {
  value: FeatureMode;
  onChange: (value: FeatureMode) => void;
  className?: string;
}

const FEATURE_OPTIONS: Array<{
  value: FeatureMode;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}> = [
  {
    value: "chat",
    label: "Chat",
    icon: MessageSquare,
    description: "일반 대화",
  },
  {
    value: "rag",
    label: "RAG",
    icon: FileText,
    description: "문서 검색 & 질의응답",
  },
  {
    value: "multi-agent",
    label: "Multi-Agent",
    icon: Users,
    description: "에이전트 협업",
  },
  {
    value: "knowledge-graph",
    label: "Knowledge Graph",
    icon: Network,
    description: "지식 그래프 구축 & 탐색",
  },
  {
    value: "audio",
    label: "Audio",
    icon: Mic,
    description: "음성 인식 & 전사",
  },
  {
    value: "ocr",
    label: "OCR",
    icon: ScanText,
    description: "이미지 텍스트 인식",
  },
  {
    value: "google",
    label: "Google Workspace",
    icon: FolderOpen,
    description: "Docs, Drive, Gmail 연동",
  },
  {
    value: "web-search",
    label: "Web Search",
    icon: Search,
    description: "웹 검색",
  },
];

export function FeatureSelector({
  value,
  onChange,
  className,
}: FeatureSelectorProps) {
  const selectedFeature = FEATURE_OPTIONS.find((f) => f.value === value);

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className={className}>
        <div className="flex items-center gap-2">
          {selectedFeature?.icon && (
            <selectedFeature.icon className="h-4 w-4" />
          )}
          <SelectValue placeholder="Select feature" />
        </div>
      </SelectTrigger>
      <SelectContent>
        {FEATURE_OPTIONS.map((feature) => {
          const Icon = feature.icon;
          return (
            <SelectItem key={feature.value} value={feature.value}>
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4 text-muted-foreground" />
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
