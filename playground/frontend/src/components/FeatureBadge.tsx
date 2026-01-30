"use client";

import { FeatureMode } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
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

interface FeatureBadgeProps {
  feature: FeatureMode;
  className?: string;
  showIcon?: boolean;
}

const FEATURE_CONFIG: Record<
  FeatureMode,
  {
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    color: string;
  }
> = {
  agentic: {
    label: "Agentic",
    icon: Sparkles,
    color: "bg-indigo-500/10 text-indigo-600 border-indigo-500/20",
  },
  chat: {
    label: "Chat",
    icon: MessageSquare,
    color: "bg-blue-500/10 text-blue-600 border-blue-500/20",
  },
  rag: {
    label: "RAG",
    icon: FileText,
    color: "bg-green-500/10 text-green-600 border-green-500/20",
  },
  "multi-agent": {
    label: "Multi-Agent",
    icon: Users,
    color: "bg-purple-500/10 text-purple-600 border-purple-500/20",
  },
  "knowledge-graph": {
    label: "KG",
    icon: Network,
    color: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20",
  },
  audio: {
    label: "Audio",
    icon: Mic,
    color: "bg-pink-500/10 text-pink-600 border-pink-500/20",
  },
  ocr: {
    label: "OCR",
    icon: ScanText,
    color: "bg-orange-500/10 text-orange-600 border-orange-500/20",
  },
  google: {
    label: "Google",
    icon: FolderOpen,
    color: "bg-red-500/10 text-red-600 border-red-500/20",
  },
  "web-search": {
    label: "Web",
    icon: Search,
    color: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
  },
};

export function FeatureBadge({
  feature,
  className,
  showIcon = true,
}: FeatureBadgeProps) {
  const config = FEATURE_CONFIG[feature];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={cn(
        "font-medium text-[10px] px-1.5 py-0.5 gap-1",
        config.color,
        className
      )}
    >
      {showIcon && <Icon className="h-2.5 w-2.5" />}
      {config.label}
    </Badge>
  );
}
