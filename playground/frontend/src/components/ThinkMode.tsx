"use client";

import { useState } from "react";
import { Brain, ChevronDown, ChevronUp, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface ThinkModeProps {
  thoughts?: string[];
  steps?: Array<{ step: number; thought?: string; action?: string; result?: string }>;
  className?: string;
}

/** 카드 없음. border + spacing + 타이포 위계만. */
export function ThinkMode({ thoughts, steps, className }: ThinkModeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!thoughts?.length && !steps?.length) return null;

  const itemCount = thoughts?.length || steps?.length || 0;

  return (
    <div className={cn("border border-border/40 rounded-lg overflow-hidden", className)}>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full justify-between h-auto py-3 px-3 hover:bg-muted/30 rounded-none border-b border-border/40"
      >
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Thinking</span>
          <span className="text-xs text-muted-foreground">
            {itemCount} {itemCount === 1 ? "step" : "steps"}
          </span>
        </div>
        {isExpanded ? <ChevronUp className="h-4 w-4 text-muted-foreground shrink-0" /> : <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />}
      </Button>

      {isExpanded && (
        <div className="p-3 space-y-2 max-h-[400px] overflow-y-auto">
          {thoughts?.map((thought, idx) => (
            <div key={idx} className="text-sm py-2 px-3 rounded border-l-2 border-border bg-muted/20">
              <span className="text-xs font-medium text-muted-foreground">Thought {idx + 1}</span>
              <div className="text-foreground leading-relaxed whitespace-pre-wrap mt-1">{thought}</div>
            </div>
          ))}
          {steps?.map((step, idx) => (
            <div key={idx} className="text-sm py-2 px-3 rounded border-l-2 border-border bg-muted/20 space-y-1.5">
              <span className="text-xs font-medium text-muted-foreground">Step {step.step ?? idx + 1}</span>
              {step.thought && <div><span className="text-muted-foreground">Thought </span><span className="whitespace-pre-wrap">{step.thought}</span></div>}
              {step.action && <div><span className="text-muted-foreground">Action </span><span className="font-mono text-xs">{step.action}</span></div>}
              {step.result && <div><span className="text-muted-foreground">Result </span><span className="whitespace-pre-wrap">{step.result}</span></div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
