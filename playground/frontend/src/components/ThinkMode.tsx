"use client";

import { useState } from "react";
import { Brain, ChevronDown, ChevronUp, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface ThinkModeProps {
  thoughts?: string[];
  steps?: Array<{ step: number; thought?: string; action?: string; result?: string }>;
  className?: string;
}

export function ThinkMode({ thoughts, steps, className }: ThinkModeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!thoughts?.length && !steps?.length) return null;

  const itemCount = thoughts?.length || steps?.length || 0;

  return (
    <Card className={cn("border-border/50 bg-card/80 backdrop-blur-sm shadow-sm relative z-[60]", className)}>
      <CardContent className="p-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full justify-between h-auto p-4 hover:bg-accent/50"
        >
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-md bg-primary/10">
              <Brain className="h-4 w-4 text-primary" />
            </div>
            <div className="flex flex-col items-start">
              <span className="text-sm font-semibold text-foreground">Thinking Process</span>
              <span className="text-xs text-muted-foreground">
                {itemCount} {itemCount === 1 ? "step" : "steps"}
              </span>
            </div>
          </div>
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </Button>

        {isExpanded && (
          <div className="px-4 pb-4 space-y-3 max-h-[500px] overflow-y-auto custom-scrollbar">
            {thoughts?.map((thought, idx) => (
              <div
                key={idx}
                className="text-sm p-3.5 rounded-lg bg-muted/40 border-l-3 border-primary/60 shadow-sm"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="h-3.5 w-3.5 text-primary/70" />
                  <div className="font-medium text-xs text-muted-foreground uppercase tracking-wide">
                    Thought {idx + 1}
                  </div>
                </div>
                <div className="text-foreground leading-relaxed whitespace-pre-wrap">{thought}</div>
              </div>
            ))}

            {steps?.map((step, idx) => (
              <div
                key={idx}
                className="text-sm p-3.5 rounded-lg bg-muted/40 border-l-3 border-accent/60 shadow-sm"
              >
                <div className="flex items-center gap-2 mb-2.5">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-semibold text-primary">
                    {step.step || idx + 1}
                  </div>
                  <div className="font-medium text-xs text-muted-foreground uppercase tracking-wide">
                    Step {step.step || idx + 1}
                  </div>
                </div>
                <div className="space-y-2">
                  {step.thought && (
                    <div className="p-2 rounded-md bg-background/50 border border-border/50 hover:border-border">
                      <div className="font-medium text-xs text-muted-foreground mb-1">💭 Thought</div>
                      <div className="text-foreground leading-relaxed whitespace-pre-wrap">{step.thought}</div>
                    </div>
                  )}
                  {step.action && (
                    <div className="p-2 rounded-md bg-background/50 border border-border/50 hover:border-border">
                      <div className="font-medium text-xs text-muted-foreground mb-1">⚡ Action</div>
                      <div className="text-foreground font-mono text-xs">{step.action}</div>
                    </div>
                  )}
                  {step.result && (
                    <div className="p-2 rounded-md bg-background/50 border border-border/50 hover:border-border">
                      <div className="font-medium text-xs text-muted-foreground mb-1">📊 Result</div>
                      <div className="text-foreground leading-relaxed whitespace-pre-wrap">{step.result}</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
