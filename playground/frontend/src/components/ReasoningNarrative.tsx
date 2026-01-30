"use client";

import { cn } from "@/lib/utils";

interface ReasoningNarrativeProps {
  text: string;
  state?: "reasoning" | "completed";
  className?: string;
}

/**
 * Think-mode narrative block: user-facing reasoning description.
 * Typography subtle, left-accent for "thought in progress". No Deep Research.
 */
export function ReasoningNarrative({
  text,
  state = "reasoning",
  className,
}: ReasoningNarrativeProps) {
  if (!text?.trim()) return null;

  return (
    <div
      className={cn(
        "border-l-2 pl-3 py-1 text-sm text-muted-foreground",
        state === "reasoning" && "border-primary/30",
        state === "completed" && "border-muted-foreground/30",
        className
      )}
      role="status"
      aria-live="polite"
      aria-label={text}
    >
      {text}
    </div>
  );
}
