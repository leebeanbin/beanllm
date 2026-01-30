"use client";

import { cn } from "@/lib/utils";

/**
 * 타이핑/생성 중 인디케이터
 * Design: Clarity over cleverness. 말풍선 + 점 3개만. 장식 최소화.
 * (Functional minimalism, 3초 안에 이해 가능)
 */
export function TypingIndicator({ className }: { className?: string }) {
  return (
    <div className={cn("relative inline-flex", className)}>
      <div
        className={cn(
          "rounded-2xl px-4 py-3",
          "bg-muted/50 border border-border/40",
          "inline-flex items-center justify-center gap-1.5",
          "min-w-[72px]"
        )}
        role="status"
        aria-label="Generating response"
      >
        <span className="flex items-center gap-1" aria-hidden>
          <span
            className="h-1.5 w-1.5 rounded-full bg-muted-foreground/70 animate-[typing-bounce_1.4s_ease-in-out_infinite]"
            style={{ animationDelay: "0ms" }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-muted-foreground/70 animate-[typing-bounce_1.4s_ease-in-out_infinite]"
            style={{ animationDelay: "160ms" }}
          />
          <span
            className="h-1.5 w-1.5 rounded-full bg-muted-foreground/70 animate-[typing-bounce_1.4s_ease-in-out_infinite]"
            style={{ animationDelay: "320ms" }}
          />
        </span>
      </div>
      {/* 말풍선 꼬리 - 아래 왼쪽 */}
      <div
        className="absolute -bottom-1.5 left-4 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-muted/50"
        aria-hidden
      />
    </div>
  );
}
