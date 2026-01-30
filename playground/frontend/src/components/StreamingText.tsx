"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

/**
 * ChatGPT/Gemini 스타일의 스트리밍 텍스트 컴포넌트
 * 타이핑 애니메이션과 커서 효과 제공
 */
export function StreamingText({ content, isStreaming = false, className }: StreamingTextProps) {
  const [displayedContent, setDisplayedContent] = useState("");
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    if (!isStreaming) {
      // 스트리밍이 끝나면 즉시 전체 내용 표시
      setDisplayedContent(content);
      setShowCursor(false);
      return;
    }

    // 스트리밍 중: 부드러운 타이핑 효과
    if (content.length > displayedContent.length) {
      const newContent = content.slice(0, displayedContent.length + 1);
      setDisplayedContent(newContent);
    }

    // 커서 깜빡임 효과
    const cursorInterval = setInterval(() => {
      setShowCursor((prev) => !prev);
    }, 530);

    return () => clearInterval(cursorInterval);
  }, [content, isStreaming, displayedContent.length]);

  return (
    <span className={cn("relative", className)}>
      {displayedContent}
      {isStreaming && showCursor && (
        <span className="inline-block w-0.5 h-[1.2em] bg-primary ml-0.5 animate-pulse" />
      )}
    </span>
  );
}
