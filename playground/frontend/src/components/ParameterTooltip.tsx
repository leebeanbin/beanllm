"use client";

import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface ParameterTooltipProps {
  parameter: "temperature" | "max_tokens" | "top_p" | "frequency_penalty" | "presence_penalty";
  children: React.ReactNode;
}

const parameterInfo: Record<string, { title: string; examples: string[] }> = {
  temperature: {
    title: "Temperature",
    examples: [
      "낮음 (0.0-0.3): 일관된 답변, 사실 기반 질문에 적합",
      "중간 (0.4-0.7): 균형잡힌 창의성, 일반적인 대화에 적합",
      "높음 (0.8-2.0): 창의적인 답변, 아이디어 생성에 적합",
    ],
  },
  max_tokens: {
    title: "Max Tokens",
    examples: [
      "낮음 (100-500): 짧은 답변, 요약에 적합",
      "중간 (500-2000): 일반적인 답변 길이",
      "높음 (2000+): 긴 답변, 상세한 설명에 적합",
    ],
  },
  top_p: {
    title: "Top P (Nucleus Sampling)",
    examples: [
      "낮음 (0.1-0.5): 집중된 답변, 일관성 높음",
      "중간 (0.6-0.9): 다양한 표현, 자연스러운 대화",
      "높음 (0.95-1.0): 매우 다양한 답변, 창의성 높음",
    ],
  },
  frequency_penalty: {
    title: "Frequency Penalty",
    examples: [
      "음수 (-2.0~0): 반복 허용, 패턴 학습에 유용",
      "0: 중립, 반복에 영향 없음",
      "양수 (0.1-2.0): 반복 감소, 다양한 표현 유도",
    ],
  },
  presence_penalty: {
    title: "Presence Penalty",
    examples: [
      "음수 (-2.0~0): 기존 주제 유지",
      "0: 중립, 주제 전환에 영향 없음",
      "양수 (0.1-2.0): 새로운 주제 도입, 다양성 증가",
    ],
  },
};

export function ParameterTooltip({ parameter, children }: ParameterTooltipProps) {
  const info = parameterInfo[parameter];
  if (!info) return <>{children}</>;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex items-center gap-1.5" role="group" aria-label={`${info.title} 정보`}>
          {children}
          <Info 
            className="h-3.5 w-3.5 text-muted-foreground cursor-help" 
            aria-hidden="true"
            aria-label={`${info.title} 도움말`}
          />
        </div>
      </TooltipTrigger>
      <TooltipContent
        side="right"
        className="max-w-xs bg-popover text-popover-foreground border-border/50 shadow-lg p-3"
        role="tooltip"
        aria-label={`${info.title} 설명`}
      >
        <div className="space-y-2">
          <div className="font-semibold text-sm">{info.title}</div>
          <div className="space-y-1.5 text-xs text-muted-foreground" role="list">
            {info.examples.map((example, idx) => (
              <div key={idx} className="leading-relaxed" role="listitem">
                • {example}
              </div>
            ))}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
