"use client";

import { cn } from "@/lib/utils";

export interface ReasoningStep {
  id: string;
  label: string;
}

interface ReasoningStepperProps {
  steps: ReasoningStep[];
  currentIndex: number;
  variant?: "horizontal" | "vertical";
  className?: string;
}

/**
 * Think-mode reasoning phase visualization.
 * Shows step labels with current step highlighted. No Deep Research — Think only.
 */
export function ReasoningStepper({
  steps,
  currentIndex,
  variant = "horizontal",
  className,
}: ReasoningStepperProps) {
  if (!steps.length) return null;

  const clamp = Math.min(Math.max(0, currentIndex), steps.length - 1);

  return (
    <div
      className={cn(
        "flex items-center",
        variant === "horizontal" ? "gap-1 flex-wrap" : "flex-col gap-2",
        className
      )}
      role="progressbar"
      aria-valuenow={clamp + 1}
      aria-valuemin={1}
      aria-valuemax={steps.length}
      aria-label={`Reasoning step ${clamp + 1} of ${steps.length}: ${steps[clamp].label}`}
    >
      {steps.map((step, idx) => {
        const isPast = idx < clamp;
        const isCurrent = idx === clamp;
        const isFuture = idx > clamp;

        return (
          <div key={step.id} className="flex items-center shrink-0">
            <div
              className={cn(
                "flex items-center justify-center rounded-full text-xs font-medium transition-colors shrink-0",
                "w-5 h-5",
                isPast && "bg-muted-foreground/20 text-muted-foreground",
                isCurrent && "bg-primary text-primary-foreground",
                isFuture && "bg-muted/50 text-muted-foreground"
              )}
            >
              {isPast ? "✓" : idx + 1}
            </div>
            <span
              className={cn(
                "text-sm ml-1.5 transition-colors",
                isCurrent && "text-foreground font-medium",
                (isPast || isFuture) && "text-muted-foreground"
              )}
            >
              {step.label}
            </span>
            {variant === "horizontal" && idx < steps.length - 1 && (
              <div
                className={cn(
                  "w-3 h-px mx-2 shrink-0",
                  isPast ? "bg-muted-foreground/40" : "bg-border"
                )}
                aria-hidden
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Default Think-mode phases (no Deep Research). */
export const DEFAULT_REASONING_STEPS: ReasoningStep[] = [
  { id: "understanding", label: "Understanding" },
  { id: "reasoning", label: "Reasoning" },
  { id: "responding", label: "Responding" },
];
