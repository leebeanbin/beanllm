"use client";

import { useState, useEffect, useLayoutEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { X, ChevronRight, ChevronLeft, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

export interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  target?: string; // CSS selector for highlighting
  position?: "top" | "bottom" | "left" | "right" | "center";
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface OnboardingGuideProps {
  steps: OnboardingStep[];
  storageKey: string; // localStorage key to track if onboarding was completed
  onComplete?: () => void;
  onSkip?: () => void;
  onStepChange?: (stepIndex: number) => void; // Callback when step changes
  forceShow?: boolean; // Force show even if completed
}

export function OnboardingGuide({
  steps,
  storageKey,
  onComplete,
  onSkip,
  onStepChange,
  forceShow = false,
}: OnboardingGuideProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [highlightedElement, setHighlightedElement] = useState<HTMLElement | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ top: number; left: number; transform: string } | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // If forceShow is true, always show
    if (forceShow && steps.length > 0) {
      setIsVisible(true);
      setTimeout(() => {
        highlightStep(0);
      }, 300);
      return;
    }

    // Check if onboarding was already completed
    const completed = localStorage.getItem(storageKey);
    if (!completed && steps.length > 0) {
      setIsVisible(true);
      // Start highlighting after a short delay to ensure DOM is ready
      setTimeout(() => {
        highlightStep(0);
      }, 300);
    }
  }, [storageKey, steps.length, forceShow]);

  // Recalculate position when tooltip is rendered (useLayoutEffect for synchronous DOM updates)
  useLayoutEffect(() => {
    if (!isVisible || !highlightedElement || !tooltipRef.current) return;

    // Calculate position after tooltip is rendered
    const timer = setTimeout(() => {
      calculateTooltipPosition();
    }, 100);

    return () => clearTimeout(timer);
  }, [isVisible, highlightedElement, currentStep]);

  // Recalculate position on window resize or scroll (with debounce)
  useEffect(() => {
    if (!isVisible || !highlightedElement) return;

    let resizeTimer: NodeJS.Timeout;
    let scrollTimer: NodeJS.Timeout;

    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (tooltipRef.current) {
          calculateTooltipPosition();
        }
      }, 150);
    };

    const handleScroll = () => {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => {
        if (tooltipRef.current) {
          calculateTooltipPosition();
        }
      }, 100);
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("scroll", handleScroll, true);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("scroll", handleScroll, true);
      clearTimeout(resizeTimer);
      clearTimeout(scrollTimer);
    };
  }, [isVisible, highlightedElement, currentStep]);

  const calculateTooltipPosition = () => {
    if (!highlightedElement || !tooltipRef.current) {
      return;
    }

    // Use requestAnimationFrame for smooth position updates
    requestAnimationFrame(() => {
      if (!highlightedElement || !tooltipRef.current) return;

      const rect = highlightedElement.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const gap = 20;
      const padding = 24; // Increased padding for better viewport margins
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      
      // Get tooltip dimensions (use actual or fallback)
      const tooltipWidth = tooltipRect.width || 384; // w-96 = 384px
      const tooltipHeight = tooltipRect.height || 304; // Approximate height
      
      const currentStepData = steps[currentStep];
      const position = currentStepData?.position || "bottom";

      let top = 0;
      let left = 0;
      let transform = "";

      switch (position) {
        case "top": {
        top = rect.top - gap;
        left = rect.left + rect.width / 2;
        transform = "translate(-50%, -100%)";
        
        // Check if tooltip goes above viewport
        const tooltipTop = top - tooltipHeight;
        if (tooltipTop < padding) {
          // Move to bottom instead
          top = rect.bottom + gap;
          transform = "translate(-50%, 0)";
        }
        
        // Check horizontal bounds
        const tooltipLeft = left - tooltipWidth / 2;
        const tooltipRight = left + tooltipWidth / 2;
        if (tooltipLeft < padding) {
          left = padding + tooltipWidth / 2;
        } else if (tooltipRight > viewportWidth - padding) {
          left = viewportWidth - padding - tooltipWidth / 2;
        }
        break;
      }
      case "bottom": {
        top = rect.bottom + gap;
        left = rect.left + rect.width / 2;
        transform = "translate(-50%, 0)";
        
        // Check if tooltip goes below viewport
        const tooltipBottom = top + tooltipHeight;
        if (tooltipBottom > viewportHeight - padding) {
          // Move to top instead
          top = rect.top - gap;
          transform = "translate(-50%, -100%)";
        }
        
        // Check horizontal bounds
        const tooltipLeft = left - tooltipWidth / 2;
        const tooltipRight = left + tooltipWidth / 2;
        if (tooltipLeft < padding) {
          left = padding + tooltipWidth / 2;
        } else if (tooltipRight > viewportWidth - padding) {
          left = viewportWidth - padding - tooltipWidth / 2;
        }
        break;
      }
      case "left": {
        top = rect.top + rect.height / 2;
        left = rect.left - gap;
        transform = "translate(-100%, -50%)";
        
        // Check if tooltip goes left of viewport
        const tooltipLeft = left - tooltipWidth;
        if (tooltipLeft < padding) {
          // Move to right instead
          left = rect.right + gap;
          transform = "translate(0, -50%)";
        }
        
        // Check vertical bounds
        const tooltipTop = top - tooltipHeight / 2;
        const tooltipBottom = top + tooltipHeight / 2;
        if (tooltipTop < padding) {
          top = padding + tooltipHeight / 2;
        } else if (tooltipBottom > viewportHeight - padding) {
          top = viewportHeight - padding - tooltipHeight / 2;
        }
        break;
      }
      case "right": {
        top = rect.top + rect.height / 2;
        left = rect.right + gap;
        transform = "translate(0, -50%)";
        
        // Check if tooltip goes right of viewport
        const tooltipRight = left + tooltipWidth;
        if (tooltipRight > viewportWidth - padding) {
          // Move to left instead
          left = rect.left - gap;
          transform = "translate(-100%, -50%)";
        }
        
        // Check vertical bounds
        const tooltipTop = top - tooltipHeight / 2;
        const tooltipBottom = top + tooltipHeight / 2;
        if (tooltipTop < padding) {
          top = padding + tooltipHeight / 2;
        } else if (tooltipBottom > viewportHeight - padding) {
          top = viewportHeight - padding - tooltipHeight / 2;
        }
        break;
      }
      case "center":
      default: {
        top = rect.top + rect.height / 2;
        left = rect.left + rect.width / 2;
        transform = "translate(-50%, -50%)";
        
        // Check horizontal bounds
        const tooltipLeft = left - tooltipWidth / 2;
        const tooltipRight = left + tooltipWidth / 2;
        if (tooltipLeft < padding) {
          left = padding + tooltipWidth / 2;
        } else if (tooltipRight > viewportWidth - padding) {
          left = viewportWidth - padding - tooltipWidth / 2;
        }
        
        // Check vertical bounds
        const tooltipTop = top - tooltipHeight / 2;
        const tooltipBottom = top + tooltipHeight / 2;
        if (tooltipTop < padding) {
          top = padding + tooltipHeight / 2;
        } else if (tooltipBottom > viewportHeight - padding) {
          top = viewportHeight - padding - tooltipHeight / 2;
        }
        break;
      }
    }

      // Final bounds check - ensure everything is within viewport
      // Calculate actual tooltip bounds based on transform
      let actualTop = top;
      let actualLeft = left;
      
      // Calculate actual position based on transform
      if (transform.includes("translate(-50%")) {
        actualLeft = left - tooltipWidth / 2;
      } else if (transform.includes("translate(-100%")) {
        actualLeft = left - tooltipWidth;
      } else {
        actualLeft = left;
      }
      
      if (transform.includes("translateY(-50%)") || transform.includes("translate(-50%, -50%)")) {
        actualTop = top - tooltipHeight / 2;
      } else if (transform.includes("translateY(-100%)") || transform.includes("translate(-50%, -100%)")) {
        actualTop = top - tooltipHeight;
      } else {
        actualTop = top;
      }
      
      // Ensure tooltip stays within viewport with padding
      // Adjust if out of bounds
      if (actualTop < padding) {
        const offset = transform.includes("translateY(-50%)") || transform.includes("translate(-50%, -50%)") 
          ? tooltipHeight / 2 
          : transform.includes("translateY(-100%)") || transform.includes("translate(-50%, -100%)")
          ? tooltipHeight
          : 0;
        top = padding + offset;
      } else if (actualTop + tooltipHeight > viewportHeight - padding) {
        const offset = transform.includes("translateY(-50%)") || transform.includes("translate(-50%, -50%)")
          ? tooltipHeight / 2
          : 0;
        top = viewportHeight - padding - tooltipHeight + offset;
      }
      
      if (actualLeft < padding) {
        const offset = transform.includes("translate(-50%") ? tooltipWidth / 2 : transform.includes("translate(-100%") ? tooltipWidth : 0;
        left = padding + offset;
      } else if (actualLeft + tooltipWidth > viewportWidth - padding) {
        const offset = transform.includes("translate(-50%") ? tooltipWidth / 2 : 0;
        left = viewportWidth - padding - tooltipWidth + offset;
      }
      
      // Final clamp to ensure we never go outside viewport
      // But respect the transform origin
      const finalTop = Math.max(padding, Math.min(top, viewportHeight - padding));
      const finalLeft = Math.max(padding, Math.min(left, viewportWidth - padding));

      setTooltipPosition({
        top: finalTop,
        left: finalLeft,
        transform,
      });
    });
  };

  const highlightStep = (stepIndex: number) => {
    const step = steps[stepIndex];
    if (!step || !step.target) {
      setHighlightedElement(null);
      setTooltipPosition(null);
      return;
    }

    // Wait a bit for DOM to be ready
    setTimeout(() => {
      const element = document.querySelector(step.target!) as HTMLElement;
      if (element) {
        setHighlightedElement(element);
        // Scroll to element with better positioning
        element.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
        
        // Calculate position after delays to ensure DOM is stable
        // First delay: wait for scroll to complete
        setTimeout(() => {
          // Second delay: wait for tooltip to render
          setTimeout(() => {
            calculateTooltipPosition();
          }, 300);
        }, 500);
      } else {
        setHighlightedElement(null);
        setTooltipPosition(null);
      }
    }, 100);
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      const nextStep = currentStep + 1;
      setCurrentStep(nextStep);
      highlightStep(nextStep);
      
      // Notify parent component about step change
      onStepChange?.(nextStep);
      
      // Execute action if present
      if (steps[nextStep].action) {
        steps[nextStep].action!.onClick();
      }
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      setCurrentStep(prevStep);
      highlightStep(prevStep);
      
      // Notify parent component about step change
      onStepChange?.(prevStep);
    }
  };

  const handleComplete = () => {
    localStorage.setItem(storageKey, "completed");
    setIsVisible(false);
    setHighlightedElement(null);
    onComplete?.();
  };

  const handleSkip = () => {
    localStorage.setItem(storageKey, "completed");
    setIsVisible(false);
    setHighlightedElement(null);
    onSkip?.();
  };

  // Keyboard navigation support (KRDS guideline)
  useEffect(() => {
    if (!isVisible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // ESC: Close the guide
      if (e.key === "Escape") {
        e.preventDefault();
        handleSkip();
        return;
      }

      // Arrow keys: Navigate between steps
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        if (currentStep < steps.length - 1) {
          handleNext();
        }
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        if (currentStep > 0) {
          handlePrevious();
        }
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isVisible, currentStep, steps.length]);

  // Focus management (KRDS guideline - focus trap)
  useEffect(() => {
    if (!isVisible || !tooltipRef.current) return;

    const tooltip = tooltipRef.current;
    const focusableElements = tooltip.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    // Focus first element when guide opens
    if (firstElement) {
      setTimeout(() => firstElement.focus(), 100);
    }

    // Handle Tab key for focus trap
    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    tooltip.addEventListener("keydown", handleTabKey);
    return () => tooltip.removeEventListener("keydown", handleTabKey);
  }, [isVisible, currentStep]);

  if (!isVisible || steps.length === 0) {
    return null;
  }

  const step = steps[currentStep];
  
  // Default position if not calculated yet
  const defaultPosition = {
    top: "50vh",
    left: "50vw",
    transform: "translate(-50%, -50%)",
  };

  const tooltipStyle = tooltipPosition
    ? {
        top: `${tooltipPosition.top}px`,
        left: `${tooltipPosition.left}px`,
        transform: tooltipPosition.transform,
      }
    : defaultPosition;

  const stepId = `onboarding-step-${currentStep}`;
  const titleId = `${stepId}-title`;
  const descriptionId = `${stepId}-description`;

  return (
    <>
      {/* Overlay */}
      <div
        ref={overlayRef}
        className="fixed inset-0 z-50 bg-black/20"
        onClick={(e) => {
          // Only close if clicking the overlay itself, not the tooltip
          if (e.target === overlayRef.current) {
            handleSkip();
          }
        }}
        aria-hidden="true"
      >
        {/* Highlighted element border */}
        {highlightedElement && (
          <div
            className="fixed border-2 border-primary rounded-lg pointer-events-none animate-pulse z-40"
            style={{
              top: `${highlightedElement.getBoundingClientRect().top - 4}px`,
              left: `${highlightedElement.getBoundingClientRect().left - 4}px`,
              width: `${highlightedElement.getBoundingClientRect().width + 8}px`,
              height: `${highlightedElement.getBoundingClientRect().height + 8}px`,
              boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.4)",
            }}
            aria-hidden="true"
          />
        )}

        {/* Tooltip Card - KRDS guideline: dialog role with ARIA attributes */}
        <Card
          ref={tooltipRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          aria-describedby={descriptionId}
          aria-label={`온보딩 가이드: ${step.title}`}
          className={cn(
            "fixed z-50 w-96 max-w-[calc(100vw-2rem)] shadow-2xl border-2 border-primary/50",
            "focus:outline-none"
          )}
          style={tooltipStyle}
          onClick={(e) => e.stopPropagation()}
        >
          <CardContent className="pt-6 pb-4 px-5">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-primary/10">
                  <Sparkles className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-foreground">
                    Step {currentStep + 1} of {steps.length}
                  </div>
                  <div className="text-xs text-muted-foreground">Interactive Guide</div>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSkip}
                className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
                aria-label="가이드 닫기"
              >
                <X className="h-3.5 w-3.5" aria-hidden="true" />
              </Button>
            </div>

            {/* Content */}
            <div className="space-y-3 mb-5">
              <h3 id={titleId} className="text-base font-semibold text-foreground">
                {step.title}
              </h3>
              <p id={descriptionId} className="text-sm text-muted-foreground leading-relaxed">
                {step.description}
              </p>
            </div>

            {/* Progress bar */}
            <div className="mb-5">
              <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                />
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePrevious}
                  disabled={currentStep === 0}
                  className="h-8 text-xs"
                  aria-label="이전 단계"
                >
                  <ChevronLeft className="h-3.5 w-3.5 mr-1" aria-hidden="true" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSkip}
                  className="h-8 text-xs"
                  aria-label="가이드 건너뛰기"
                >
                  Skip
                </Button>
              </div>
              <Button
                onClick={handleNext}
                size="sm"
                className="h-8 text-xs"
                aria-label={currentStep === steps.length - 1 ? "가이드 완료" : "다음 단계"}
              >
                {currentStep === steps.length - 1 ? "Complete" : "Next"}
                {currentStep < steps.length - 1 && (
                  <ChevronRight className="h-3.5 w-3.5 ml-1" aria-hidden="true" />
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
