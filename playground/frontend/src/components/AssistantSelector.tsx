/**
 * Assistant Selector - Minimal & Functional Design
 * Less is more. Focus on usability.
 */

import { useStreamContext } from "@/providers/Stream";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";
import { useState, useRef, useEffect } from "react";

const ASSISTANTS = [
  { id: "chat", name: "Chat", desc: "General conversation" },
  { id: "kg", name: "Knowledge Graph", desc: "Entity extraction" },
  { id: "rag", name: "RAG", desc: "Document Q&A" },
  { id: "agent", name: "Agent", desc: "Autonomous tasks" },
  { id: "web", name: "Web Search", desc: "Search the web" },
  { id: "rag_debug", name: "RAG Debug", desc: "Debug pipeline" },
  { id: "optimizer", name: "Optimizer", desc: "Optimize performance" },
  { id: "multi_agent", name: "Multi-Agent", desc: "Agent collaboration" },
  { id: "orchestrator", name: "Orchestrator", desc: "Workflow management" },
] as const;

export function AssistantSelector() {
  const { assistantId, setAssistantId } = useStreamContext();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentAssistant = ASSISTANTS.find(a => a.id === assistantId) || ASSISTANTS[0];

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Current selection button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "group flex items-center gap-3 px-4 py-2.5 rounded-xl",
          "border border-gray-200/60 dark:border-gray-700/60",
          "bg-gradient-to-br from-white to-gray-50/50 dark:from-gray-800 dark:to-gray-800/50",
          "hover:border-gray-300 dark:hover:border-gray-600",
          "hover:shadow-sm",
          "transition-all duration-200",
          "focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50"
        )}
      >
        <div className="flex flex-col items-start">
          <span className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Assistant
          </span>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {currentAssistant.name}
          </span>
        </div>
        <ChevronDown
          className={cn(
            "h-4 w-4 text-gray-400 transition-transform duration-200 ml-auto",
            "group-hover:text-gray-600 dark:group-hover:text-gray-300",
            isOpen && "rotate-180"
          )}
        />
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div
          className={cn(
            "absolute top-full mt-2 right-0 z-50",
            "w-72 rounded-xl",
            "border border-gray-200/80 dark:border-gray-700/80",
            "bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl",
            "shadow-2xl shadow-gray-200/50 dark:shadow-gray-900/50",
            "animate-in fade-in slide-in-from-top-2 duration-200"
          )}
        >
          <div className="p-2">
            {ASSISTANTS.map((assistant) => {
              const isActive = assistantId === assistant.id;

              return (
                <button
                  key={assistant.id}
                  onClick={() => {
                    setAssistantId(assistant.id);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "group w-full px-3 py-2.5 text-left rounded-lg transition-all duration-150",
                    isActive
                      ? "bg-gradient-to-r from-blue-50 to-blue-100/50 dark:from-blue-950/40 dark:to-blue-900/20 shadow-sm"
                      : "hover:bg-gradient-to-r hover:from-gray-50 hover:to-gray-100/50 dark:hover:from-gray-800/50 dark:hover:to-gray-700/30"
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className={cn(
                        "text-sm font-semibold truncate",
                        isActive
                          ? "text-blue-700 dark:text-blue-300"
                          : "text-gray-900 dark:text-gray-100 group-hover:text-gray-900 dark:group-hover:text-white"
                      )}>
                        {assistant.name}
                      </p>
                      <p className={cn(
                        "text-xs mt-0.5 line-clamp-1",
                        isActive
                          ? "text-blue-600/80 dark:text-blue-400/80"
                          : "text-gray-500 dark:text-gray-400"
                      )}>
                        {assistant.desc}
                      </p>
                    </div>
                    {isActive && (
                      <div className="flex-shrink-0 h-2 w-2 rounded-full bg-blue-500 dark:bg-blue-400 shadow-sm shadow-blue-500/50" />
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
