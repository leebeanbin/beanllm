/**
 * Assistant Input Panel - Clean & Minimal Design
 * Show only what's necessary. Hide complexity.
 */

import { useStreamContext } from "@/providers/Stream";
import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

const ASSISTANT_GUIDES = {
  chat: {
    quick: "Ask me anything. I can help with questions, writing, problem-solving, and more.",
    examples: [
      "Explain quantum computing",
      "Help me write a professional email",
      "Solve this math problem: ...",
    ],
  },
  kg: {
    quick: "I extract entities and relationships from text to build knowledge graphs.",
    examples: [
      "Build a knowledge graph from: [your text]",
      "Show all entities",
      "Find relationship between X and Y",
    ],
  },
  rag: {
    quick: "I help you query your documents using semantic search and AI.",
    examples: [
      "Build RAG index from: [your documents]",
      "What are the main topics in my documents?",
      "Find information about X in the documents",
    ],
  },
  agent: {
    quick: "I autonomously complete complex tasks by breaking them down into steps.",
    examples: [
      "Research the latest AI trends and summarize",
      "Find and analyze competitors in the market",
      "Create a project plan for building an app",
    ],
  },
  web: {
    quick: "I search the web using DuckDuckGo and other engines to find information.",
    examples: [
      "Search for latest news about quantum computing",
      "Find documentation for React hooks",
      "Research best practices for API design",
    ],
  },
  rag_debug: {
    quick: "I analyze your RAG pipeline to identify issues and improve retrieval quality.",
    examples: [
      "Analyze my RAG pipeline performance",
      "Why are my retrieval results not relevant?",
      "How can I improve my embeddings?",
    ],
  },
  optimizer: {
    quick: "I automatically find the best parameters and configurations for your AI system.",
    examples: [
      "Optimize my RAG system for latency",
      "Find best parameters for accuracy",
      "Improve my agent's performance",
    ],
  },
  multi_agent: {
    quick: "I coordinate multiple AI agents to work together on complex problems.",
    examples: [
      "Run a debate between 3 agents on: [topic]",
      "Parallel consensus on: [decision]",
      "Hierarchical task execution for: [project]",
    ],
  },
  orchestrator: {
    quick: "I manage and execute complex multi-step workflows with multiple agents.",
    examples: [
      "Research and write a report on: [topic]",
      "Run a parallel consensus workflow",
      "Execute a debate workflow on: [topic]",
    ],
  },
} as const;

export function AssistantInputPanel() {
  const { assistantId } = useStreamContext();
  const [showExamples, setShowExamples] = useState(false);

  const guide = ASSISTANT_GUIDES[assistantId as keyof typeof ASSISTANT_GUIDES] || ASSISTANT_GUIDES.chat;

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      {/* Main description */}
      <div className="text-center space-y-2">
        <p className="text-base text-gray-700 dark:text-gray-300 leading-relaxed">
          {guide.quick}
        </p>
      </div>

      {/* Examples toggle */}
      <div className="flex justify-center">
        <button
          onClick={() => setShowExamples(!showExamples)}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg",
            "text-sm font-medium",
            "text-gray-600 dark:text-gray-400",
            "hover:text-gray-900 dark:hover:text-gray-200",
            "hover:bg-gray-100 dark:hover:bg-gray-800",
            "transition-colors duration-150"
          )}
        >
          {showExamples ? (
            <>
              Hide examples
              <ChevronUp className="h-4 w-4" />
            </>
          ) : (
            <>
              Show examples
              <ChevronDown className="h-4 w-4" />
            </>
          )}
        </button>
      </div>

      {/* Examples (collapsible) */}
      {showExamples && (
        <div className="space-y-2 animate-in fade-in slide-in-from-top-2 duration-200">
          {guide.examples.map((example, i) => (
            <div
              key={i}
              className={cn(
                "px-4 py-3 rounded-lg",
                "bg-gray-50 dark:bg-gray-800/50",
                "border border-gray-200 dark:border-gray-700",
                "text-sm text-gray-700 dark:text-gray-300"
              )}
            >
              <code className="font-mono">{example}</code>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
