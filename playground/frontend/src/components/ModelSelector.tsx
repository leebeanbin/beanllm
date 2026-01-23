/**
 * Model Selector - Choose AI Model
 */

import { useState, useEffect } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useStream } from "@/providers/Stream";

interface Model {
  name: string;
  display_name: string;
  description: string;
  use_case: string;
  max_tokens: number;
  type: string;
}

interface ModelsGrouped {
  [provider: string]: Model[];
}

export function ModelSelector() {
  const { model, setModel } = useStream();
  const [isOpen, setIsOpen] = useState(false);
  const [models, setModels] = useState<ModelsGrouped>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch models from backend
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/api/models`)
      .then((res) => res.json())
      .then((data) => {
        setModels(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch models:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <span className="text-sm text-gray-500">Loading models...</span>
      </div>
    );
  }

  const currentModel = Object.values(models)
    .flat()
    .find((m) => m.name === model);

  return (
    <div className="relative">
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
            Model
          </span>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {currentModel?.display_name || model}
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
            "w-96 max-h-[32rem] overflow-y-auto rounded-xl",
            "border border-gray-200/80 dark:border-gray-700/80",
            "bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl",
            "shadow-2xl shadow-gray-200/50 dark:shadow-gray-900/50",
            "animate-in fade-in slide-in-from-top-2 duration-200"
          )}
        >
          {Object.entries(models).map(([provider, providerModels]) => (
            <div key={provider} className="border-b border-gray-100 dark:border-gray-800 last:border-b-0">
              <div className="sticky top-0 px-4 py-2.5 bg-gradient-to-r from-gray-50 to-gray-100/50 dark:from-gray-800 dark:to-gray-800/50 backdrop-blur-sm">
                <p className="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wide">
                  {provider}
                </p>
              </div>
              <div className="p-2">
                {providerModels.map((modelItem) => {
                  const isActive = model === modelItem.name;

                  return (
                    <button
                      key={modelItem.name}
                      onClick={() => {
                        setModel(modelItem.name);
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
                          <p
                            className={cn(
                              "text-sm font-semibold truncate",
                              isActive
                                ? "text-blue-700 dark:text-blue-300"
                                : "text-gray-900 dark:text-gray-100 group-hover:text-gray-900 dark:group-hover:text-white"
                            )}
                          >
                            {modelItem.display_name}
                          </p>
                          <p className={cn(
                            "text-xs mt-0.5 line-clamp-1",
                            isActive
                              ? "text-blue-600/80 dark:text-blue-400/80"
                              : "text-gray-500 dark:text-gray-400"
                          )}>
                            {modelItem.description}
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
          ))}
        </div>
      )}
    </div>
  );
}
