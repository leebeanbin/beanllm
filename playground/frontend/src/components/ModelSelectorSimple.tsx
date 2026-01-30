"use client";

import { useState, useEffect, useRef } from "react";
import { ChevronDown, Download, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface Model {
  name: string;
  display_name: string;
  description?: string;
  installed?: boolean;
  provider?: string;
}

interface ModelsGrouped {
  [provider: string]: Model[];
}

interface ModelSelectorSimpleProps {
  value?: string;
  onChange?: (model: string) => void;
  className?: string;
}

export function ModelSelectorSimple({ value, onChange, className }: ModelSelectorSimpleProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [models, setModels] = useState<ModelsGrouped>({});
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<Set<string>>(new Set());
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  // 각 모델의 AbortController를 추적
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map());

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    
    // Default fallback models (빈 배열로 시작 - API에서 가져온 모델만 표시)
    const defaultModels: ModelsGrouped = {
      OpenAI: [
        { name: "gpt-4o-mini", display_name: "GPT-4o Mini", description: "Cost-effective" },
        { name: "gpt-4o", display_name: "GPT-4o", description: "High performance" },
      ],
    };

    // Set default models immediately for better UX (Ollama는 API에서 가져온 것만)
    setModels(defaultModels);
    setLoading(false);

    // Try to fetch from API in the background
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    fetch(`${apiUrl}/api/models`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        // Only update if we got valid data
        if (data && typeof data === "object" && Object.keys(data).length > 0) {
          setModels(data);
        }
      })
      .catch((err) => {
        // Silently fail - we already have default models
        if (err.name !== "AbortError") {
          console.debug("Failed to fetch models from API, using defaults:", err.message);
        }
      })
      .finally(() => {
        clearTimeout(timeoutId);
      });
  }, []);

  const handleCancelDownload = (modelName: string) => {
    const abortController = abortControllersRef.current.get(modelName);
    if (abortController) {
      abortController.abort();
      abortControllersRef.current.delete(modelName);
      
      // 상태 정리
      setDownloading((prev) => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
      setDownloadProgress((prev) => {
        const next = { ...prev };
        delete next[modelName];
        return next;
      });
      
      // localStorage에서도 제거
      try {
        const savedDownloading = localStorage.getItem("model-downloading");
        const savedProgress = localStorage.getItem("model-download-progress");
        
        if (savedDownloading) {
          const downloadingSet = new Set(JSON.parse(savedDownloading) as string[]);
          downloadingSet.delete(modelName);
          localStorage.setItem("model-downloading", JSON.stringify(Array.from(downloadingSet)));
        }
        
        if (savedProgress) {
          const progress = JSON.parse(savedProgress) as Record<string, number>;
          delete progress[modelName];
          localStorage.setItem("model-download-progress", JSON.stringify(progress));
        }
      } catch (error) {
        console.error("Failed to update download state:", error);
      }
      
      toast.info(`${modelName} model download cancelled`);
    }
  };

  const handleDownloadModel = async (modelName: string) => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    
    // 이미 다운로드 중이면 중복 시작 방지
    if (downloading.has(modelName)) {
      toast.info(`${modelName} is already downloading`);
      return;
    }
    
    setDownloading((prev) => new Set(prev).add(modelName));
    setDownloadProgress((prev) => ({ ...prev, [modelName]: 0 }));
    
    // AbortController로 취소 가능하게 만들되, 페이지 이동 시에도 계속 진행되도록
    const abortController = new AbortController();
    abortControllersRef.current.set(modelName, abortController);
    
    try {
      const response = await fetch(`${apiUrl}/api/models/${encodeURIComponent(modelName)}/pull`, {
        method: "POST",
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("Cannot read stream");
      }

      let buffer = "";
      let hasReceivedData = false;
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (!hasReceivedData) {
            console.warn("No data received from SSE stream");
          }
          break;
        }

        hasReceivedData = true;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        
        // 마지막 줄은 완전하지 않을 수 있으므로 버퍼에 보관
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim() === "") continue;
          
          if (line.startsWith("data: ")) {
            try {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;
              
              const data = JSON.parse(jsonStr);
              console.debug("SSE data received:", data);
              
              if (data.status === "completed") {
                // 다운로드 상태 제거 (먼저 상태 업데이트)
                setDownloading((prev) => {
                  const next = new Set(prev);
                  next.delete(modelName);
                  return next;
                });
                setDownloadProgress((prev) => {
                  const next = { ...prev };
                  delete next[modelName];
                  return next;
                });
                
                // localStorage에서 다운로드 상태 제거
                try {
                  const savedDownloading = localStorage.getItem("model-downloading");
                  const savedProgress = localStorage.getItem("model-download-progress");
                  
                  if (savedDownloading) {
                    const downloadingSet = new Set(JSON.parse(savedDownloading) as string[]);
                    downloadingSet.delete(modelName);
                    localStorage.setItem("model-downloading", JSON.stringify(Array.from(downloadingSet)));
                  }
                  
                  if (savedProgress) {
                    const progress = JSON.parse(savedProgress) as Record<string, number>;
                    delete progress[modelName];
                    localStorage.setItem("model-download-progress", JSON.stringify(progress));
                  }
                } catch (error) {
                  console.error("Failed to update download state:", error);
                }
                
                // 브라우저 알림 (페이지가 백그라운드에 있어도)
                if ("Notification" in window && Notification.permission === "granted") {
                  new Notification("Model download complete", {
                    body: `${modelName} has been downloaded`,
                    icon: "/favicon.ico",
                  });
                } else if ("Notification" in window && Notification.permission !== "denied") {
                  Notification.requestPermission().then((permission) => {
                    if (permission === "granted") {
                      new Notification("Model download complete", {
                        body: `${modelName} has been downloaded`,
                        icon: "/favicon.ico",
                      });
                    }
                  });
                }
                
                toast.success(`${modelName} download complete`);
                // 모델 목록 새로고침 (다운로드 완료 후 설치 상태 업데이트)
                // 약간의 지연을 두어 Ollama가 모델 목록을 업데이트할 시간을 줌
                setTimeout(async () => {
                  try {
                    const res = await fetch(`${apiUrl}/api/models`);
                    if (res.ok) {
                      const updatedModels = await res.json();
                      setModels(updatedModels);
                    }
                  } catch (error) {
                    console.error("Failed to refresh models after download:", error);
                  }
                }, 1000); // 1초 후 새로고침
                return; // 성공적으로 완료
              } else if (data.status === "error") {
                throw new Error(data.error || "Download failed");
              } else if (data.progress !== undefined) {
                console.debug(`Progress update: ${data.progress}%`);
                setDownloadProgress((prev) => ({ ...prev, [modelName]: Math.min(100, Math.max(0, data.progress)) }));
              } else if (data.completed !== undefined && data.total !== undefined && data.total > 0) {
                // progress가 없으면 completed와 total로 계산
                const calculatedProgress = (data.completed / data.total) * 100;
                console.debug(`Progress calculated: ${calculatedProgress}% (${data.completed}/${data.total})`);
                setDownloadProgress((prev) => ({ ...prev, [modelName]: Math.min(100, Math.max(0, calculatedProgress)) }));
              } else if (data.status) {
                // status만 있는 경우 (예: "pulling", "downloading")
                console.debug(`Status update: ${data.status}`);
                // status만 있어도 최소한 진행 중임을 표시
                if (data.status !== "completed" && data.status !== "error") {
                  // 진행 중이지만 정확한 진행률을 모를 때는 1%로 표시 (0%가 아닌)
                  setDownloadProgress((prev) => {
                    const current = prev[modelName] || 0;
                    return { ...prev, [modelName]: Math.max(1, current) };
                  });
                }
              }
            } catch (e) {
              // JSON 파싱 실패는 무시 (디버깅용 로그)
              console.warn("Failed to parse SSE data:", line, e);
            }
          } else {
            // "data: "로 시작하지 않는 줄도 로그
            console.debug("Non-SSE line received:", line);
          }
        }
      }
      
      // 스트림이 끝났는데 완료 메시지가 없으면 확인
      if (buffer.trim()) {
        const line = buffer.trim();
        if (line.startsWith("data: ")) {
          try {
            const jsonStr = line.slice(6).trim();
            if (jsonStr) {
              const data = JSON.parse(jsonStr);
              if (data.status === "completed") {
                setDownloadProgress((prev) => ({ ...prev, [modelName]: 100 }));
                toast.success(`${modelName} download complete`);
                const res = await fetch(`${apiUrl}/api/models`);
                if (res.ok) {
                  const updatedModels = await res.json();
                  setModels(updatedModels);
                }
                return;
              }
            }
          } catch (e) {
            console.debug("Failed to parse final SSE data:", e);
          }
        }
      }
    } catch (error) {
      console.error("Download error:", error);
      toast.error(`Model download failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setDownloading((prev) => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
      setTimeout(() => {
        setDownloadProgress((prev) => {
          const next = { ...prev };
          delete next[modelName];
          return next;
        });
      }, 1000);
    }
  };

  const currentModel = Object.values(models)
    .flat()
    .find((m) => m.name === value) || Object.values(models).flat()[0];

  if (loading) {
    return (
      <div className={cn("flex items-center gap-2 px-3 py-2 rounded-lg border", className)}>
        <span className="text-sm text-muted-foreground">Loading models...</span>
      </div>
    );
  }

  const dropdownId = `model-selector-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div className={cn("relative", className)}>
      <Button
        variant="outline"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full justify-between gap-2 h-9 rounded-lg border-border/40 text-[12px] font-medium tracking-tight"
        aria-expanded={isOpen}
        aria-controls={dropdownId}
        aria-haspopup="listbox"
        aria-label={`Select model: ${currentModel?.display_name || value || "Not selected"}`}
      >
        <span className="text-sm truncate flex-1 min-w-0 text-left">
          {currentModel?.display_name || value || "Select model"}
        </span>
        <ChevronDown 
          className={cn("h-4 w-4 shrink-0 transition-transform", isOpen && "rotate-180")} 
          aria-hidden="true"
        />
      </Button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
            aria-hidden="true"
          />
          <div 
            id={dropdownId}
            role="listbox"
            className="absolute top-full mt-2 left-0 right-0 z-50 max-h-64 overflow-y-auto rounded-lg border border-border/40 bg-background shadow-sm"
            aria-label="Model list"
          >
            {Object.entries(models).map(([provider, providerModels]) => (
              <div key={provider} className="border-b last:border-b-0" role="group" aria-label={provider}>
                <div className="px-3 py-2 bg-muted text-xs font-semibold uppercase" role="presentation">
                  {provider}
                </div>
                {providerModels.map((model) => {
                  const isOllama = provider.toLowerCase() === "ollama" || model.provider === "ollama";
                  // installed가 명시적으로 true인 경우만 설치된 것으로 간주
                  const isInstalled = model.installed === true;
                  const isDownloading = downloading.has(model.name);
                  const progress = downloadProgress[model.name] || 0;
                  
                  return (
                    <div
                      key={model.name}
                      className={cn(
                        "w-full px-3 py-2 text-left text-sm hover:bg-accent transition-colors",
                        value === model.name && "bg-accent"
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <button
                          role="option"
                          aria-selected={value === model.name}
                          onClick={() => {
                            if (!isDownloading) {
                              onChange?.(model.name);
                              setIsOpen(false);
                            }
                          }}
                          className={cn(
                            "flex-1 text-left focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 rounded",
                            isDownloading && "opacity-50 cursor-not-allowed"
                          )}
                          disabled={isDownloading}
                        >
                          <div className="font-medium">{model.display_name}</div>
                          {model.description && (
                            <div className="text-xs text-muted-foreground">{model.description}</div>
                          )}
                          {isDownloading && (
                            <div className="mt-1">
                              <div className="h-1 bg-muted rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-primary transition-all duration-300"
                                  style={{ width: `${progress}%` }}
                                />
                              </div>
                              <div className="text-xs text-muted-foreground mt-1">
                                Downloading... {Math.round(progress)}%
                              </div>
                            </div>
                          )}
                        </button>
                        {isOllama && !isInstalled && !isDownloading && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={async (e) => {
                              e.stopPropagation();
                              await handleDownloadModel(model.name);
                            }}
                            aria-label={`Download ${model.display_name} model`}
                          >
                            <Download className="h-4 w-4" aria-hidden="true" />
                          </Button>
                        )}
                        {isDownloading && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCancelDownload(model.name);
                            }}
                            aria-label={`Cancel ${model.display_name} download`}
                          >
                            <X className="h-4 w-4 text-destructive" aria-hidden="true" />
                          </Button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
