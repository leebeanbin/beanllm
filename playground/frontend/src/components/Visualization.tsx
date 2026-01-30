"use client";

import { useMemo } from "react";
import { Network, BarChart3, TrendingUp, GitBranch } from "lucide-react";
import { cn } from "@/lib/utils";

/** 카드 없음. border + spacing + 타이포 위계만. */

interface GraphNode {
  id: string;
  label: string;
  type?: string;
  metadata?: Record<string, any>;
}

interface GraphEdge {
  source: string;
  target: string;
  label?: string;
  type?: string;
}

interface GraphVisualizationProps {
  nodes?: GraphNode[];
  edges?: GraphEdge[];
  title?: string;
  interactive?: boolean;
}

export function GraphVisualization({
  nodes = [],
  edges = [],
  title = "Graph",
  interactive = false
}: GraphVisualizationProps) {
  // 노드 타입별 색상 매핑
  const getNodeColor = (type?: string) => {
    const colors: Record<string, string> = {
      PERSON: "bg-primary/20 border-primary/40 text-primary",
      ORGANIZATION: "bg-accent/20 border-accent/40 text-accent-foreground",
      LOCATION: "bg-chart-1/20 border-chart-1/40",
      EVENT: "bg-chart-2/20 border-chart-2/40",
      CONCEPT: "bg-chart-3/20 border-chart-3/40",
      default: "bg-muted/50 border-border/50",
    };
    return colors[type || "default"] || colors.default;
  };

  // 간단한 그래프 레이아웃 계산 (force-directed simulation 대신 간단한 그리드)
  const layoutNodes = useMemo(() => {
    if (nodes.length === 0) return [];

    const cols = Math.ceil(Math.sqrt(nodes.length));
    return nodes.map((node, idx) => {
      const row = Math.floor(idx / cols);
      const col = idx % cols;
      return {
        ...node,
        x: col * 200 + 100,
        y: row * 150 + 100,
      };
    });
  }, [nodes]);

  if (nodes.length === 0) return null;

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between py-3 px-3 border-b border-border/40">
        <div className="flex items-center gap-2">
          <Network className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
        <span className="text-xs text-muted-foreground">{nodes.length} nodes · {edges.length} edges</span>
      </div>
      <div className="p-3">
        {interactive && nodes.length > 0 ? (
          <div className="relative w-full h-[500px] border border-border/50 rounded-lg bg-muted/20 overflow-hidden">
            <svg className="w-full h-full" viewBox="0 0 1000 600">
              {/* Edges */}
              <g className="edges">
                {edges.map((edge, idx) => {
                  const sourceNode = layoutNodes.find(n => n.id === edge.source);
                  const targetNode = layoutNodes.find(n => n.id === edge.target);
                  if (!sourceNode || !targetNode) return null;
                  
                  return (
                    <g key={idx}>
                      <line
                        x1={sourceNode.x}
                        y1={sourceNode.y}
                        x2={targetNode.x}
                        y2={targetNode.y}
                        stroke="oklch(0.65 0.08 240 / 0.3)"
                        strokeWidth="2"
                        markerEnd="url(#arrowhead)"
                      />
                      {edge.label && (
                        <text
                          x={(sourceNode.x + targetNode.x) / 2}
                          y={(sourceNode.y + targetNode.y) / 2 - 5}
                          textAnchor="middle"
                          className="text-xs fill-muted-foreground"
                          fontSize="10"
                        >
                          {edge.label}
                        </text>
                      )}
                    </g>
                  );
                })}
                <defs>
                  <marker
                    id="arrowhead"
                    markerWidth="10"
                    markerHeight="10"
                    refX="9"
                    refY="3"
                    orient="auto"
                  >
                    <polygon
                      points="0 0, 10 3, 0 6"
                      fill="oklch(0.65 0.08 240 / 0.5)"
                    />
                  </marker>
                </defs>
              </g>
              
              {/* Nodes */}
              <g className="nodes">
                {layoutNodes.map((node) => (
                  <g key={node.id}>
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r="30"
                      className={cn(
                        "fill-current stroke-2",
                        getNodeColor(node.type)
                      )}
                    />
                    <text
                      x={node.x}
                      y={node.y + 5}
                      textAnchor="middle"
                      className="text-xs fill-foreground font-medium"
                      fontSize="10"
                    >
                      {node.label.length > 8 ? node.label.substring(0, 8) + "..." : node.label}
                    </text>
                  </g>
                ))}
              </g>
            </svg>
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
            {nodes.slice(0, 50).map((node) => (
              <div
                key={node.id}
                className={cn(
                  "p-3 rounded-lg border border-border/40 bg-muted/20",
                  getNodeColor(node.type)
                )}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{node.label}</div>
                    {node.type && (
                      <div className="text-xs text-muted-foreground mt-0.5">
                        Type: {node.type}
                      </div>
                    )}
                  </div>
                  {node.metadata && Object.keys(node.metadata).length > 0 && (
                    <div className="text-xs text-muted-foreground ml-2">
                      {Object.keys(node.metadata).length} props
                    </div>
                  )}
                </div>
              </div>
            ))}
            {nodes.length > 50 && (
              <div className="text-xs text-muted-foreground text-center py-2">
                ... and {nodes.length - 50} more nodes
              </div>
            )}
          </div>
        )}
        
        {edges.length > 0 && !interactive && (
          <div className="mt-4 pt-4 border-t border-border/50">
            <div className="text-xs font-medium text-muted-foreground mb-2">Relations</div>
            <div className="space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar">
              {edges.slice(0, 20).map((edge, idx) => {
                const source = nodes.find(n => n.id === edge.source);
                const target = nodes.find(n => n.id === edge.target);
                return (
                  <div
                    key={idx}
                    className="text-xs p-2 rounded bg-muted/30 border border-border/30"
                  >
                    <span className="font-medium">{source?.label || edge.source}</span>
                    <span className="mx-2 text-muted-foreground">
                      {edge.label ? `--[{edge.label}]-->` : "-->"}
                    </span>
                    <span className="font-medium">{target?.label || edge.target}</span>
                  </div>
                );
              })}
              {edges.length > 20 && (
                <div className="text-xs text-muted-foreground text-center py-1">
                  ... and {edges.length - 20} more relations
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface MetricsVisualizationProps {
  metrics: Record<string, number>;
  title?: string;
}

export function MetricsVisualization({ metrics, title = "Metrics" }: MetricsVisualizationProps) {
  const entries = Object.entries(metrics || {});

  if (entries.length === 0) return null;

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 py-3 px-3 border-b border-border/40">
        <BarChart3 className="h-4 w-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">{title}</span>
      </div>
      <div className="p-3 space-y-4">
          {entries.map(([key, value]) => (
            <div key={key} className="space-y-1.5">
              <div className="flex justify-between items-center text-sm">
                <span className="font-medium text-foreground">{key}</span>
                <span className="text-muted-foreground font-mono">
                  {typeof value === "number" ? value.toFixed(3) : String(value)}
                </span>
              </div>
              {typeof value === "number" && value >= 0 && value <= 1 && (
                <div className="w-full bg-muted/50 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-primary/60 h-full rounded-full transition-all duration-300"
                    style={{ width: `${value * 100}%` }}
                  />
                </div>
              )}
              {typeof value === "number" && (value < 0 || value > 1) && (
                <div className="text-xs text-muted-foreground">
                  Raw value: {value.toLocaleString()}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface StepsVisualizationProps {
  steps: Array<{ step: number; name?: string; status?: string; duration?: number }>;
  title?: string;
}

export function StepsVisualization({ steps, title = "Execution Steps" }: StepsVisualizationProps) {
  if (steps.length === 0) return null;

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between py-3 px-3 border-b border-border/40">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
        <span className="text-xs text-muted-foreground">{steps.length} steps</span>
      </div>
      <div className="p-3 space-y-2">
          {steps.map((step, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 p-3 rounded-lg border border-border/40 bg-muted/20"
            >
              <div className="flex-shrink-0 w-7 h-7 rounded-full bg-muted flex items-center justify-center text-xs font-medium text-muted-foreground">
                {step.step || idx + 1}
              </div>
              <div className="flex-1 min-w-0">
                {step.name && (
                  <div className="font-medium text-sm text-foreground truncate">{step.name}</div>
                )}
                {step.status && (
                  <div className="text-xs text-muted-foreground mt-0.5">{step.status}</div>
                )}
              </div>
              {step.duration !== undefined && (
                <div className="flex-shrink-0 text-xs text-muted-foreground font-mono">
                  {step.duration.toFixed(2)}s
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface PipelineVisualizationProps {
  steps: Array<{ name: string; type: string; status?: string }>;
  title?: string;
}

export function PipelineVisualization({ steps, title = "Pipeline" }: PipelineVisualizationProps) {
  if (steps.length === 0) return null;

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between py-3 px-3 border-b border-border/40">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
        <span className="text-xs text-muted-foreground">{steps.length} steps</span>
      </div>
      <div className="p-3">
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {steps.map((step, idx) => (
            <div key={idx} className="flex items-center flex-shrink-0">
              <div className="flex flex-col items-center">
                <div className={cn(
                  "px-3 py-2 rounded-lg border text-sm font-medium",
                  step.status === "completed" 
                    ? "bg-muted/40 border-border/40 text-foreground"
                    : step.status === "running"
                    ? "bg-muted/40 border-border/60 text-foreground"
                    : "bg-muted/20 border-border/40 text-muted-foreground"
                )}>
                  {step.name}
                </div>
                {step.type && (
                  <div className="text-xs text-muted-foreground mt-1">{step.type}</div>
                )}
              </div>
              {idx < steps.length - 1 && (
                <div className="mx-2 text-muted-foreground">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
