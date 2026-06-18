"use client";

import React, { useMemo, useEffect, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Node,
  Edge,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import { TelemetryEvent } from "@/hooks/useTelemetry";
import { Brain, Tool, MessageSquare, CheckCircle, Activity, User, Play, ListTodo } from "lucide-react";
import { Button } from "@/components/ui/button";

interface WorkflowMapProps {
  events: TelemetryEvent[];
  isConnected: boolean;
  onResume?: (modifiedPlan?: any[]) => void;
}

export function WorkflowMap({ events, isConnected, onResume }: WorkflowMapProps) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [currentPlan, setCurrentPlan] = useState<any[] | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  // Process events to build nodes and edges
  useEffect(() => {
    const agentNodes: Record<string, Node> = {};
    const agentEdges: Edge[] = [];

    events.forEach((event) => {
      const { agent_id, event_type, content, metadata } = event;

      // Handle Plan Ready
      if (event_type === "plan_ready") {
        setCurrentPlan(content);
        if (metadata?.wait_for_approval) {
          setIsPaused(true);
        }
      }

      if (!agentNodes[agent_id]) {
        agentNodes[agent_id] = {
          id: agent_id,
          data: { label: agent_id, status: "idle" },
          position: { x: Object.keys(agentNodes).length * 250, y: 100 },
          style: { 
            background: "#fff", 
            color: "#333", 
            border: "1px solid #ccc",
            borderRadius: "8px",
            padding: "10px",
            width: 180
          },
        };
      }

      // Update node status
      if (event_type === "agent_started") {
        agentNodes[agent_id].data.status = "running";
      } else if (event_type === "agent_finished") {
        agentNodes[agent_id].data.status = "finished";
      } else if (event_type === "thinking_chunk") {
        agentNodes[agent_id].data.status = "thinking";
      } else if (event_type === "tool_started") {
        agentNodes[agent_id].data.status = "using_tool";
      } else if (event_type === "message_sent") {
        const receiver = metadata.receiver;
        if (receiver && receiver !== "broadcast") {
          const edgeId = `${agent_id}-${receiver}`;
          if (!agentEdges.find(e => e.id === edgeId)) {
            agentEdges.push({
              id: edgeId,
              source: agent_id,
              target: receiver,
              label: metadata.message_type || "message",
              animated: true,
              markerEnd: { type: MarkerType.ArrowClosed },
            });
          }
        }
      }
    });

    // Custom node rendering (simulated here with style updates)
    Object.values(agentNodes).forEach(node => {
      const status = node.data.status;
      if (status === "running" || status === "thinking" || status === "using_tool") {
        node.style = { ...node.style, border: "2px solid #3b82f6", boxShadow: "0 0 10px rgba(59, 130, 246, 0.5)" };
      } else if (status === "finished") {
        node.style = { ...node.style, border: "1px solid #10b981", opacity: 0.8 };
      }
    });

    setNodes(Object.values(agentNodes));
    setEdges(agentEdges);
  }, [events]);

  const handleResumeClick = () => {
    setIsPaused(false);
    if (onResume) {
      onResume(currentPlan || undefined);
    }
  };

  const updatePlanStep = (idx: number, field: string, value: string) => {
    if (!currentPlan) return;
    const newPlan = [...currentPlan];
    newPlan[idx] = { ...newPlan[idx], [field]: value };
    setCurrentPlan(newPlan);
  };

  return (
    <div className="w-full h-full min-h-[400px] border border-border/40 rounded-lg bg-muted/5 relative overflow-hidden flex flex-col">
      {/* Header Overlay */}
      <div className="absolute top-2 left-2 z-10 flex gap-2">
        {!isConnected && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-destructive/10 text-destructive text-[10px] font-medium border border-destructive/20">
            <Activity className="h-3 w-3 animate-pulse" />
            Disconnected
          </div>
        )}
        {isConnected && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-500/10 text-green-600 dark:text-green-500 text-[10px] font-medium border border-green-500/20">
            <Activity className="h-3 w-3" />
            Live
          </div>
        )}
        {isPaused && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-500 text-[10px] font-medium border border-amber-500/20 animate-pulse">
            <Activity className="h-3 w-3" />
            Waiting for Approval
          </div>
        )}
      </div>

      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          className="workflow-flow"
        >
          <Background color="#ccc" gap={20} />
          <Controls />
        </ReactFlow>
      </div>

      {/* Plan / Kanban View - Editable */}
      {currentPlan && (
        <div className="border-t border-border/40 bg-background/50 backdrop-blur-sm p-3 max-h-56 overflow-y-auto">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <ListTodo className="h-4 w-4 text-primary" />
              <h5 className="text-xs font-bold text-foreground">Execution Plan (Editable)</h5>
            </div>
            {isPaused && (
              <Button size="sm" className="h-7 px-3 text-[11px] gap-1.5" onClick={handleResumeClick}>
                <Play className="h-3 w-3 fill-current" />
                Approve & Resume
              </Button>
            )}
          </div>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {currentPlan.map((step, idx) => (
              <div key={idx} className="flex-shrink-0 w-56 p-2 rounded-md border border-border/40 bg-background/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold text-muted-foreground uppercase">Step {idx + 1}</span>
                  <select 
                    className="text-[9px] bg-primary/5 text-primary border border-primary/20 rounded px-1 outline-none"
                    value={step.assigned_agent_id}
                    onChange={(e) => updatePlanStep(idx, 'assigned_agent_id', e.target.value)}
                    disabled={!isPaused}
                  >
                    <option value={step.assigned_agent_id}>{step.assigned_agent_id}</option>
                    {/* Unique agents from nodes could be used here to allow reassignment */}
                  </select>
                </div>
                <input 
                  className="w-full text-[11px] font-semibold text-foreground bg-transparent border-none p-0 focus:ring-0"
                  value={step.title}
                  onChange={(e) => updatePlanStep(idx, 'title', e.target.value)}
                  readOnly={!isPaused}
                  placeholder="Task Title"
                />
                <textarea 
                  className="w-full text-[10px] text-muted-foreground bg-transparent border-none p-0 focus:ring-0 resize-none min-h-[40px]"
                  value={step.description}
                  onChange={(e) => updatePlanStep(idx, 'description', e.target.value)}
                  readOnly={!isPaused}
                  placeholder="Task Description"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Legend / Status Overlay */}
      <div className="absolute bottom-2 left-2 z-10 p-2 rounded-lg bg-background/80 backdrop-blur-sm border border-border/40 space-y-2 pointer-events-none">
        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-1">Agent Status</p>
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            <span className="text-[11px] text-foreground">Thinking / Running</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-[11px] text-foreground">Finished</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-gray-300" />
            <span className="text-[11px] text-foreground">Idle</span>
          </div>
        </div>
      </div>
    </div>
  );
}
