"""
Agent Response Schemas

Response models for Agent, Multi-Agent, and Orchestrator APIs.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class AgentStepResponse(BaseModel):
    """Single step in agent execution"""
    step: int = Field(..., description="Step number")
    thought: str = Field(..., description="Agent's reasoning")
    action: Optional[str] = Field(None, description="Action taken")


class AgentRunResponse(BaseModel):
    """Response from single agent execution"""
    task: str
    result: str
    steps: List[AgentStepResponse] = Field(default_factory=list)
    iterations: int = Field(default=0)


class AgentOutputResponse(BaseModel):
    """Individual agent output in multi-agent execution"""
    agent_id: str
    output: str
    role: Optional[str] = Field(None, description="Agent role (manager, worker, etc.)")


class MultiAgentRunResponse(BaseModel):
    """Response from multi-agent execution"""
    task: str
    strategy: str
    final_result: str
    intermediate_results: List[Dict[str, Any]] = Field(default_factory=list)
    all_steps: List[Dict[str, Any]] = Field(default_factory=list)
    agent_outputs: List[AgentOutputResponse] = Field(default_factory=list)


class WorkflowRunResponse(BaseModel):
    """Response from orchestrator workflow execution"""
    workflow_id: str = Field(default="wf_001")
    result: str
    execution_time: float = Field(default=0.0)
    steps_executed: int = Field(default=0)
