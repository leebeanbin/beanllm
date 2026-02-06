"""
Agent Request Schemas

Request models for Agent, Multi-Agent, and Orchestrator APIs.
Uses Python best practices: type hints, Field descriptions, validators.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request to run a single agent task"""

    task: str = Field(..., description="Task description for the agent")
    tools: Optional[List[str]] = Field(None, description="List of tool names to use")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Maximum iterations")
    model: Optional[str] = Field(None, description="LLM model to use")


class MultiAgentRequest(BaseModel):
    """Request for multi-agent coordination"""

    task: str = Field(..., description="Task for agents to work on")
    num_agents: int = Field(default=3, ge=2, le=10, description="Number of agents")
    strategy: str = Field(
        default="sequential",
        description="Execution strategy: sequential, parallel, hierarchical, debate",
    )
    model: Optional[str] = Field(None, description="Default LLM model for all agents")
    agent_configs: Optional[List[Dict[str, Any]]] = Field(
        None, description="Custom configuration per agent"
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)


class WorkflowRequest(BaseModel):
    """Request for orchestrator workflow execution"""

    workflow_type: str = Field(
        ..., description="Workflow type: research_write, parallel_consensus, debate"
    )
    task: str = Field(..., description="Task to execute")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Additional input data")
    model: Optional[str] = Field(None, description="LLM model for agents")
    num_agents: int = Field(default=2, ge=2, le=10, description="Number of agents")
