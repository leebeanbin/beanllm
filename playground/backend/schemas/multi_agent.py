"""
Multi-Agent Request Schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MultiAgentRequest(BaseModel):
    """Request to run multi-agent system"""

    task: str
    num_agents: int = 3
    strategy: str = "sequential"  # sequential, parallel, hierarchical, debate
    model: Optional[str] = None
    agent_configs: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class WorkflowRequest(BaseModel):
    """Request to run orchestrator workflow"""

    workflow_type: str  # research_write, parallel_consensus, debate
    task: str
    input_data: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    num_agents: int = 2


class ChainRequest(BaseModel):
    """Request to run a chain"""

    input: str
    chain_id: Optional[str] = None
    chain_type: str = "basic"  # basic, prompt
    template: Optional[str] = None
    model: Optional[str] = None
