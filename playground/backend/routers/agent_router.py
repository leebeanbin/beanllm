"""
Agent Router

Agent, Multi-Agent, and Orchestrator endpoints.
Uses Python best practices: duck typing, type hints, comprehensions.
"""

import logging
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException

from common import get_orchestrator
from schemas.agent import AgentRequest, MultiAgentRequest, WorkflowRequest
from schemas.responses.agent import (
    AgentRunResponse,
    AgentStepResponse,
    MultiAgentRunResponse,
    AgentOutputResponse,
    WorkflowRunResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Agent"])


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_step(step: Any, index: int) -> Dict[str, Any]:
    """Extract step data using duck typing"""
    if hasattr(step, "step_number"):
        return {
            "step": getattr(step, "step_number", index),
            "thought": getattr(step, "thought", ""),
            "action": getattr(step, "action", None),
        }
    elif isinstance(step, dict):
        return {
            "step": step.get("step_number", step.get("step", index)),
            "thought": step.get("thought", ""),
            "action": step.get("action"),
        }
    return {"step": index, "thought": str(step), "action": None}


def _extract_result(result: Any) -> Dict[str, Any]:
    """Extract result using duck typing for various result types"""
    if isinstance(result, str):
        return {"final_result": result, "intermediate_results": [], "all_steps": []}

    if isinstance(result, dict):
        return {
            "final_result": result.get("final_result", ""),
            "intermediate_results": result.get("intermediate_results", []),
            "all_steps": result.get("all_steps", []),
        }

    # Object with attributes
    return {
        "final_result": getattr(result, "result", str(result)),
        "intermediate_results": getattr(result, "intermediate_results", []),
        "all_steps": getattr(result, "all_steps", []),
    }


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute or dict key using duck typing"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ============================================================================
# Single Agent Endpoints
# ============================================================================

@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(request: AgentRequest) -> AgentRunResponse:
    """
    Run a single agent task.

    The agent will iteratively work on the task using available tools.
    """
    try:
        from beanllm.facade.core.agent_facade import Agent

        model = request.model or "gpt-4o-mini"
        agent = Agent(
            model=model,
            max_iterations=request.max_iterations,
            verbose=True,
        )

        result = await agent.run(task=request.task)

        # Extract steps using duck typing
        steps = [
            AgentStepResponse(**_extract_step(step, idx))
            for idx, step in enumerate(getattr(result, "steps", []))
        ]

        return AgentRunResponse(
            task=request.task,
            result=getattr(result, "answer", str(result)),
            steps=steps,
            iterations=getattr(result, "total_steps", len(steps)),
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(500, f"Agent error: {str(e)}")


# ============================================================================
# Multi-Agent Endpoints
# ============================================================================

@router.post("/multi_agent/run", response_model=MultiAgentRunResponse)
async def multi_agent_run(request: MultiAgentRequest) -> MultiAgentRunResponse:
    """
    Run multi-agent coordination task.

    Strategies:
    - sequential: Agents execute one after another
    - parallel: Agents execute simultaneously
    - hierarchical: Manager-worker pattern
    - debate: Agents debate to reach consensus
    """
    try:
        from beanllm.facade.core.agent_facade import Agent
        from beanllm.facade.advanced.multi_agent_facade import MultiAgentCoordinator

        model = request.model or "gpt-4o-mini"
        agents: Dict[str, Agent] = {}

        # Create agents using configuration or defaults
        if request.agent_configs:
            agents = {
                config.get("agent_id", f"agent_{i}"): Agent(
                    model=config.get("model", model),
                    tools=config.get("tools", []),
                    max_iterations=config.get("max_iterations", 10),
                    verbose=config.get("verbose", False),
                )
                for i, config in enumerate(request.agent_configs)
            }
        else:
            agents = {
                f"agent_{i}": Agent(model=model, max_iterations=10, verbose=False)
                for i in range(request.num_agents)
            }

        coordinator = MultiAgentCoordinator(agents=agents)
        agent_ids = list(agents.keys())

        # Strategy dispatch using dict mapping (more Pythonic than if-elif chain)
        strategy_handlers = {
            "sequential": _execute_sequential,
            "parallel": _execute_parallel,
            "hierarchical": _execute_hierarchical,
            "debate": _execute_debate,
        }

        handler = strategy_handlers.get(request.strategy, _execute_sequential)
        return await handler(coordinator, request.task, agent_ids)

    except Exception as e:
        logger.error(f"Multi-agent error: {e}", exc_info=True)
        raise HTTPException(500, f"Multi-agent error: {str(e)}")


async def _execute_sequential(
    coordinator: Any, task: str, agent_ids: List[str]
) -> MultiAgentRunResponse:
    """Execute agents sequentially"""
    result = await coordinator.execute_sequential(task=task, agent_order=agent_ids)
    extracted = _extract_result(result)

    return MultiAgentRunResponse(
        task=task,
        strategy="sequential",
        final_result=extracted["final_result"],
        intermediate_results=extracted["intermediate_results"],
        all_steps=extracted["all_steps"],
        agent_outputs=[
            AgentOutputResponse(
                agent_id=agent_id,
                output=_get_intermediate_output(extracted["intermediate_results"], i),
            )
            for i, agent_id in enumerate(agent_ids)
        ],
    )


async def _execute_parallel(
    coordinator: Any, task: str, agent_ids: List[str]
) -> MultiAgentRunResponse:
    """Execute agents in parallel"""
    result = await coordinator.execute_parallel(
        task=task, agent_ids=agent_ids, aggregation="vote"
    )

    return MultiAgentRunResponse(
        task=task,
        strategy="parallel",
        final_result=_safe_get(result, "final_result", ""),
        agent_outputs=[
            AgentOutputResponse(agent_id=aid, output=f"Completed task: {task}")
            for aid in agent_ids
        ],
    )


async def _execute_hierarchical(
    coordinator: Any, task: str, agent_ids: List[str]
) -> MultiAgentRunResponse:
    """Execute with hierarchical (manager-worker) pattern"""
    if len(agent_ids) < 2:
        raise HTTPException(
            400, "Hierarchical strategy requires at least 2 agents"
        )

    manager_id, *worker_ids = agent_ids  # Unpacking for cleaner code
    result = await coordinator.execute_hierarchical(
        task=task, manager_id=manager_id, worker_ids=worker_ids
    )

    return MultiAgentRunResponse(
        task=task,
        strategy="hierarchical",
        final_result=_safe_get(result, "final_result", ""),
        agent_outputs=[
            AgentOutputResponse(agent_id=manager_id, output="Coordinated all tasks", role="manager"),
            *[
                AgentOutputResponse(agent_id=wid, output="Completed subtask", role="worker")
                for wid in worker_ids
            ],
        ],
    )


async def _execute_debate(
    coordinator: Any, task: str, agent_ids: List[str]
) -> MultiAgentRunResponse:
    """Execute debate among agents"""
    result = await coordinator.execute_debate(task=task, agent_ids=agent_ids, rounds=3)

    return MultiAgentRunResponse(
        task=task,
        strategy="debate",
        final_result=_safe_get(result, "final_result", ""),
        agent_outputs=[
            AgentOutputResponse(agent_id=aid, output=f"Argument presented for: {task}")
            for aid in agent_ids
        ],
    )


def _get_intermediate_output(intermediate_results: List[Any], index: int) -> str:
    """Extract output from intermediate results"""
    if index >= len(intermediate_results):
        return f"Step {index + 1} completed"

    item = intermediate_results[index]
    if isinstance(item, dict):
        return item.get("result", f"Step {index + 1} completed")
    return str(item)


# ============================================================================
# Orchestrator Endpoints
# ============================================================================

@router.post("/orchestrator/run", response_model=WorkflowRunResponse)
async def orchestrator_run(request: WorkflowRequest) -> WorkflowRunResponse:
    """
    Run orchestrator workflow.

    Workflow types:
    - research_write: Research then write pattern
    - parallel_consensus: Parallel execution with voting
    - debate: Multi-round debate pattern
    """
    try:
        from beanllm.facade.core.agent_facade import Agent

        orchestrator = get_orchestrator()
        model = request.model or "gpt-4o-mini"

        # Workflow type dispatch
        workflow_handlers = {
            "research_write": lambda: _run_research_write(orchestrator, model, request.task),
            "parallel_consensus": lambda: _run_parallel_consensus(
                orchestrator, model, request.task, request.num_agents
            ),
            "debate": lambda: _run_debate(
                orchestrator, model, request.task, request.num_agents
            ),
        }

        handler = workflow_handlers.get(request.workflow_type)

        if handler:
            response = await handler()
        else:
            response = await orchestrator.run_full_workflow(
                workflow_type=request.workflow_type,
                input_data=request.input_data or {"task": request.task},
            )

        return WorkflowRunResponse(
            workflow_id=_safe_get(response, "workflow_id", "wf_001"),
            result=_safe_get(response, "result", str(response)),
            execution_time=_safe_get(response, "execution_time", 0.0),
            steps_executed=_safe_get(response, "steps", 0),
        )

    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        raise HTTPException(500, f"Orchestrator error: {str(e)}")


async def _run_research_write(orchestrator: Any, model: str, task: str) -> Any:
    """Run research-write workflow"""
    from beanllm.facade.core.agent_facade import Agent

    researcher = Agent(model=model, max_iterations=10)
    writer = Agent(model=model, max_iterations=10)
    return await orchestrator.quick_research_write(
        researcher_agent=researcher, writer_agent=writer, task=task
    )


async def _run_parallel_consensus(
    orchestrator: Any, model: str, task: str, num_agents: int
) -> Any:
    """Run parallel consensus workflow"""
    from beanllm.facade.core.agent_facade import Agent

    agents = [Agent(model=model, max_iterations=10) for _ in range(num_agents)]
    return await orchestrator.quick_parallel_consensus(
        agents=agents, task=task, aggregation="vote"
    )


async def _run_debate(
    orchestrator: Any, model: str, task: str, num_agents: int
) -> Any:
    """Run debate workflow"""
    from beanllm.facade.core.agent_facade import Agent

    debaters = [Agent(model=model, max_iterations=10) for _ in range(num_agents - 1)]
    judge = Agent(model=model, max_iterations=10)
    return await orchestrator.quick_debate(
        debater_agents=debaters, judge_agent=judge, task=task, rounds=3
    )
