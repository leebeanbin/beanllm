"""
Multi-Agent Tools - 기존 beanllm Multi-Agent 기능을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from beanllm.facade.advanced import MultiAgent

# 기존 beanllm 코드 import (wrapping 대상)
from beanllm.facade.core import Agent
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스 생성
mcp = FastMCP("Multi-Agent Tools")

# 전역 Multi-Agent 시스템 캐시
_multiagent_systems: Dict[str, MultiAgent] = {}
_agents_cache: Dict[str, Dict[str, Agent]] = {}


@mcp.tool()
async def create_multiagent_system(
    system_name: str,
    agent_configs: List[Dict[str, Any]],
    strategy: str = "sequential",
    max_rounds: int = 3,
) -> dict:
    """
    다중 에이전트 시스템 생성 (기존 코드 재사용)

    Args:
        system_name: 시스템 식별 이름
        agent_configs: 에이전트 설정 목록
            [
                {
                    "name": "researcher",
                    "role": "Research specialist",
                    "model": "qwen2.5:0.5b",
                    "temperature": 0.3
                },
                ...
            ]
        strategy: 통신 전략 (sequential, parallel, debate, hierarchical)
        max_rounds: 최대 라운드 수

    Returns:
        dict: 성공 여부, 에이전트 개수

    Example:
        User: "연구자, 작가, 비평가 에이전트를 만들어서 토론하게 해줘"
        → create_multiagent_system(
            system_name="debate_team",
            agent_configs=[
                {"name": "researcher", "role": "Research", ...},
                {"name": "writer", "role": "Writing", ...},
                {"name": "critic", "role": "Review", ...}
            ],
            strategy="debate"
        )
    """
    try:
        # 🎯 기존 beanllm 코드 재사용!
        # 1. Agent 객체 생성
        agents: Dict[str, Agent] = {}
        for config in agent_configs:
            agent_name = config["name"]
            model = config.get("model", MCPServerConfig.DEFAULT_CHAT_MODEL)
            system_prompt = config.get("system_prompt") or config.get("role", agent_name)
            temperature = config.get("temperature", 0.7)

            agent = Agent(
                model=model,
                system_prompt=f"You are {agent_name}. {system_prompt}",
                temperature=temperature,
            )
            agents[agent_name] = agent

        # 2. MultiAgent (MultiAgentCoordinator) 생성
        coordinator = MultiAgent(agents=agents)

        # 3. 캐시에 저장 (strategy와 max_rounds는 메타데이터로 저장)
        _multiagent_systems[system_name] = coordinator
        _agents_cache[system_name] = {
            "agents": agents,
            "strategy": strategy,
            "max_rounds": max_rounds,
            "configs": agent_configs,
        }

        return {
            "success": True,
            "system_name": system_name,
            "agent_count": len(agents),
            "agent_names": list(agents.keys()),
            "strategy": strategy,
            "max_rounds": max_rounds,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def run_multiagent_task(
    system_name: str,
    task: str,
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    다중 에이전트 시스템에 작업 실행 (기존 코드 재사용)

    Args:
        system_name: 사용할 시스템 이름
        task: 실행할 작업 설명
        context: 추가 컨텍스트 (선택)

    Returns:
        dict: 각 에이전트의 응답, 최종 결과, 대화 히스토리

    Example:
        User: "AI의 미래에 대해 토론해줘"
        → run_multiagent_task(
            system_name="debate_team",
            task="AI의 미래에 대해 각자의 관점에서 토론하세요"
        )
    """
    try:
        # 1. 캐시에서 시스템 가져오기
        if system_name not in _multiagent_systems:
            return {
                "success": False,
                "error": f"Multi-agent system '{system_name}' not found. Please create it first.",
            }

        coordinator = _multiagent_systems[system_name]
        meta = _agents_cache.get(system_name, {})
        strategy = meta.get("strategy", "sequential")
        max_rounds = meta.get("max_rounds", 3)
        agent_names = list(meta.get("agents", {}).keys())

        # 2. 🎯 기존 MultiAgent의 실행 메서드 사용!
        if strategy == "sequential":
            result = await coordinator.execute_sequential(
                task=task,
                agent_order=agent_names,
            )
        elif strategy == "parallel":
            result = await coordinator.execute_parallel(
                task=task,
                agents=agent_names,
                aggregation="concatenate",
            )
        elif strategy == "debate":
            result = await coordinator.execute_debate(
                topic=task,
                participants=agent_names,
                rounds=max_rounds,
            )
        elif strategy == "hierarchical":
            result = await coordinator.execute_hierarchical(
                task=task,
                leader=agent_names[0] if agent_names else "leader",
                workers=agent_names[1:] if len(agent_names) > 1 else [],
            )
        else:
            # 기본: sequential
            result = await coordinator.execute_sequential(
                task=task,
                agent_order=agent_names,
            )

        # 3. 결과 포매팅
        agent_responses = []
        if isinstance(result, dict):
            for agent_name, response in result.items():
                if agent_name not in ["final_result", "metadata"]:
                    agent_responses.append(
                        {
                            "agent": agent_name,
                            "content": str(response),
                        }
                    )
            final_result = result.get("final_result", str(result))
        else:
            final_result = str(result)

        return {
            "success": True,
            "system_name": system_name,
            "task": task,
            "strategy": strategy,
            "agent_responses": agent_responses,
            "final_result": final_result,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_multiagent_stats(system_name: str) -> dict:
    """
    다중 에이전트 시스템 통계 조회

    Args:
        system_name: 시스템 이름

    Returns:
        dict: 에이전트 정보, 전략, 설정

    Example:
        User: "debate_team 시스템 정보 알려줘"
        → get_multiagent_stats(system_name="debate_team")
    """
    try:
        if system_name not in _multiagent_systems:
            return {
                "success": False,
                "error": f"Multi-agent system '{system_name}' not found.",
            }

        meta = _agents_cache.get(system_name, {})
        configs = meta.get("configs", [])

        # 에이전트 정보 수집
        agents_info = []
        for config in configs:
            agents_info.append(
                {
                    "name": config.get("name"),
                    "role": config.get("role", config.get("name")),
                    "model": config.get("model", MCPServerConfig.DEFAULT_CHAT_MODEL),
                    "temperature": config.get("temperature", 0.7),
                }
            )

        return {
            "success": True,
            "system_name": system_name,
            "agent_count": len(agents_info),
            "agents": agents_info,
            "strategy": meta.get("strategy", "sequential"),
            "max_rounds": meta.get("max_rounds", 3),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def list_multiagent_systems() -> dict:
    """
    생성된 다중 에이전트 시스템 목록 조회

    Returns:
        dict: 시스템 이름 목록

    Example:
        User: "어떤 에이전트 시스템들이 있어?"
        → list_multiagent_systems()
    """
    return {
        "success": True,
        "systems": list(_multiagent_systems.keys()),
        "count": len(_multiagent_systems),
    }


@mcp.tool()
async def delete_multiagent_system(system_name: str) -> dict:
    """
    다중 에이전트 시스템 삭제

    Args:
        system_name: 삭제할 시스템 이름

    Returns:
        dict: 성공 여부

    Example:
        User: "debate_team 시스템 삭제해줘"
        → delete_multiagent_system(system_name="debate_team")
    """
    try:
        if system_name not in _multiagent_systems:
            return {
                "success": False,
                "error": f"Multi-agent system '{system_name}' not found.",
            }

        del _multiagent_systems[system_name]
        if system_name in _agents_cache:
            del _agents_cache[system_name]

        return {
            "success": True,
            "message": f"Multi-agent system '{system_name}' deleted successfully.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
