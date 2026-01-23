"""
Multi-Agent Tools - 기존 beanllm Multi-Agent 기능을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""
import asyncio
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

# 기존 beanllm 코드 import (wrapping 대상)
from beanllm.facade.advanced import MultiAgentFacade
from beanllm.dto.request.multi_agent import (
    MultiAgentRequest,
    AgentConfig,
    CommunicationStrategy,
)
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스 생성
mcp = FastMCP("Multi-Agent Tools")

# 전역 Multi-Agent 시스템 캐시
_multiagent_systems: Dict[str, MultiAgentFacade] = {}


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
        strategy: 통신 전략 (sequential, round_robin, debate, hierarchical)
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
        # 1. AgentConfig 객체 생성
        agents = []
        for config in agent_configs:
            agent = AgentConfig(
                name=config["name"],
                role=config.get("role", config["name"]),
                model=config.get("model", MCPServerConfig.DEFAULT_CHAT_MODEL),
                temperature=config.get("temperature", 0.7),
                system_prompt=config.get("system_prompt"),
                tools=config.get("tools", []),
            )
            agents.append(agent)

        # 2. CommunicationStrategy enum 변환
        strategy_map = {
            "sequential": CommunicationStrategy.SEQUENTIAL,
            "round_robin": CommunicationStrategy.ROUND_ROBIN,
            "debate": CommunicationStrategy.DEBATE,
            "hierarchical": CommunicationStrategy.HIERARCHICAL,
        }
        comm_strategy = strategy_map.get(strategy, CommunicationStrategy.SEQUENTIAL)

        # 3. MultiAgentFacade 생성
        facade = MultiAgentFacade(
            agents=agents,
            strategy=comm_strategy,
            max_rounds=max_rounds,
        )

        # 4. 캐시에 저장
        _multiagent_systems[system_name] = facade

        return {
            "success": True,
            "system_name": system_name,
            "agent_count": len(agents),
            "agent_names": [a.name for a in agents],
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

        facade = _multiagent_systems[system_name]

        # 2. 🎯 기존 MultiAgentFacade.run() 메서드 사용!
        request = MultiAgentRequest(
            task=task,
            context=context or {},
        )

        result = await facade.run(request)

        # 3. 결과 포매팅
        agent_responses = []
        for msg in result.messages:
            agent_responses.append(
                {
                    "agent": msg.agent_name,
                    "role": msg.role,
                    "content": msg.content,
                    "round": msg.metadata.get("round", 0),
                }
            )

        return {
            "success": True,
            "system_name": system_name,
            "task": task,
            "agent_responses": agent_responses,
            "final_result": result.final_result,
            "total_rounds": result.metadata.get("total_rounds", 0),
            "strategy": result.metadata.get("strategy", "unknown"),
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

        facade = _multiagent_systems[system_name]

        # 에이전트 정보 수집
        agents_info = []
        for agent in facade._agents:
            agents_info.append(
                {
                    "name": agent.name,
                    "role": agent.role,
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "tools_count": len(agent.tools) if agent.tools else 0,
                }
            )

        return {
            "success": True,
            "system_name": system_name,
            "agent_count": len(agents_info),
            "agents": agents_info,
            "strategy": facade._strategy.value,
            "max_rounds": facade._max_rounds,
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

        return {
            "success": True,
            "message": f"Multi-agent system '{system_name}' deleted successfully.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
