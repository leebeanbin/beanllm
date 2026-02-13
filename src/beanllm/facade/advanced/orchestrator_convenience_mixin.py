"""
Orchestrator Convenience Mixin - quick_* 원스톱 워크플로우 메서드

이 mixin은 Orchestrator Facade에 혼합되어 사용됩니다.
create_and_execute를 활용한 quick_research_write, quick_parallel_consensus,
quick_debate 등 짧은 이름의 편의 메서드를 제공합니다.

호스트 클래스는 create_and_execute 메서드를 제공해야 합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, cast

from beanllm.dto.response.advanced.orchestrator_response import ExecuteWorkflowResponse

if TYPE_CHECKING:
    from typing import Protocol

    class _OrchestratorConvenienceHost(Protocol):
        """Protocol for host class: must provide create_and_execute."""

        async def create_and_execute(
            self,
            name: str,
            strategy: str,
            agents: Dict[str, Any],
            task: str,
            config: Optional[Dict[str, Any]] = None,
            tools: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]: ...


class OrchestratorConvenienceMixin:
    """
    Orchestrator Facade용 편의 mixin.

    quick_* 메서드는 내부적으로 self.create_and_execute를 호출하므로,
    이 mixin을 사용하는 클래스는 create_workflow, execute, create_and_execute 및
    _ensure_handler() (및 self._handler)를 제공해야 합니다.
    """

    async def quick_research_write(
        self,
        researcher_agent: Any,
        writer_agent: Any,
        task: str,
        reviewer_agent: Optional[Any] = None,
        name: str = "Research & Write",
    ) -> ExecuteWorkflowResponse:
        """
        빠른 Research & Write 워크플로우

        Args:
            researcher_agent: Researcher agent
            writer_agent: Writer agent
            task: 연구 주제
            reviewer_agent: Reviewer agent (optional)
            name: 워크플로우 이름

        Returns:
            ExecuteWorkflowResponse: 실행 결과

        Example:
            ```python
            result = await orchestrator.quick_research_write(
                researcher_agent=researcher,
                writer_agent=writer,
                task="The future of AI in healthcare",
                reviewer_agent=reviewer
            )
            ```
        """
        agents: Dict[str, Any] = {
            "researcher": researcher_agent,
            "writer": writer_agent,
        }
        if reviewer_agent:
            agents["reviewer"] = reviewer_agent

        config: Dict[str, str] = {
            "researcher_id": "researcher",
            "writer_id": "writer",
        }
        if reviewer_agent:
            config["reviewer_id"] = "reviewer"

        create_and_execute = cast(
            Callable[..., Awaitable[Dict[str, Any]]],
            getattr(self, "create_and_execute"),
        )
        result = await create_and_execute(
            name=name,
            strategy="research_write",
            agents=agents,
            task=task,
            config=config,
        )
        execution = result.get("execution")
        return cast(ExecuteWorkflowResponse, execution)

    async def quick_parallel_consensus(
        self,
        agents: List[Any],
        task: str,
        aggregation: str = "vote",
        name: str = "Parallel Consensus",
    ) -> ExecuteWorkflowResponse:
        """
        빠른 Parallel Consensus 워크플로우

        Args:
            agents: Agent 리스트
            task: 태스크
            aggregation: 집계 방법 ("vote", "consensus")
            name: 워크플로우 이름

        Returns:
            ExecuteWorkflowResponse: 실행 결과

        Example:
            ```python
            result = await orchestrator.quick_parallel_consensus(
                agents=[agent1, agent2, agent3],
                task="Evaluate this proposal",
                aggregation="vote"
            )
            ```
        """
        agents_dict = {f"agent{i}": agent for i, agent in enumerate(agents)}
        agent_ids = list(agents_dict.keys())

        create_and_execute = cast(
            Callable[..., Awaitable[Dict[str, Any]]],
            getattr(self, "create_and_execute"),
        )
        result = await create_and_execute(
            name=name,
            strategy="parallel",
            agents=agents_dict,
            task=task,
            config={
                "agent_ids": agent_ids,
                "aggregation": aggregation,
            },
        )
        execution = result.get("execution")
        return cast(ExecuteWorkflowResponse, execution)

    async def quick_debate(
        self,
        debater_agents: List[Any],
        judge_agent: Any,
        task: str,
        rounds: int = 3,
        name: str = "Debate & Judge",
    ) -> ExecuteWorkflowResponse:
        """
        빠른 Debate & Judge 워크플로우

        Args:
            debater_agents: Debater agent 리스트
            judge_agent: Judge agent
            task: 논쟁 주제
            rounds: 논쟁 라운드 수
            name: 워크플로우 이름

        Returns:
            ExecuteWorkflowResponse: 실행 결과

        Example:
            ```python
            result = await orchestrator.quick_debate(
                debater_agents=[debater1, debater2],
                judge_agent=judge,
                task="Should AI be regulated?",
                rounds=3
            )
            ```
        """
        agents_dict = {f"debater{i}": agent for i, agent in enumerate(debater_agents)}
        agents_dict["judge"] = judge_agent

        debater_ids = [f"debater{i}" for i in range(len(debater_agents))]

        create_and_execute = cast(
            Callable[..., Awaitable[Dict[str, Any]]],
            getattr(self, "create_and_execute"),
        )
        result = await create_and_execute(
            name=name,
            strategy="debate",
            agents=agents_dict,
            task=task,
            config={
                "debater_ids": debater_ids,
                "judge_id": "judge",
                "rounds": rounds,
            },
        )
        execution = result.get("execution")
        return cast(ExecuteWorkflowResponse, execution)
