"""
OrchestratorConvenienceMixin - 오케스트레이터 편의 메서드 모음

quick_research_write, quick_parallel_consensus, quick_debate 등
자주 사용하는 워크플로우 패턴을 원스톱으로 제공합니다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from beanllm.dto.response.advanced.orchestrator_response import ExecuteWorkflowResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OrchestratorConvenienceMixin:
    """오케스트레이터 편의 메서드 Mixin

    Orchestrator 클래스에 믹스인되어 `quick_*` 계열 편의 메서드를 제공합니다.
    본 클래스 단독으로는 사용할 수 없으며, ``create_workflow`` / ``execute`` /
    ``monitor`` / ``analyze`` 를 제공하는 클래스와 함께 사용해야 합니다.
    """

    async def create_and_execute(
        self,
        name: str,
        strategy: str,
        agents: Dict[str, Any],
        task: str,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        워크플로우 생성 및 실행 (원스톱)

        Args:
            name: 워크플로우 이름
            strategy: 전략
            agents: Agent 인스턴스 딕셔너리
            task: 실행할 태스크
            config: 전략별 설정
            tools: 사용 가능한 도구

        Returns:
            Dict: 생성 및 실행 결과 (``workflow``, ``execution`` 키)
        """
        logger.info(f"Creating and executing workflow: {name}")

        workflow = await self.create_workflow(  # type: ignore[attr-defined]
            name=name,
            strategy=strategy,
            config=config,
        )

        execution = await self.execute(  # type: ignore[attr-defined]
            workflow_id=workflow.workflow_id,
            agents=agents,
            task=task,
            tools=tools,
        )

        return {
            "workflow": workflow,
            "execution": execution,
        }

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
        """
        agents: Dict[str, Any] = {
            "researcher": researcher_agent,
            "writer": writer_agent,
        }
        if reviewer_agent:
            agents["reviewer"] = reviewer_agent

        config: Dict[str, Any] = {
            "researcher_id": "researcher",
            "writer_id": "writer",
        }
        if reviewer_agent:
            config["reviewer_id"] = "reviewer"

        result = await self.create_and_execute(
            name=name,
            strategy="research_write",
            agents=agents,
            task=task,
            config=config,
        )

        return result["execution"]

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
        """
        agents_dict = {f"agent{i}": agent for i, agent in enumerate(agents)}
        agent_ids = list(agents_dict.keys())

        result = await self.create_and_execute(
            name=name,
            strategy="parallel",
            agents=agents_dict,
            task=task,
            config={
                "agent_ids": agent_ids,
                "aggregation": aggregation,
            },
        )

        return result["execution"]

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
        """
        agents_dict: Dict[str, Any] = {
            f"debater{i}": agent for i, agent in enumerate(debater_agents)
        }
        agents_dict["judge"] = judge_agent

        debater_ids = [f"debater{i}" for i in range(len(debater_agents))]

        result = await self.create_and_execute(
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

        return result["execution"]

    async def run_full_workflow(
        self,
        workflow_id: str,
        agents: Dict[str, Any],
        task: str,
        tools: Optional[Dict[str, Any]] = None,
        monitor: bool = True,
        analyze: bool = True,
    ) -> Dict[str, Any]:
        """
        전체 워크플로우 실행 (실행 + 모니터링 + 분석)

        Args:
            workflow_id: 워크플로우 ID
            agents: Agent 인스턴스 딕셔너리
            task: 실행할 태스크
            tools: 사용 가능한 도구
            monitor: 모니터링 실행 여부
            analyze: 분석 실행 여부

        Returns:
            Dict: 실행, 모니터링, 분석 결과
        """
        logger.info(f"Running full workflow: {workflow_id}")

        execution = await self.execute(  # type: ignore[attr-defined]
            workflow_id=workflow_id,
            agents=agents,
            task=task,
            tools=tools,
        )

        results: Dict[str, Any] = {"execution": execution}

        if monitor and execution.execution_id:
            results["monitor"] = await self.monitor(  # type: ignore[attr-defined]
                workflow_id=workflow_id,
                execution_id=execution.execution_id,
            )

        if analyze:
            results["analytics"] = await self.analyze(workflow_id)  # type: ignore[attr-defined]

        logger.info("Full workflow completed")

        return results
