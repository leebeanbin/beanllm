"""
WorkflowTemplates - 사전 정의된 워크플로우 템플릿
SOLID 원칙:
- SRP: 템플릿 생성만 담당
- OCP: 새로운 템플릿 추가 가능
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .workflow_graph import NodeType, WorkflowGraph


class WorkflowTemplates:
    """
    사전 정의된 워크플로우 템플릿 모음

    책임:
    - 일반적인 패턴의 워크플로우 제공
    - 빠른 프로토타이핑 지원

    Example:
        ```python
        # Research & Write template
        workflow = WorkflowTemplates.research_and_write(
            researcher_id="researcher",
            writer_id="writer"
        )

        # Multi-stage pipeline
        workflow = WorkflowTemplates.pipeline(
            stages=["gather", "analyze", "summarize", "present"]
        )
        ```
    """

    @staticmethod
    def research_and_write(
        researcher_id: str = "researcher",
        writer_id: str = "writer",
        reviewer_id: Optional[str] = None,
    ) -> WorkflowGraph:
        """
        연구 → 작성 워크플로우

        Args:
            researcher_id: Researcher agent ID
            writer_id: Writer agent ID
            reviewer_id: Reviewer agent ID (optional)

        Returns:
            WorkflowGraph: Research & Write workflow

        Workflow:
            START → Researcher → Writer → [Reviewer] → END
        """
        workflow = WorkflowGraph(name="Research & Write")

        # Nodes
        start = workflow.add_node(NodeType.START, "start")
        research = workflow.add_node(
            NodeType.AGENT,
            "researcher",
            config={"agent_id": researcher_id},
        )
        write = workflow.add_node(
            NodeType.AGENT,
            "writer",
            config={"agent_id": writer_id},
        )

        # Edges
        workflow.add_edge(start, research)
        workflow.add_edge(research, write)

        # Optional reviewer
        if reviewer_id:
            review = workflow.add_node(
                NodeType.AGENT,
                "reviewer",
                config={"agent_id": reviewer_id},
            )
            workflow.add_edge(write, review)

            end = workflow.add_node(NodeType.END, "end")
            workflow.add_edge(review, end)
        else:
            end = workflow.add_node(NodeType.END, "end")
            workflow.add_edge(write, end)

        return workflow

    @staticmethod
    def parallel_consensus(
        agent_ids: List[str],
        aggregation: str = "vote",
    ) -> WorkflowGraph:
        """
        병렬 실행 → 합의 워크플로우

        Args:
            agent_ids: Agent IDs
            aggregation: 집계 방법 ("vote", "consensus")

        Returns:
            WorkflowGraph: Parallel consensus workflow

        Workflow:
            START → [Agent1, Agent2, Agent3] → Merge → END
        """
        workflow = WorkflowGraph(name=f"Parallel Consensus ({aggregation})")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Parallel agents
        parallel = workflow.add_node(
            NodeType.PARALLEL,
            "parallel_agents",
            config={"agent_ids": agent_ids, "aggregation": aggregation},
        )
        workflow.add_edge(start, parallel)

        # Merge (optional - could be implicit in parallel node)
        merge = workflow.add_node(
            NodeType.MERGE,
            "merge_results",
            config={"method": aggregation},
        )
        workflow.add_edge(parallel, merge)

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(merge, end)

        return workflow

    @staticmethod
    def hierarchical_delegation(
        manager_id: str,
        worker_ids: List[str],
    ) -> WorkflowGraph:
        """
        계층적 위임 워크플로우

        Args:
            manager_id: Manager agent ID
            worker_ids: Worker agent IDs

        Returns:
            WorkflowGraph: Hierarchical workflow

        Workflow:
            START → Manager (분해) → [Workers] → Manager (종합) → END
        """
        workflow = WorkflowGraph(name="Hierarchical Delegation")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Manager: Task decomposition
        decompose = workflow.add_node(
            NodeType.AGENT,
            "manager_decompose",
            config={
                "agent_id": manager_id,
                "role": "decompose",
            },
        )
        workflow.add_edge(start, decompose)

        # Hierarchical execution (parallel workers)
        execute = workflow.add_node(
            NodeType.HIERARCHICAL,
            "execute_tasks",
            config={
                "manager_id": manager_id,
                "worker_ids": worker_ids,
            },
        )
        workflow.add_edge(decompose, execute)

        # Manager: Synthesis
        synthesize = workflow.add_node(
            NodeType.AGENT,
            "manager_synthesize",
            config={
                "agent_id": manager_id,
                "role": "synthesize",
            },
        )
        workflow.add_edge(execute, synthesize)

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(synthesize, end)

        return workflow

    @staticmethod
    def debate_and_judge(
        debater_ids: List[str],
        judge_id: str,
        rounds: int = 3,
    ) -> WorkflowGraph:
        """
        토론 → 판정 워크플로우

        Args:
            debater_ids: Debater agent IDs
            judge_id: Judge agent ID
            rounds: 토론 라운드 수

        Returns:
            WorkflowGraph: Debate workflow

        Workflow:
            START → [Debaters x rounds] → Judge → END
        """
        workflow = WorkflowGraph(name=f"Debate ({rounds} rounds)")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Debate
        debate = workflow.add_node(
            NodeType.DEBATE,
            "debate",
            config={
                "agent_ids": debater_ids,
                "rounds": rounds,
            },
        )
        workflow.add_edge(start, debate)

        # Judge
        judge = workflow.add_node(
            NodeType.AGENT,
            "judge",
            config={"agent_id": judge_id},
        )
        workflow.add_edge(debate, judge)

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(judge, end)

        return workflow

    @staticmethod
    def pipeline(
        stages: List[str],
        agent_ids: Optional[List[str]] = None,
    ) -> WorkflowGraph:
        """
        순차 파이프라인 워크플로우

        Args:
            stages: Stage 이름 리스트
            agent_ids: Agent IDs (None이면 stage 이름 사용)

        Returns:
            WorkflowGraph: Pipeline workflow

        Workflow:
            START → Stage1 → Stage2 → ... → END
        """
        workflow = WorkflowGraph(name="Pipeline")

        if agent_ids is None:
            agent_ids = stages

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Stages
        prev_node = start
        for i, (stage_name, agent_id) in enumerate(zip(stages, agent_ids)):
            stage = workflow.add_node(
                NodeType.AGENT,
                stage_name,
                config={"agent_id": agent_id, "stage": i + 1},
            )
            workflow.add_edge(prev_node, stage)
            prev_node = stage

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(prev_node, end)

        return workflow

    @staticmethod
    def conditional_branch(
        condition_agent_id: str,
        branch_a_id: str,
        branch_b_id: str,
    ) -> WorkflowGraph:
        """
        조건부 분기 워크플로우

        Args:
            condition_agent_id: 조건 평가 agent
            branch_a_id: Branch A agent (조건 true)
            branch_b_id: Branch B agent (조건 false)

        Returns:
            WorkflowGraph: Conditional workflow

        Workflow:
            START → Decision → [Branch A | Branch B] → END
        """
        workflow = WorkflowGraph(name="Conditional Branch")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Decision node
        decision = workflow.add_node(
            NodeType.DECISION,
            "decision",
            config={"agent_id": condition_agent_id},
        )
        workflow.add_edge(start, decision)

        # Branch A
        branch_a = workflow.add_node(
            NodeType.AGENT,
            "branch_a",
            config={"agent_id": branch_a_id},
        )

        # Branch B
        branch_b = workflow.add_node(
            NodeType.AGENT,
            "branch_b",
            config={"agent_id": branch_b_id},
        )

        # Conditional edges (Note: actual condition function would be added separately)
        from .workflow_graph import EdgeCondition

        workflow.add_edge(decision, branch_a, condition=EdgeCondition.ON_SUCCESS)
        workflow.add_edge(decision, branch_b, condition=EdgeCondition.ON_FAILURE)

        # Merge
        merge = workflow.add_node(NodeType.MERGE, "merge")
        workflow.add_edge(branch_a, merge)
        workflow.add_edge(branch_b, merge)

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(merge, end)

        return workflow

    @staticmethod
    def iterative_refinement(
        agent_id: str,
        max_iterations: int = 3,
        quality_checker_id: Optional[str] = None,
    ) -> WorkflowGraph:
        """
        반복적 개선 워크플로우

        Args:
            agent_id: Main agent ID
            max_iterations: 최대 반복 횟수
            quality_checker_id: Quality checker agent (optional)

        Returns:
            WorkflowGraph: Iterative refinement workflow

        Workflow:
            START → Agent → [Quality Check → Agent] × N → END
        """
        workflow = WorkflowGraph(name=f"Iterative Refinement ({max_iterations}x)")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Initial generation
        prev_node = start
        for i in range(max_iterations):
            # Generate/Refine
            generate = workflow.add_node(
                NodeType.AGENT,
                f"iteration_{i + 1}",
                config={
                    "agent_id": agent_id,
                    "iteration": i + 1,
                },
            )
            workflow.add_edge(prev_node, generate)

            # Optional quality check
            if quality_checker_id and i < max_iterations - 1:
                check = workflow.add_node(
                    NodeType.DECISION,
                    f"check_{i + 1}",
                    config={"agent_id": quality_checker_id},
                )
                workflow.add_edge(generate, check)
                prev_node = check
            else:
                prev_node = generate

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(prev_node, end)

        return workflow

    @staticmethod
    def map_reduce(
        mapper_ids: List[str],
        reducer_id: str,
    ) -> WorkflowGraph:
        """
        Map-Reduce 워크플로우

        Args:
            mapper_ids: Mapper agent IDs
            reducer_id: Reducer agent ID

        Returns:
            WorkflowGraph: Map-Reduce workflow

        Workflow:
            START → [Mappers] → Reducer → END
        """
        workflow = WorkflowGraph(name="Map-Reduce")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Map phase (parallel)
        map_parallel = workflow.add_node(
            NodeType.PARALLEL,
            "map_phase",
            config={"agent_ids": mapper_ids},
        )
        workflow.add_edge(start, map_parallel)

        # Reduce phase
        reduce = workflow.add_node(
            NodeType.AGENT,
            "reduce_phase",
            config={"agent_id": reducer_id},
        )
        workflow.add_edge(map_parallel, reduce)

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(reduce, end)

        return workflow

    @staticmethod
    def code_review_pipeline(
        coder_id: str,
        reviewer_ids: List[str],
        final_approver_id: str,
    ) -> WorkflowGraph:
        """
        코드 리뷰 파이프라인

        Args:
            coder_id: Coder agent ID
            reviewer_ids: Reviewer agent IDs
            final_approver_id: Final approver ID

        Returns:
            WorkflowGraph: Code review workflow

        Workflow:
            START → Coder → [Reviewers] → Approver → END
        """
        workflow = WorkflowGraph(name="Code Review Pipeline")

        # Start
        start = workflow.add_node(NodeType.START, "start")

        # Coder
        code = workflow.add_node(
            NodeType.AGENT,
            "coder",
            config={"agent_id": coder_id},
        )
        workflow.add_edge(start, code)

        # Parallel reviewers
        review = workflow.add_node(
            NodeType.PARALLEL,
            "reviewers",
            config={"agent_ids": reviewer_ids},
        )
        workflow.add_edge(code, review)

        # Final approver
        approve = workflow.add_node(
            NodeType.AGENT,
            "approver",
            config={"agent_id": final_approver_id},
        )
        workflow.add_edge(review, approve)

        # End
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(approve, end)

        return workflow

    @staticmethod
    def custom_template(
        name: str,
        structure: Dict[str, Any],
    ) -> WorkflowGraph:
        """
        사용자 정의 템플릿

        Args:
            name: 워크플로우 이름
            structure: 구조 정의
                {
                    "nodes": [
                        {"id": "n1", "type": "agent", "name": "...", "config": {...}},
                        ...
                    ],
                    "edges": [
                        {"source": "n1", "target": "n2"},
                        ...
                    ]
                }

        Returns:
            WorkflowGraph: Custom workflow
        """
        workflow = WorkflowGraph(name=name)

        # Add nodes
        node_map = {}
        for node_def in structure.get("nodes", []):
            node_id = workflow.add_node(
                node_type=NodeType(node_def["type"]),
                name=node_def["name"],
                config=node_def.get("config", {}),
                node_id=node_def.get("id"),
            )
            node_map[node_def.get("id", node_id)] = node_id

        # Add edges
        for edge_def in structure.get("edges", []):
            source = node_map[edge_def["source"]]
            target = node_map[edge_def["target"]]
            workflow.add_edge(source, target)

        return workflow


# Quick access functions
def quick_research_write(researcher: str, writer: str) -> WorkflowGraph:
    """Quick: Research & Write"""
    return WorkflowTemplates.research_and_write(researcher, writer)


def quick_parallel(agents: List[str]) -> WorkflowGraph:
    """Quick: Parallel execution"""
    return WorkflowTemplates.parallel_consensus(agents)


def quick_pipeline(stages: List[str]) -> WorkflowGraph:
    """Quick: Sequential pipeline"""
    return WorkflowTemplates.pipeline(stages)


def quick_debate(debaters: List[str], judge: str, rounds: int = 3) -> WorkflowGraph:
    """Quick: Debate"""
    return WorkflowTemplates.debate_and_judge(debaters, judge, rounds)
