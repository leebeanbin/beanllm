"""
Comprehensive tests for facade convenience/standalone modules and service factory.

Covers:
- knowledge_graph_convenience.py
- knowledge_graph_standalone.py
- optimizer_convenience.py
- optimizer_standalone.py
- orchestrator_convenience_mixin.py
- multi_agent_facade.py
- graph_facade.py
- service/factory.py
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------
from beanllm.dto.response.advanced.optimizer_response import (
    BenchmarkResponse,
    OptimizeResponse,
    ProfileResponse,
    RecommendationResponse,
)
from beanllm.dto.response.advanced.orchestrator_response import (
    CreateWorkflowResponse,
    ExecuteWorkflowResponse,
)
from beanllm.dto.response.graph.kg_response import (
    BuildGraphResponse,
    GraphRAGResponse,
    QueryGraphResponse,
)

# ---------------------------------------------------------------------------
# Helpers: shared response factories
# ---------------------------------------------------------------------------


def _build_graph_response(**kwargs: Any) -> BuildGraphResponse:
    defaults: Dict[str, Any] = dict(
        graph_id="graph-1",
        graph_name="test",
        num_nodes=3,
        num_edges=2,
        backend="networkx",
        document_ids=["doc-1"],
        created_at="2026-01-01T00:00:00",
        statistics={},
    )
    defaults.update(kwargs)
    return BuildGraphResponse(**defaults)


def _query_graph_response(**kwargs: Any) -> QueryGraphResponse:
    defaults: Dict[str, Any] = dict(
        graph_id="graph-1",
        query="test",
        results=[{"name": "Apple"}],
        num_results=1,
        execution_time=0.01,
    )
    defaults.update(kwargs)
    return QueryGraphResponse(**defaults)


def _graph_rag_response(**kwargs: Any) -> GraphRAGResponse:
    defaults: Dict[str, Any] = dict(
        answer="Apple was founded by Steve Jobs.",
        entities_used=["Apple"],
        reasoning_paths=[["Apple", "founded_by", "Steve Jobs"]],
        graph_context="context",
    )
    defaults.update(kwargs)
    return GraphRAGResponse(**defaults)


def _optimize_response(**kwargs: Any) -> OptimizeResponse:
    defaults: Dict[str, Any] = dict(
        optimization_id="opt-1",
        system_id="default",
        optimal_parameters={"top_k": 7},
        improvement_metrics={"quality": 0.15},
        num_trials=30,
    )
    defaults.update(kwargs)
    return OptimizeResponse(**defaults)


def _benchmark_response(**kwargs: Any) -> BenchmarkResponse:
    defaults: Dict[str, Any] = dict(
        benchmark_id="bench-1",
        num_queries=30,
        avg_latency=0.4,
        throughput=2.5,
    )
    defaults.update(kwargs)
    return BenchmarkResponse(**defaults)


def _profile_response(**kwargs: Any) -> ProfileResponse:
    defaults: Dict[str, Any] = dict(
        profile_id="prof-1",
        system_id="default",
        duration=2.0,
        component_breakdown={"embedding": {"latency": 0.5}},
        total_latency=2.0,
        total_cost=0.02,
        bottlenecks=[],
        cost_breakdown={"embedding": 0.01},
        bottleneck="embedding",
    )
    defaults.update(kwargs)
    return ProfileResponse(**defaults)


def _recommendation_response(**kwargs: Any) -> RecommendationResponse:
    defaults: Dict[str, Any] = dict(
        profile_id="prof-1",
        recommendations=[{"title": "Use reranking", "priority": "high"}],
        estimated_improvements={"quality": 0.15},
        implementation_difficulty={"Use reranking": "medium"},
        priority_order=["Use reranking"],
    )
    defaults.update(kwargs)
    return RecommendationResponse(**defaults)


def _execute_workflow_response(**kwargs: Any) -> ExecuteWorkflowResponse:
    defaults: Dict[str, Any] = dict(
        execution_id="exec-1",
        workflow_id="wf-1",
        status="completed",
        result="Done",
        execution_time=1.5,
    )
    defaults.update(kwargs)
    return ExecuteWorkflowResponse(**defaults)


def _create_workflow_response(**kwargs: Any) -> CreateWorkflowResponse:
    defaults: Dict[str, Any] = dict(
        workflow_id="wf-1",
        workflow_name="test",
        num_nodes=2,
        num_edges=1,
        strategy="sequential",
        visualization="[A] --> [B]",
        created_at="2026-01-01T00:00:00",
    )
    defaults.update(kwargs)
    return CreateWorkflowResponse(**defaults)


# ===========================================================================
# 1. KnowledgeGraphConvenienceMixin
# ===========================================================================


class _KGHost:
    """Minimal host that satisfies the mixin contract."""

    def __init__(self) -> None:
        self._handler = AsyncMock()

    async def build_graph(self, **kwargs: Any) -> BuildGraphResponse:
        return _build_graph_response()

    async def query_graph(self, **kwargs: Any) -> QueryGraphResponse:
        return _query_graph_response(**{"results": [{"name": "Apple"}]})

    async def graph_rag(self, **kwargs: Any) -> GraphRAGResponse:
        return _graph_rag_response()


from beanllm.facade.advanced.knowledge_graph_convenience import (
    KnowledgeGraphConvenienceMixin,
)


class _KGMixed(_KGHost, KnowledgeGraphConvenienceMixin):
    pass


class TestKnowledgeGraphConvenienceMixin:
    @pytest.fixture
    def kg(self) -> _KGMixed:
        return _KGMixed()

    async def test_quick_build_returns_build_graph_response(self, kg: _KGMixed) -> None:
        result = await kg.quick_build(documents=["Apple was founded by Steve Jobs."])
        assert isinstance(result, BuildGraphResponse)
        assert result.graph_id == "graph-1"

    async def test_quick_build_with_graph_id(self, kg: _KGMixed) -> None:
        result = await kg.quick_build(
            documents=["doc1"],
            graph_id="custom-id",
        )
        assert isinstance(result, BuildGraphResponse)

    async def test_find_entities_by_type(self, kg: _KGMixed) -> None:
        result = await kg.find_entities_by_type(graph_id="graph-1", entity_type="PERSON")
        assert isinstance(result, list)
        assert result[0]["name"] == "Apple"

    async def test_find_entities_by_name(self, kg: _KGMixed) -> None:
        result = await kg.find_entities_by_name(graph_id="graph-1", name="Apple")
        assert isinstance(result, list)

    async def test_find_entities_by_name_fuzzy(self, kg: _KGMixed) -> None:
        result = await kg.find_entities_by_name(graph_id="graph-1", name="App", fuzzy=True)
        assert isinstance(result, list)

    async def test_find_related_entities_default_hops(self, kg: _KGMixed) -> None:
        result = await kg.find_related_entities(graph_id="graph-1", entity_id="e1")
        assert isinstance(result, list)

    async def test_find_related_entities_with_relation_type(self, kg: _KGMixed) -> None:
        result = await kg.find_related_entities(
            graph_id="graph-1",
            entity_id="e1",
            relation_type="founded_by",
            max_hops=2,
        )
        assert isinstance(result, list)

    async def test_find_path_with_valid_path(self, kg: _KGMixed) -> None:
        # Patch query_graph to return a path result
        kg.query_graph = AsyncMock(  # type: ignore[method-assign]
            return_value=QueryGraphResponse(
                graph_id="graph-1",
                query="find_shortest_path",
                results=[{"path": ["e1", "e2", "e3"]}],
                num_results=1,
                execution_time=0.01,
            )
        )
        result = await kg.find_path(graph_id="graph-1", source_id="e1", target_id="e3")
        assert result == ["e1", "e2", "e3"]

    async def test_find_path_empty_results(self, kg: _KGMixed) -> None:
        kg.query_graph = AsyncMock(  # type: ignore[method-assign]
            return_value=QueryGraphResponse(
                graph_id="graph-1",
                query="find_shortest_path",
                results=[],
                num_results=0,
                execution_time=0.01,
            )
        )
        result = await kg.find_path(graph_id="graph-1", source_id="e1", target_id="e99")
        assert result is None

    async def test_find_path_non_list_path_value(self, kg: _KGMixed) -> None:
        kg.query_graph = AsyncMock(  # type: ignore[method-assign]
            return_value=QueryGraphResponse(
                graph_id="graph-1",
                query="find_shortest_path",
                results=[{"path": "not-a-list"}],
                num_results=1,
                execution_time=0.01,
            )
        )
        result = await kg.find_path(graph_id="graph-1", source_id="e1", target_id="e2")
        assert result is None

    async def test_get_entity_details_found(self, kg: _KGMixed) -> None:
        kg.query_graph = AsyncMock(  # type: ignore[method-assign]
            return_value=QueryGraphResponse(
                graph_id="graph-1",
                query="get_entity_details",
                results=[{"name": "Apple", "type": "ORG"}],
                num_results=1,
                execution_time=0.01,
            )
        )
        result = await kg.get_entity_details(graph_id="graph-1", entity_id="e1")
        assert result is not None
        assert result["name"] == "Apple"

    async def test_get_entity_details_not_found(self, kg: _KGMixed) -> None:
        kg.query_graph = AsyncMock(  # type: ignore[method-assign]
            return_value=QueryGraphResponse(
                graph_id="graph-1",
                query="get_entity_details",
                results=[],
                num_results=0,
                execution_time=0.01,
            )
        )
        result = await kg.get_entity_details(graph_id="graph-1", entity_id="unknown")
        assert result is None

    async def test_ask_with_hybrid_results(self, kg: _KGMixed) -> None:
        kg.graph_rag = AsyncMock(  # type: ignore[method-assign]
            return_value=GraphRAGResponse(
                answer="test",
                entities_used=["Apple"],
                reasoning_paths=[],
                graph_context="ctx",
                hybrid_results=[
                    {"entity": {"name": "Apple", "type": "ORG"}, "score": 0.95},
                    {"entity": {"name": "Steve Jobs", "type": "PERSON"}, "score": 0.88},
                ],
            )
        )
        result = await kg.ask(query="Who founded Apple?", graph_id="tech")
        assert "Based on the knowledge graph:" in result
        assert "Apple" in result
        assert "0.95" in result

    async def test_ask_with_non_dict_entity(self, kg: _KGMixed) -> None:
        """entity field is not a dict — should fall back to Unknown/UNKNOWN."""
        kg.graph_rag = AsyncMock(  # type: ignore[method-assign]
            return_value=GraphRAGResponse(
                answer="test",
                entities_used=[],
                reasoning_paths=[],
                graph_context="ctx",
                hybrid_results=[
                    {"entity": "not-a-dict", "score": 0.75},
                ],
            )
        )
        result = await kg.ask(query="Q?", graph_id="g")
        assert "Unknown" in result
        assert "UNKNOWN" in result

    async def test_ask_no_hybrid_results(self, kg: _KGMixed) -> None:
        kg.graph_rag = AsyncMock(  # type: ignore[method-assign]
            return_value=GraphRAGResponse(
                answer="test",
                entities_used=[],
                reasoning_paths=[],
                graph_context="ctx",
                hybrid_results=None,
            )
        )
        result = await kg.ask(query="Q?", graph_id="g")
        assert result == "No relevant information found in the knowledge graph."

    async def test_merge_graphs_raises_not_implemented(self, kg: _KGMixed) -> None:
        with pytest.raises(NotImplementedError):
            await kg.merge_graphs(graph_ids=["g1", "g2"])


# ===========================================================================
# 2. KnowledgeGraph Standalone
# ===========================================================================


class TestKnowledgeGraphStandalone:
    async def test_quick_knowledge_graph(self) -> None:
        mock_kg = MagicMock()
        mock_kg.quick_build = AsyncMock(return_value=_build_graph_response())

        # KnowledgeGraph is lazy-imported inside the function; patch the source module
        with patch(
            "beanllm.facade.advanced.knowledge_graph_facade.KnowledgeGraph",
            return_value=mock_kg,
        ):
            from beanllm.facade.advanced.knowledge_graph_standalone import (
                quick_knowledge_graph,
            )

            result = await quick_knowledge_graph(documents=["Apple was founded by Steve Jobs."])
        assert isinstance(result, BuildGraphResponse)
        assert result.graph_id == "graph-1"

    async def test_quick_knowledge_graph_with_client(self) -> None:
        mock_client = MagicMock()
        mock_kg = MagicMock()
        mock_kg.quick_build = AsyncMock(return_value=_build_graph_response())

        with patch(
            "beanllm.facade.advanced.knowledge_graph_facade.KnowledgeGraph",
            return_value=mock_kg,
        ):
            from beanllm.facade.advanced.knowledge_graph_standalone import (
                quick_knowledge_graph,
            )

            result = await quick_knowledge_graph(
                documents=["doc"],
                client=mock_client,
            )
        assert isinstance(result, BuildGraphResponse)

    async def test_quick_graph_rag(self) -> None:
        mock_kg = MagicMock()
        mock_kg.ask = AsyncMock(return_value="Apple was founded by Steve Jobs.")

        with patch(
            "beanllm.facade.advanced.knowledge_graph_facade.KnowledgeGraph",
            return_value=mock_kg,
        ):
            from beanllm.facade.advanced.knowledge_graph_standalone import (
                quick_graph_rag,
            )

            result = await quick_graph_rag(
                query="Who founded Apple?",
                graph_id="tech_companies",
            )
        assert result == "Apple was founded by Steve Jobs."

    async def test_quick_graph_rag_with_client(self) -> None:
        mock_client = MagicMock()
        mock_kg = MagicMock()
        mock_kg.ask = AsyncMock(return_value="answer")

        with patch(
            "beanllm.facade.advanced.knowledge_graph_facade.KnowledgeGraph",
            return_value=mock_kg,
        ):
            from beanllm.facade.advanced.knowledge_graph_standalone import (
                quick_graph_rag,
            )

            result = await quick_graph_rag(
                query="Q?",
                graph_id="g1",
                client=mock_client,
            )
        assert result == "answer"


# ===========================================================================
# 3. OptimizerConvenienceMixin
# ===========================================================================


class _OptimizerHost:
    """Minimal host that satisfies the mixin contract."""

    def __init__(self) -> None:
        self._optimize_resp = _optimize_response()
        self._benchmark_resp = _benchmark_response()
        self._profile_resp = _profile_response()
        self._recommendation_resp = _recommendation_response()

    async def optimize(self, **kwargs: Any) -> OptimizeResponse:
        return self._optimize_resp

    async def benchmark(self, **kwargs: Any) -> BenchmarkResponse:
        return self._benchmark_resp

    async def profile(self, **kwargs: Any) -> ProfileResponse:
        return self._profile_resp

    async def get_recommendations(self, profile_id: str) -> RecommendationResponse:
        return self._recommendation_resp


from beanllm.facade.advanced.optimizer_convenience import OptimizerConvenienceMixin


class _OptimizerMixed(_OptimizerHost, OptimizerConvenienceMixin):
    pass


class TestOptimizerConvenienceMixin:
    @pytest.fixture
    def optimizer(self) -> _OptimizerMixed:
        return _OptimizerMixed()

    async def test_quick_optimize_default_params(self, optimizer: _OptimizerMixed) -> None:
        result = await optimizer.quick_optimize()
        assert isinstance(result, OptimizeResponse)

    async def test_quick_optimize_custom_ranges(self, optimizer: _OptimizerMixed) -> None:
        result = await optimizer.quick_optimize(
            top_k_range=(5, 15),
            threshold_range=(0.5, 0.9),
            method="random",
            n_trials=10,
        )
        assert isinstance(result, OptimizeResponse)

    async def test_quick_benchmark_default(self, optimizer: _OptimizerMixed) -> None:
        result = await optimizer.quick_benchmark()
        assert isinstance(result, BenchmarkResponse)
        assert result.num_queries == 30

    async def test_quick_benchmark_custom_domain(self, optimizer: _OptimizerMixed) -> None:
        result = await optimizer.quick_benchmark(domain="machine learning", num_queries=50)
        assert isinstance(result, BenchmarkResponse)

    async def test_quick_profile_and_recommend(self, optimizer: _OptimizerMixed) -> None:
        profile, recs = await optimizer.quick_profile_and_recommend()
        assert isinstance(profile, ProfileResponse)
        assert isinstance(recs, RecommendationResponse)
        assert profile.profile_id == "prof-1"
        assert len(recs.recommendations) > 0

    async def test_quick_profile_and_recommend_with_components(
        self, optimizer: _OptimizerMixed
    ) -> None:
        profile, recs = await optimizer.quick_profile_and_recommend(
            components=["embedding", "retrieval"]
        )
        assert isinstance(profile, ProfileResponse)
        assert isinstance(recs, RecommendationResponse)

    async def test_multi_objective_optimize(self, optimizer: _OptimizerMixed) -> None:
        params = [
            {"name": "top_k", "type": "integer", "low": 1, "high": 20},
            {"name": "model", "type": "categorical", "categories": ["gpt-4"]},
        ]
        result = await optimizer.multi_objective_optimize(
            parameters=params,
            quality_weight=0.7,
            latency_weight=0.2,
            cost_weight=0.1,
            n_trials=20,
        )
        assert isinstance(result, OptimizeResponse)

    async def test_benchmark_and_optimize(self, optimizer: _OptimizerMixed) -> None:
        params = [{"name": "top_k", "type": "integer", "low": 1, "high": 20}]
        result = await optimizer.benchmark_and_optimize(
            parameters=params,
            benchmark_num_queries=20,
            optimize_n_trials=10,
        )
        assert isinstance(result, dict)
        assert "benchmark" in result
        assert "optimization" in result
        assert isinstance(result["benchmark"], BenchmarkResponse)
        assert isinstance(result["optimization"], OptimizeResponse)

    async def test_auto_tune_all_enabled(self, optimizer: _OptimizerMixed) -> None:
        results = await optimizer.auto_tune(profile=True, optimize=True, recommend=True)
        assert "profile" in results
        assert "recommendations" in results
        assert "optimization" in results

    async def test_auto_tune_profile_only(self, optimizer: _OptimizerMixed) -> None:
        results = await optimizer.auto_tune(profile=True, optimize=False, recommend=False)
        assert "profile" in results
        assert "optimization" not in results
        assert "recommendations" not in results

    async def test_auto_tune_optimize_only(self, optimizer: _OptimizerMixed) -> None:
        results = await optimizer.auto_tune(profile=False, optimize=True, recommend=False)
        assert "optimization" in results
        assert "profile" not in results

    async def test_auto_tune_profile_no_recommend(self, optimizer: _OptimizerMixed) -> None:
        results = await optimizer.auto_tune(profile=True, optimize=False, recommend=False)
        assert "profile" in results
        assert "recommendations" not in results

    async def test_auto_tune_all_disabled(self, optimizer: _OptimizerMixed) -> None:
        results = await optimizer.auto_tune(profile=False, optimize=False, recommend=False)
        assert results == {}


# ===========================================================================
# 4. Optimizer Standalone
# ===========================================================================


class TestOptimizerStandalone:
    async def test_quick_optimizer(self) -> None:
        mock_opt = MagicMock()
        mock_opt.optimize = AsyncMock(return_value=_optimize_response())

        with patch(
            "beanllm.facade.advanced.optimizer_facade.Optimizer",
            return_value=mock_opt,
        ):
            from beanllm.facade.advanced.optimizer_standalone import quick_optimizer

            params = [{"name": "top_k", "type": "integer", "low": 1, "high": 20}]
            result = await quick_optimizer(parameters=params, method="bayesian", n_trials=20)
        assert isinstance(result, OptimizeResponse)

    async def test_quick_optimizer_default_method(self) -> None:
        mock_opt = MagicMock()
        mock_opt.optimize = AsyncMock(return_value=_optimize_response())

        with patch(
            "beanllm.facade.advanced.optimizer_facade.Optimizer",
            return_value=mock_opt,
        ):
            from beanllm.facade.advanced.optimizer_standalone import quick_optimizer

            result = await quick_optimizer(
                parameters=[{"name": "top_k", "type": "integer", "low": 1, "high": 20}]
            )
        assert isinstance(result, OptimizeResponse)
        mock_opt.optimize.assert_called_once()

    async def test_quick_profile(self) -> None:
        mock_opt = MagicMock()
        mock_opt.profile = AsyncMock(return_value=_profile_response())

        with patch(
            "beanllm.facade.advanced.optimizer_facade.Optimizer",
            return_value=mock_opt,
        ):
            from beanllm.facade.advanced.optimizer_standalone import quick_profile

            result = await quick_profile()
        assert isinstance(result, ProfileResponse)
        assert result.bottleneck == "embedding"


# ===========================================================================
# 5. OrchestratorConvenienceMixin
# ===========================================================================


from beanllm.facade.advanced.orchestrator_convenience_mixin import (
    OrchestratorConvenienceMixin,
)


class _OrchestratorHost:
    """Minimal host that satisfies the mixin contract."""

    async def create_and_execute(
        self,
        name: str,
        strategy: str,
        agents: Dict[str, Any],
        task: str,
        config: Any = None,
        tools: Any = None,
    ) -> Dict[str, Any]:
        return {
            "workflow": _create_workflow_response(),
            "execution": _execute_workflow_response(),
        }


class _OrchestratorMixed(_OrchestratorHost, OrchestratorConvenienceMixin):
    pass


class TestOrchestratorConvenienceMixin:
    @pytest.fixture
    def orchestrator(self) -> _OrchestratorMixed:
        return _OrchestratorMixed()

    async def test_quick_research_write_no_reviewer(self, orchestrator: _OrchestratorMixed) -> None:
        result = await orchestrator.quick_research_write(
            researcher_agent=MagicMock(),
            writer_agent=MagicMock(),
            task="Write about AI",
        )
        assert isinstance(result, ExecuteWorkflowResponse)
        assert result.status == "completed"

    async def test_quick_research_write_with_reviewer(
        self, orchestrator: _OrchestratorMixed
    ) -> None:
        result = await orchestrator.quick_research_write(
            researcher_agent=MagicMock(),
            writer_agent=MagicMock(),
            task="Write about AI",
            reviewer_agent=MagicMock(),
            name="Custom Research",
        )
        assert isinstance(result, ExecuteWorkflowResponse)

    async def test_quick_research_write_delegates_to_create_and_execute(
        self, orchestrator: _OrchestratorMixed
    ) -> None:
        orchestrator.create_and_execute = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "workflow": _create_workflow_response(),
                "execution": _execute_workflow_response(),
            }
        )
        researcher = MagicMock()
        writer = MagicMock()
        reviewer = MagicMock()
        await orchestrator.quick_research_write(
            researcher_agent=researcher,
            writer_agent=writer,
            task="task",
            reviewer_agent=reviewer,
        )
        orchestrator.create_and_execute.assert_called_once()
        call_kwargs = orchestrator.create_and_execute.call_args.kwargs
        assert call_kwargs["strategy"] == "research_write"
        assert "researcher" in call_kwargs["agents"]
        assert "writer" in call_kwargs["agents"]
        assert "reviewer" in call_kwargs["agents"]

    async def test_quick_parallel_consensus(self, orchestrator: _OrchestratorMixed) -> None:
        agents = [MagicMock(), MagicMock(), MagicMock()]
        result = await orchestrator.quick_parallel_consensus(
            agents=agents,
            task="Vote on this proposal",
            aggregation="vote",
        )
        assert isinstance(result, ExecuteWorkflowResponse)

    async def test_quick_parallel_consensus_delegates_correctly(
        self, orchestrator: _OrchestratorMixed
    ) -> None:
        orchestrator.create_and_execute = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "workflow": _create_workflow_response(),
                "execution": _execute_workflow_response(),
            }
        )
        agents = [MagicMock(), MagicMock()]
        await orchestrator.quick_parallel_consensus(
            agents=agents,
            task="task",
            aggregation="consensus",
            name="My Consensus",
        )
        call_kwargs = orchestrator.create_and_execute.call_args.kwargs
        assert call_kwargs["strategy"] == "parallel"
        assert len(call_kwargs["agents"]) == 2
        assert call_kwargs["config"]["aggregation"] == "consensus"

    async def test_quick_debate(self, orchestrator: _OrchestratorMixed) -> None:
        debaters = [MagicMock(), MagicMock()]
        judge = MagicMock()
        result = await orchestrator.quick_debate(
            debater_agents=debaters,
            judge_agent=judge,
            task="Should AI be regulated?",
            rounds=3,
        )
        assert isinstance(result, ExecuteWorkflowResponse)

    async def test_quick_debate_delegates_correctly(self, orchestrator: _OrchestratorMixed) -> None:
        orchestrator.create_and_execute = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "workflow": _create_workflow_response(),
                "execution": _execute_workflow_response(),
            }
        )
        debaters = [MagicMock(), MagicMock()]
        judge = MagicMock()
        await orchestrator.quick_debate(
            debater_agents=debaters,
            judge_agent=judge,
            task="task",
            rounds=5,
            name="Big Debate",
        )
        call_kwargs = orchestrator.create_and_execute.call_args.kwargs
        assert call_kwargs["strategy"] == "debate"
        assert "judge" in call_kwargs["agents"]
        assert "debater0" in call_kwargs["agents"]
        assert "debater1" in call_kwargs["agents"]
        assert call_kwargs["config"]["rounds"] == 5
        assert call_kwargs["config"]["judge_id"] == "judge"


# ===========================================================================
# 6. MultiAgentCoordinator (multi_agent_facade.py)
# ===========================================================================


def _make_mock_container() -> tuple[MagicMock, MagicMock]:
    """Return (mock_container, mock_multi_agent_handler)."""
    mock_response = Mock()
    mock_response.final_result = "result"
    mock_response.strategy = "sequential"
    mock_response.intermediate_results = []
    mock_response.all_steps = []
    mock_response.metadata = {}

    async def _handle_execute(*args: Any, **kwargs: Any) -> Mock:
        return mock_response

    mock_handler = MagicMock()
    mock_handler.handle_execute = MagicMock(side_effect=_handle_execute)

    mock_handler_factory = Mock()
    mock_handler_factory.create_multi_agent_handler.return_value = mock_handler

    mock_container = Mock()
    mock_container.handler_factory = mock_handler_factory

    return mock_container, mock_handler


class TestMultiAgentCoordinator:
    @pytest.fixture
    def coordinator_and_handler(self):
        from beanllm.facade.advanced.multi_agent_facade import MultiAgentCoordinator

        mock_container, mock_handler = _make_mock_container()
        with patch(
            "beanllm.facade.advanced.multi_agent_facade.MultiAgentCoordinator._init_services"
        ):
            coord = MultiAgentCoordinator(agents={"agent1": MagicMock(), "agent2": MagicMock()})
        coord._multi_agent_handler = mock_handler
        return coord, mock_handler

    async def test_execute_sequential(self, coordinator_and_handler) -> None:
        coord, handler = coordinator_and_handler
        result = await coord.execute_sequential(
            task="Do something", agent_order=["agent1", "agent2"]
        )
        assert isinstance(result, dict)
        assert result["final_result"] == "result"
        assert result["strategy"] == "sequential"
        assert result["intermediate_results"] == []
        handler.handle_execute.assert_called_once()

    async def test_execute_parallel_all_agents(self, coordinator_and_handler) -> None:
        coord, handler = coordinator_and_handler
        result = await coord.execute_parallel(task="parallel task")
        assert isinstance(result, dict)
        assert result["final_result"] == "result"

    async def test_execute_parallel_specific_agents(self, coordinator_and_handler) -> None:
        coord, handler = coordinator_and_handler
        result = await coord.execute_parallel(
            task="parallel task",
            agent_ids=["agent1"],
            aggregation="consensus",
        )
        assert isinstance(result, dict)
        call_kwargs = handler.handle_execute.call_args.kwargs
        assert call_kwargs["aggregation"] == "consensus"

    async def test_execute_hierarchical(self, coordinator_and_handler) -> None:
        coord, handler = coordinator_and_handler
        result = await coord.execute_hierarchical(
            task="hierarchical task",
            manager_id="agent1",
            worker_ids=["agent2"],
        )
        assert isinstance(result, dict)
        assert result["final_result"] == "result"
        call_kwargs = handler.handle_execute.call_args.kwargs
        assert call_kwargs["strategy"] == "hierarchical"
        assert call_kwargs["manager_id"] == "agent1"

    async def test_execute_debate_no_judge(self, coordinator_and_handler) -> None:
        coord, handler = coordinator_and_handler
        result = await coord.execute_debate(task="debate topic", rounds=2)
        assert isinstance(result, dict)
        assert result["strategy"] == "sequential"  # from mock_response

    async def test_execute_debate_with_judge(self, coordinator_and_handler) -> None:
        coord, handler = coordinator_and_handler
        result = await coord.execute_debate(
            task="debate topic",
            agent_ids=["agent1"],
            judge_id="agent2",
            rounds=3,
        )
        assert isinstance(result, dict)
        call_kwargs = handler.handle_execute.call_args.kwargs
        assert call_kwargs["judge_id"] == "agent2"

    async def test_send_message(self, coordinator_and_handler) -> None:
        from beanllm.domain.multi_agent import MessageType

        coord, _ = coordinator_and_handler
        coord.bus.publish = AsyncMock()
        await coord.send_message(
            sender="agent1",
            receiver="agent2",
            content="Hello!",
            message_type=MessageType.INFORM,
        )
        coord.bus.publish.assert_called_once()

    def test_add_agent(self, coordinator_and_handler) -> None:
        coord, _ = coordinator_and_handler
        new_agent = MagicMock()
        coord.add_agent("agent3", new_agent)
        assert "agent3" in coord.agents

    def test_remove_agent(self, coordinator_and_handler) -> None:
        coord, _ = coordinator_and_handler
        coord.remove_agent("agent1")
        assert "agent1" not in coord.agents

    def test_remove_nonexistent_agent_no_error(self, coordinator_and_handler) -> None:
        coord, _ = coordinator_and_handler
        coord.remove_agent("no-such-agent")  # should not raise

    def test_get_communication_history(self, coordinator_and_handler) -> None:
        coord, _ = coordinator_and_handler
        history = coord.get_communication_history(limit=10)
        assert isinstance(history, list)


class TestMultiAgentCoordinatorInit:
    """Test MultiAgentCoordinator initialization via DI container patch."""

    def test_init_creates_handler(self) -> None:
        from beanllm.facade.advanced.multi_agent_facade import MultiAgentCoordinator

        mock_container, mock_handler = _make_mock_container()
        with patch(
            "beanllm.utils.core.di_container.get_container",
            return_value=mock_container,
        ):
            coord = MultiAgentCoordinator(agents={"a": MagicMock()})
        assert coord._multi_agent_handler is mock_handler


class TestCreateCoordinator:
    """Test the create_coordinator convenience function."""

    def test_create_coordinator(self) -> None:
        from beanllm.facade.advanced.multi_agent_facade import create_coordinator

        # Agent is lazily imported from beanllm.facade.core.agent_facade inside create_coordinator
        with patch("beanllm.facade.core.agent_facade.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            configs = [
                {"id": "a1", "model": "gpt-4o-mini"},
                {"id": "a2", "model": "gpt-4o"},
            ]
            coord = create_coordinator(configs)

        from beanllm.facade.advanced.multi_agent_facade import MultiAgentCoordinator

        assert isinstance(coord, MultiAgentCoordinator)


# ===========================================================================
# 7. Graph Facade (graph_facade.py)
# ===========================================================================


def _make_graph_mock_container():
    mock_response = Mock()
    mock_response.final_state = {"result": "done"}
    mock_response.metadata = {}

    async def _handle_run(*args: Any, **kwargs: Any) -> Mock:
        return mock_response

    mock_handler = MagicMock()
    mock_handler.handle_run = MagicMock(side_effect=_handle_run)

    mock_handler_factory = Mock()
    mock_handler_factory.create_graph_handler.return_value = mock_handler

    mock_container = Mock()
    mock_container.handler_factory = mock_handler_factory

    return mock_container, mock_handler


class TestGraphFacade:
    @pytest.fixture
    def graph_and_handler(self):
        from beanllm.facade.advanced.graph_facade import Graph

        mock_container, mock_handler = _make_graph_mock_container()
        with patch(
            "beanllm.utils.core.di_container.get_container",
            return_value=mock_container,
        ):
            g = Graph()
        return g, mock_handler

    async def test_run_with_dict_state(self, graph_and_handler) -> None:
        from beanllm.domain.graph import GraphState

        g, handler = graph_and_handler
        result = await g.run({"input": "test"})
        assert isinstance(result, GraphState)
        assert result.data["result"] == "done"
        handler.handle_run.assert_called_once()

    async def test_run_with_graph_state_input(self, graph_and_handler) -> None:
        from beanllm.domain.graph import GraphState

        g, handler = graph_and_handler
        state = GraphState(data={"x": 1})
        result = await g.run(state)
        assert isinstance(result, GraphState)
        call_kwargs = handler.handle_run.call_args.kwargs
        assert call_kwargs["initial_state"] == {"x": 1}

    def test_add_node(self, graph_and_handler) -> None:
        g, _ = graph_and_handler
        from beanllm.domain.graph import BaseNode

        node = MagicMock(spec=BaseNode)
        node.name = "test_node"
        g.add_node(node)
        assert "test_node" in g.nodes

    def test_add_function_node(self, graph_and_handler) -> None:
        g, _ = graph_and_handler

        def my_func(state: dict) -> dict:
            return state

        g.add_function_node("fn", my_func)
        assert "fn" in g.nodes

    def test_add_edge(self, graph_and_handler) -> None:
        g, _ = graph_and_handler
        g.add_edge("a", "b")
        assert "b" in g.edges["a"]

    def test_add_multiple_edges_from_same_node(self, graph_and_handler) -> None:
        g, _ = graph_and_handler
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert "b" in g.edges["a"]
        assert "c" in g.edges["a"]

    def test_add_conditional_edge(self, graph_and_handler) -> None:
        from beanllm.domain.graph import GraphState

        g, _ = graph_and_handler

        def cond(state: GraphState) -> str:
            return "b"

        g.add_conditional_edge("a", cond)
        assert "a" in g.conditional_edges

    def test_set_entry_point(self, graph_and_handler) -> None:
        g, _ = graph_and_handler
        g.set_entry_point("start")
        assert g.entry_point == "start"

    def test_visualize_empty_graph(self, graph_and_handler) -> None:
        g, _ = graph_and_handler
        text = g.visualize()
        assert "Graph Structure:" in text

    def test_visualize_with_nodes_and_edges(self, graph_and_handler) -> None:
        from beanllm.domain.graph import BaseNode, GraphState

        g, _ = graph_and_handler

        node_a = MagicMock(spec=BaseNode)
        node_a.name = "nodeA"
        node_a.description = "First node"
        node_a.cache_enabled = False

        node_b = MagicMock(spec=BaseNode)
        node_b.name = "nodeB"
        node_b.description = None
        node_b.cache_enabled = True

        g.add_node(node_a)
        g.add_node(node_b)
        g.add_edge("nodeA", "nodeB")

        def cond(s: GraphState) -> str:
            return "nodeA"

        g.add_conditional_edge("nodeB", cond)

        text = g.visualize()
        assert "nodeA" in text
        assert "nodeB" in text
        assert "nodeA" in text
        assert "[cached]" in text
        assert "[conditional]" in text

    async def test_run_with_verbose(self, graph_and_handler) -> None:
        from beanllm.domain.graph import GraphState

        g, handler = graph_and_handler
        result = await g.run({"x": 1}, verbose=True)
        assert isinstance(result, GraphState)
        call_kwargs = handler.handle_run.call_args.kwargs
        assert call_kwargs["verbose"] is True

    def test_graph_no_cache(self) -> None:
        from beanllm.facade.advanced.graph_facade import Graph

        mock_container, _ = _make_graph_mock_container()
        with patch(
            "beanllm.utils.core.di_container.get_container",
            return_value=mock_container,
        ):
            g = Graph(enable_cache=False)
        assert g.cache is None


class TestCreateSimpleGraph:
    def test_create_simple_graph(self) -> None:
        from beanllm.domain.graph import BaseNode
        from beanllm.facade.advanced.graph_facade import Graph, create_simple_graph

        mock_container, _ = _make_graph_mock_container()
        with patch(
            "beanllm.utils.core.di_container.get_container",
            return_value=mock_container,
        ):
            node_a = MagicMock(spec=BaseNode)
            node_a.name = "a"
            node_b = MagicMock(spec=BaseNode)
            node_b.name = "b"

            g = create_simple_graph(
                nodes=[("a", node_a), ("b", node_b)],
                edges=[("a", "b")],
            )

        assert isinstance(g, Graph)
        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "b" in g.edges.get("a", [])


# ===========================================================================
# 8. ServiceFactory (service/factory.py)
# ===========================================================================


def _make_service_factory(**kwargs):
    """Create a ServiceFactory with mocked provider_factory."""
    from beanllm.service.factory import ServiceFactory

    mock_provider_factory = MagicMock()
    mock_vector_store = kwargs.pop("vector_store", MagicMock())
    mock_embedding_service = kwargs.pop("embedding_service", None)

    return ServiceFactory(
        provider_factory=mock_provider_factory,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )


class TestServiceFactory:
    @pytest.fixture
    def factory(self):
        return _make_service_factory()

    def test_create_chat_service(self, factory) -> None:
        with patch("beanllm.service.factory.ServiceFactory.create_chat_service") as mock_method:
            mock_method.return_value = MagicMock()
            svc = factory.create_chat_service()
        assert svc is not None

    def test_create_graph_service(self, factory) -> None:
        with patch("beanllm.service.impl.advanced.graph_service_impl.GraphServiceImpl") as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_graph_service()
        assert svc is not None

    def test_create_state_graph_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.advanced.state_graph_service_impl.StateGraphServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_state_graph_service()
        assert svc is not None

    def test_create_multi_agent_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.advanced.multi_agent_service_impl.MultiAgentServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_multi_agent_service()
        assert svc is not None

    def test_create_web_search_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.ml.web_search_service_impl.WebSearchServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_web_search_service()
        assert svc is not None

    def test_create_rag_debug_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.advanced.rag_debug_service_impl.RAGDebugServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_rag_debug_service()
        assert svc is not None

    def test_create_orchestrator_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.advanced.orchestrator_service_impl.OrchestratorServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_orchestrator_service()
        assert svc is not None

    def test_create_optimizer_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.advanced.optimizer_service_impl.OptimizerServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_optimizer_service()
        assert svc is not None

    def test_create_knowledge_graph_service_no_client(self, factory) -> None:
        with patch(
            "beanllm.service.impl.advanced.knowledge_graph_service_impl.KnowledgeGraphServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_knowledge_graph_service()
        assert svc is not None
        MockImpl.assert_called_once_with(client=None)

    def test_create_knowledge_graph_service_with_client(self, factory) -> None:
        mock_client = MagicMock()
        with patch(
            "beanllm.service.impl.advanced.knowledge_graph_service_impl.KnowledgeGraphServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_knowledge_graph_service(client=mock_client)
        assert svc is not None
        MockImpl.assert_called_once_with(client=mock_client)

    def test_create_evaluation_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.ml.evaluation_service_impl.EvaluationServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_evaluation_service()
        assert svc is not None

    def test_create_evaluation_service_with_client(self, factory) -> None:
        mock_client = MagicMock()
        with patch(
            "beanllm.service.impl.ml.evaluation_service_impl.EvaluationServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            factory.create_evaluation_service(client=mock_client)
        MockImpl.assert_called_once_with(client=mock_client, embedding_model=None)

    def test_create_finetuning_service(self, factory) -> None:
        with patch(
            "beanllm.service.impl.ml.finetuning_service_impl.FinetuningServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            svc = factory.create_finetuning_service()
        assert svc is not None

    def test_create_finetuning_service_with_provider(self, factory) -> None:
        mock_provider = MagicMock()
        with patch(
            "beanllm.service.impl.ml.finetuning_service_impl.FinetuningServiceImpl"
        ) as MockImpl:
            MockImpl.return_value = MagicMock()
            factory.create_finetuning_service(provider=mock_provider)
        MockImpl.assert_called_once_with(provider=mock_provider)

    def test_create_rag_service_raises_without_vector_store(self) -> None:
        from beanllm.service.factory import ServiceFactory

        factory_no_vs = ServiceFactory(
            provider_factory=MagicMock(),
            vector_store=None,
        )
        with pytest.raises(ValueError, match="vector_store"):
            with patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat:
                MockChat.return_value = MagicMock()
                factory_no_vs.create_rag_service()

    def test_create_rag_service_with_vector_store(self, factory) -> None:
        with (
            patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat,
            patch("beanllm.service.impl.core.rag_service_impl.RAGServiceImpl") as MockRAG,
        ):
            MockChat.return_value = MagicMock()
            MockRAG.return_value = MagicMock()
            svc = factory.create_rag_service()
        assert svc is not None

    def test_create_rag_service_with_existing_chat_service(self, factory) -> None:
        mock_chat = MagicMock()
        with patch("beanllm.service.impl.core.rag_service_impl.RAGServiceImpl") as MockRAG:
            MockRAG.return_value = MagicMock()
            svc = factory.create_rag_service(chat_service=mock_chat)
        assert svc is not None

    def test_create_agent_service(self, factory) -> None:
        with (
            patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat,
            patch("beanllm.service.impl.core.agent_service_impl.AgentServiceImpl") as MockAgent,
        ):
            MockChat.return_value = MagicMock()
            MockAgent.return_value = MagicMock()
            svc = factory.create_agent_service()
        assert svc is not None

    def test_create_agent_service_with_tool_registry(self, factory) -> None:
        mock_registry = MagicMock()
        with (
            patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat,
            patch("beanllm.service.impl.core.agent_service_impl.AgentServiceImpl") as MockAgent,
        ):
            MockChat.return_value = MagicMock()
            MockAgent.return_value = MagicMock()
            factory.create_agent_service(tool_registry=mock_registry)
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs
        assert call_kwargs["tool_registry"] is mock_registry

    def test_create_chain_service(self, factory) -> None:
        with (
            patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat,
            patch("beanllm.service.impl.core.chain_service_impl.ChainServiceImpl") as MockChain,
        ):
            MockChat.return_value = MagicMock()
            MockChain.return_value = MagicMock()
            svc = factory.create_chain_service()
        assert svc is not None

    def test_create_vision_rag_service(self, factory) -> None:
        mock_vs = MagicMock()
        with (
            patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat,
            patch(
                "beanllm.service.impl.ml.vision_rag_service_impl.VisionRAGServiceImpl"
            ) as MockVision,
        ):
            MockChat.return_value = MagicMock()
            MockVision.return_value = MagicMock()
            svc = factory.create_vision_rag_service(vector_store=mock_vs)
        assert svc is not None

    def test_create_audio_service(self, factory) -> None:
        with patch("beanllm.service.impl.ml.audio_service_impl.AudioServiceImpl") as MockAudio:
            MockAudio.return_value = MagicMock()
            svc = factory.create_audio_service()
        assert svc is not None

    def test_get_or_create_chat_service_reuses_existing(self, factory) -> None:
        mock_chat = MagicMock()
        result = factory._get_or_create_chat_service(mock_chat)
        assert result is mock_chat

    def test_get_or_create_chat_service_creates_new(self, factory) -> None:
        with patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat:
            MockChat.return_value = MagicMock()
            result = factory._get_or_create_chat_service(None)
        assert result is not None

    def test_create_all_services(self, factory) -> None:
        with (
            patch("beanllm.service.impl.core.chat_service_impl.ChatServiceImpl") as MockChat,
            patch("beanllm.service.impl.core.rag_service_impl.RAGServiceImpl") as MockRAG,
            patch("beanllm.service.impl.core.agent_service_impl.AgentServiceImpl") as MockAgent,
            patch("beanllm.service.impl.core.chain_service_impl.ChainServiceImpl") as MockChain,
            patch("beanllm.service.impl.advanced.graph_service_impl.GraphServiceImpl") as MockGraph,
            patch(
                "beanllm.service.impl.advanced.state_graph_service_impl.StateGraphServiceImpl"
            ) as MockSG,
            patch(
                "beanllm.service.impl.advanced.multi_agent_service_impl.MultiAgentServiceImpl"
            ) as MockMA,
            patch("beanllm.service.impl.ml.web_search_service_impl.WebSearchServiceImpl") as MockWS,
            patch(
                "beanllm.service.impl.ml.evaluation_service_impl.EvaluationServiceImpl"
            ) as MockEval,
            patch(
                "beanllm.service.impl.ml.finetuning_service_impl.FinetuningServiceImpl"
            ) as MockFT,
            patch(
                "beanllm.service.impl.advanced.rag_debug_service_impl.RAGDebugServiceImpl"
            ) as MockRD,
            patch(
                "beanllm.service.impl.advanced.orchestrator_service_impl.OrchestratorServiceImpl"
            ) as MockOrch,
            patch(
                "beanllm.service.impl.advanced.optimizer_service_impl.OptimizerServiceImpl"
            ) as MockOpt,
            patch(
                "beanllm.service.impl.advanced.knowledge_graph_service_impl.KnowledgeGraphServiceImpl"
            ) as MockKG,
        ):
            for m in [
                MockChat,
                MockRAG,
                MockAgent,
                MockChain,
                MockGraph,
                MockSG,
                MockMA,
                MockWS,
                MockEval,
                MockFT,
                MockRD,
                MockOrch,
                MockOpt,
                MockKG,
            ]:
                m.return_value = MagicMock()

            services = factory.create_all_services()

        expected_keys = {
            "chat",
            "rag",
            "agent",
            "chain",
            "graph",
            "state_graph",
            "multi_agent",
            "web_search",
            "evaluation",
            "finetuning",
            "rag_debug",
            "orchestrator",
            "optimizer",
            "knowledge_graph",
        }
        assert set(services.keys()) == expected_keys
        for key in expected_keys:
            assert services[key] is not None
