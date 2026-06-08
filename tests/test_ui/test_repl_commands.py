"""Tests for REPL command classes."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_progress_patch(module_path: str):
    """Return a patch context that replaces rich.Progress in a command module."""
    mock_progress = MagicMock()
    mock_progress.__enter__ = lambda s: s
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.update = MagicMock()
    return patch(f"{module_path}.Progress", return_value=mock_progress)


# ── KnowledgeGraphCommands ─────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.knowledge_graph_commands import KnowledgeGraphCommands

    KG_CMD_AVAILABLE = True
except ImportError:
    KG_CMD_AVAILABLE = False


@pytest.mark.skipif(not KG_CMD_AVAILABLE, reason="KnowledgeGraphCommands not available")
class TestKnowledgeGraphCommands:
    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.build_graph = AsyncMock(
            return_value=MagicMock(
                graph_id="g-001",
                num_nodes=10,
                num_edges=15,
                build_time=1.2,
                entity_types=["person", "org"],
            )
        )
        kg.query_graph = AsyncMock(
            return_value=MagicMock(
                results=[{"id": "e1", "name": "Alice", "type": "person"}],
                query_type="find_entities_by_type",
                execution_time=0.05,
            )
        )
        kg.graph_rag = AsyncMock(
            return_value=MagicMock(
                answer="Alice is a person.",
                sources=["doc1"],
                entities_used=["Alice"],
                relations_used=[],
                execution_time=0.3,
            )
        )
        kg.get_graph_stats = AsyncMock(
            return_value=MagicMock(
                graph_id="g-001",
                stats={
                    "graph_id": "g-001",
                    "num_nodes": 10,
                    "num_edges": 15,
                    "density": 0.3,
                    "average_degree": 3.0,
                    "num_connected_components": 1,
                },
            )
        )
        kg.list_graphs = AsyncMock(
            return_value=MagicMock(
                graphs=[
                    MagicMock(graph_id="g-001", created_at="2026-01-01", num_nodes=10, num_edges=15)
                ],
            )
        )
        return kg

    @pytest.fixture
    def cmd(self, mock_kg):
        console = MagicMock()
        return KnowledgeGraphCommands(knowledge_graph=mock_kg, console=console)

    async def test_build_graph_no_documents(self, cmd):
        await cmd.cmd_build_graph(documents=None)
        cmd.console.print.assert_called()

    async def test_build_graph_with_documents(self, cmd, mock_kg):
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_build_graph(
                documents=["Alice founded Apple in 1976."],
                graph_id="tech",
                entity_types=["person", "org"],
            )
        mock_kg.build_graph.assert_called_once()

    async def test_build_graph_exception(self, cmd, mock_kg):
        mock_kg.build_graph.side_effect = Exception("LLM error")
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_build_graph(documents=["some text"])
        cmd.console.print.assert_called()

    async def test_query_graph(self, cmd, mock_kg):
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_query(
                graph_id="g-001",
                query_type="find_entities_by_type",
                entity_type="person",
            )
        mock_kg.query_graph.assert_called_once()

    async def test_query_graph_exception(self, cmd, mock_kg):
        mock_kg.query_graph.side_effect = Exception("query failed")
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_query(
                graph_id="g-001", query_type="find_entities_by_type", entity_type="person"
            )
        cmd.console.print.assert_called()

    async def test_graph_rag(self, cmd, mock_kg):
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_graph_rag(query="Who is Alice?", graph_id="g-001")
        mock_kg.graph_rag.assert_called_once()

    async def test_graph_rag_exception(self, cmd, mock_kg):
        mock_kg.graph_rag.side_effect = Exception("rag error")
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_graph_rag(query="Who?", graph_id="g-001")
        cmd.console.print.assert_called()

    async def test_cmd_stats(self, cmd, mock_kg):
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_stats(graph_id="g-001")
        mock_kg.get_graph_stats.assert_called_once()

    async def test_cmd_stats_exception(self, cmd, mock_kg):
        mock_kg.get_graph_stats.side_effect = Exception("stats error")
        module = "beanllm.ui.repl.knowledge_graph_commands"
        with _make_progress_patch(module):
            await cmd.cmd_stats(graph_id="g-001")
        cmd.console.print.assert_called()

    async def test_cmd_list_graphs(self, cmd, mock_kg):
        service = MagicMock()
        service.list_graphs = AsyncMock(
            return_value=[{"id": "g-001", "num_nodes": 5, "num_edges": 3}]
        )
        cmd._kg._handler._service = service
        with patch("asyncio.run", return_value=[{"id": "g-001", "num_nodes": 5, "num_edges": 3}]):
            await cmd.cmd_list_graphs()
        cmd.console.print.assert_called()

    async def test_cmd_list_graphs_exception(self, cmd):
        cmd._kg._handler._service = MagicMock()
        with patch("asyncio.run", side_effect=Exception("list error")):
            await cmd.cmd_list_graphs()
        cmd.console.print.assert_called()


# ── OptimizerCommands ──────────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.optimizer_commands import OptimizerCommands

    OPT_CMD_AVAILABLE = True
except ImportError:
    OPT_CMD_AVAILABLE = False


@pytest.mark.skipif(not OPT_CMD_AVAILABLE, reason="OptimizerCommands not available")
class TestOptimizerCommands:
    @pytest.fixture
    def mock_optimizer(self):
        opt = MagicMock()
        opt.benchmark = AsyncMock(
            return_value=MagicMock(
                benchmark_id="bench-001",
                num_queries=50,
                avg_latency=0.2,
                p50_latency=0.18,
                p95_latency=0.45,
                p99_latency=0.9,
                throughput=5.0,
                avg_score=0.85,
                min_score=0.6,
                max_score=0.98,
                total_duration=10.0,
                queries=["q1", "q2"],
            )
        )
        opt.optimize = AsyncMock(
            return_value=MagicMock(
                optimization_id="opt-001",
                best_score=0.92,
                n_trials=20,
                best_params={"top_k": 5},
                convergence_data=[],
            )
        )
        opt.profile = AsyncMock(
            return_value=MagicMock(
                profile_id="prof-001",
                total_duration_ms=250.0,
                total_tokens=1500,
                total_cost=0.003,
                bottleneck="embedding",
                components=[],
                breakdown={},
                recommendations=[],
            )
        )
        opt.ab_test = AsyncMock(
            return_value=MagicMock(
                winner="B",
                lift=9.0,
                p_value=0.02,
                is_significant=True,
                confidence_level=0.95,
                variant_a_mean=0.75,
                variant_b_mean=0.82,
            )
        )
        opt.get_recommendations = AsyncMock(
            return_value=MagicMock(
                recommendations=["tip1"],
                summary={"critical": 0, "high": 1, "medium": 2, "low": 0},
            )
        )
        return opt

    @pytest.fixture
    def cmd(self, mock_optimizer):
        console = MagicMock()
        return OptimizerCommands(optimizer=mock_optimizer, console=console)

    async def test_benchmark_basic(self, cmd, mock_optimizer):
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_benchmark(num_queries=10)
        mock_optimizer.benchmark.assert_called_once()

    async def test_benchmark_exception(self, cmd, mock_optimizer):
        mock_optimizer.benchmark.side_effect = Exception("bench error")
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_benchmark(num_queries=10)
        cmd.console.print.assert_called()

    async def test_optimize_basic(self, cmd, mock_optimizer):
        params = [{"name": "top_k", "type": "integer", "low": 1, "high": 20}]
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_optimize(parameters=params)
        mock_optimizer.optimize.assert_called_once()

    async def test_optimize_exception(self, cmd, mock_optimizer):
        mock_optimizer.optimize.side_effect = Exception("opt error")
        params = [{"name": "top_k", "type": "integer", "low": 1, "high": 20}]
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_optimize(parameters=params)
        cmd.console.print.assert_called()

    async def test_profile_basic(self, cmd, mock_optimizer):
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_profile()
        mock_optimizer.profile.assert_called_once()

    async def test_profile_exception(self, cmd, mock_optimizer):
        mock_optimizer.profile.side_effect = Exception("profile error")
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_profile()
        cmd.console.print.assert_called()

    async def test_ab_test_basic(self, cmd, mock_optimizer):
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_ab_test(
                variant_a_name="baseline",
                variant_b_name="new",
                num_queries=20,
            )
        mock_optimizer.ab_test.assert_called_once()

    async def test_ab_test_exception(self, cmd, mock_optimizer):
        mock_optimizer.ab_test.side_effect = Exception("ab error")
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_ab_test(variant_a_name="A", variant_b_name="B")
        cmd.console.print.assert_called()

    async def test_recommendations_basic(self, cmd, mock_optimizer):
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_recommendations(profile_id="prof-001")
        mock_optimizer.get_recommendations.assert_called_once()

    async def test_recommendations_exception(self, cmd, mock_optimizer):
        mock_optimizer.get_recommendations.side_effect = Exception("rec error")
        module = "beanllm.ui.repl.optimizer_commands"
        with _make_progress_patch(module):
            await cmd.cmd_recommendations(profile_id="prof-001")
        cmd.console.print.assert_called()


# ── OrchestratorCommands ───────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.orchestrator_commands import OrchestratorCommands

    ORCH_CMD_AVAILABLE = True
except ImportError:
    ORCH_CMD_AVAILABLE = False


@pytest.mark.skipif(not ORCH_CMD_AVAILABLE, reason="OrchestratorCommands not available")
class TestOrchestratorCommands:
    @pytest.fixture
    def mock_orchestrator(self):
        orch = MagicMock()
        orch.get_templates = AsyncMock(
            return_value=MagicMock(
                templates=[
                    MagicMock(
                        template_id="t1", name="Sequential RAG", description="Basic sequential"
                    ),
                ]
            )
        )
        orch.create_workflow = AsyncMock(
            return_value=MagicMock(
                workflow_id="wf-001",
                workflow_name="test",
                strategy="sequential",
                num_nodes=2,
                num_edges=1,
                created_at="2026-01-01",
                metadata=None,
            )
        )
        orch.execute = AsyncMock(
            return_value=MagicMock(
                execution_id="exec-001",
                workflow_id="wf-001",
                status="completed",
                execution_time=1.5,
                result={"answer": "done"},
                error=None,
                node_results=[],
            )
        )
        orch.monitor = AsyncMock(
            return_value=MagicMock(
                execution_id="exec-001",
                current_node="done",
                progress=1.0,
                nodes_completed=["n1"],
                nodes_pending=[],
                elapsed_time=1.5,
            )
        )
        orch.analyze = AsyncMock(
            return_value=MagicMock(
                workflow_id="wf-001",
                total_executions=5,
                avg_execution_time=2.0,
                success_rate=0.9,
                bottlenecks=[],
                agent_utilization=None,
            )
        )
        orch.visualize = AsyncMock(
            return_value=MagicMock(
                workflow_id="wf-001",
                diagram="[node1] -> [node2]",
            )
        )
        return orch

    @pytest.fixture
    def cmd(self, mock_orchestrator):
        console = MagicMock()
        return OrchestratorCommands(orchestrator=mock_orchestrator, console=console)

    async def test_templates(self, cmd, mock_orchestrator):
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_templates()
        mock_orchestrator.get_templates.assert_called_once()

    async def test_templates_exception(self, cmd, mock_orchestrator):
        mock_orchestrator.get_templates.side_effect = Exception("error")
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_templates()
        cmd.console.print.assert_called()

    async def test_create_workflow(self, cmd, mock_orchestrator):
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_create(
                name="my-workflow",
                strategy="sequential",
                nodes=["node1", "node2"],
                edges=[("node1", "node2")],
            )
        mock_orchestrator.create_workflow.assert_called_once()

    async def test_create_workflow_exception(self, cmd, mock_orchestrator):
        mock_orchestrator.create_workflow.side_effect = Exception("create error")
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_create(name="wf", strategy="sequential", nodes=[], edges=[])
        cmd.console.print.assert_called()

    async def test_execute(self, cmd, mock_orchestrator):
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_execute(
                workflow_id="wf-001",
                agents=["agent1"],
                task="Do something",
            )
        mock_orchestrator.execute.assert_called_once()

    async def test_execute_exception(self, cmd, mock_orchestrator):
        mock_orchestrator.execute.side_effect = Exception("exec error")
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_execute(workflow_id="wf-001", agents=[], task="")
        cmd.console.print.assert_called()

    async def test_analyze(self, cmd, mock_orchestrator):
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_analyze(workflow_id="wf-001")
        mock_orchestrator.analyze.assert_called_once()

    async def test_analyze_exception(self, cmd, mock_orchestrator):
        mock_orchestrator.analyze.side_effect = Exception("analyze error")
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_analyze(workflow_id="wf-001")
        cmd.console.print.assert_called()

    async def test_visualize(self, cmd, mock_orchestrator):
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_visualize(workflow_id="wf-001")
        mock_orchestrator.visualize.assert_called_once()

    async def test_visualize_exception(self, cmd, mock_orchestrator):
        mock_orchestrator.visualize.side_effect = Exception("viz error")
        module = "beanllm.ui.repl.orchestrator_commands"
        with _make_progress_patch(module):
            await cmd.cmd_visualize(workflow_id="wf-001")
        cmd.console.print.assert_called()


# ── RAGDebugCommands ───────────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.rag_commands import RAGDebugCommands

    RAG_CMD_AVAILABLE = True
except ImportError:
    RAG_CMD_AVAILABLE = False


@pytest.mark.skipif(not RAG_CMD_AVAILABLE, reason="RAGDebugCommands not available")
class TestRAGDebugCommands:
    @pytest.fixture
    def mock_debug(self):
        debug = MagicMock()
        debug.start_session = AsyncMock(
            return_value=MagicMock(
                session_id="sess-001",
                session_name="test",
                num_documents=10,
                num_embeddings=10,
                embedding_dim=1536,
                created_at="2026-01-01",
            )
        )
        debug.analyze_embeddings = AsyncMock(
            return_value=MagicMock(
                method="kmeans",
                num_clusters=3,
                outliers=[1, 5],
                silhouette_score=0.75,
                cluster_sizes={0: 4, 1: 3, 2: 3},
            )
        )
        debug.validate_chunks = AsyncMock(
            return_value=MagicMock(
                total_chunks=10,
                valid_chunks=9,
                invalid_chunks=1,
                issues=[{"chunk_id": 1, "issue": "too short"}],
            )
        )
        debug.tune_parameters = AsyncMock(
            return_value=MagicMock(
                best_params={"chunk_size": 512, "overlap": 50},
                best_score=0.88,
                iterations=5,
                history=[],
            )
        )
        debug.export_session = AsyncMock(
            return_value=MagicMock(
                path="session.json",
            )
        )
        return debug

    @pytest.fixture
    def cmd(self, mock_debug):
        console = MagicMock()
        cmd = RAGDebugCommands(console=console)
        cmd._debug = mock_debug
        cmd._session_active = True
        return cmd

    async def test_analyze_basic(self, cmd, mock_debug):
        module = "beanllm.ui.repl.rag_commands"
        with _make_progress_patch(module):
            await cmd.cmd_analyze(method="kmeans")
        mock_debug.analyze_embeddings.assert_called_once()

    async def test_analyze_no_session(self, cmd):
        cmd._session_active = False
        await cmd.cmd_analyze()
        cmd.console.print.assert_called()

    async def test_analyze_exception(self, cmd, mock_debug):
        mock_debug.analyze_embeddings.side_effect = Exception("analysis error")
        module = "beanllm.ui.repl.rag_commands"
        with _make_progress_patch(module):
            await cmd.cmd_analyze()
        cmd.console.print.assert_called()

    async def test_validate_basic(self, cmd, mock_debug):
        module = "beanllm.ui.repl.rag_commands"
        with _make_progress_patch(module):
            await cmd.cmd_validate()
        mock_debug.validate_chunks.assert_called_once()

    async def test_validate_no_session(self, cmd):
        cmd._session_active = False
        await cmd.cmd_validate()
        cmd.console.print.assert_called()

    async def test_tune_basic(self, cmd, mock_debug):
        mock_debug.tune_parameters = AsyncMock(
            return_value=MagicMock(
                best_params={"chunk_size": 512},
                best_score=0.88,
                iterations=5,
                history=[],
            )
        )
        module = "beanllm.ui.repl.rag_commands"
        with _make_progress_patch(module):
            await cmd.cmd_tune(parameters={"top_k": 10, "score_threshold": 0.7})
        mock_debug.tune_parameters.assert_called_once()

    async def test_tune_no_session(self, cmd):
        cmd._session_active = False
        await cmd.cmd_tune(parameters={"top_k": 5})
        cmd.console.print.assert_called()

    async def test_export_basic(self, cmd, mock_debug):
        mock_debug.export_report = AsyncMock(
            return_value=MagicMock(
                output_dir="./reports",
                files=["report.json"],
            )
        )
        module = "beanllm.ui.repl.rag_commands"
        with _make_progress_patch(module):
            await cmd.cmd_export(output_dir="./reports")
        mock_debug.export_report.assert_called_once()

    async def test_export_no_session(self, cmd):
        cmd._session_active = False
        await cmd.cmd_export(output_dir="./reports")
        cmd.console.print.assert_called()
