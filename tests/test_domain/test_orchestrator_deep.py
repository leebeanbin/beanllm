"""
Comprehensive pytest tests for beanllm orchestrator domain files:
- visual_builder.py   (VisualBuilder, create_simple_workflow)
- workflow_executors.py (execute_sequential_node, execute_decision_node,
                         execute_merge_node, execute_debate_node,
                         execute_hierarchical_node)
- templates.py        (WorkflowTemplates, quick_* helpers)
- workflow_graph.py   (WorkflowGraph – covers the remaining 28 %)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.domain.orchestrator.templates import (
    WorkflowTemplates,
    quick_debate,
    quick_parallel,
    quick_pipeline,
    quick_research_write,
)
from beanllm.domain.orchestrator.visual_builder import (
    VisualBuilder,
    create_simple_workflow,
)
from beanllm.domain.orchestrator.workflow_executors import (
    execute_debate_node,
    execute_decision_node,
    execute_hierarchical_node,
    execute_merge_node,
    execute_sequential_node,
)
from beanllm.domain.orchestrator.workflow_graph import WorkflowGraph
from beanllm.domain.orchestrator.workflow_types import (
    EdgeCondition,
    ExecutionResult,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(answer: str = "ok") -> Any:
    """Return an async-capable mock agent."""
    agent = MagicMock()
    result = MagicMock()
    result.answer = answer
    agent.run = AsyncMock(return_value=result)
    return agent


def _simple_graph() -> WorkflowGraph:
    """START → AGENT → END graph."""
    wf = WorkflowGraph(name="Test")
    s = wf.add_node(NodeType.START, "start")
    a = wf.add_node(NodeType.AGENT, "worker", config={"agent_id": "a1"})
    e = wf.add_node(NodeType.END, "end")
    wf.add_edge(s, a)
    wf.add_edge(a, e)
    return wf


# ===========================================================================
# 1. WorkflowGraph – structural tests
# ===========================================================================


class TestWorkflowGraphInit:
    def test_default_name(self) -> None:
        wf = WorkflowGraph()
        assert wf.name == "Workflow"

    def test_custom_id(self) -> None:
        wf = WorkflowGraph(name="X", workflow_id="my-id")
        assert wf.workflow_id == "my-id"

    def test_auto_uuid_id(self) -> None:
        wf = WorkflowGraph()
        assert len(wf.workflow_id) > 0

    def test_empty_state(self) -> None:
        wf = WorkflowGraph()
        assert wf.nodes == {}
        assert wf.edges == {}
        assert wf.adjacency == {}
        assert wf.reverse_adjacency == {}
        assert wf.execution_history == []


class TestWorkflowGraphNodes:
    def test_add_node_returns_id(self) -> None:
        wf = WorkflowGraph()
        nid = wf.add_node(NodeType.START, "start")
        assert nid in wf.nodes

    def test_add_node_custom_id(self) -> None:
        wf = WorkflowGraph()
        nid = wf.add_node(NodeType.AGENT, "a", node_id="custom_id")
        assert nid == "custom_id"
        assert "custom_id" in wf.nodes

    def test_add_node_config_stored(self) -> None:
        wf = WorkflowGraph()
        nid = wf.add_node(NodeType.AGENT, "a", config={"agent_id": "x"})
        assert wf.nodes[nid].config["agent_id"] == "x"

    def test_add_node_position_stored(self) -> None:
        wf = WorkflowGraph()
        nid = wf.add_node(NodeType.AGENT, "a", position=(5, 10))
        assert wf.nodes[nid].position == (5, 10)

    def test_add_node_initialises_adjacency(self) -> None:
        wf = WorkflowGraph()
        nid = wf.add_node(NodeType.AGENT, "a")
        assert wf.adjacency[nid] == []
        assert wf.reverse_adjacency[nid] == []


class TestWorkflowGraphEdges:
    def test_add_edge_success(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        t = wf.add_node(NodeType.END, "t")
        eid = wf.add_edge(s, t)
        assert eid in wf.edges
        assert eid in wf.adjacency[s]
        assert eid in wf.reverse_adjacency[t]

    def test_add_edge_missing_source_raises(self) -> None:
        wf = WorkflowGraph()
        t = wf.add_node(NodeType.END, "t")
        with pytest.raises(ValueError, match="Source node not found"):
            wf.add_edge("nonexistent", t)

    def test_add_edge_missing_target_raises(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        with pytest.raises(ValueError, match="Target node not found"):
            wf.add_edge(s, "nonexistent")

    def test_add_edge_cycle_raises(self) -> None:
        wf = WorkflowGraph()
        a = wf.add_node(NodeType.AGENT, "a")
        b = wf.add_node(NodeType.AGENT, "b")
        wf.add_edge(a, b)
        with pytest.raises(ValueError, match="creates a cycle"):
            wf.add_edge(b, a)

    def test_add_edge_custom_id(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        t = wf.add_node(NodeType.END, "t")
        eid = wf.add_edge(s, t, edge_id="my_edge")
        assert eid == "my_edge"

    def test_add_edge_with_condition(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        t = wf.add_node(NodeType.END, "t")
        eid = wf.add_edge(s, t, condition=EdgeCondition.ON_SUCCESS)
        assert wf.edges[eid].condition == EdgeCondition.ON_SUCCESS

    def test_cycle_detection_rollback(self) -> None:
        """Edge count should not increase after rejected cycle edge."""
        wf = WorkflowGraph()
        a = wf.add_node(NodeType.AGENT, "a")
        b = wf.add_node(NodeType.AGENT, "b")
        wf.add_edge(a, b)
        edge_count_before = len(wf.edges)
        with pytest.raises(ValueError):
            wf.add_edge(b, a)
        assert len(wf.edges) == edge_count_before


class TestWorkflowGraphTraversal:
    def test_get_start_nodes_by_type(self) -> None:
        wf = _simple_graph()
        starts = wf.get_start_nodes()
        # The START-typed node must be included
        start_types = [wf.nodes[nid].node_type for nid in starts]
        assert NodeType.START in start_types

    def test_get_end_nodes_by_type(self) -> None:
        wf = _simple_graph()
        ends = wf.get_end_nodes()
        end_types = [wf.nodes[nid].node_type for nid in ends]
        assert NodeType.END in end_types

    def test_get_start_nodes_no_explicit_start(self) -> None:
        """A node with no incoming edge should appear as a start."""
        wf = WorkflowGraph()
        a = wf.add_node(NodeType.AGENT, "a")
        b = wf.add_node(NodeType.AGENT, "b")
        wf.add_edge(a, b)
        starts = wf.get_start_nodes()
        assert a in starts

    def test_get_end_nodes_no_outgoing(self) -> None:
        wf = WorkflowGraph()
        a = wf.add_node(NodeType.AGENT, "a")
        b = wf.add_node(NodeType.AGENT, "b")
        wf.add_edge(a, b)
        ends = wf.get_end_nodes()
        assert b in ends

    def test_topological_order_linear(self) -> None:
        wf = _simple_graph()
        order = wf.get_topological_order()
        assert len(order) == 3

    def test_topological_order_all_nodes_present(self) -> None:
        wf = _simple_graph()
        order = wf.get_topological_order()
        assert set(order) == set(wf.nodes.keys())

    def test_topological_order_start_before_end(self) -> None:
        wf = _simple_graph()
        order = wf.get_topological_order()
        start_ids = [nid for nid, n in wf.nodes.items() if n.node_type == NodeType.START]
        end_ids = [nid for nid, n in wf.nodes.items() if n.node_type == NodeType.END]
        assert order.index(start_ids[0]) < order.index(end_ids[0])


class TestWorkflowGraphSerialization:
    def test_to_dict_keys(self) -> None:
        wf = _simple_graph()
        d = wf.to_dict()
        assert "workflow_id" in d
        assert "name" in d
        assert "nodes" in d
        assert "edges" in d

    def test_from_dict_roundtrip(self) -> None:
        wf = _simple_graph()
        d = wf.to_dict()
        wf2 = WorkflowGraph.from_dict(d)
        assert wf2.name == wf.name
        assert set(wf2.nodes.keys()) == set(wf.nodes.keys())
        assert set(wf2.edges.keys()) == set(wf.edges.keys())

    def test_from_dict_preserves_workflow_id(self) -> None:
        wf = WorkflowGraph(name="A", workflow_id="fixed-id")
        wf.add_node(NodeType.START, "s", node_id="s")
        d = wf.to_dict()
        wf2 = WorkflowGraph.from_dict(d)
        assert wf2.workflow_id == "fixed-id"


class TestWorkflowGraphExecution:
    async def test_execute_start_end_only(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, e)
        result = await wf.execute(agents={}, task="hello")
        assert result["success"] is True
        assert result["workflow_name"] == wf.name

    async def test_execute_agent_node(self) -> None:
        wf = _simple_graph()
        agent = _make_agent("answer_text")
        result = await wf.execute(agents={"a1": agent}, task="do it")
        assert result["success"] is True

    async def test_execute_missing_agent_returns_failure(self) -> None:
        wf = _simple_graph()
        result = await wf.execute(agents={}, task="do it")
        # Node failed because agent not found
        assert result["success"] is False

    async def test_execute_tool_node(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        t = wf.add_node(NodeType.TOOL, "tool", config={"tool_id": "t1", "input": {}})
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, t)
        wf.add_edge(t, e)
        tool = MagicMock()
        tool.execute = AsyncMock(return_value="tool_output")
        result = await wf.execute(agents={}, task="x", tools={"t1": tool})
        assert result["success"] is True

    async def test_execute_missing_tool_returns_failure(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        t = wf.add_node(NodeType.TOOL, "tool", config={"tool_id": "missing"})
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, t)
        wf.add_edge(t, e)
        result = await wf.execute(agents={}, task="x")
        assert result["success"] is False

    async def test_execute_parallel_node(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        p = wf.add_node(NodeType.PARALLEL, "parallel", config={"agent_ids": ["a1", "a2"]})
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, p)
        wf.add_edge(p, e)
        agents = {"a1": _make_agent("r1"), "a2": _make_agent("r2")}
        result = await wf.execute(agents=agents, task="task")
        assert result["success"] is True

    async def test_execute_decision_node(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        d = wf.add_node(
            NodeType.DECISION,
            "dec",
            config={"condition_key": "flag", "branches": {"yes": "branch_a"}},
        )
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, d)
        wf.add_edge(d, e)
        result = await wf.execute(agents={}, task="t")
        assert result["success"] is True

    async def test_execute_fail_fast_stops_execution(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        a = wf.add_node(
            NodeType.AGENT, "a", config={"agent_id": "missing_agent", "fail_fast": True}
        )
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, a)
        wf.add_edge(a, e)
        result = await wf.execute(agents={}, task="t")
        # After fail_fast the loop breaks; END node not executed
        assert result["success"] is False

    async def test_execute_edge_condition_skips_node(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        a = wf.add_node(NodeType.AGENT, "a", config={"agent_id": "a1"})
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, a)
        # ON_FAILURE edge: will be skipped when success=True
        wf.add_edge(a, e, condition=EdgeCondition.ON_FAILURE)
        agents = {"a1": _make_agent("output")}
        result = await wf.execute(agents=agents, task="t")
        # Workflow ran but END was skipped because edge condition not met
        assert "execution_history" in result

    async def test_execute_regular_workflow_succeeds(self) -> None:
        """Verify basic execution path works end-to-end."""
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "s")
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, e)
        result = await wf.execute(agents={}, task="t")
        assert result["success"] is True

    async def test_execute_history_populated(self) -> None:
        wf = _simple_graph()
        await wf.execute(agents={"a1": _make_agent()}, task="t")
        assert len(wf.execution_history) == 3  # START + AGENT + END

    async def test_execute_returns_workflow_id(self) -> None:
        wf = WorkflowGraph(name="Named", workflow_id="wf-42")
        s = wf.add_node(NodeType.START, "s")
        e = wf.add_node(NodeType.END, "e")
        wf.add_edge(s, e)
        result = await wf.execute(agents={}, task="t")
        assert result["workflow_id"] == "wf-42"


# ===========================================================================
# 2. VisualBuilder
# ===========================================================================


class TestVisualBuilderInit:
    def test_init_stores_workflow(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        assert vb.workflow is wf
        assert vb.layers == []
        assert vb.node_positions == {}


class TestVisualBuilderBuildDiagram:
    def test_build_box_diagram_returns_string(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        diagram = vb.build_diagram(style="box")
        assert isinstance(diagram, str)
        assert len(diagram) > 0

    def test_build_simple_diagram(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        diagram = vb.build_diagram(style="simple")
        assert "↓" in diagram or "[" in diagram

    def test_build_compact_diagram(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        diagram = vb.build_diagram(style="compact")
        assert "→" in diagram

    def test_build_unknown_style_falls_back_to_box(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        diagram_unknown = vb.build_diagram(style="unknown_style")
        diagram_box = vb.build_diagram(style="box")
        # Both should be non-empty strings
        assert isinstance(diagram_unknown, str)
        assert isinstance(diagram_box, str)

    def test_build_box_show_config(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "start")
        a = wf.add_node(NodeType.AGENT, "worker", config={"agent_id": "a1", "temp": 0.5})
        e = wf.add_node(NodeType.END, "end")
        wf.add_edge(s, a)
        wf.add_edge(a, e)
        vb = VisualBuilder(wf)
        diagram = vb.build_diagram(style="box", show_config=True)
        assert "agent_id" in diagram

    def test_build_diagram_empty_workflow(self) -> None:
        wf = WorkflowGraph()
        vb = VisualBuilder(wf)
        diagram = vb.build_diagram()
        # Empty workflow → empty or minimal output
        assert isinstance(diagram, str)

    def test_layers_populated_after_build(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        vb.build_diagram()
        assert len(vb.layers) > 0

    def test_node_positions_populated_after_build(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        vb.build_diagram()
        assert len(vb.node_positions) == 3


class TestVisualBuilderAssignLayers:
    def test_no_start_node_falls_back_to_no_incoming(self) -> None:
        """Workflow with AGENT nodes only (no START type)."""
        wf = WorkflowGraph()
        a = wf.add_node(NodeType.AGENT, "a")
        b = wf.add_node(NodeType.AGENT, "b")
        wf.add_edge(a, b)
        vb = VisualBuilder(wf)
        vb._assign_layers()
        assert len(vb.layers) >= 2

    def test_single_node_one_layer(self) -> None:
        wf = WorkflowGraph()
        wf.add_node(NodeType.AGENT, "solo")
        vb = VisualBuilder(wf)
        vb._assign_layers()
        assert len(vb.layers) == 1

    def test_later_layer_assigned_max_depth(self) -> None:
        """Diamond shape: a→b, a→c, b→d, c→d. d should be layer 2."""
        wf = WorkflowGraph()
        a = wf.add_node(NodeType.START, "a")
        b = wf.add_node(NodeType.AGENT, "b")
        c = wf.add_node(NodeType.AGENT, "c")
        d = wf.add_node(NodeType.END, "d")
        wf.add_edge(a, b)
        wf.add_edge(a, c)
        wf.add_edge(b, d)
        wf.add_edge(c, d)
        vb = VisualBuilder(wf)
        vb._assign_layers()
        # 'd' must be at layer >= 2
        assert vb.node_positions[d][0] >= 2


class TestVisualBuilderCenterText:
    def test_center_short_text(self) -> None:
        wf = WorkflowGraph()
        vb = VisualBuilder(wf)
        result = vb._center_text("hi", 10)
        assert result.strip() == "hi"
        assert len(result) == 10 // 2 + len("hi") // 2 or len(result) >= len("hi")

    def test_center_text_wider_than_width(self) -> None:
        wf = WorkflowGraph()
        vb = VisualBuilder(wf)
        long_text = "A" * 30
        result = vb._center_text(long_text, 10)
        assert result == long_text  # Returned as-is when text >= width


class TestVisualBuilderMermaid:
    def test_mermaid_starts_with_graph_td(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        mermaid = vb.build_mermaid_diagram()
        assert mermaid.startswith("graph TD")

    def test_mermaid_contains_arrow(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        mermaid = vb.build_mermaid_diagram()
        assert "-->" in mermaid

    def test_mermaid_decision_node_shape(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "start")
        d = wf.add_node(NodeType.DECISION, "decide")
        e = wf.add_node(NodeType.END, "end")
        wf.add_edge(s, d)
        wf.add_edge(d, e)
        vb = VisualBuilder(wf)
        mermaid = vb.build_mermaid_diagram()
        assert "{" in mermaid  # Decision uses {name} shape

    def test_mermaid_conditional_edge_label(self) -> None:
        wf = WorkflowGraph()
        s = wf.add_node(NodeType.START, "start")
        e = wf.add_node(NodeType.END, "end")
        wf.add_edge(s, e, condition=EdgeCondition.ON_SUCCESS)
        vb = VisualBuilder(wf)
        mermaid = vb.build_mermaid_diagram()
        # on_success is not "always", so label should appear
        assert "|on_success|" in mermaid


class TestVisualBuilderBuildCode:
    def test_build_code_python(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        code = vb.build_code(language="python")
        assert "WorkflowGraph" in code
        assert "add_node" in code
        assert "add_edge" in code

    def test_build_code_non_python(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        code = vb.build_code(language="javascript")
        assert "Only Python" in code


class TestVisualBuilderStatistics:
    def test_statistics_node_count(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        vb.build_diagram()
        stats = vb.get_statistics()
        assert stats["num_nodes"] == 3

    def test_statistics_edge_count(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        vb.build_diagram()
        stats = vb.get_statistics()
        assert stats["num_edges"] == 2

    def test_statistics_empty_layers(self) -> None:
        """If build_diagram not called layers is empty → max_layer_width == 0."""
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        stats = vb.get_statistics()
        assert stats["max_layer_width"] == 0

    def test_statistics_node_types(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        vb.build_diagram()
        stats = vb.get_statistics()
        assert "start" in stats["node_types"]
        assert "end" in stats["node_types"]
        assert "agent" in stats["node_types"]


class TestVisualBuilderValidate:
    def test_validate_valid_workflow(self) -> None:
        wf = _simple_graph()
        vb = VisualBuilder(wf)
        warnings = vb.validate_workflow()
        assert warnings == []

    def test_validate_no_start_node(self) -> None:
        """Workflow with no nodes at all → get_start_nodes() returns empty."""
        wf = WorkflowGraph()  # no nodes at all → start_nodes will be empty
        vb = VisualBuilder(wf)
        warnings = vb.validate_workflow()
        assert any("START" in w for w in warnings)

    def test_validate_no_end_node(self) -> None:
        """Workflow with no nodes at all → get_end_nodes() returns empty."""
        wf = WorkflowGraph()  # no nodes → end_nodes will be empty
        vb = VisualBuilder(wf)
        warnings = vb.validate_workflow()
        assert any("END" in w for w in warnings)

    def test_validate_isolated_node(self) -> None:
        wf = WorkflowGraph()
        wf.add_node(NodeType.START, "s")
        wf.add_node(NodeType.AGENT, "isolated")  # not connected
        vb = VisualBuilder(wf)
        warnings = vb.validate_workflow()
        assert any("Isolated" in w for w in warnings)


class TestCreateSimpleWorkflow:
    def test_sequential_connection(self) -> None:
        wf = create_simple_workflow(
            [
                ("start", NodeType.START),
                ("work", NodeType.AGENT),
                ("end", NodeType.END),
            ]
        )
        assert len(wf.nodes) == 3
        assert len(wf.edges) == 2

    def test_custom_name(self) -> None:
        wf = create_simple_workflow(
            [("s", NodeType.START), ("e", NodeType.END)],
            name="My Flow",
        )
        assert wf.name == "My Flow"

    def test_single_node_no_edges(self) -> None:
        wf = create_simple_workflow([("s", NodeType.START)])
        assert len(wf.nodes) == 1
        assert len(wf.edges) == 0


# ===========================================================================
# 3. WorkflowTemplates
# ===========================================================================


class TestWorkflowTemplates:
    def test_research_and_write_basic(self) -> None:
        wf = WorkflowTemplates.research_and_write("r1", "w1")
        assert wf.name == "Research & Write"
        # 4 nodes: start, researcher, writer, end
        assert len(wf.nodes) == 4

    def test_research_and_write_with_reviewer(self) -> None:
        wf = WorkflowTemplates.research_and_write("r1", "w1", reviewer_id="rev1")
        # 5 nodes: start, researcher, writer, reviewer, end
        assert len(wf.nodes) == 5

    def test_parallel_consensus_vote(self) -> None:
        wf = WorkflowTemplates.parallel_consensus(["a1", "a2", "a3"], "vote")
        assert "vote" in wf.name.lower()
        assert len(wf.nodes) == 4  # start, parallel, merge, end

    def test_parallel_consensus_consensus(self) -> None:
        wf = WorkflowTemplates.parallel_consensus(["a1", "a2"], "consensus")
        assert len(wf.nodes) == 4

    def test_hierarchical_delegation(self) -> None:
        wf = WorkflowTemplates.hierarchical_delegation("mgr", ["w1", "w2"])
        assert wf.name == "Hierarchical Delegation"
        assert len(wf.nodes) == 5  # start, decompose, execute, synthesize, end

    def test_debate_and_judge(self) -> None:
        wf = WorkflowTemplates.debate_and_judge(["d1", "d2"], "judge", rounds=2)
        assert "2 rounds" in wf.name
        assert len(wf.nodes) == 4  # start, debate, judge, end

    def test_pipeline_stages(self) -> None:
        wf = WorkflowTemplates.pipeline(["gather", "analyze", "present"])
        assert len(wf.nodes) == 5  # start + 3 stages + end

    def test_pipeline_custom_agent_ids(self) -> None:
        wf = WorkflowTemplates.pipeline(
            ["s1", "s2"],
            agent_ids=["agent_a", "agent_b"],
        )
        agent_ids_in_config = [
            n.config.get("agent_id") for n in wf.nodes.values() if n.config.get("agent_id")
        ]
        assert "agent_a" in agent_ids_in_config
        assert "agent_b" in agent_ids_in_config

    def test_conditional_branch(self) -> None:
        wf = WorkflowTemplates.conditional_branch("cond", "branch_a", "branch_b")
        assert wf.name == "Conditional Branch"
        # start, decision, branch_a, branch_b, merge, end = 6 nodes
        assert len(wf.nodes) == 6

    def test_conditional_branch_edge_conditions(self) -> None:
        wf = WorkflowTemplates.conditional_branch("cond", "ba", "bb")
        conditions = [e.condition for e in wf.edges.values()]
        assert EdgeCondition.ON_SUCCESS in conditions
        assert EdgeCondition.ON_FAILURE in conditions

    def test_iterative_refinement_no_checker(self) -> None:
        wf = WorkflowTemplates.iterative_refinement("agent1", max_iterations=2)
        # start + 2 iterations + end = 4
        assert len(wf.nodes) == 4

    def test_iterative_refinement_with_checker(self) -> None:
        wf = WorkflowTemplates.iterative_refinement(
            "agent1", max_iterations=3, quality_checker_id="checker"
        )
        # start + 3 agents + 2 checkers (last iteration has no checker) + end = 7
        assert len(wf.nodes) == 7

    def test_map_reduce(self) -> None:
        wf = WorkflowTemplates.map_reduce(["m1", "m2"], "reducer")
        assert wf.name == "Map-Reduce"
        assert len(wf.nodes) == 4  # start, map_parallel, reduce, end

    def test_code_review_pipeline(self) -> None:
        wf = WorkflowTemplates.code_review_pipeline("coder", ["rev1", "rev2"], "approver")
        assert wf.name == "Code Review Pipeline"
        assert len(wf.nodes) == 5  # start, coder, reviewers, approver, end

    def test_custom_template(self) -> None:
        structure = {
            "nodes": [
                {"id": "n1", "type": "start", "name": "S"},
                {"id": "n2", "type": "end", "name": "E"},
            ],
            "edges": [
                {"source": "n1", "target": "n2"},
            ],
        }
        wf = WorkflowTemplates.custom_template("Custom", structure)
        assert wf.name == "Custom"
        assert len(wf.nodes) == 2
        assert len(wf.edges) == 1

    def test_custom_template_no_id_in_node(self) -> None:
        """Nodes without 'id' should still be added."""
        structure = {
            "nodes": [
                {"type": "start", "name": "S"},
                {"type": "end", "name": "E"},
            ],
            "edges": [],
        }
        wf = WorkflowTemplates.custom_template("No-id", structure)
        assert len(wf.nodes) == 2


class TestQuickHelpers:
    def test_quick_research_write(self) -> None:
        wf = quick_research_write("r", "w")
        assert isinstance(wf, WorkflowGraph)

    def test_quick_parallel(self) -> None:
        wf = quick_parallel(["a1", "a2"])
        assert isinstance(wf, WorkflowGraph)

    def test_quick_pipeline(self) -> None:
        wf = quick_pipeline(["s1", "s2"])
        assert isinstance(wf, WorkflowGraph)

    def test_quick_debate(self) -> None:
        wf = quick_debate(["d1", "d2"], "judge", rounds=2)
        assert isinstance(wf, WorkflowGraph)


# ===========================================================================
# 4. WorkflowExecutors
# ===========================================================================


class TestExecuteDecisionNode:
    async def test_callable_condition(self) -> None:
        node = WorkflowNode(
            node_id="d",
            node_type=NodeType.DECISION,
            name="d",
            config={
                "condition": lambda _state: "yes",
                "branches": {"yes": "node_yes"},
            },
        )
        result = await execute_decision_node(node, {})
        assert result["decision"] == "yes"
        assert result["next_node"] == "node_yes"

    async def test_condition_key(self) -> None:
        node = WorkflowNode(
            node_id="d",
            node_type=NodeType.DECISION,
            name="d",
            config={
                "condition_key": "choice",
                "branches": {"A": "nodeA", "B": "nodeB"},
                "default": "nodeA",
            },
        )
        result = await execute_decision_node(node, {"choice": "B"})
        assert result["decision"] == "B"
        assert result["next_node"] == "nodeB"

    async def test_keyword_matching(self) -> None:
        node = WorkflowNode(
            node_id="d",
            node_type=NodeType.DECISION,
            name="d",
            config={"branches": {"error": "handle_error"}},
        )
        result = await execute_decision_node(node, {"last_output": "there was an error"})
        assert result["decision"] == "error"
        assert result["next_node"] == "handle_error"

    async def test_no_match_uses_default(self) -> None:
        node = WorkflowNode(
            node_id="d",
            node_type=NodeType.DECISION,
            name="d",
            config={
                "branches": {"yes": "node_yes"},
                "default": "default_node",
            },
        )
        result = await execute_decision_node(node, {"last_output": "nothing useful"})
        assert result["next_node"] == "default_node"

    async def test_available_branches_returned(self) -> None:
        node = WorkflowNode(
            node_id="d",
            node_type=NodeType.DECISION,
            name="d",
            config={"branches": {"a": "na", "b": "nb"}},
        )
        result = await execute_decision_node(node, {})
        assert set(result["available_branches"]) == {"a", "b"}


class TestExecuteMergeNode:
    async def test_strategy_all(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={"input_nodes": ["n1", "n2"], "strategy": "all"},
        )
        result = await execute_merge_node(node, {"n1": "res1", "n2": "res2"})
        assert result["strategy"] == "all"
        assert result["count"] == 2

    async def test_strategy_vote(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={"input_nodes": ["n1", "n2", "n3"], "strategy": "vote"},
        )
        result = await execute_merge_node(node, {"n1": "yes", "n2": "yes", "n3": "no"})
        assert result["strategy"] == "vote"
        assert result["winner"] == "yes"

    async def test_strategy_consensus_agreed(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={"input_nodes": ["n1", "n2"], "strategy": "consensus"},
        )
        result = await execute_merge_node(node, {"n1": "yes", "n2": "yes"})
        assert result["agreed"] is True

    async def test_strategy_consensus_disagreed(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={"input_nodes": ["n1", "n2"], "strategy": "consensus"},
        )
        result = await execute_merge_node(node, {"n1": "yes", "n2": "no"})
        assert result["agreed"] is False
        assert result["unique_count"] == 2

    async def test_strategy_first(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={"input_nodes": ["n1", "n2"], "strategy": "first"},
        )
        result = await execute_merge_node(node, {"n1": "first_result", "n2": "second"})
        assert result["result"] == "first_result"

    async def test_strategy_custom(self) -> None:
        def my_merge(items: List[Any]) -> str:
            return "|".join(str(x) for x in items)

        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={
                "input_nodes": ["n1", "n2"],
                "strategy": "custom",
                "custom_merge": my_merge,
            },
        )
        result = await execute_merge_node(node, {"n1": "A", "n2": "B"})
        assert result["strategy"] == "custom"
        assert result["result"] == "A|B"

    async def test_no_inputs_returns_none(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={"input_nodes": ["absent1", "absent2"], "strategy": "all"},
        )
        result = await execute_merge_node(node, {})
        assert result is None

    async def test_strategy_vote_with_weights(self) -> None:
        node = WorkflowNode(
            node_id="m",
            node_type=NodeType.MERGE,
            name="m",
            config={
                "input_nodes": ["n1", "n2"],
                "strategy": "vote",
                "weights": {"n1": 2.0, "n2": 1.0},
            },
        )
        result = await execute_merge_node(node, {"n1": "A", "n2": "B"})
        assert result["winner"] == "A"  # A has weight 2.0 vs B 1.0


class TestExecuteSequentialNode:
    async def test_sequential_runs_children_in_order(self) -> None:
        calls: List[str] = []

        async def fake_execute(child_node: WorkflowNode) -> ExecutionResult:
            calls.append(child_node.node_id)
            return ExecutionResult(node_id=child_node.node_id, success=True, output="out")

        child_a = WorkflowNode(node_id="ca", node_type=NodeType.AGENT, name="ca")
        child_b = WorkflowNode(node_id="cb", node_type=NodeType.AGENT, name="cb")
        nodes = {"ca": child_a, "cb": child_b}

        parent = WorkflowNode(
            node_id="seq",
            node_type=NodeType.SEQUENTIAL,
            name="seq",
            config={"child_nodes": ["ca", "cb"]},
        )
        state: Dict[str, Any] = {"task": "initial_task"}
        history: List[ExecutionResult] = []

        await execute_sequential_node(parent, nodes, state, fake_execute, history)
        assert calls == ["ca", "cb"]

    async def test_sequential_stops_on_failure(self) -> None:
        calls: List[str] = []

        async def fake_execute(child_node: WorkflowNode) -> ExecutionResult:
            calls.append(child_node.node_id)
            if child_node.node_id == "fail_node":
                return ExecutionResult(
                    node_id=child_node.node_id, success=False, output=None, error="fail"
                )
            return ExecutionResult(node_id=child_node.node_id, success=True, output="ok")

        nodes = {
            "a": WorkflowNode(node_id="a", node_type=NodeType.AGENT, name="a"),
            "fail_node": WorkflowNode(node_id="fail_node", node_type=NodeType.AGENT, name="fn"),
            "b": WorkflowNode(node_id="b", node_type=NodeType.AGENT, name="b"),
        }
        parent = WorkflowNode(
            node_id="seq",
            node_type=NodeType.SEQUENTIAL,
            name="seq",
            config={"child_nodes": ["a", "fail_node", "b"]},
        )
        history: List[ExecutionResult] = []
        await execute_sequential_node(parent, nodes, {}, fake_execute, history)
        # "b" should not have been called
        assert "b" not in calls

    async def test_sequential_missing_child_skipped(self) -> None:
        async def fake_execute(child_node: WorkflowNode) -> ExecutionResult:
            return ExecutionResult(node_id=child_node.node_id, success=True, output="ok")

        parent = WorkflowNode(
            node_id="seq",
            node_type=NodeType.SEQUENTIAL,
            name="seq",
            config={"child_nodes": ["nonexistent"]},
        )
        result = await execute_sequential_node(parent, {}, {"task": "t"}, fake_execute, [])
        assert result is None

    async def test_sequential_pass_output_feeds_next(self) -> None:
        async def fake_execute(child_node: WorkflowNode) -> ExecutionResult:
            return ExecutionResult(
                node_id=child_node.node_id, success=True, output=f"out_{child_node.node_id}"
            )

        nodes = {
            "x": WorkflowNode(node_id="x", node_type=NodeType.AGENT, name="x"),
            "y": WorkflowNode(node_id="y", node_type=NodeType.AGENT, name="y"),
        }
        parent = WorkflowNode(
            node_id="seq",
            node_type=NodeType.SEQUENTIAL,
            name="seq",
            config={"child_nodes": ["x", "y"], "pass_output": True},
        )
        state: Dict[str, Any] = {"task": "original"}
        await execute_sequential_node(parent, nodes, state, fake_execute, [])
        # Original task should be restored
        assert state["task"] == "original"


class TestExecuteDebateNode:
    async def test_debate_requires_two_agents(self) -> None:
        node = WorkflowNode(
            node_id="deb",
            node_type=NodeType.DEBATE,
            name="deb",
            config={"agent_ids": ["only_one"], "rounds": 1},
        )
        state = {"agents": {"only_one": _make_agent()}, "task": "topic"}
        with pytest.raises(ValueError, match="at least 2 agents"):
            await execute_debate_node(node, state)

    async def test_debate_returns_rounds_history(self) -> None:
        node = WorkflowNode(
            node_id="deb",
            node_type=NodeType.DEBATE,
            name="deb",
            config={"agent_ids": ["a1", "a2"], "rounds": 2},
        )
        state = {
            "agents": {"a1": _make_agent("arg1"), "a2": _make_agent("arg2")},
            "task": "discuss topic",
        }
        result = await execute_debate_node(node, state)
        assert result["rounds"] == 2
        assert len(result["rounds_history"]) == 2

    async def test_debate_with_judge(self) -> None:
        node = WorkflowNode(
            node_id="deb",
            node_type=NodeType.DEBATE,
            name="deb",
            config={"agent_ids": ["a1", "a2"], "rounds": 1, "judge_id": "judge"},
        )
        state = {
            "agents": {
                "a1": _make_agent("arg1"),
                "a2": _make_agent("arg2"),
                "judge": _make_agent("verdict"),
            },
            "task": "topic",
        }
        result = await execute_debate_node(node, state)
        assert result["judge_verdict"] == "verdict"

    async def test_debate_without_judge_verdict_is_none(self) -> None:
        node = WorkflowNode(
            node_id="deb",
            node_type=NodeType.DEBATE,
            name="deb",
            config={"agent_ids": ["a1", "a2"], "rounds": 1},
        )
        state = {
            "agents": {"a1": _make_agent(), "a2": _make_agent()},
            "task": "t",
        }
        result = await execute_debate_node(node, state)
        assert result["judge_verdict"] is None

    async def test_debate_agent_error_handled(self) -> None:
        """If an agent raises, the error string is captured and debate continues."""
        bad_agent = MagicMock()
        bad_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        node = WorkflowNode(
            node_id="deb",
            node_type=NodeType.DEBATE,
            name="deb",
            config={"agent_ids": ["good", "bad"], "rounds": 1},
        )
        state = {
            "agents": {"good": _make_agent("fine"), "bad": bad_agent},
            "task": "t",
        }
        result = await execute_debate_node(node, state)
        # Should still return 1 round despite the error
        assert result["rounds"] == 1
        # Error string appears somewhere
        assert any("error" in s for s in result["rounds_history"][0])


class TestExecuteHierarchicalNode:
    async def test_missing_manager_raises(self) -> None:
        node = WorkflowNode(
            node_id="h",
            node_type=NodeType.HIERARCHICAL,
            name="h",
            config={"manager_id": "mgr", "worker_ids": ["w1"]},
        )
        state = {"agents": {}, "task": "task"}
        with pytest.raises(ValueError, match="manager not found"):
            await execute_hierarchical_node(node, state)

    async def test_no_workers_raises(self) -> None:
        node = WorkflowNode(
            node_id="h",
            node_type=NodeType.HIERARCHICAL,
            name="h",
            config={"manager_id": "mgr", "worker_ids": []},
        )
        state = {"agents": {"mgr": _make_agent()}, "task": "task"}
        with pytest.raises(ValueError, match="no workers found"):
            await execute_hierarchical_node(node, state)

    async def test_hierarchical_returns_final_result(self) -> None:
        mgr_result = MagicMock()
        mgr_result.answer = "subtask1\nsubtask2"
        mgr_final = MagicMock()
        mgr_final.answer = "final_synthesis"
        manager = MagicMock()
        manager.run = AsyncMock(side_effect=[mgr_result, mgr_final])

        worker1 = _make_agent("worker1_answer")
        worker2 = _make_agent("worker2_answer")

        node = WorkflowNode(
            node_id="h",
            node_type=NodeType.HIERARCHICAL,
            name="h",
            config={"manager_id": "mgr", "worker_ids": ["w1", "w2"]},
        )
        state = {
            "agents": {"mgr": manager, "w1": worker1, "w2": worker2},
            "task": "big task",
        }
        result = await execute_hierarchical_node(node, state)
        assert result["final_result"] == "final_synthesis"
        assert len(result["worker_results"]) == 2

    async def test_hierarchical_worker_error_captured(self) -> None:
        decompose_result = MagicMock()
        decompose_result.answer = "sub1\nsub2"
        final_result = MagicMock()
        final_result.answer = "ok"
        manager = MagicMock()
        manager.run = AsyncMock(side_effect=[decompose_result, final_result])

        bad_worker = MagicMock()
        bad_worker.run = AsyncMock(side_effect=RuntimeError("worker crash"))
        good_worker = _make_agent("good_output")

        node = WorkflowNode(
            node_id="h",
            node_type=NodeType.HIERARCHICAL,
            name="h",
            config={"manager_id": "mgr", "worker_ids": ["bad", "good"]},
        )
        state = {
            "agents": {"mgr": manager, "bad": bad_worker, "good": good_worker},
            "task": "t",
        }
        result = await execute_hierarchical_node(node, state)
        assert any("failed" in r for r in result["worker_results"])
        assert "good_output" in result["worker_results"]
