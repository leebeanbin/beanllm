"""Tests for domain/state_graph: checkpoint, execution, config."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from beanllm.domain.state_graph.checkpoint import Checkpoint
from beanllm.domain.state_graph.config import GraphConfig
from beanllm.domain.state_graph.execution import END, GraphExecution, NodeExecution

# ---------------------------------------------------------------------------
# GraphConfig
# ---------------------------------------------------------------------------


class TestGraphConfig:
    def test_defaults(self):
        cfg = GraphConfig()
        assert cfg.max_iterations == 100
        assert cfg.enable_checkpointing is False
        assert cfg.checkpoint_dir is None
        assert cfg.debug is False

    def test_custom_values(self):
        cfg = GraphConfig(max_iterations=50, enable_checkpointing=True, debug=True)
        assert cfg.max_iterations == 50
        assert cfg.enable_checkpointing is True
        assert cfg.debug is True

    def test_checkpoint_dir(self, tmp_path):
        cfg = GraphConfig(checkpoint_dir=tmp_path / "ckpts")
        assert cfg.checkpoint_dir == tmp_path / "ckpts"


# ---------------------------------------------------------------------------
# NodeExecution / GraphExecution / END
# ---------------------------------------------------------------------------


class TestNodeExecution:
    def test_basic(self):
        ne = NodeExecution(
            node_name="step1",
            input_state={"x": 1},
            output_state={"x": 2},
        )
        assert ne.node_name == "step1"
        assert ne.input_state == {"x": 1}
        assert ne.output_state == {"x": 2}
        assert ne.error is None

    def test_with_error(self):
        err = ValueError("boom")
        ne = NodeExecution(node_name="n", input_state={}, output_state={}, error=err)
        assert ne.error is err

    def test_timestamp_set(self):
        ne = NodeExecution(node_name="n", input_state={}, output_state={})
        assert isinstance(ne.timestamp, datetime)


class TestGraphExecution:
    def test_basic(self):
        ge = GraphExecution(execution_id="exec-1", start_time=datetime.now())
        assert ge.execution_id == "exec-1"
        assert ge.end_time is None
        assert ge.nodes_executed == []
        assert ge.final_state is None
        assert ge.error is None

    def test_with_nodes(self):
        ne = NodeExecution(node_name="n", input_state={}, output_state={"done": True})
        ge = GraphExecution(
            execution_id="exec-2",
            start_time=datetime.now(),
            nodes_executed=[ne],
            final_state={"done": True},
        )
        assert len(ge.nodes_executed) == 1
        assert ge.final_state == {"done": True}


class TestEND:
    def test_is_class(self):
        # END is a marker class, not an instance
        assert isinstance(END, type)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        state = {"messages": ["hello"], "counter": 5}
        ckpt.save("exec-1", state, "node_A")

        loaded = ckpt.load("exec-1", "node_A")
        assert loaded == state

    def test_load_missing_returns_none(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        result = ckpt.load("nonexistent", "node")
        assert result is None

    def test_saves_valid_json(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        ckpt.save("exec-1", {"key": "value"}, "node_B")

        json_file = tmp_path / "exec-1_node_B.json"
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert data["execution_id"] == "exec-1"
        assert data["node_name"] == "node_B"
        assert data["state"] == {"key": "value"}
        assert "timestamp" in data

    def test_list_checkpoints(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        ckpt.save("exec-A", {"x": 1}, "node1")
        ckpt.save("exec-A", {"x": 2}, "node2")
        ckpt.save("exec-B", {"x": 3}, "node1")

        checkpoints = ckpt.list_checkpoints("exec-A")
        assert len(checkpoints) == 2
        assert all("exec-A" in c for c in checkpoints)

    def test_list_checkpoints_empty(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        result = ckpt.list_checkpoints("no-exec")
        assert result == []

    def test_clear_specific_execution(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        ckpt.save("exec-1", {"a": 1}, "nodeX")
        ckpt.save("exec-2", {"b": 2}, "nodeY")

        ckpt.clear("exec-1")

        assert ckpt.load("exec-1", "nodeX") is None
        assert ckpt.load("exec-2", "nodeY") == {"b": 2}

    def test_clear_all(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        ckpt.save("exec-1", {}, "nodeA")
        ckpt.save("exec-2", {}, "nodeB")

        ckpt.clear()

        assert ckpt.load("exec-1", "nodeA") is None
        assert ckpt.load("exec-2", "nodeB") is None

    def test_auto_creates_dir(self, tmp_path):
        new_dir = tmp_path / "subdir"
        assert not new_dir.exists()
        Checkpoint(checkpoint_dir=new_dir)
        assert new_dir.exists()

    def test_overwrite_checkpoint(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        ckpt.save("exec-1", {"v": 1}, "node")
        ckpt.save("exec-1", {"v": 99}, "node")

        loaded = ckpt.load("exec-1", "node")
        assert loaded == {"v": 99}

    def test_state_with_non_serializable_default_str(self, tmp_path):
        ckpt = Checkpoint(checkpoint_dir=tmp_path)
        # datetime objects are handled via default=str
        from datetime import datetime as dt

        state = {"ts": dt(2024, 1, 1)}
        ckpt.save("exec-1", state, "node")
        # Should not raise; timestamp gets serialized as string
        data = ckpt.load("exec-1", "node")
        assert "ts" in data
