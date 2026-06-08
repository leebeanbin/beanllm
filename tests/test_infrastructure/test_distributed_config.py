"""Tests for infrastructure/distributed/config.py and event_integration.py."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.config import (
    DistributedConfig,
    get_distributed_config,
    get_pipeline_config,
    reset_pipeline_config,
    set_distributed_config,
    update_pipeline_config,
)

# ---------------------------------------------------------------------------
# DistributedConfig init
# ---------------------------------------------------------------------------


class TestDistributedConfigInit:
    def setup_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def teardown_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def test_default_non_distributed_mode(self):
        config = DistributedConfig()
        assert config.use_distributed is False

    def test_distributed_mode_env_var(self, monkeypatch):
        monkeypatch.setenv("USE_DISTRIBUTED", "true")
        config = DistributedConfig()
        assert config.use_distributed is True

    def test_distributed_mode_activates_pipelines(self, monkeypatch):
        monkeypatch.setenv("USE_DISTRIBUTED", "true")
        config = DistributedConfig()
        assert config.ocr.use_distributed_queue is True
        assert config.ocr.enable_event_streaming is True
        assert config.vision_rag.use_distributed_queue is True
        assert config.multi_agent.use_kafka_bus is True
        assert config.chain.use_distributed_queue is True
        assert config.graph.use_distributed_queue is True

    def test_kafka_env_vars_applied(self, monkeypatch):
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka1:9092")
        monkeypatch.setenv("KAFKA_TOPIC_PREFIX", "myapp")
        config = DistributedConfig()
        assert config.kafka_bootstrap_servers == "kafka1:9092"
        assert config.kafka_topic_prefix == "myapp"


class TestGetDistributedConfig:
    def setup_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def teardown_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def test_returns_distributed_config(self):
        config = get_distributed_config()
        assert isinstance(config, DistributedConfig)

    def test_cached_singleton(self):
        c1 = get_distributed_config()
        c2 = get_distributed_config()
        assert c1 is c2

    def test_set_replaces_global_config(self):
        new_config = DistributedConfig()
        set_distributed_config(new_config)
        retrieved = get_distributed_config()
        assert retrieved is new_config


class TestUpdatePipelineConfig:
    def setup_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def teardown_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def test_update_known_pipeline_attribute(self):
        update_pipeline_config("ocr", enable_rate_limiting=False)
        config = get_pipeline_config("ocr")
        assert config.enable_rate_limiting is False

    def test_update_restores_pipeline_attribute(self):
        update_pipeline_config("ocr", enable_rate_limiting=True)
        config = get_pipeline_config("ocr")
        assert config.enable_rate_limiting is True

    def test_update_unknown_pipeline_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline type"):
            update_pipeline_config("nonexistent_xyz")

    def test_update_global_attribute_when_not_on_pipeline(self):
        update_pipeline_config("ocr", redis_host="newhost")
        config = get_distributed_config()
        assert config.redis_host == "newhost"

    def test_update_unknown_key_warns_and_not_raises(self):
        # Should log warning, not raise
        update_pipeline_config("ocr", completely_unknown_key_xyz=True)


class TestGetPipelineConfig:
    def test_returns_ocr_config(self):
        cfg = get_pipeline_config("ocr")
        assert hasattr(cfg, "enable_rate_limiting")

    def test_returns_vision_rag_config(self):
        cfg = get_pipeline_config("vision_rag")
        assert hasattr(cfg, "enable_rate_limiting")

    def test_returns_multi_agent_config(self):
        cfg = get_pipeline_config("multi_agent")
        assert cfg is not None

    def test_returns_chain_config(self):
        cfg = get_pipeline_config("chain")
        assert cfg is not None

    def test_returns_graph_config(self):
        cfg = get_pipeline_config("graph")
        assert cfg is not None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline type"):
            get_pipeline_config("totally_unknown_xyz")


class TestResetPipelineConfig:
    def setup_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def teardown_method(self):
        from beanllm.infrastructure.distributed import config as cfg_mod

        cfg_mod._global_distributed_config = None

    def test_reset_ocr_restores_defaults(self):
        update_pipeline_config("ocr", enable_rate_limiting=False)
        reset_pipeline_config("ocr")
        cfg = get_pipeline_config("ocr")
        assert cfg.enable_rate_limiting is True  # default is True

    def test_reset_vision_rag(self):
        reset_pipeline_config("vision_rag")
        cfg = get_pipeline_config("vision_rag")
        assert cfg is not None

    def test_reset_multi_agent(self):
        reset_pipeline_config("multi_agent")
        cfg = get_pipeline_config("multi_agent")
        assert cfg is not None

    def test_reset_chain(self):
        reset_pipeline_config("chain")
        cfg = get_pipeline_config("chain")
        assert cfg is not None

    def test_reset_graph(self):
        reset_pipeline_config("graph")
        cfg = get_pipeline_config("graph")
        assert cfg is not None

    def test_reset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline type"):
            reset_pipeline_config("nonexistent_type")
