"""Tests for utils/logging/structured_logger.py — StructuredLogger, LogLevel."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from beanllm.utils.logging.structured_logger import (
    LogLevel,
    StructuredLogger,
    get_structured_logger,
)

# ---------------------------------------------------------------------------
# LogLevel enum
# ---------------------------------------------------------------------------


class TestLogLevel:
    def test_debug_value(self):
        assert LogLevel.DEBUG == "debug"

    def test_info_value(self):
        assert LogLevel.INFO == "info"

    def test_warning_value(self):
        assert LogLevel.WARNING == "warning"

    def test_error_value(self):
        assert LogLevel.ERROR == "error"

    def test_critical_value(self):
        assert LogLevel.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------


class TestStructuredLogger:
    def _make_logger(self, enable_structured=True):
        sl = StructuredLogger("test.module", enable_structured=enable_structured)
        sl.logger = MagicMock()
        return sl

    def test_init_enables_structured(self):
        sl = StructuredLogger("test.module")
        assert sl.enable_structured is True

    def test_init_structured_disabled(self):
        sl = StructuredLogger("test.module", enable_structured=False)
        assert sl.enable_structured is False

    def test_log_operation_structured_format(self):
        sl = self._make_logger(enable_structured=True)
        sl.log_operation("info", "api_call", "success", provider="openai")
        sl.logger.info.assert_called_once()
        call_arg = sl.logger.info.call_args.args[0]
        assert isinstance(call_arg, dict)
        assert call_arg["operation"] == "api_call"
        assert call_arg["status"] == "success"
        assert call_arg["provider"] == "openai"

    def test_log_operation_plain_format(self):
        sl = self._make_logger(enable_structured=False)
        sl.log_operation("info", "file_load", "success", filepath="/tmp/file")
        sl.logger.info.assert_called_once()
        call_arg = sl.logger.info.call_args.args[0]
        assert isinstance(call_arg, str)
        assert "file_load" in call_arg
        assert "success" in call_arg

    def test_log_operation_no_context_plain(self):
        sl = self._make_logger(enable_structured=False)
        sl.log_operation("debug", "op", "ok")
        sl.logger.debug.assert_called_once()
        call_arg = sl.logger.debug.call_args.args[0]
        assert "op ok" in call_arg

    def test_log_operation_uses_correct_level(self):
        sl = self._make_logger()
        sl.log_operation("warning", "check", "alert")
        sl.logger.warning.assert_called_once()

    def test_log_operation_error_level(self):
        sl = self._make_logger()
        sl.log_operation("error", "check", "failed")
        sl.logger.error.assert_called_once()


class TestLogFileLOad:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_success_with_count(self):
        sl = self._make_logger()
        sl.log_file_load("/path/file.pdf", count=5)
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert arg["document_count"] == 5

    def test_success_without_count(self):
        sl = self._make_logger()
        sl.log_file_load("/path/file.pdf")
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert "document_count" not in arg

    def test_failure_uses_error_level(self):
        sl = self._make_logger()
        sl.log_file_load("/path/bad.pdf", success=False, error="Not found")
        sl.logger.error.assert_called_once()
        arg = sl.logger.error.call_args.args[0]
        assert arg["status"] == "failed"
        assert arg["error"] == "Not found"

    def test_filepath_in_context(self):
        sl = self._make_logger()
        sl.log_file_load("/my/path.txt", count=2)
        arg = sl.logger.info.call_args.args[0]
        assert arg["filepath"] == "/my/path.txt"


class TestLogApiCall:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_success_call(self):
        sl = self._make_logger()
        sl.log_api_call("openai", "gpt-4o", success=True, latency_ms=250, tokens_used=1500)
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert arg["provider"] == "openai"
        assert arg["model"] == "gpt-4o"
        assert arg["latency_ms"] == 250
        assert arg["tokens_used"] == 1500

    def test_failed_call(self):
        sl = self._make_logger()
        sl.log_api_call("anthropic", "claude-3", success=False, error="Rate limit")
        sl.logger.error.assert_called_once()
        arg = sl.logger.error.call_args.args[0]
        assert arg["error"] == "Rate limit"

    def test_without_optional_fields(self):
        sl = self._make_logger()
        sl.log_api_call("ollama", "llama3")
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert "latency_ms" not in arg
        assert "tokens_used" not in arg


class TestLogEmbeddingGeneration:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_success(self):
        sl = self._make_logger()
        sl.log_embedding_generation(100, 1536, latency_ms=500)
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert arg["text_count"] == 100
        assert arg["embedding_dim"] == 1536
        assert arg["latency_ms"] == 500

    def test_failure(self):
        sl = self._make_logger()
        sl.log_embedding_generation(50, 768, success=False, error="OOM")
        sl.logger.error.assert_called_once()
        arg = sl.logger.error.call_args.args[0]
        assert arg["error"] == "OOM"

    def test_no_latency(self):
        sl = self._make_logger()
        sl.log_embedding_generation(10, 512)
        arg = sl.logger.info.call_args.args[0]
        assert "latency_ms" not in arg


class TestLogVectorSearch:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_basic_search(self):
        sl = self._make_logger()
        sl.log_vector_search("What is AI?", 5)
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert arg["result_count"] == 5
        assert arg["search_type"] == "similarity"

    def test_hybrid_search(self):
        sl = self._make_logger()
        sl.log_vector_search("AI query", 3, search_type="hybrid", latency_ms=50)
        arg = sl.logger.info.call_args.args[0]
        assert arg["search_type"] == "hybrid"
        assert arg["latency_ms"] == 50

    def test_long_query_truncated(self):
        sl = self._make_logger()
        long_query = "x" * 200
        sl.log_vector_search(long_query, 4)
        arg = sl.logger.info.call_args.args[0]
        assert len(arg["query"]) <= 100


class TestLogCacheOperation:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_cache_hit(self):
        sl = self._make_logger()
        sl.log_cache_operation("embedding", "get", hit=True)
        sl.logger.debug.assert_called_once()
        arg = sl.logger.debug.call_args.args[0]
        assert arg["cache_hit"] is True
        assert arg["status"] == "hit"

    def test_cache_miss_with_key(self):
        sl = self._make_logger()
        sl.log_cache_operation("prompt", "get", hit=False, key="tmpl_123")
        arg = sl.logger.debug.call_args.args[0]
        assert arg["cache_hit"] is False
        assert arg["key"] == "tmpl_123"

    def test_cache_set_operation(self):
        sl = self._make_logger()
        sl.log_cache_operation("model", "set", hit=False)
        arg = sl.logger.debug.call_args.args[0]
        assert arg["cache_operation"] == "set"

    def test_no_key_no_key_field(self):
        sl = self._make_logger()
        sl.log_cache_operation("embedding", "get", hit=True)
        arg = sl.logger.debug.call_args.args[0]
        assert "key" not in arg


class TestLogDuration:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_success_logs_duration(self):
        sl = self._make_logger()
        with sl.log_duration("pdf_parsing", file="doc.pdf") as ctx:
            ctx["page_count"] = 42
        sl.logger.info.assert_called_once()
        arg = sl.logger.info.call_args.args[0]
        assert arg["operation"] == "pdf_parsing"
        assert "duration_ms" in arg
        assert arg["page_count"] == 42

    def test_failure_logs_error(self):
        sl = self._make_logger()
        with pytest.raises(ValueError):
            with sl.log_duration("operation") as ctx:
                raise ValueError("test error")
        sl.logger.error.assert_called_once()
        arg = sl.logger.error.call_args.args[0]
        assert arg["status"] == "failed"
        assert "test error" in arg["error"]
        assert "duration_ms" in arg


class TestConvenienceMethods:
    def _make_logger(self):
        sl = StructuredLogger("test.module")
        sl.logger = MagicMock()
        return sl

    def test_debug(self):
        sl = self._make_logger()
        sl.debug("test message")
        sl.logger.debug.assert_called_once()

    def test_info(self):
        sl = self._make_logger()
        sl.info("info message", key="value")
        sl.logger.info.assert_called_once()

    def test_warning(self):
        sl = self._make_logger()
        sl.warning("warning message")
        sl.logger.warning.assert_called_once()

    def test_error(self):
        sl = self._make_logger()
        sl.error("error message")
        sl.logger.error.assert_called_once()


class TestGetStructuredLogger:
    def test_returns_structured_logger_instance(self):
        logger = get_structured_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_with_structured_disabled(self):
        logger = get_structured_logger("test.module", enable_structured=False)
        assert logger.enable_structured is False
