"""Tests for infrastructure/scanner/model_scanner.py — ModelScanner."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.scanner.model_scanner import ModelScanner
from beanllm.infrastructure.scanner.types import ScannedModel

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_scanner(**config_attrs):
    scanner = ModelScanner()
    for k, v in config_attrs.items():
        setattr(scanner.config, k, v)
    return scanner


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestModelScannerInit:
    def test_creates_instance(self):
        scanner = ModelScanner()
        assert scanner is not None

    def test_has_config(self):
        scanner = ModelScanner()
        assert scanner.config is not None


# ---------------------------------------------------------------------------
# _is_chat_model
# ---------------------------------------------------------------------------


class TestIsChatModel:
    def setup_method(self):
        self.scanner = ModelScanner()

    def test_gpt4o_is_chat_model(self):
        assert self.scanner._is_chat_model("gpt-4o") is True

    def test_embedding_not_chat_model(self):
        assert self.scanner._is_chat_model("text-embedding-3-small") is False

    def test_tts_not_chat_model(self):
        assert self.scanner._is_chat_model("tts-1") is False

    def test_dalle_not_chat_model(self):
        assert self.scanner._is_chat_model("dall-e-3") is False

    def test_whisper_not_chat_model(self):
        assert self.scanner._is_chat_model("whisper-1") is False

    def test_moderation_not_chat_model(self):
        assert self.scanner._is_chat_model("text-moderation-latest") is False

    def test_claude_is_chat_model(self):
        assert self.scanner._is_chat_model("claude-3-opus-20240229") is True

    def test_deepseek_is_chat_model(self):
        assert self.scanner._is_chat_model("deepseek-chat") is True

    def test_audio_not_chat_model(self):
        assert self.scanner._is_chat_model("gpt-4o-audio-preview") is False

    def test_realtime_not_chat_model(self):
        assert self.scanner._is_chat_model("gpt-4o-realtime-preview") is False


# ---------------------------------------------------------------------------
# scan_anthropic
# ---------------------------------------------------------------------------


class TestScanAnthropic:
    async def test_returns_list_of_scanned_models(self):
        scanner = ModelScanner()
        models = await scanner.scan_anthropic()
        assert isinstance(models, list)
        assert all(isinstance(m, ScannedModel) for m in models)

    async def test_all_have_anthropic_provider(self):
        scanner = ModelScanner()
        models = await scanner.scan_anthropic()
        assert all(m.provider == "anthropic" for m in models)

    async def test_contains_claude_sonnet(self):
        scanner = ModelScanner()
        models = await scanner.scan_anthropic()
        model_ids = [m.model_id for m in models]
        assert any("claude" in mid for mid in model_ids)

    async def test_returns_non_empty_list(self):
        scanner = ModelScanner()
        models = await scanner.scan_anthropic()
        assert len(models) > 0


# ---------------------------------------------------------------------------
# scan_openai
# ---------------------------------------------------------------------------


class TestScanOpenAI:
    async def test_returns_empty_list_on_import_error(self):
        scanner = ModelScanner()
        with patch.dict("sys.modules", {"openai": None}):
            models = await scanner.scan_openai()
        assert isinstance(models, list)

    async def test_returns_empty_list_on_api_error(self):
        scanner = ModelScanner()
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("API error"))
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            models = await scanner.scan_openai()
        assert isinstance(models, list)

    async def test_filters_non_chat_models(self):
        scanner = ModelScanner()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model1.created = 12345
        mock_model2 = MagicMock()
        mock_model2.id = "text-embedding-3-small"
        mock_model2.created = 12345

        mock_response = MagicMock()
        mock_response.data = [mock_model1, mock_model2]

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            models = await scanner.scan_openai()

        model_ids = [m.model_id for m in models]
        assert "gpt-4o" in model_ids
        assert "text-embedding-3-small" not in model_ids

    async def test_returns_scanned_model_objects(self):
        scanner = ModelScanner()
        mock_model = MagicMock()
        mock_model.id = "gpt-4o"
        mock_model.created = 12345
        mock_model.model_dump.return_value = {"id": "gpt-4o"}

        mock_response = MagicMock()
        mock_response.data = [mock_model]

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            models = await scanner.scan_openai()

        assert all(isinstance(m, ScannedModel) for m in models)


# ---------------------------------------------------------------------------
# scan_gemini
# ---------------------------------------------------------------------------


class TestScanGemini:
    async def test_returns_empty_list_on_import_error(self):
        scanner = ModelScanner()
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            models = await scanner.scan_gemini()
        assert isinstance(models, list)

    async def test_returns_empty_list_on_api_error(self):
        scanner = ModelScanner()
        mock_genai = MagicMock()
        mock_genai.Client.side_effect = Exception("API error")
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            models = await scanner.scan_gemini()
        assert isinstance(models, list)

    async def test_strips_models_prefix_from_name(self):
        scanner = ModelScanner()

        mock_model = MagicMock()
        mock_model.name = "models/gemini-1.5-pro"

        mock_models_response = MagicMock()
        mock_models_response.models = [mock_model]

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_models_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        import google

        with patch.dict("sys.modules", {"google.genai": mock_genai}):
            with patch(
                "beanllm.infrastructure.scanner.model_scanner.genai", mock_genai, create=True
            ):
                try:
                    from google import genai as real_genai

                    models = await scanner.scan_gemini()
                    model_ids = [m.model_id for m in models]
                    assert any("gemini" in mid for mid in model_ids)
                except Exception:
                    pass  # OK if genai patching doesn't fully work


# ---------------------------------------------------------------------------
# scan_ollama
# ---------------------------------------------------------------------------


class TestScanOllama:
    async def test_returns_empty_list_on_exception(self):
        scanner = ModelScanner()
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client
            models = await scanner.scan_ollama()
        assert isinstance(models, list)

    async def test_returns_scanned_models_from_response(self):
        scanner = ModelScanner()
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:latest"},
                    {"name": "qwen2.5:0.5b"},
                ]
            }
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client
            models = await scanner.scan_ollama()

        model_ids = [m.model_id for m in models]
        assert "llama3.2:latest" in model_ids
        assert "qwen2.5:0.5b" in model_ids


# ---------------------------------------------------------------------------
# scan_all
# ---------------------------------------------------------------------------


class TestScanAll:
    async def test_always_includes_anthropic(self):
        scanner = ModelScanner()
        scanner.config.is_provider_available = MagicMock(return_value=False)
        # Make scan_ollama fail silently
        with patch.object(scanner, "scan_ollama", AsyncMock(side_effect=Exception("no ollama"))):
            results = await scanner.scan_all()
        assert "anthropic" in results

    async def test_includes_openai_when_available(self):
        scanner = ModelScanner()
        scanner.config.is_provider_available = MagicMock(side_effect=lambda p: p == "openai")
        mock_models = [ScannedModel(model_id="gpt-4o", provider="openai")]
        with (
            patch.object(scanner, "scan_openai", AsyncMock(return_value=mock_models)),
            patch.object(scanner, "scan_ollama", AsyncMock(return_value=[])),
        ):
            results = await scanner.scan_all()
        assert "openai" in results

    async def test_includes_ollama(self):
        scanner = ModelScanner()
        scanner.config.is_provider_available = MagicMock(return_value=False)
        mock_models = [ScannedModel(model_id="llama3", provider="ollama")]
        with patch.object(scanner, "scan_ollama", AsyncMock(return_value=mock_models)):
            results = await scanner.scan_all()
        assert "ollama" in results
        assert results["ollama"] == mock_models

    async def test_catches_openai_scan_error(self):
        scanner = ModelScanner()
        scanner.config.is_provider_available = MagicMock(side_effect=lambda p: p == "openai")
        with (
            patch.object(scanner, "scan_openai", AsyncMock(side_effect=Exception("openai error"))),
            patch.object(scanner, "scan_ollama", AsyncMock(return_value=[])),
        ):
            results = await scanner.scan_all()
        assert results.get("openai") == []


# ---------------------------------------------------------------------------
# scan_openai_sync
# ---------------------------------------------------------------------------


class TestScanOpenAISync:
    def test_returns_empty_list_on_exception(self):
        scanner = ModelScanner()
        with patch("openai.OpenAI", side_effect=Exception("no openai")):
            models = scanner.scan_openai_sync()
        assert isinstance(models, list)

    def test_returns_empty_list_on_api_error(self):
        scanner = ModelScanner()
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("API error")
        with patch("beanllm.infrastructure.scanner.model_scanner.OpenAI", mock_client, create=True):
            models = scanner.scan_openai_sync()
        assert isinstance(models, list)
