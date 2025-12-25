"""
Model Scanner - 모델 스캐너 구현
"""

from typing import Dict, List

from .types import ScannedModel

try:
    from ...utils.config import EnvConfig
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)

    class EnvConfig:
        def __init__(self):
            pass

        def is_provider_available(self, provider: str) -> bool:
            return False

        @property
        def OPENAI_API_KEY(self):
            import os

            return os.getenv("OPENAI_API_KEY")

        @property
        def GEMINI_API_KEY(self):
            import os

            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        @property
        def OLLAMA_HOST(self):
            import os

            return os.getenv("OLLAMA_HOST", "http://localhost:11434")


logger = get_logger(__name__)


class ModelScanner:
    """
    각 Provider API에서 실시간으로 모델 목록 가져오기
    """

    def __init__(self):
        self.config = EnvConfig()

    async def scan_all(self) -> Dict[str, List[ScannedModel]]:
        """
        모든 활성화된 Provider 스캔

        Returns:
            {provider_name: [ScannedModel, ...]}
        """
        results = {}

        # OpenAI
        if self.config.is_provider_available("openai"):
            try:
                results["openai"] = await self.scan_openai()
            except Exception as e:
                logger.error(f"OpenAI scan failed: {e}")
                results["openai"] = []

        # Anthropic (API 없음, 로컬 목록만)
        results["anthropic"] = await self.scan_anthropic()

        # Gemini
        if self.config.is_provider_available("gemini"):
            try:
                results["google"] = await self.scan_gemini()
            except Exception as e:
                logger.error(f"Gemini scan failed: {e}")
                results["google"] = []

        # Ollama
        try:
            results["ollama"] = await self.scan_ollama()
        except Exception as e:
            logger.error(f"Ollama scan failed: {e}")
            results["ollama"] = []

        return results

    async def scan_openai(self) -> List[ScannedModel]:
        """OpenAI API에서 모델 목록 가져오기"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.config.OPENAI_API_KEY)
            response = await client.models.list()

            models = []
            for model in response.data:
                # 채팅 모델만 필터링
                if self._is_chat_model(model.id):
                    models.append(
                        ScannedModel(
                            model_id=model.id,
                            provider="openai",
                            created_at=str(model.created) if hasattr(model, "created") else None,
                            raw_data=model.model_dump() if hasattr(model, "model_dump") else None,
                        )
                    )

            logger.info(f"✅ OpenAI: {len(models)} chat models found")
            return models

        except ImportError:
            logger.warning("OpenAI SDK not installed. Run: pip install beanllm[openai]")
            return []
        except Exception as e:
            logger.error(f"OpenAI scan error: {e}")
            return []

    def _is_chat_model(self, model_id: str) -> bool:
        """
        채팅 모델인지 확인 (embedding, tts, whisper 등 제외)
        """
        excluded = [
            "embedding",
            "tts",
            "dall-e",
            "whisper",
            "codex",
            "audio",
            "realtime",
            "image",
            "moderation",
            "diarize",
            "transcribe",
        ]
        return not any(x in model_id.lower() for x in excluded)

    async def scan_anthropic(self) -> List[ScannedModel]:
        """
        Anthropic는 API로 모델 목록 제공 안함
        공식 모델 목록만 반환
        """
        # 공식 문서 기반 모델 목록
        official_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]

        models = [ScannedModel(model_id=m, provider="anthropic") for m in official_models]

        logger.info(f"✅ Anthropic: {len(models)} models (official list)")
        return models

    async def scan_gemini(self) -> List[ScannedModel]:
        """Google Gemini API에서 모델 목록 가져오기"""
        try:
            from google import genai

            client = genai.Client(api_key=self.config.GEMINI_API_KEY)

            # Sync version for now (async support varies)
            models_response = client.models.list()

            models = []
            for model in models_response.models:
                # "models/gemini-2.5-flash" → "gemini-2.5-flash"
                model_id = model.name.split("/")[-1] if "/" in model.name else model.name

                models.append(
                    ScannedModel(
                        model_id=model_id, provider="google", raw_data={"name": model.name}
                    )
                )

            logger.info(f"✅ Gemini: {len(models)} models found")
            return models

        except ImportError:
            logger.warning("Gemini SDK not installed. Run: pip install beanllm[gemini]")
            return []
        except Exception as e:
            logger.error(f"Gemini scan error: {e}")
            return []

    async def scan_ollama(self) -> List[ScannedModel]:
        """Ollama 로컬 모델 스캔"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.OLLAMA_HOST}/api/tags")
                data = response.json()

                models = []
                for model in data.get("models", []):
                    models.append(
                        ScannedModel(model_id=model["name"], provider="ollama", raw_data=model)
                    )

                logger.info(f"✅ Ollama: {len(models)} local models found")
                return models

        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return []

    def scan_openai_sync(self) -> List[ScannedModel]:
        """OpenAI API 동기 버전"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            response = client.models.list()

            models = []
            for model in response.data:
                if self._is_chat_model(model.id):
                    models.append(
                        ScannedModel(
                            model_id=model.id,
                            provider="openai",
                            created_at=str(model.created) if hasattr(model, "created") else None,
                        )
                    )

            return models

        except Exception as e:
            logger.error(f"OpenAI sync scan error: {e}")
            return []
