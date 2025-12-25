"""
Model Registry
모델 레지스트리 - 활성화된 모델 정보 관리
"""

import logging
from typing import Any, Dict, List, Optional

from ...infrastructure.models import (
    ModelCapabilityInfo,
    ModelStatus,
    ParameterInfo,
    ProviderInfo,
    get_all_models,
    get_default_model,
    get_models_by_provider,
)
from ...utils.config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _providers: Dict[str, ProviderInfo] = {}
    _models: Dict[str, ModelCapabilityInfo] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._scan_providers()
        self._scan_models()

    def _scan_providers(self):
        provider_configs = [
            ("openai", "OPENAI_API_KEY", Config.OPENAI_API_KEY),
            ("anthropic", "ANTHROPIC_API_KEY", Config.ANTHROPIC_API_KEY),
            ("google", "GEMINI_API_KEY", Config.GEMINI_API_KEY),
            ("ollama", "OLLAMA_HOST", Config.OLLAMA_HOST),
        ]
        for name, env_key, env_value in provider_configs:
            try:
                is_available = bool(env_value) if name != "ollama" else True
                status = ModelStatus.ACTIVE if is_available else ModelStatus.INACTIVE
                available_models = []
                default_model = None
                if status == ModelStatus.ACTIVE:
                    models = get_models_by_provider(name)
                    available_models = list(models.keys())
                    if available_models:
                        default_model = get_default_model(provider=name, model_type="llm")
                        if not default_model and available_models:
                            default_model = available_models[0]
                self._providers[name] = ProviderInfo(
                    name=name,
                    status=status,
                    env_key=env_key,
                    env_value_set=bool(env_value),
                    available_models=available_models,
                    default_model=default_model,
                )
            except Exception as e:
                logger.error(f"Error scanning provider {name}: {e}")
                self._providers[name] = ProviderInfo(
                    name=name,
                    status=ModelStatus.ERROR,
                    env_key=env_key,
                    env_value_set=bool(env_value),
                    error_message=str(e),
                )

    def _scan_models(self):
        all_models = get_all_models()
        for model_name, model_config in all_models.items():
            try:
                parameters = []
                supports_temp = model_config.get("supports_temperature", True)
                default_temp = model_config.get("temperature", 0.0)
                parameters.append(
                    ParameterInfo(
                        name="temperature",
                        type="float",
                        description="응답의 창의성/랜덤성 조절 (0.0-2.0)",
                        default=default_temp,
                        required=False,
                        supported=supports_temp,
                        notes=(
                            "일부 모델(gpt-5-mini, o3 등)은 temperature 미지원"
                            if not supports_temp
                            else None
                        ),
                    )
                )
                supports_max_tokens = model_config.get("supports_max_tokens", True)
                uses_max_completion = model_config.get("uses_max_completion_tokens", False)
                default_max_tokens = model_config.get("max_tokens")
                if uses_max_completion:
                    parameters.append(
                        ParameterInfo(
                            name="max_completion_tokens",
                            type="int",
                            description="생성할 최대 토큰 수 (새로운 모델용)",
                            default=default_max_tokens,
                            required=False,
                            supported=supports_max_tokens,
                            notes="새로운 모델(gpt-5, gpt-4.1 시리즈)은 max_completion_tokens 사용",
                        )
                    )
                else:
                    parameters.append(
                        ParameterInfo(
                            name="max_tokens",
                            type="int",
                            description="생성할 최대 토큰 수",
                            default=default_max_tokens,
                            required=False,
                            supported=supports_max_tokens,
                            notes=(
                                "일부 모델은 max_tokens 미지원" if not supports_max_tokens else None
                            ),
                        )
                    )
                example_usage = self._generate_example_usage(model_name, model_config)
                self._models[model_name] = ModelCapabilityInfo(
                    model_name=model_name,
                    display_name=model_config.get("display_name", model_name),
                    provider=model_config["provider"],
                    model_type=model_config["type"],
                    supports_streaming=True,
                    supports_temperature=supports_temp,
                    supports_max_tokens=supports_max_tokens,
                    uses_max_completion_tokens=uses_max_completion,
                    max_tokens=default_max_tokens,
                    default_temperature=default_temp,
                    description=model_config.get("description", ""),
                    use_case=model_config.get("use_case", ""),
                    parameters=parameters,
                    example_usage=example_usage,
                )
            except Exception as e:
                logger.error(f"Error scanning model {model_name}: {e}")

    def _generate_example_usage(self, model_name: str, model_config: dict) -> str:
        provider = model_config["provider"]
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
            "ollama": "OLLAMA_HOST",
        }
        env_key = env_key_map.get(provider, f"{provider.upper()}_API_KEY")
        example = f"""# {model_name} 사용 예제

## 환경변수 설정
```bash
export {env_key}="your-api-key"
```

## 기본 사용법 (insightstock-ai-service)
```python
from src.services.llm_service import LLMService
from src.models.model_config import ModelConfigManager

model_config = ModelConfigManager.get_model_config("{model_name}")
llm_service = LLMService(model_config)
response = await llm_service.chat(messages=[{{"role": "user", "content": "안녕하세요"}}])
print(response)
```

## 스트리밍 사용법
```python
async for chunk in llm_service.stream_chat(messages=[{{"role": "user", "content": "안녕하세요"}}]):
    print(chunk, end="", flush=True)
```

## 파라미터 설정
```python
"""
        if model_config.get("supports_temperature", True):
            example += f"temperature = {model_config.get('temperature', 0.0)}\n"
        if model_config.get("uses_max_completion_tokens", False):
            example += f"max_completion_tokens = {model_config.get('max_tokens', 1000)}\n"
        elif model_config.get("supports_max_tokens", True):
            example += f"max_tokens = {model_config.get('max_tokens', 1000)}\n"
        example += "```\n"
        return example

    def get_active_providers(self) -> List[ProviderInfo]:
        return [p for p in self._providers.values() if p.status == ModelStatus.ACTIVE]

    def get_all_providers(self) -> Dict[str, ProviderInfo]:
        return self._providers.copy()

    def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        name_map = {"claude": "anthropic", "gemini": "google"}
        normalized = name_map.get(provider_name.lower(), provider_name.lower())
        return self._providers.get(normalized)

    def get_available_models(self, provider: Optional[str] = None) -> List[ModelCapabilityInfo]:
        if provider:
            name_map = {"claude": "anthropic", "gemini": "google"}
            normalized = name_map.get(provider.lower(), provider.lower())
            return [m for m in self._models.values() if m.provider == normalized]
        return list(self._models.values())

    def get_model_info(self, model_name: str) -> Optional[ModelCapabilityInfo]:
        return self._models.get(model_name)

    def refresh(self):
        self._providers.clear()
        self._models.clear()
        self._initialize()

    def get_summary(self) -> Dict[str, Any]:
        active = self.get_active_providers()
        return {
            "total_providers": len(self._providers),
            "active_providers": len(active),
            "total_models": len(self._models),
            "providers": {
                p.name: {
                    "status": p.status.value,
                    "available_models_count": len(p.available_models),
                    "default_model": p.default_model,
                }
                for p in self._providers.values()
            },
            "active_provider_names": [p.name for p in active],
        }


def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
