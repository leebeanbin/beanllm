"""
Parameter Adapter
Provider별 파라미터 자동 변환
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...infrastructure.models import MODELS
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AdaptedParameters:
    """변환된 파라미터"""

    params: Dict[str, Any]
    removed: Dict[str, str]  # 제거된 파라미터와 이유
    warnings: list[str]


class ParameterAdapter:
    """
    Provider별 파라미터 자동 변환

    기능:
    1. 파라미터 이름 매핑 (max_tokens → max_output_tokens)
    2. 값 범위 조정 (temperature)
    3. 지원하지 않는 파라미터 제거
    4. 모델별 특수 처리
    """

    # Provider별 파라미터 매핑
    PARAM_MAPPING = {
        "openai": {
            "max_tokens": "max_tokens",  # 기본
            "temperature": "temperature",
            "top_p": "top_p",
            "stream": "stream",
        },
        "anthropic": {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "stream": "stream",
        },
        "google": {
            "max_tokens": "max_output_tokens",  # 변환 필요!
            "temperature": "temperature",
            "top_p": "top_p",
            "stream": "stream",
        },
        "ollama": {
            "max_tokens": "num_predict",  # 변환 필요!
            "temperature": "temperature",
            "top_p": "top_p",
            "stream": "stream",
        },
    }

    def __init__(self):
        pass

    def adapt(self, provider: str, model: str, params: Dict[str, Any]) -> AdaptedParameters:
        """
        파라미터 자동 변환

        Args:
            provider: Provider 이름
            model: 모델 ID
            params: 원본 파라미터

        Returns:
            AdaptedParameters: 변환된 파라미터 + 제거된 것들 + 경고
        """
        logger.debug(f"Adapting parameters for {provider}/{model}: {params}")

        adapted = {}
        removed = {}
        warnings = []

        # 1. 모델 메타데이터 가져오기
        model_config = self._get_model_config(provider, model)

        # 2. 파라미터별 처리
        for key, value in params.items():
            # 파라미터 이름 매핑
            mapped_key = self._map_parameter_name(provider, key)

            if mapped_key is None:
                # 알 수 없는 파라미터
                warnings.append(f"Unknown parameter: {key}")
                adapted[key] = value  # 그대로 전달
                continue

            # 모델이 지원하는지 확인
            if not self._is_parameter_supported(model_config, key, model):
                removed[key] = f"Model {model} does not support {key}"
                continue

            # 값 변환
            converted_value = self._convert_parameter_value(
                provider, model, key, value, model_config
            )

            if converted_value is None:
                removed[key] = f"Invalid value for {key}: {value}"
                continue

            adapted[mapped_key] = converted_value

        # 3. 특수 처리 (GPT-5 시리즈)
        if provider == "openai" and model_config:
            if model_config.get("uses_max_completion_tokens"):
                # max_tokens → max_completion_tokens
                if "max_tokens" in adapted:
                    adapted["max_completion_tokens"] = adapted.pop("max_tokens")
                    logger.debug(f"Converted max_tokens → max_completion_tokens for {model}")

        logger.debug(f"Adapted: {adapted}, Removed: {removed}")

        return AdaptedParameters(params=adapted, removed=removed, warnings=warnings)

    def _get_model_config(self, provider: str, model: str) -> Optional[Dict]:
        """모델 설정 가져오기"""
        # 날짜 버전 제거 (gpt-5-nano-2025-08-07 → gpt-5-nano)
        base_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)

        # MODELS에서 찾기
        if base_model in MODELS:
            return MODELS[base_model]

        # 원본 모델 이름으로 찾기
        if model in MODELS:
            return MODELS[model]

        return None

    def _map_parameter_name(self, provider: str, param_name: str) -> Optional[str]:
        """파라미터 이름 매핑"""
        provider_mapping = self.PARAM_MAPPING.get(provider)
        if not provider_mapping:
            return param_name  # 알 수 없는 provider

        return provider_mapping.get(param_name, param_name)

    def _is_parameter_supported(
        self, model_config: Optional[Dict], param_name: str, model: str
    ) -> bool:
        """모델이 파라미터를 지원하는지 확인"""
        if not model_config:
            # 설정이 없으면 지원한다고 가정
            return True

        # temperature 체크
        if param_name == "temperature":
            return model_config.get("supports_temperature", True)

        # max_tokens 체크
        if param_name == "max_tokens":
            # uses_max_completion_tokens가 True면 변환할 것이므로 지원함
            if model_config.get("uses_max_completion_tokens"):
                return True
            return model_config.get("supports_max_tokens", True)

        # 기타 파라미터는 지원
        return True

    def _convert_parameter_value(
        self, provider: str, model: str, param_name: str, value: Any, model_config: Optional[Dict]
    ) -> Optional[Any]:
        """파라미터 값 변환"""

        # temperature 범위 조정
        if param_name == "temperature":
            if not isinstance(value, (int, float)):
                return None

            # Anthropic: 0.0-1.0 엄격
            if provider == "anthropic":
                if value < 0.0:
                    logger.warning(f"Temperature {value} < 0.0, setting to 0.0")
                    return 0.0
                if value > 1.0:
                    logger.warning(f"Temperature {value} > 1.0, setting to 1.0")
                    return 1.0

            return value

        # max_tokens 체크
        if param_name == "max_tokens":
            if not isinstance(value, int):
                return None

            if value <= 0:
                return None

            # 모델의 max_tokens 제한 확인
            if model_config:
                max_allowed = model_config.get("max_tokens")
                if max_allowed and value > max_allowed:
                    logger.warning(
                        f"max_tokens {value} exceeds model limit {max_allowed}, "
                        f"setting to {max_allowed}"
                    )
                    return max_allowed

            return value

        # top_p
        if param_name == "top_p":
            if not isinstance(value, (int, float)):
                return None
            if value < 0.0 or value > 1.0:
                return None
            return value

        # stream
        if param_name == "stream":
            return bool(value)

        # 기타
        return value

    def validate_parameters(
        self, provider: str, model: str, params: Dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        파라미터 검증

        Returns:
            (is_valid, errors)
        """
        errors = []

        # 모델 설정 가져오기
        model_config = self._get_model_config(provider, model)

        for key, value in params.items():
            # 지원 여부 확인
            if not self._is_parameter_supported(model_config, key, model):
                errors.append(f"Parameter '{key}' not supported by model '{model}'")

            # 값 유효성 확인
            converted = self._convert_parameter_value(provider, model, key, value, model_config)
            if converted is None:
                errors.append(f"Invalid value for parameter '{key}': {value}")

        return len(errors) == 0, errors


# 전역 인스턴스
_adapter = ParameterAdapter()


def adapt_parameters(provider: str, model: str, params: Dict[str, Any]) -> AdaptedParameters:
    """파라미터 변환 (편의 함수)"""
    return _adapter.adapt(provider, model, params)


def validate_parameters(
    provider: str, model: str, params: Dict[str, Any]
) -> tuple[bool, list[str]]:
    """파라미터 검증 (편의 함수)"""
    return _adapter.validate_parameters(provider, model, params)
