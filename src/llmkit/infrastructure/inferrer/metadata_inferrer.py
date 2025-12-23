"""
Metadata Inferrer - 메타데이터 추론기 구현
"""

import re
from datetime import datetime
from typing import Dict

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class MetadataInferrer:
    """
    패턴 기반으로 모델 메타데이터 추론

    새로운 모델이 발견되었을 때, 모델 이름 패턴을 분석해서
    지원하는 파라미터를 추론합니다.
    """

    # 추론 규칙 DB
    INFERENCE_RULES = {
        "openai": {
            "patterns": [
                {
                    "match": r"gpt-5.*|gpt-4\.1.*",
                    "name": "GPT-5/4.1 Series",
                    "rules": {
                        "uses_max_completion_tokens": True,
                        "supports_max_tokens": False,
                    },
                },
                {
                    "match": r".*nano.*",
                    "name": "Nano Models",
                    "rules": {
                        "supports_temperature": False,
                        "max_tokens": 8192,
                        "tier": "nano",
                        "speed": "fastest",
                        "notes": "Temperature parameter not supported",
                    },
                },
                {
                    "match": r".*mini.*",
                    "name": "Mini Models",
                    "rules": {
                        "supports_temperature": True,
                        "max_tokens": 16384,
                        "tier": "mini",
                        "speed": "fast",
                    },
                },
                {
                    "match": r"o3.*|o4.*",
                    "name": "O-Series (Reasoning)",
                    "rules": {
                        "supports_temperature": False,
                        "max_tokens": 16384,
                        "notes": "Reasoning models, temperature not supported",
                    },
                },
            ],
            "defaults": {
                "supports_streaming": True,
                "supports_temperature": True,
                "temperature_range": [0.0, 2.0],
                "supports_max_tokens": True,
                "uses_max_completion_tokens": False,
                "max_tokens": 128000,
            },
        },
        "anthropic": {
            "patterns": [
                {
                    "match": r"claude-4.*",
                    "name": "Claude 4 Series",
                    "rules": {
                        "max_tokens": 16384,
                        "description": "Claude 4 시리즈 (최신)",
                    },
                },
                {
                    "match": r"claude-3-5.*",
                    "name": "Claude 3.5 Series",
                    "rules": {
                        "max_tokens": 8192,
                        "description": "Claude 3.5 시리즈",
                    },
                },
                {
                    "match": r".*opus.*",
                    "name": "Opus Tier",
                    "rules": {
                        "tier": "opus",
                        "max_tokens": 4096,
                        "description": "최고 성능 모델",
                    },
                },
                {
                    "match": r".*sonnet.*",
                    "name": "Sonnet Tier",
                    "rules": {
                        "tier": "sonnet",
                        "max_tokens": 8192,
                        "description": "균형잡힌 모델",
                    },
                },
                {
                    "match": r".*haiku.*",
                    "name": "Haiku Tier",
                    "rules": {
                        "tier": "haiku",
                        "max_tokens": 4096,
                        "description": "빠른 모델",
                    },
                },
            ],
            "defaults": {
                "supports_streaming": True,
                "supports_temperature": True,
                "temperature_range": [0.0, 1.0],
                "supports_max_tokens": True,
                "uses_max_completion_tokens": False,
                "max_tokens": 8192,
            },
        },
        "google": {
            "patterns": [
                {
                    "match": r"gemini-2\.5.*",
                    "name": "Gemini 2.5 Series",
                    "rules": {
                        "supports_thinking": True,
                        "max_tokens": 8192,
                        "description": "Gemini 2.5 (Thinking 모드 지원)",
                    },
                },
                {
                    "match": r"gemini-2\.0.*",
                    "name": "Gemini 2.0 Series",
                    "rules": {
                        "supports_thinking": False,
                        "max_tokens": 8192,
                        "description": "Gemini 2.0",
                    },
                },
                {
                    "match": r"gemini-1\.5.*",
                    "name": "Gemini 1.5 Series",
                    "rules": {
                        "supports_thinking": False,
                        "max_tokens": 8192,
                        "description": "Gemini 1.5",
                    },
                },
                {
                    "match": r".*flash.*",
                    "name": "Flash Tier",
                    "rules": {
                        "tier": "flash",
                        "speed": "fast",
                    },
                },
                {
                    "match": r".*pro.*",
                    "name": "Pro Tier",
                    "rules": {
                        "tier": "pro",
                        "speed": "balanced",
                    },
                },
            ],
            "defaults": {
                "supports_streaming": True,
                "supports_temperature": True,
                "temperature_range": [0.0, 2.0],
                "uses_max_output_tokens": True,
                "max_tokens": 8192,
            },
        },
        "ollama": {
            "defaults": {
                "supports_streaming": True,
                "supports_temperature": True,
                "uses_num_predict": True,
                "description": "Ollama 로컬 모델",
            }
        },
    }

    def infer(self, provider: str, model_id: str) -> Dict:
        """
        패턴 기반으로 모델 메타데이터 추론

        Args:
            provider: Provider 이름 (openai, anthropic, google, ollama)
            model_id: 모델 ID

        Returns:
            추론된 메타데이터 딕셔너리
        """
        # 날짜 제거 (기본 모델 이름 추출)
        base_model = self._extract_base_model(model_id)

        # Provider 설정 가져오기
        provider_config = self.INFERENCE_RULES.get(provider, {})

        # 기본 메타데이터
        metadata = {
            "model_id": model_id,
            "display_name": model_id,
            "provider": provider,
            "base_model": base_model if base_model != model_id else None,
            "is_inferred": True,
            "inferred_at": datetime.now().isoformat(),
            "inference_confidence": 0.0,
            "matched_patterns": [],
        }

        # Defaults 적용
        if "defaults" in provider_config:
            metadata.update(provider_config["defaults"])

        # 패턴 매칭
        patterns = provider_config.get("patterns", [])
        matched_rules = []

        for pattern_rule in patterns:
            pattern = pattern_rule["match"]
            if re.match(pattern, base_model, re.IGNORECASE):
                matched_rules.append(pattern_rule)
                metadata["matched_patterns"].append(pattern_rule["name"])
                # 규칙 적용
                metadata.update(pattern_rule["rules"])

        # 신뢰도 계산
        if matched_rules:
            # 매칭된 패턴이 많을수록 신뢰도 높음
            metadata["inference_confidence"] = min(0.9, 0.5 + len(matched_rules) * 0.2)
        else:
            # 매칭 없으면 defaults만 사용
            metadata["inference_confidence"] = 0.3

        logger.debug(
            f"Inferred metadata for {model_id}: "
            f"confidence={metadata['inference_confidence']:.2f}, "
            f"matched={len(matched_rules)} patterns"
        )

        return metadata

    def _extract_base_model(self, model_id: str) -> str:
        """
        모델 ID에서 기본 모델 이름 추출 (날짜 제거)

        Examples:
            gpt-5-nano-2025-08-07 → gpt-5-nano
            claude-3-5-sonnet-20241022 → claude-3-5-sonnet
            gemini-2.5-flash → gemini-2.5-flash (변경 없음)
        """
        base = model_id

        # YYYY-MM-DD 형식 제거
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base)

        # YYYYMMDD 형식 제거
        base = re.sub(r"-\d{8}$", "", base)

        # YYYY 형식 제거
        base = re.sub(r"-\d{4}$", "", base)

        return base

    def get_inference_rules(self, provider: str) -> Dict:
        """특정 Provider의 추론 규칙 조회"""
        return self.INFERENCE_RULES.get(provider, {})

    def add_inference_rule(self, provider: str, pattern: str, name: str, rules: Dict):
        """추론 규칙 동적 추가"""
        if provider not in self.INFERENCE_RULES:
            self.INFERENCE_RULES[provider] = {"patterns": [], "defaults": {}}

        if "patterns" not in self.INFERENCE_RULES[provider]:
            self.INFERENCE_RULES[provider]["patterns"] = []

        self.INFERENCE_RULES[provider]["patterns"].append(
            {"match": pattern, "name": name, "rules": rules}
        )

        logger.info(f"Added inference rule for {provider}: {name}")
