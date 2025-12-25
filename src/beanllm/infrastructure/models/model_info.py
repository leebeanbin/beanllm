"""
Model Information
모델 정보 데이터 클래스
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class ModelStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class ParameterInfo:
    name: str
    type: str
    description: str
    default: Any
    required: bool
    supported: bool
    notes: Optional[str] = None


@dataclass
class ProviderInfo:
    name: str
    status: ModelStatus
    env_key: str
    env_value_set: bool
    available_models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status.value,
            "env_key": self.env_key,
            "env_value_set": self.env_value_set,
            "available_models": self.available_models,
            "default_model": self.default_model,
            "error_message": self.error_message,
        }


@dataclass
class ModelCapabilityInfo:
    model_name: str
    display_name: str
    provider: str
    model_type: str
    supports_streaming: bool
    supports_temperature: bool
    supports_max_tokens: bool
    uses_max_completion_tokens: bool
    max_tokens: int
    default_temperature: float
    description: str
    use_case: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    example_usage: Optional[str] = None

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "provider": self.provider,
            "type": self.model_type,
            "supports_streaming": self.supports_streaming,
            "supports_temperature": self.supports_temperature,
            "supports_max_tokens": self.supports_max_tokens,
            "uses_max_completion_tokens": self.uses_max_completion_tokens,
            "max_tokens": self.max_tokens,
            "default_temperature": self.default_temperature,
            "description": self.description,
            "use_case": self.use_case,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "default": p.default,
                    "required": p.required,
                    "supported": p.supported,
                    "notes": p.notes,
                }
                for p in self.parameters
            ],
            "example_usage": self.example_usage,
        }
