"""
Finetuning Enums - 파인튜닝 관련 열거형
"""

from enum import Enum


class FineTuningStatus(Enum):
    """파인튜닝 작업 상태"""

    CREATED = "created"
    VALIDATING = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelProvider(Enum):
    """지원 프로바이더"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
