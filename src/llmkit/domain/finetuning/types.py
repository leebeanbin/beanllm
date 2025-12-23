"""
Finetuning Types - 파인튜닝 데이터 타입
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import FineTuningStatus


@dataclass
class TrainingExample:
    """훈련 예제"""

    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {"messages": self.messages}

    def to_jsonl(self) -> str:
        """JSONL 형식으로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        """딕셔너리에서 생성"""
        return cls(messages=data["messages"], metadata=data.get("metadata", {}))


@dataclass
class FineTuningConfig:
    """파인튜닝 설정"""

    model: str
    training_file: str
    validation_file: Optional[str] = None
    n_epochs: int = 3
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    suffix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTuningJob:
    """파인튜닝 작업"""

    job_id: str
    model: str
    status: FineTuningStatus
    created_at: int
    finished_at: Optional[int] = None
    fine_tuned_model: Optional[str] = None
    training_file: Optional[str] = None
    validation_file: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    result_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """완료 여부"""
        return self.status in [
            FineTuningStatus.SUCCEEDED,
            FineTuningStatus.FAILED,
            FineTuningStatus.CANCELLED,
        ]

    def is_success(self) -> bool:
        """성공 여부"""
        return self.status == FineTuningStatus.SUCCEEDED


@dataclass
class FineTuningMetrics:
    """파인튜닝 메트릭"""

    step: int
    train_loss: Optional[float] = None
    valid_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    valid_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
