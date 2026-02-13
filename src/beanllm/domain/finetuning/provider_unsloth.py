"""
Unsloth Fine-tuning Provider - 로컬 파인튜닝 (Unsloth)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from beanllm.domain.finetuning.enums import FineTuningStatus
from beanllm.domain.finetuning.providers import BaseFineTuningProvider
from beanllm.domain.finetuning.types import (
    FineTuningConfig,
    FineTuningJob,
    FineTuningMetrics,
    TrainingExample,
)

try:
    from beanllm.utils.constants import DEFAULT_MAX_SEQ_LENGTH
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class UnslothProvider(BaseFineTuningProvider):
    """
    Unsloth 파인튜닝 프로바이더 (로컬).

    Unsloth AI의 초고속 파인튜닝 프레임워크.
    BaseFineTuningProvider를 상속하여 표준 인터페이스 제공.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: Union[str, Path],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        dtype: Optional[str] = None,
        load_in_4bit: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            model_name: 모델 이름 (unsloth/... 또는 HuggingFace)
            output_dir: 출력 디렉토리
            max_seq_length: 최대 시퀀스 길이
            dtype: 데이터 타입 (None=auto, float16, bfloat16)
            load_in_4bit: 4-bit 양자화 로드
            **kwargs: 추가 Unsloth 설정
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.kwargs = kwargs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, FineTuningJob] = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """의존성 확인."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            logger.warning("unsloth not installed. Install it with: pip install unsloth")

    def prepare_data(self, examples: List[TrainingExample], output_path: str) -> str:
        """훈련 데이터 준비 (JSONL)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(example.to_jsonl() + "\n")
        logger.info(f"Prepared {len(examples)} examples at {output_file}")
        return str(output_file)

    def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """파인튜닝 작업 생성."""
        job_id = f"unsloth_{int(time.time())}"
        job = FineTuningJob(
            job_id=job_id,
            model=config.model,
            status=FineTuningStatus.CREATED,
            created_at=int(time.time()),
            training_file=config.training_file,
            validation_file=config.validation_file,
            hyperparameters=config.metadata or {},
            metadata={
                "output_dir": str(self.output_dir / job_id),
                "provider": "unsloth",
                "max_seq_length": self.max_seq_length,
                "load_in_4bit": self.load_in_4bit,
            },
        )
        self._jobs[job_id] = job
        logger.info(f"Unsloth job created: {job_id}")
        return job

    def get_job(self, job_id: str) -> FineTuningJob:
        """작업 상태 조회."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        return self._jobs[job_id]

    def list_jobs(self, limit: int = 20) -> List[FineTuningJob]:
        """작업 목록 조회."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> FineTuningJob:
        """작업 취소."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        job = self._jobs[job_id]
        job.status = FineTuningStatus.CANCELLED
        job.finished_at = int(time.time())
        logger.info(f"Job {job_id} cancelled")
        return job

    def get_metrics(self, job_id: str) -> List[FineTuningMetrics]:
        """훈련 메트릭 조회."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        return []

    def __repr__(self) -> str:
        return f"UnslothProvider(model={self.model_name}, 4bit={self.load_in_4bit})"
