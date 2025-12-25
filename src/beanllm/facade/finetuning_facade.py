"""
Finetuning Facade - 기존 Finetuning API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, List, Optional

from ..domain.finetuning.providers import BaseFineTuningProvider, OpenAIFineTuningProvider
from ..domain.finetuning.types import FineTuningJob, TrainingExample
from ..handler.finetuning_handler import FinetuningHandler

if TYPE_CHECKING:
    pass


class FineTuningManagerFacade:
    """
    파인튜닝 통합 매니저 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용
    """

    def __init__(self, provider: BaseFineTuningProvider):
        """
        Args:
            provider: 파인튜닝 프로바이더
        """
        self.provider = provider

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..service.impl.finetuning_service_impl import FinetuningServiceImpl

        # FinetuningService 생성 (커스텀 의존성)
        finetuning_service = FinetuningServiceImpl(provider=self.provider)

        # FinetuningHandler 생성 (직접 생성 - 커스텀 Service 사용)
        self._finetuning_handler = FinetuningHandler(finetuning_service)

    def prepare_and_upload(
        self, examples: List[TrainingExample], output_path: str, validate: bool = True
    ) -> str:
        """데이터 준비 및 업로드"""
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._finetuning_handler.handle_prepare_data(
                examples=examples, output_path=output_path, validate=validate
            )
        )
        return response.file_id

    def start_training(
        self, model: str, training_file: str, validation_file: Optional[str] = None, **kwargs
    ) -> FineTuningJob:
        """훈련 시작"""
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._finetuning_handler.handle_start_training(
                model=model,
                training_file=training_file,
                validation_file=validation_file,
                **kwargs,
            )
        )
        return response.job

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: Optional[int] = None,
        callback: Optional[Callable[[FineTuningJob], None]] = None,
    ) -> FineTuningJob:
        """작업 완료 대기"""
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._finetuning_handler.handle_wait_for_completion(
                job_id=job_id,
                poll_interval=poll_interval,
                timeout=timeout,
                callback=callback,
            )
        )
        return response.job

    def get_training_progress(self, job_id: str) -> dict:
        """훈련 진행상황 조회"""
        # 동기 메서드이지만 내부적으로는 비동기 사용
        job_response = asyncio.run(self._finetuning_handler.handle_get_job(job_id=job_id))
        metrics_response = asyncio.run(self._finetuning_handler.handle_get_metrics(job_id=job_id))

        return {
            "job": job_response.job,
            "metrics": metrics_response.metrics,
            "latest_metric": metrics_response.metrics[-1] if metrics_response.metrics else None,
        }


# 편의 함수들 (기존 API 유지)


def create_finetuning_provider(provider: str = "openai", **kwargs) -> BaseFineTuningProvider:
    """
    파인튜닝 프로바이더 생성

    Args:
        provider: "openai", "anthropic", "google", "local"
        **kwargs: 프로바이더별 설정

    Returns:
        파인튜닝 프로바이더
    """
    if provider == "openai":
        return OpenAIFineTuningProvider(**kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported yet")


def quick_finetune(
    training_data: List[TrainingExample],
    model: str = "gpt-3.5-turbo",
    validation_split: float = 0.1,
    n_epochs: int = 3,
    wait: bool = True,
    **kwargs,
) -> FineTuningJob:
    """
    빠른 파인튜닝 시작

    Args:
        training_data: 훈련 데이터
        model: 베이스 모델
        validation_split: 검증 데이터 비율
        n_epochs: 에폭 수
        wait: 완료 대기 여부

    Returns:
        파인튜닝 작업
    """
    # Handler/Service 초기화
    from ..service.impl.finetuning_service_impl import FinetuningServiceImpl

    finetuning_service = FinetuningServiceImpl()
    handler = FinetuningHandler(finetuning_service)

    # 동기 메서드이지만 내부적으로는 비동기 사용
    response = asyncio.run(
        handler.handle_quick_finetune(
            training_data=training_data,
            model=model,
            validation_split=validation_split,
            n_epochs=n_epochs,
            wait=wait,
            **kwargs,
        )
    )
    return response.job
