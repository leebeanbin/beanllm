"""
Finetuning Providers - 파인튜닝 프로바이더
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .enums import FineTuningStatus
from .types import FineTuningConfig, FineTuningJob, FineTuningMetrics, TrainingExample


class BaseFineTuningProvider(ABC):
    """파인튜닝 프로바이더 베이스 클래스"""

    @abstractmethod
    def prepare_data(self, examples: List[TrainingExample], output_path: str) -> str:
        """훈련 데이터 준비"""
        pass

    @abstractmethod
    def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """파인튜닝 작업 생성"""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> FineTuningJob:
        """작업 상태 조회"""
        pass

    @abstractmethod
    def list_jobs(self, limit: int = 20) -> List[FineTuningJob]:
        """작업 목록 조회"""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> FineTuningJob:
        """작업 취소"""
        pass

    @abstractmethod
    def get_metrics(self, job_id: str) -> List[FineTuningMetrics]:
        """훈련 메트릭 조회"""
        pass


class OpenAIFineTuningProvider(BaseFineTuningProvider):
    """
    OpenAI 파인튜닝 프로바이더

    OpenAI의 fine-tuning API 통합
    """

    def __init__(self, api_key: Optional[str] = None):
        import os

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required")

        # OpenAI client lazy loading
        self._client = None

    def _get_client(self):
        """OpenAI client 가져오기"""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI SDK required. Install with: pip install openai")
        return self._client

    def prepare_data(self, examples: List[TrainingExample], output_path: str) -> str:
        """
        OpenAI 형식으로 데이터 준비

        Args:
            examples: 훈련 예제 리스트
            output_path: 출력 파일 경로 (.jsonl)

        Returns:
            파일 경로
        """
        # JSONL 형식으로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(example.to_jsonl() + "\n")

        return output_path

    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """
        파일 업로드

        Args:
            file_path: 파일 경로
            purpose: 파일 용도 ("fine-tune")

        Returns:
            파일 ID
        """
        client = self._get_client()

        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose=purpose)

        return response.id

    def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """
        파인튜닝 작업 생성

        Args:
            config: 파인튜닝 설정

        Returns:
            파인튜닝 작업
        """
        client = self._get_client()

        # Hyperparameters 구성
        hyperparameters = {}
        if config.n_epochs:
            hyperparameters["n_epochs"] = config.n_epochs
        if config.batch_size:
            hyperparameters["batch_size"] = config.batch_size
        if config.learning_rate_multiplier:
            hyperparameters["learning_rate_multiplier"] = config.learning_rate_multiplier

        # 작업 생성
        response = client.fine_tuning.jobs.create(
            training_file=config.training_file,
            validation_file=config.validation_file,
            model=config.model,
            hyperparameters=hyperparameters or None,
            suffix=config.suffix,
        )

        # FineTuningJob으로 변환
        return self._parse_job_response(response)

    def get_job(self, job_id: str) -> FineTuningJob:
        """작업 상태 조회"""
        client = self._get_client()
        response = client.fine_tuning.jobs.retrieve(job_id)
        return self._parse_job_response(response)

    def list_jobs(self, limit: int = 20) -> List[FineTuningJob]:
        """작업 목록 조회"""
        client = self._get_client()
        response = client.fine_tuning.jobs.list(limit=limit)
        return [self._parse_job_response(job) for job in response.data]

    def cancel_job(self, job_id: str) -> FineTuningJob:
        """작업 취소"""
        client = self._get_client()
        response = client.fine_tuning.jobs.cancel(job_id)
        return self._parse_job_response(response)

    def get_metrics(self, job_id: str) -> List[FineTuningMetrics]:
        """훈련 메트릭 조회"""
        client = self._get_client()

        try:
            # Events에서 메트릭 추출
            events = client.fine_tuning.jobs.list_events(job_id, limit=100)

            metrics = []
            for event in events.data:
                if event.type == "metrics":
                    data = event.data
                    metrics.append(
                        FineTuningMetrics(
                            step=data.get("step", 0),
                            train_loss=data.get("train_loss"),
                            valid_loss=data.get("valid_loss"),
                            train_accuracy=data.get("train_accuracy"),
                            valid_accuracy=data.get("valid_accuracy"),
                            learning_rate=data.get("learning_rate"),
                        )
                    )

            return metrics
        except Exception:
            return []

    def _parse_job_response(self, response) -> FineTuningJob:
        """OpenAI 응답을 FineTuningJob으로 변환"""
        return FineTuningJob(
            job_id=response.id,
            model=response.model,
            status=FineTuningStatus(response.status),
            created_at=response.created_at,
            finished_at=response.finished_at,
            fine_tuned_model=response.fine_tuned_model,
            training_file=response.training_file,
            validation_file=response.validation_file,
            hyperparameters=response.hyperparameters.to_dict() if response.hyperparameters else {},
            result_files=response.result_files or [],
            error=response.error.message if response.error else None,
        )
