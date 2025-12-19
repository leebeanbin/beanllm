"""
llmkit.finetuning - Fine-tuning Support
파인튜닝 지원 모듈

이 모듈은 LLM 파인튜닝을 위한 도구를 제공합니다.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


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


# ===== Base Fine-tuning Provider =====


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


# ===== OpenAI Fine-tuning Provider =====


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


# ===== Data Preparation Utilities =====


class DatasetBuilder:
    """
    파인튜닝 데이터셋 빌더

    다양한 형식의 데이터를 훈련 예제로 변환
    """

    @staticmethod
    def from_conversations(conversations: List[List[Dict[str, str]]]) -> List[TrainingExample]:
        """
        대화 데이터에서 훈련 예제 생성

        Args:
            conversations: [
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
                ...
            ]
        """
        examples = []
        for conv in conversations:
            examples.append(TrainingExample(messages=conv))
        return examples

    @staticmethod
    def from_qa_pairs(
        qa_pairs: List[Dict[str, str]], system_message: Optional[str] = None
    ) -> List[TrainingExample]:
        """
        Q&A 쌍에서 훈련 예제 생성

        Args:
            qa_pairs: [{"question": "...", "answer": "..."}, ...]
            system_message: 시스템 메시지 (선택)
        """
        examples = []
        for pair in qa_pairs:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": pair["question"]})
            messages.append({"role": "assistant", "content": pair["answer"]})

            examples.append(TrainingExample(messages=messages))

        return examples

    @staticmethod
    def from_instructions(
        instructions: List[Dict[str, str]], system_template: str = "You are a helpful assistant."
    ) -> List[TrainingExample]:
        """
        Instruction-following 데이터에서 훈련 예제 생성

        Args:
            instructions: [{"instruction": "...", "output": "..."}, ...]
            system_template: 시스템 메시지 템플릿
        """
        examples = []
        for inst in instructions:
            messages = [
                {"role": "system", "content": system_template},
                {"role": "user", "content": inst["instruction"]},
                {"role": "assistant", "content": inst["output"]},
            ]
            examples.append(TrainingExample(messages=messages))

        return examples

    @staticmethod
    def from_json_file(file_path: str) -> List[TrainingExample]:
        """JSON 파일에서 훈련 예제 로드"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [TrainingExample.from_dict(item) for item in data]
        else:
            raise ValueError("JSON file must contain a list of examples")

    @staticmethod
    def from_jsonl_file(file_path: str) -> List[TrainingExample]:
        """JSONL 파일에서 훈련 예제 로드"""
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                examples.append(TrainingExample.from_dict(data))
        return examples

    @staticmethod
    def split_dataset(
        examples: List[TrainingExample], train_ratio: float = 0.8, shuffle: bool = True
    ) -> tuple[List[TrainingExample], List[TrainingExample]]:
        """데이터셋 분할 (훈련/검증)"""
        import random

        if shuffle:
            examples = examples.copy()
            random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)
        train_set = examples[:split_idx]
        val_set = examples[split_idx:]

        return train_set, val_set


class DataValidator:
    """
    훈련 데이터 검증기

    OpenAI 형식 요구사항 검증
    """

    @staticmethod
    def validate_example(example: TrainingExample) -> List[str]:
        """
        개별 예제 검증

        Returns:
            에러 메시지 리스트 (빈 리스트 = 유효함)
        """
        errors = []

        if not example.messages:
            errors.append("Example must have at least one message")
            return errors

        # 메시지 검증
        for i, msg in enumerate(example.messages):
            if "role" not in msg:
                errors.append(f"Message {i} missing 'role'")
            elif msg["role"] not in ["system", "user", "assistant"]:
                errors.append(f"Message {i} has invalid role: {msg['role']}")

            if "content" not in msg:
                errors.append(f"Message {i} missing 'content'")
            elif not isinstance(msg["content"], str):
                errors.append(f"Message {i} content must be string")

        # 첫 메시지는 system 또는 user여야 함
        if example.messages[0]["role"] not in ["system", "user"]:
            errors.append("First message must be 'system' or 'user'")

        # Assistant 메시지가 최소 하나 있어야 함
        has_assistant = any(m["role"] == "assistant" for m in example.messages)
        if not has_assistant:
            errors.append("Must have at least one 'assistant' message")

        return errors

    @staticmethod
    def validate_dataset(examples: List[TrainingExample]) -> Dict[str, Any]:
        """
        전체 데이터셋 검증

        Returns:
            검증 리포트
        """
        total = len(examples)
        errors_per_example = []

        for i, example in enumerate(examples):
            errors = DataValidator.validate_example(example)
            if errors:
                errors_per_example.append((i, errors))

        is_valid = len(errors_per_example) == 0

        return {
            "is_valid": is_valid,
            "total_examples": total,
            "invalid_count": len(errors_per_example),
            "errors": errors_per_example,
        }

    @staticmethod
    def estimate_tokens(examples: List[TrainingExample]) -> Dict[str, Any]:
        """토큰 수 추정 (간단한 휴리스틱)"""
        total_tokens = 0

        for example in examples:
            for msg in example.messages:
                # 대략 1 token = 0.75 words
                words = len(msg["content"].split())
                tokens = int(words / 0.75)
                total_tokens += tokens

        return {
            "total_tokens": total_tokens,
            "average_per_example": total_tokens / len(examples) if examples else 0,
        }


# ===== Fine-tuning Manager =====


class FineTuningManager:
    """
    파인튜닝 통합 매니저

    데이터 준비부터 훈련, 배포까지 전체 워크플로우 관리
    """

    def __init__(self, provider: BaseFineTuningProvider):
        self.provider = provider

    def prepare_and_upload(
        self, examples: List[TrainingExample], output_path: str, validate: bool = True
    ) -> str:
        """
        데이터 준비 및 업로드

        Args:
            examples: 훈련 예제
            output_path: 로컬 저장 경로
            validate: 검증 여부

        Returns:
            업로드된 파일 ID
        """
        # 검증
        if validate:
            report = DataValidator.validate_dataset(examples)
            if not report["is_valid"]:
                raise ValueError(
                    f"Dataset validation failed: " f"{report['invalid_count']} invalid examples"
                )

        # 데이터 준비
        self.provider.prepare_data(examples, output_path)

        # 업로드 (OpenAI의 경우)
        if isinstance(self.provider, OpenAIFineTuningProvider):
            file_id = self.provider.upload_file(output_path)
            return file_id
        else:
            return output_path

    def start_training(
        self, model: str, training_file: str, validation_file: Optional[str] = None, **kwargs
    ) -> FineTuningJob:
        """
        훈련 시작

        Args:
            model: 베이스 모델
            training_file: 훈련 파일 ID
            validation_file: 검증 파일 ID (선택)
            **kwargs: 추가 설정 (n_epochs, batch_size 등)

        Returns:
            파인튜닝 작업
        """
        config = FineTuningConfig(
            model=model, training_file=training_file, validation_file=validation_file, **kwargs
        )

        return self.provider.create_job(config)

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: Optional[int] = None,
        callback: Optional[Callable[[FineTuningJob], None]] = None,
    ) -> FineTuningJob:
        """
        작업 완료 대기

        Args:
            job_id: 작업 ID
            poll_interval: 폴링 간격 (초)
            timeout: 타임아웃 (초)
            callback: 상태 변경시 호출할 콜백

        Returns:
            완료된 작업
        """
        start_time = time.time()

        while True:
            job = self.provider.get_job(job_id)

            # 콜백 호출
            if callback:
                callback(job)

            # 완료 확인
            if job.is_complete():
                return job

            # 타임아웃 확인
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            # 대기
            time.sleep(poll_interval)

    def get_training_progress(self, job_id: str) -> Dict[str, Any]:
        """훈련 진행상황 조회"""
        job = self.provider.get_job(job_id)
        metrics = self.provider.get_metrics(job_id)

        return {"job": job, "metrics": metrics, "latest_metric": metrics[-1] if metrics else None}


# ===== Cost Estimation =====


class FineTuningCostEstimator:
    """파인튜닝 비용 추정"""

    # OpenAI 파인튜닝 가격 (2024년 기준, tokens per 1M)
    OPENAI_PRICES = {
        "gpt-3.5-turbo": {"training": 8.00, "inference": 3.00},
        "gpt-4": {"training": 30.00, "inference": 60.00},
        "gpt-4o-mini": {"training": 3.00, "inference": 1.50},
    }

    @staticmethod
    def estimate_training_cost(
        model: str, n_tokens: int, n_epochs: int = 3, provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        훈련 비용 추정

        Args:
            model: 모델 이름
            n_tokens: 총 토큰 수
            n_epochs: 에폭 수
            provider: 프로바이더

        Returns:
            비용 정보
        """
        if provider == "openai":
            prices = FineTuningCostEstimator.OPENAI_PRICES.get(model, {})
            training_price = prices.get("training", 0)

            total_tokens = n_tokens * n_epochs
            cost = (total_tokens / 1_000_000) * training_price

            return {
                "model": model,
                "total_tokens": total_tokens,
                "price_per_1m": training_price,
                "estimated_cost_usd": cost,
                "epochs": n_epochs,
            }
        else:
            return {"error": f"Provider {provider} not supported"}


# ===== 유틸리티 함수 =====


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
    # 데이터 분할
    train_examples, val_examples = DatasetBuilder.split_dataset(
        training_data, train_ratio=1 - validation_split
    )

    # 프로바이더 생성
    provider = create_finetuning_provider("openai", **kwargs)
    manager = FineTuningManager(provider)

    # 데이터 업로드
    train_file = manager.prepare_and_upload(train_examples, "train.jsonl")

    val_file = None
    if val_examples:
        val_file = manager.prepare_and_upload(val_examples, "val.jsonl")

    # 훈련 시작
    job = manager.start_training(
        model=model, training_file=train_file, validation_file=val_file, n_epochs=n_epochs
    )

    # 대기
    if wait:

        def progress_callback(j):
            print(f"Status: {j.status.value}, Model: {j.fine_tuned_model or 'N/A'}")

        job = manager.wait_for_completion(job.job_id, callback=progress_callback)

    return job
