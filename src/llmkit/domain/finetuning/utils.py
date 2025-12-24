"""
Finetuning Utils - 파인튜닝 유틸리티 클래스
"""

import json
import random
import time
from typing import Any, Callable, Dict, List, Optional

from .providers import BaseFineTuningProvider, OpenAIFineTuningProvider
from .types import FineTuningConfig, FineTuningJob, TrainingExample


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
                    f"Dataset validation failed: {report['invalid_count']} invalid examples"
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
