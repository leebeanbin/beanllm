"""
Local Fine-tuning Providers - 로컬 파인튜닝 프로바이더 (2024-2025)

Axolotl과 Unsloth를 사용한 로컬 LLM 파인튜닝.
BaseFineTuningProvider를 상속하여 인터페이스 통일.

주요 프레임워크:
- Axolotl: 종합 파인튜닝 프레임워크 (8K+ stars)
- Unsloth: 2-5x 빠른 파인튜닝 (10K+ stars)

Requirements:
    pip install axolotl-core  # Axolotl
    pip install unsloth  # Unsloth
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .enums import FineTuningStatus
from .providers import BaseFineTuningProvider
from .types import FineTuningConfig, FineTuningJob, FineTuningMetrics, TrainingExample

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class AxolotlProvider(BaseFineTuningProvider):
    """
    Axolotl 파인튜닝 프로바이더 (로컬)

    OpenAccess AI Collective의 Axolotl을 사용한 종합 파인튜닝 프레임워크.
    BaseFineTuningProvider를 상속하여 표준 인터페이스 제공.

    Axolotl 특징:
    - LoRA, QLoRA, Full Fine-tuning 지원
    - Flash Attention 2 지원
    - 다양한 모델 아키텍처 (Llama, Mistral, Qwen 등)
    - YAML 기반 설정
    - W&B, MLflow 통합
    - 8K+ GitHub stars

    Example:
        ```python
        from beanllm.domain.finetuning import AxolotlProvider, FineTuningConfig, TrainingExample

        # Provider 생성
        provider = AxolotlProvider(
            base_model="meta-llama/Llama-3.2-1B",
            output_dir="./outputs/llama-lora"
        )

        # 훈련 데이터 준비
        examples = [
            TrainingExample(messages=[
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."}
            ])
        ]
        data_file = provider.prepare_data(examples, "train.jsonl")

        # 작업 생성
        config = FineTuningConfig(
            model="meta-llama/Llama-3.2-1B",
            training_file=data_file,
            n_epochs=3,
            metadata={
                "adapter": "lora",
                "lora_r": 16,
                "lora_alpha": 32,
            }
        )
        job = provider.create_job(config)

        # 작업 상태 확인
        job_status = provider.get_job(job.job_id)
        print(job_status.status)
        ```
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Union[str, Path],
        use_flash_attention: bool = True,
        device_map: str = "auto",
        **kwargs,
    ):
        """
        Args:
            base_model: 기본 모델 (HuggingFace model ID)
            output_dir: 출력 디렉토리
            use_flash_attention: Flash Attention 2 사용 여부
            device_map: 디바이스 맵 (auto/cuda/cpu)
            **kwargs: 추가 Axolotl 설정
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.use_flash_attention = use_flash_attention
        self.device_map = device_map
        self.kwargs = kwargs

        # Output directory 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Jobs 추적
        self._jobs: Dict[str, FineTuningJob] = {}

        # Axolotl 설치 확인
        self._check_dependencies()

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            import axolotl
        except ImportError:
            logger.warning(
                "axolotl not installed. "
                "Install it with: pip install axolotl-core"
            )

    def prepare_data(self, examples: List[TrainingExample], output_path: str) -> str:
        """
        훈련 데이터 준비

        Args:
            examples: 훈련 예제 리스트
            output_path: 출력 파일 경로 (.jsonl)

        Returns:
            파일 경로
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # JSONL 형식으로 저장 (Alpaca 형식)
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                # Alpaca 형식 변환
                if len(example.messages) >= 2:
                    instruction = example.messages[0].get("content", "")
                    response = example.messages[-1].get("content", "")

                    alpaca_format = {
                        "instruction": instruction,
                        "output": response,
                        "input": "",  # Alpaca format requires this
                    }
                    f.write(json.dumps(alpaca_format, ensure_ascii=False) + "\n")

        logger.info(f"Prepared {len(examples)} examples at {output_file}")
        return str(output_file)

    def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """
        파인튜닝 작업 생성

        Args:
            config: 파인튜닝 설정

        Returns:
            파인튜닝 작업
        """
        # Axolotl config 생성
        axolotl_config = self._create_axolotl_config(config)

        # Config 파일 저장
        job_id = f"axolotl_{int(time.time())}"
        config_path = self.output_dir / f"{job_id}_config.yml"

        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(axolotl_config, f, default_flow_style=False, allow_unicode=True)

        # FineTuningJob 생성
        job = FineTuningJob(
            job_id=job_id,
            model=config.model,
            status=FineTuningStatus.CREATED,
            created_at=int(time.time()),
            training_file=config.training_file,
            validation_file=config.validation_file,
            hyperparameters=config.metadata,
            metadata={
                "config_path": str(config_path),
                "output_dir": str(self.output_dir),
                "provider": "axolotl",
            },
        )

        # Jobs 추적에 추가
        self._jobs[job_id] = job

        logger.info(f"Axolotl job created: {job_id}")

        return job

    def get_job(self, job_id: str) -> FineTuningJob:
        """
        작업 상태 조회

        Args:
            job_id: 작업 ID

        Returns:
            파인튜닝 작업
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]

        # 로그 파일에서 상태 업데이트 (선택적)
        log_file = self.output_dir / f"{job_id}.log"
        if log_file.exists():
            job = self._update_job_from_log(job, log_file)

        return job

    def list_jobs(self, limit: int = 20) -> List[FineTuningJob]:
        """
        작업 목록 조회

        Args:
            limit: 최대 개수

        Returns:
            작업 목록
        """
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> FineTuningJob:
        """
        작업 취소

        Args:
            job_id: 작업 ID

        Returns:
            파인튜닝 작업
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]

        # 프로세스 kill (실제 구현에서는 PID 추적 필요)
        job.status = FineTuningStatus.CANCELLED
        job.finished_at = int(time.time())

        logger.info(f"Job {job_id} cancelled")

        return job

    def get_metrics(self, job_id: str) -> List[FineTuningMetrics]:
        """
        훈련 메트릭 조회

        Args:
            job_id: 작업 ID

        Returns:
            메트릭 리스트
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        # 로그 파일에서 메트릭 추출
        log_file = self.output_dir / f"{job_id}.log"
        if not log_file.exists():
            return []

        metrics = self._extract_metrics_from_log(log_file)

        return metrics

    def _create_axolotl_config(self, config: FineTuningConfig) -> Dict[str, Any]:
        """Axolotl 설정 생성"""
        metadata = config.metadata or {}

        axolotl_config = {
            # Base model
            "base_model": config.model,
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",

            # Dataset
            "datasets": [
                {
                    "path": config.training_file,
                    "type": "alpaca",
                }
            ],

            # Adapter
            "adapter": metadata.get("adapter", "lora"),
            "lora_r": metadata.get("lora_r", 16),
            "lora_alpha": metadata.get("lora_alpha", 32),
            "lora_dropout": metadata.get("lora_dropout", 0.05),
            "lora_target_modules": metadata.get("lora_target_modules", [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),

            # Training
            "sequence_len": metadata.get("max_seq_length", 2048),
            "num_epochs": config.n_epochs,
            "micro_batch_size": config.batch_size or 4,
            "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", 4),
            "learning_rate": metadata.get("learning_rate", 2e-4),
            "warmup_steps": metadata.get("warmup_steps", 100),
            "save_steps": metadata.get("save_steps", 100),
            "logging_steps": metadata.get("logging_steps", 10),

            # Optimizer
            "optimizer": metadata.get("optimizer", "adamw_torch"),
            "lr_scheduler": metadata.get("lr_scheduler", "cosine"),

            # Performance
            "flash_attention": self.use_flash_attention,
            "device_map": self.device_map,
            "bf16": metadata.get("bf16", True),
            "fp16": metadata.get("fp16", False),

            # Output
            "output_dir": str(self.output_dir),

            # W&B (optional)
            "wandb_project": metadata.get("wandb_project"),
            "wandb_run_name": metadata.get("wandb_run_name"),
        }

        return axolotl_config

    def _update_job_from_log(self, job: FineTuningJob, log_file: Path) -> FineTuningJob:
        """로그 파일에서 작업 상태 업데이트"""
        # 로그 파일 파싱 로직 (간단한 구현)
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read()

            if "Training completed" in log_content:
                job.status = FineTuningStatus.SUCCEEDED
                job.finished_at = int(time.time())
            elif "Error" in log_content or "Failed" in log_content:
                job.status = FineTuningStatus.FAILED
                job.finished_at = int(time.time())
            else:
                job.status = FineTuningStatus.RUNNING

        except Exception as e:
            logger.warning(f"Failed to update job from log: {e}")

        return job

    def _extract_metrics_from_log(self, log_file: Path) -> List[FineTuningMetrics]:
        """로그 파일에서 메트릭 추출"""
        metrics = []

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    # 간단한 파싱 (실제로는 더 정교해야 함)
                    if "loss" in line.lower():
                        # Parse loss values
                        # This is a placeholder - actual implementation depends on log format
                        pass

        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")

        return metrics

    def train(
        self,
        job_id: str,
        accelerate: bool = False,
        deepspeed: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        훈련 실행 (추가 헬퍼 메서드)

        Args:
            job_id: 작업 ID
            accelerate: Accelerate 사용 여부
            deepspeed: DeepSpeed 설정 파일 경로

        Returns:
            subprocess.CompletedProcess
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]
        config_path = job.metadata.get("config_path")

        if not config_path:
            raise ValueError("Config path not found in job metadata")

        # 명령어 구성
        if accelerate:
            cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
        elif deepspeed:
            cmd = ["deepspeed", "--config_file", deepspeed, "-m", "axolotl.cli.train", config_path]
        else:
            cmd = ["python", "-m", "axolotl.cli.train", config_path]

        logger.info(f"Running Axolotl training: {' '.join(cmd)}")

        # 작업 상태 업데이트
        job.status = FineTuningStatus.RUNNING

        # 실행
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 상태 업데이트
        if result.returncode == 0:
            job.status = FineTuningStatus.SUCCEEDED
            logger.info("Axolotl training completed successfully")
        else:
            job.status = FineTuningStatus.FAILED
            job.error = result.stderr
            logger.error(f"Axolotl training failed: {result.stderr}")

        job.finished_at = int(time.time())

        return result

    def __repr__(self) -> str:
        return (
            f"AxolotlProvider(base_model={self.base_model}, "
            f"output_dir={self.output_dir})"
        )


class UnslothProvider(BaseFineTuningProvider):
    """
    Unsloth 파인튜닝 프로바이더 (로컬)

    Unsloth AI의 초고속 파인튜닝 프레임워크.
    BaseFineTuningProvider를 상속하여 표준 인터페이스 제공.

    Unsloth 특징:
    - 2-5x 빠른 훈련 속도
    - 80% 메모리 절약
    - Flash Attention + 커스텀 커널
    - LoRA, QLoRA 최적화
    - Llama, Mistral, Qwen, Gemma 지원
    - 10K+ GitHub stars

    Example:
        ```python
        from beanllm.domain.finetuning import UnslothProvider, FineTuningConfig, TrainingExample

        # Provider 생성
        provider = UnslothProvider(
            model_name="unsloth/llama-3.2-1b-bnb-4bit",
            output_dir="./outputs/unsloth"
        )

        # 훈련 데이터 준비
        examples = [...]
        data_file = provider.prepare_data(examples, "train.jsonl")

        # 작업 생성
        config = FineTuningConfig(
            model="unsloth/llama-3.2-1b-bnb-4bit",
            training_file=data_file,
            n_epochs=3,
        )
        job = provider.create_job(config)
        ```
    """

    def __init__(
        self,
        model_name: str,
        output_dir: Union[str, Path],
        max_seq_length: int = 2048,
        dtype: Optional[str] = None,
        load_in_4bit: bool = True,
        **kwargs,
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

        # Output directory 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Jobs 추적
        self._jobs: Dict[str, FineTuningJob] = {}

        # Unsloth 설치 확인
        self._check_dependencies()

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            logger.warning(
                "unsloth not installed. "
                "Install it with: pip install unsloth"
            )

    def prepare_data(self, examples: List[TrainingExample], output_path: str) -> str:
        """
        훈련 데이터 준비

        Args:
            examples: 훈련 예제 리스트
            output_path: 출력 파일 경로 (.jsonl)

        Returns:
            파일 경로
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # JSONL 형식으로 저장
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                # Unsloth 형식 (chat template)
                f.write(example.to_jsonl() + "\n")

        logger.info(f"Prepared {len(examples)} examples at {output_file}")
        return str(output_file)

    def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """
        파인튜닝 작업 생성

        Args:
            config: 파인튜닝 설정

        Returns:
            파인튜닝 작업
        """
        job_id = f"unsloth_{int(time.time())}"

        # FineTuningJob 생성
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

        # Jobs 추적에 추가
        self._jobs[job_id] = job

        logger.info(f"Unsloth job created: {job_id}")

        return job

    def get_job(self, job_id: str) -> FineTuningJob:
        """작업 상태 조회"""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        return self._jobs[job_id]

    def list_jobs(self, limit: int = 20) -> List[FineTuningJob]:
        """작업 목록 조회"""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> FineTuningJob:
        """작업 취소"""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id]
        job.status = FineTuningStatus.CANCELLED
        job.finished_at = int(time.time())

        logger.info(f"Job {job_id} cancelled")

        return job

    def get_metrics(self, job_id: str) -> List[FineTuningMetrics]:
        """훈련 메트릭 조회"""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        # Unsloth는 Trainer 로그에서 메트릭 추출
        # 실제 구현에서는 wandb 또는 로그 파일 파싱
        return []

    def __repr__(self) -> str:
        return (
            f"UnslothProvider(model={self.model_name}, "
            f"4bit={self.load_in_4bit})"
        )
