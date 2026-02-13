"""
Axolotl Fine-tuning Provider - 로컬 파인튜닝 (Axolotl)
"""

from __future__ import annotations

import json
import logging
import subprocess
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

        provider = AxolotlProvider(
            base_model="meta-llama/Llama-3.2-1B",
            output_dir="./outputs/llama-lora"
        )
        examples = [...]
        data_file = provider.prepare_data(examples, "train.jsonl")
        config = FineTuningConfig(model="...", training_file=data_file, n_epochs=3)
        job = provider.create_job(config)
        ```
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Union[str, Path],
        use_flash_attention: bool = True,
        device_map: str = "auto",
        **kwargs: Any,
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, FineTuningJob] = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """의존성 확인"""
        try:
            import axolotl
        except ImportError:
            logger.warning("axolotl not installed. Install it with: pip install axolotl-core")

    def prepare_data(self, examples: List[TrainingExample], output_path: str) -> str:
        """훈련 데이터 준비 (Alpaca JSONL)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                if len(example.messages) >= 2:
                    instruction = example.messages[0].get("content", "")
                    response = example.messages[-1].get("content", "")
                    alpaca_format = {
                        "instruction": instruction,
                        "output": response,
                        "input": "",
                    }
                    f.write(json.dumps(alpaca_format, ensure_ascii=False) + "\n")
        logger.info(f"Prepared {len(examples)} examples at {output_file}")
        return str(output_file)

    def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """파인튜닝 작업 생성."""
        axolotl_config = self._create_axolotl_config(config)
        job_id = f"axolotl_{int(time.time())}"
        config_path = self.output_dir / f"{job_id}_config.yml"
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(axolotl_config, f, default_flow_style=False, allow_unicode=True)
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
        self._jobs[job_id] = job
        logger.info(f"Axolotl job created: {job_id}")
        return job

    def get_job(self, job_id: str) -> FineTuningJob:
        """작업 상태 조회."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        job = self._jobs[job_id]
        log_file = self.output_dir / f"{job_id}.log"
        if log_file.exists():
            job = self._update_job_from_log(job, log_file)
        return job

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
        log_file = self.output_dir / f"{job_id}.log"
        if not log_file.exists():
            return []
        return self._extract_metrics_from_log(log_file)

    def _create_axolotl_config(self, config: FineTuningConfig) -> Dict[str, Any]:
        """Axolotl 설정 생성."""
        metadata = config.metadata or {}
        return {
            "base_model": config.model,
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "datasets": [{"path": config.training_file, "type": "alpaca"}],
            "adapter": metadata.get("adapter", "lora"),
            "lora_r": metadata.get("lora_r", 16),
            "lora_alpha": metadata.get("lora_alpha", 32),
            "lora_dropout": metadata.get("lora_dropout", 0.05),
            "lora_target_modules": metadata.get(
                "lora_target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            "sequence_len": metadata.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH),
            "num_epochs": config.n_epochs,
            "micro_batch_size": config.batch_size or 4,
            "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", 4),
            "learning_rate": metadata.get("learning_rate", 2e-4),
            "warmup_steps": metadata.get("warmup_steps", 100),
            "save_steps": metadata.get("save_steps", 100),
            "logging_steps": metadata.get("logging_steps", 10),
            "optimizer": metadata.get("optimizer", "adamw_torch"),
            "lr_scheduler": metadata.get("lr_scheduler", "cosine"),
            "flash_attention": self.use_flash_attention,
            "device_map": self.device_map,
            "bf16": metadata.get("bf16", True),
            "fp16": metadata.get("fp16", False),
            "output_dir": str(self.output_dir),
            "wandb_project": metadata.get("wandb_project"),
            "wandb_run_name": metadata.get("wandb_run_name"),
        }

    def _update_job_from_log(self, job: FineTuningJob, log_file: Path) -> FineTuningJob:
        """로그 파일에서 작업 상태 업데이트."""
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
        """로그 파일에서 메트릭 추출."""
        metrics: List[FineTuningMetrics] = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "loss" in line.lower():
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
        """훈련 실행 (헬퍼 메서드)."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        job = self._jobs[job_id]
        config_path = job.metadata.get("config_path")
        if not config_path:
            raise ValueError("Config path not found in job metadata")
        if accelerate:
            cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
        elif deepspeed:
            cmd = ["deepspeed", "--config_file", deepspeed, "-m", "axolotl.cli.train", config_path]
        else:
            cmd = ["python", "-m", "axolotl.cli.train", config_path]
        logger.info(f"Running Axolotl training: {' '.join(cmd)}")
        job.status = FineTuningStatus.RUNNING
        result = subprocess.run(cmd, capture_output=True, text=True)
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
        return f"AxolotlProvider(base_model={self.base_model}, output_dir={self.output_dir})"
