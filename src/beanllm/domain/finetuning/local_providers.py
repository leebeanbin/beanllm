"""
Local Fine-tuning Providers - 로컬 파인튜닝 프로바이더 (2024-2025)

Axolotl과 Unsloth를 사용한 로컬 LLM 파인튜닝.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .enums import FineTuningStatus
from .types import FineTuningConfig, FineTuningJob, TrainingExample

try:
    from ...utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class AxolotlProvider:
    """
    Axolotl 파인튜닝 프로바이더 (로컬)

    OpenAccess AI Collective의 Axolotl을 사용한 종합 파인튜닝 프레임워크.

    Axolotl 특징:
    - LoRA, QLoRA, Full Fine-tuning 지원
    - Flash Attention 2 지원
    - 다양한 모델 아키텍처 (Llama, Mistral, Qwen 등)
    - YAML 기반 설정
    - W&B, MLflow 통합
    - 8K+ GitHub stars

    Example:
        ```python
        from beanllm.domain.finetuning import AxolotlProvider

        # 기본 LoRA 파인튜닝
        provider = AxolotlProvider(
            base_model="meta-llama/Llama-3.2-1B",
            output_dir="./outputs/llama-lora"
        )

        # YAML 설정으로 작업 생성
        config = {
            "adapter": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
        }

        job_id = provider.create_job(
            dataset_path="data/train.jsonl",
            config=config
        )

        # 훈련 실행
        provider.train(job_id)
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

    def create_config(
        self,
        dataset_path: str,
        adapter: str = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048,
        warmup_steps: int = 100,
        save_steps: int = 100,
        logging_steps: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Axolotl 설정 생성

        Args:
            dataset_path: 데이터셋 경로
            adapter: 어댑터 타입 (lora/qlora/full)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            learning_rate: 학습률
            num_epochs: 에폭 수
            batch_size: 배치 크기
            gradient_accumulation_steps: Gradient accumulation 스텝
            max_seq_length: 최대 시퀀스 길이
            warmup_steps: Warmup 스텝
            save_steps: 저장 간격
            logging_steps: 로깅 간격
            **kwargs: 추가 설정

        Returns:
            Axolotl 설정 딕셔너리
        """
        config = {
            # Base model
            "base_model": self.base_model,
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",

            # Dataset
            "datasets": [
                {
                    "path": dataset_path,
                    "type": "alpaca",  # alpaca/sharegpt/completion
                }
            ],

            # Adapter
            "adapter": adapter,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": kwargs.get("lora_target_modules", [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),

            # Training
            "sequence_len": max_seq_length,
            "num_epochs": num_epochs,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "logging_steps": logging_steps,

            # Optimizer
            "optimizer": kwargs.get("optimizer", "adamw_torch"),
            "lr_scheduler": kwargs.get("lr_scheduler", "cosine"),

            # Performance
            "flash_attention": self.use_flash_attention,
            "device_map": self.device_map,
            "bf16": kwargs.get("bf16", True),
            "fp16": kwargs.get("fp16", False),

            # Output
            "output_dir": str(self.output_dir),

            # W&B (optional)
            "wandb_project": kwargs.get("wandb_project"),
            "wandb_run_name": kwargs.get("wandb_run_name"),
        }

        # 추가 설정 병합
        config.update(kwargs)

        return config

    def save_config(self, config: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
        """
        설정을 YAML 파일로 저장

        Args:
            config: Axolotl 설정
            config_path: 설정 파일 경로 (None이면 자동 생성)

        Returns:
            설정 파일 경로
        """
        if config_path is None:
            config_path = self.output_dir / "axolotl_config.yml"

        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Axolotl config saved to: {config_path}")
        return config_path

    def create_job(
        self,
        dataset_path: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        파인튜닝 작업 생성

        Args:
            dataset_path: 데이터셋 경로
            config: Axolotl 설정 (None이면 기본값)
            **kwargs: create_config에 전달할 추가 인자

        Returns:
            작업 ID (설정 파일 경로)
        """
        # Config 생성
        if config is None:
            config = self.create_config(dataset_path, **kwargs)
        else:
            # dataset_path 추가
            if "datasets" not in config:
                config["datasets"] = [{"path": dataset_path, "type": "alpaca"}]

        # Config 저장
        config_path = self.save_config(config)

        logger.info(f"Axolotl job created: {config_path}")

        return str(config_path)

    def train(
        self,
        config_path: str,
        accelerate: bool = False,
        deepspeed: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        훈련 실행

        Args:
            config_path: Axolotl 설정 파일 경로
            accelerate: Accelerate 사용 여부
            deepspeed: DeepSpeed 설정 파일 경로

        Returns:
            subprocess.CompletedProcess

        Example:
            ```python
            # 기본 훈련
            provider.train("config.yml")

            # Accelerate로 훈련
            provider.train("config.yml", accelerate=True)

            # DeepSpeed로 훈련
            provider.train("config.yml", deepspeed="ds_config.json")
            ```
        """
        if accelerate:
            cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
        elif deepspeed:
            cmd = ["deepspeed", "--config_file", deepspeed, "-m", "axolotl.cli.train", config_path]
        else:
            cmd = ["python", "-m", "axolotl.cli.train", config_path]

        logger.info(f"Running Axolotl training: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Axolotl training failed: {result.stderr}")
        else:
            logger.info("Axolotl training completed successfully")

        return result

    def __repr__(self) -> str:
        return (
            f"AxolotlProvider(base_model={self.base_model}, "
            f"output_dir={self.output_dir})"
        )


class UnslothProvider:
    """
    Unsloth 파인튜닝 프로바이더 (로컬)

    Unsloth AI의 초고속 파인튜닝 프레임워크.

    Unsloth 특징:
    - 2-5x 빠른 훈련 속도
    - 80% 메모리 절약
    - Flash Attention + 커스텀 커널
    - LoRA, QLoRA 최적화
    - Llama, Mistral, Qwen, Gemma 지원
    - 10K+ GitHub stars

    Example:
        ```python
        from beanllm.domain.finetuning import UnslothProvider

        # Unsloth로 LoRA 파인튜닝
        provider = UnslothProvider(
            model_name="unsloth/llama-3.2-1b-bnb-4bit",
            max_seq_length=2048
        )

        # 데이터셋 로드 및 훈련
        provider.load_dataset("yahma/alpaca-cleaned")
        provider.train(
            output_dir="./outputs/unsloth-lora",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
        )

        # 모델 저장
        provider.save_model("./my-finetuned-model")
        ```
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[str] = None,
        load_in_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            model_name: 모델 이름 (unsloth/... 또는 HuggingFace)
            max_seq_length: 최대 시퀀스 길이
            dtype: 데이터 타입 (None=auto, float16, bfloat16)
            load_in_4bit: 4-bit 양자화 로드
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            **kwargs: 추가 Unsloth 설정
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.kwargs = kwargs

        # Unsloth 설치 확인
        self._check_dependencies()

        # 모델과 토크나이저 (lazy loading)
        self._model = None
        self._tokenizer = None

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth is required for UnslothProvider. "
                "Install it with: pip install unsloth"
            )

    def load_model(self):
        """모델 및 토크나이저 로드 (lazy loading)"""
        if self._model is not None:
            return self._model, self._tokenizer

        from unsloth import FastLanguageModel

        logger.info(f"Loading Unsloth model: {self.model_name}")

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            **self.kwargs,
        )

        # LoRA 적용
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth 최적화
            random_state=42,
        )

        logger.info("Unsloth model loaded with LoRA")

        return self._model, self._tokenizer

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        dataset_text_field: str = "text",
    ):
        """
        데이터셋 로드

        Args:
            dataset_name: HuggingFace 데이터셋 이름
            split: 데이터셋 split
            dataset_text_field: 텍스트 필드 이름

        Returns:
            Dataset
        """
        from datasets import load_dataset

        logger.info(f"Loading dataset: {dataset_name}")

        dataset = load_dataset(dataset_name, split=split)

        return dataset

    def train(
        self,
        output_dir: str,
        dataset: Optional[Any] = None,
        dataset_name: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 5,
        logging_steps: int = 10,
        save_steps: int = 100,
        **kwargs,
    ):
        """
        훈련 실행

        Args:
            output_dir: 출력 디렉토리
            dataset: 훈련 데이터셋 (None이면 dataset_name 사용)
            dataset_name: HuggingFace 데이터셋 이름
            num_train_epochs: 에폭 수
            per_device_train_batch_size: 배치 크기
            gradient_accumulation_steps: Gradient accumulation
            learning_rate: 학습률
            warmup_steps: Warmup 스텝
            logging_steps: 로깅 간격
            save_steps: 저장 간격
            **kwargs: 추가 TrainingArguments

        Returns:
            Trainer
        """
        from transformers import TrainingArguments
        from trl import SFTTrainer

        # 모델 로드
        model, tokenizer = self.load_model()

        # 데이터셋 로드 (선택)
        if dataset is None and dataset_name:
            dataset = self.load_dataset(dataset_name)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            optim="adamw_8bit",  # Unsloth 최적화
            weight_decay=0.01,
            fp16=not self.load_in_4bit,  # 4-bit이면 fp16 비활성화
            bf16=False,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            seed=42,
            **kwargs,
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=self.max_seq_length,
            dataset_text_field=kwargs.get("dataset_text_field", "text"),
            packing=kwargs.get("packing", False),
        )

        logger.info("Starting Unsloth training...")

        # 훈련 시작
        trainer.train()

        logger.info("Unsloth training completed")

        return trainer

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
    ):
        """
        모델 저장

        Args:
            output_dir: 출력 디렉토리
            save_method: 저장 방법
                - "merged_16bit": LoRA 병합 + 16bit
                - "merged_4bit": LoRA 병합 + 4bit
                - "lora": LoRA 어댑터만

        Returns:
            None
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Saving Unsloth model to: {output_dir} ({save_method})")

        if save_method == "merged_16bit":
            self._model.save_pretrained_merged(
                output_dir,
                self._tokenizer,
                save_method="merged_16bit"
            )
        elif save_method == "merged_4bit":
            self._model.save_pretrained_merged(
                output_dir,
                self._tokenizer,
                save_method="merged_4bit"
            )
        elif save_method == "lora":
            self._model.save_pretrained(output_dir)
            self._tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError(f"Unknown save_method: {save_method}")

        logger.info("Unsloth model saved successfully")

    def __repr__(self) -> str:
        return (
            f"UnslothProvider(model={self.model_name}, "
            f"lora_r={self.lora_r}, 4bit={self.load_in_4bit})"
        )
