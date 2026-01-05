"""
Embeddings Base - 임베딩 베이스 클래스

Template Method Pattern을 사용하여 Provider 간 중복 코드 제거
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class BaseEmbedding(ABC):
    """
    Embedding 베이스 클래스 (Template Method Pattern)

    공통 기능:
        - API 키 가져오기 및 검증
        - Import 검증
        - 에러 처리 및 로깅
        - async → sync 위임
    """

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: 모델 이름
            **kwargs: provider별 추가 파라미터
        """
        self.model = model
        self.kwargs = kwargs

    # Template Methods - 공통 헬퍼 메서드

    def _get_api_key(
        self, api_key: Optional[str], env_vars: List[str], provider_name: str
    ) -> str:
        """
        API 키 가져오기 (환경변수 fallback)

        Args:
            api_key: 직접 전달된 API 키
            env_vars: 확인할 환경변수 리스트 (우선순위 순)
            provider_name: Provider 이름 (에러 메시지용)

        Returns:
            API 키

        Raises:
            ValueError: API 키를 찾을 수 없는 경우

        Example:
            >>> self._get_api_key(
            ...     api_key=None,
            ...     env_vars=["OPENAI_API_KEY"],
            ...     provider_name="OpenAI"
            ... )
        """
        # 직접 전달된 API 키 사용
        if api_key:
            return api_key

        # 환경변수에서 찾기
        for env_var in env_vars:
            key = os.getenv(env_var)
            if key:
                return key

        # 못 찾음
        env_vars_str = " or ".join(env_vars)
        raise ValueError(
            f"{provider_name} API key not found. "
            f"Please provide api_key parameter or set {env_vars_str} environment variable"
        )

    def _validate_import(
        self, module_name: str, package_name: str, install_extra: Optional[str] = None
    ):
        """
        Import 검증 (lazy import)

        Args:
            module_name: Import할 모듈 이름
            package_name: pip 패키지 이름
            install_extra: 추가 설치 옵션 (예: "gemini", "ollama")

        Raises:
            ImportError: 모듈을 import할 수 없는 경우

        Example:
            >>> self._validate_import(
            ...     module_name="openai",
            ...     package_name="openai",
            ...     install_extra=None
            ... )

            >>> self._validate_import(
            ...     module_name="google.generativeai",
            ...     package_name="beanllm",
            ...     install_extra="gemini"
            ... )
        """
        try:
            __import__(module_name)
        except ImportError:
            if install_extra:
                install_cmd = f"pip install {package_name}[{install_extra}]"
            else:
                install_cmd = f"pip install {package_name}"

            raise ImportError(
                f"{module_name} is required for {self.__class__.__name__}. "
                f"Install it with: {install_cmd}"
            )

    def _log_embed_success(self, num_texts: int, extra_info: Optional[str] = None):
        """
        임베딩 성공 로깅 (표준 포맷)

        Args:
            num_texts: 임베딩한 텍스트 수
            extra_info: 추가 정보 (예: "usage: 100 tokens", "batch mode")
        """
        if extra_info:
            logger.info(f"Embedded {num_texts} texts using {self.model} ({extra_info})")
        else:
            logger.info(f"Embedded {num_texts} texts using {self.model}")

    def _handle_embed_error(self, provider_name: str, error: Exception):
        """
        임베딩 에러 처리 (표준 포맷)

        Args:
            provider_name: Provider 이름
            error: 발생한 에러

        Raises:
            Exception: 원본 에러를 다시 raise
        """
        logger.error(f"{provider_name} embedding failed: {error}")
        raise

    # Abstract Methods - 하위 클래스가 구현해야 함

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (비동기)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        pass

    @abstractmethod
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (동기)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        pass


class BaseAPIEmbedding(BaseEmbedding):
    """
    API 기반 Embedding Provider의 베이스 클래스

    공통 기능:
        - API 키 관리
        - async → sync 위임 (대부분의 API는 sync만 지원)

    하위 클래스:
        - OpenAIEmbedding
        - GeminiEmbedding
        - VoyageEmbedding
        - JinaEmbedding
        - MistralEmbedding
        - CohereEmbedding
    """

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (비동기)

        Note: 대부분의 API Provider는 async를 지원하지 않으므로
              sync 메서드를 호출합니다.
        """
        return self.embed_sync(texts)


class BaseLocalEmbedding(BaseEmbedding):
    """
    로컬 모델 기반 Embedding Provider의 베이스 클래스

    공통 기능:
        - Lazy loading (첫 사용 시 모델 로드)
        - GPU/CPU 자동 선택
        - async → sync 위임

    하위 클래스:
        - HuggingFaceEmbedding
        - NVEmbedEmbedding
        - Qwen3Embedding
        - CodeEmbedding
    """

    def __init__(self, model: str, use_gpu: bool = True, **kwargs):
        """
        Args:
            model: 모델 이름
            use_gpu: GPU 사용 여부
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)
        self.use_gpu = use_gpu

        # Lazy loading
        self._model = None
        self._device = None

    @abstractmethod
    def _load_model(self):
        """
        모델 로딩 (lazy loading)

        Note: 하위 클래스에서 구현해야 합니다.
              - torch.cuda.is_available() 체크
              - 모델 및 토크나이저 로드
              - device 설정
        """
        pass

    def _get_device(self) -> str:
        """
        Device 선택 (GPU/CPU)

        Returns:
            "cuda" 또는 "cpu"
        """
        try:
            import torch

            if self.use_gpu and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (비동기)

        Note: sentence-transformers/transformers는 async를 지원하지 않으므로
              sync 메서드를 호출합니다.
        """
        return self.embed_sync(texts)
