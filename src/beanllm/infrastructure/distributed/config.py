"""
분산 시스템 설정 (Distributed System Configuration)

각 파이프라인별로 분산 시스템 파라미터를 자유롭게 수정하고 적용할 수 있도록 설정 구조 제공
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRDistributedConfig:
    """OCR 분산 시스템 설정"""

    # Rate Limiting
    enable_rate_limiting: bool = True
    rate_limit_per_second: int = 10

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 초

    # Task Queue
    use_distributed_queue: bool = False  # USE_DISTRIBUTED 따라감
    max_concurrent: int = 4

    # Event Streaming
    enable_event_streaming: bool = True  # USE_DISTRIBUTED 따라감

    # Distributed Lock
    enable_distributed_lock: bool = True  # USE_DISTRIBUTED 따라감
    lock_timeout: float = 60.0  # 초


@dataclass
class VisionRAGDistributedConfig:
    """Vision RAG 분산 시스템 설정"""

    # Rate Limiting
    enable_rate_limiting: bool = True
    embedding_rate_limit: int = 20  # 초당
    llm_rate_limit: int = 10  # 초당

    # Caching
    enable_embedding_cache: bool = True
    embedding_cache_ttl: int = 7200  # 초
    enable_search_cache: bool = True
    search_cache_ttl: int = 3600  # 초

    # Task Queue
    use_distributed_queue: bool = False
    queue_threshold: int = 100  # 이미지 수가 이 값 이상이면 자동 활성화
    max_concurrent_images: int = 10

    # Event Streaming
    enable_event_streaming: bool = True

    # Distributed Lock
    enable_distributed_lock: bool = True
    lock_timeout: float = 120.0  # 초


@dataclass
class MultiAgentDistributedConfig:
    """Multi-Agent 분산 시스템 설정"""

    # Rate Limiting
    enable_rate_limiting: bool = True
    agent_rate_limit: int = 10  # 초당
    per_agent_rate_limit: bool = True  # 각 Agent별로 제한

    # Caching
    enable_agent_cache: bool = True
    agent_cache_ttl: int = 1800  # 초

    # Task Queue
    use_distributed_queue: bool = False
    queue_priority: int = 0

    # Event Streaming (Kafka)
    use_kafka_bus: bool = False  # USE_DISTRIBUTED 따라감
    kafka_topic_prefix: str = "multi_agent"

    # Distributed Lock
    enable_distributed_lock: bool = True
    lock_timeout: float = 300.0  # 초


@dataclass
class ChainDistributedConfig:
    """Chain 분산 시스템 설정"""

    # Rate Limiting
    enable_rate_limiting: bool = True
    chain_rate_limit: int = 10  # 초당

    # Caching
    enable_chain_cache: bool = True
    chain_cache_ttl: int = 3600  # 초

    # Task Queue
    use_distributed_queue: bool = False
    queue_threshold_seconds: float = 60.0  # 예상 실행 시간

    # Event Streaming
    enable_event_streaming: bool = True

    # Distributed Lock
    enable_distributed_lock: bool = True
    lock_timeout: float = 300.0  # 초


@dataclass
class GraphDistributedConfig:
    """Graph 분산 시스템 설정"""

    # Rate Limiting
    enable_rate_limiting: bool = True
    graph_rate_limit: int = 10  # 초당
    per_node_rate_limit: bool = False  # 전체 Graph에 대해 제한

    # Caching
    # NodeCache 사용 (이미 분산 캐시 지원)

    # Task Queue
    use_distributed_queue: bool = False
    queue_threshold_seconds: float = 120.0  # 예상 실행 시간

    # Event Streaming
    enable_event_streaming: bool = True

    # Distributed Lock
    enable_distributed_lock: bool = True
    lock_timeout: float = 600.0  # 초


@dataclass
class DistributedConfig:
    """전역 분산 시스템 설정"""

    # 기본 설정
    use_distributed: bool = False  # 환경변수 USE_DISTRIBUTED 따라감

    # Redis 설정
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Kafka 설정
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "llmkit"

    # OCR 설정
    ocr: OCRDistributedConfig = field(default_factory=OCRDistributedConfig)

    # Vision RAG 설정
    vision_rag: VisionRAGDistributedConfig = field(default_factory=VisionRAGDistributedConfig)

    # Multi-Agent 설정
    multi_agent: MultiAgentDistributedConfig = field(default_factory=MultiAgentDistributedConfig)

    # Chain 설정
    chain: ChainDistributedConfig = field(default_factory=ChainDistributedConfig)

    # Graph 설정
    graph: GraphDistributedConfig = field(default_factory=GraphDistributedConfig)

    def __post_init__(self):
        """환경변수에서 기본값 로드"""
        use_distributed = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"
        self.use_distributed = use_distributed

        # Redis 설정
        self.redis_host = os.getenv("REDIS_HOST", self.redis_host)
        self.redis_port = int(os.getenv("REDIS_PORT", str(self.redis_port)))
        self.redis_db = int(os.getenv("REDIS_DB", str(self.redis_db)))
        self.redis_password = os.getenv("REDIS_PASSWORD", self.redis_password)

        # Kafka 설정
        self.kafka_bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", self.kafka_bootstrap_servers
        )
        self.kafka_topic_prefix = os.getenv("KAFKA_TOPIC_PREFIX", self.kafka_topic_prefix)

        # 각 파이프라인 설정도 환경변수에서 로드 가능
        if use_distributed:
            # 기본적으로 분산 모드 활성화
            self.ocr.use_distributed_queue = True
            self.ocr.enable_event_streaming = True
            self.ocr.enable_distributed_lock = True

            self.vision_rag.use_distributed_queue = True
            self.vision_rag.enable_event_streaming = True
            self.vision_rag.enable_distributed_lock = True

            self.multi_agent.use_kafka_bus = True
            self.multi_agent.use_distributed_queue = True
            self.multi_agent.enable_distributed_lock = True

            self.chain.use_distributed_queue = True
            self.chain.enable_event_streaming = True
            self.chain.enable_distributed_lock = True

            self.graph.use_distributed_queue = True
            self.graph.enable_event_streaming = True
            self.graph.enable_distributed_lock = True


# 전역 설정 인스턴스
_global_distributed_config: Optional[DistributedConfig] = None


def get_distributed_config() -> DistributedConfig:
    """전역 분산 설정 반환"""
    global _global_distributed_config
    if _global_distributed_config is None:
        _global_distributed_config = DistributedConfig()
    return _global_distributed_config


def set_distributed_config(config: DistributedConfig) -> None:
    """전역 분산 설정 설정"""
    global _global_distributed_config
    _global_distributed_config = config


def update_pipeline_config(pipeline_type: str, **kwargs) -> None:
    """
    파이프라인별 설정 동적 수정

    Args:
        pipeline_type: 파이프라인 타입 ("ocr", "vision_rag", "multi_agent", "chain", "graph")
        **kwargs: 수정할 설정 (예: enable_rate_limiting=True, rate_limit_per_second=20)

    Example:
        ```python
        from beanllm.infrastructure.distributed import update_pipeline_config

        # Vision RAG의 Rate Limiting 비활성화
        update_pipeline_config("vision_rag", enable_rate_limiting=False)

        # Chain의 캐시 TTL 변경
        update_pipeline_config("chain", chain_cache_ttl=7200)

        # Multi-Agent의 Kafka Bus 활성화
        update_pipeline_config("multi_agent", use_kafka_bus=True)
        ```
    """
    config = get_distributed_config()
    pipeline_config = getattr(config, pipeline_type, None)

    if pipeline_config is None:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    # 설정 업데이트
    for key, value in kwargs.items():
        if hasattr(pipeline_config, key):
            setattr(pipeline_config, key, value)
            logger.info(f"Updated {pipeline_type}.{key} = {value}")
        else:
            # 전역 설정 업데이트
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Updated global.{key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key} for pipeline {pipeline_type}")


def get_pipeline_config(pipeline_type: str):
    """
    파이프라인별 설정 조회

    Args:
        pipeline_type: 파이프라인 타입

    Returns:
        파이프라인 설정 객체
    """
    config = get_distributed_config()
    pipeline_config = getattr(config, pipeline_type, None)

    if pipeline_config is None:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    return pipeline_config


def reset_pipeline_config(pipeline_type: str) -> None:
    """
    파이프라인별 설정 초기화 (기본값으로 복원)

    Args:
        pipeline_type: 파이프라인 타입
    """
    config = get_distributed_config()

    if pipeline_type == "ocr":
        config.ocr = OCRDistributedConfig()
    elif pipeline_type == "vision_rag":
        config.vision_rag = VisionRAGDistributedConfig()
    elif pipeline_type == "multi_agent":
        config.multi_agent = MultiAgentDistributedConfig()
    elif pipeline_type == "chain":
        config.chain = ChainDistributedConfig()
    elif pipeline_type == "graph":
        config.graph = GraphDistributedConfig()
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    logger.info(f"Reset {pipeline_type} config to defaults")
