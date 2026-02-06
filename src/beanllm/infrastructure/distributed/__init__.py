"""
분산 아키텍처 모듈

모든 분산 처리 컴포넌트를 추상화하여 환경에 따라 분산/인메모리 선택 가능

사용 예시:
    ```python
    from beanllm.infrastructure.distributed import (
        get_rate_limiter,
        get_cache,
        get_task_queue,
        get_event_bus,
        get_distributed_lock
    )

    # 환경변수로 자동 선택 (USE_DISTRIBUTED=true/false)
    rate_limiter = get_rate_limiter()
    cache = get_cache()
    task_queue = get_task_queue("ocr.tasks")
    producer, consumer = get_event_bus()
    lock = get_distributed_lock()
    ```
"""

from .cache_helpers import (
    get_agent_result_cache,
    get_chain_result_cache,
    get_llm_response_cache,
    get_rag_search_cache,
    set_agent_result_cache,
    set_chain_result_cache,
    set_llm_response_cache,
    set_rag_search_cache,
)
from .config import (
    ChainDistributedConfig,
    DistributedConfig,
    GraphDistributedConfig,
    MultiAgentDistributedConfig,
    OCRDistributedConfig,
    VisionRAGDistributedConfig,
    get_distributed_config,
    get_pipeline_config,
    reset_pipeline_config,
    set_distributed_config,
    update_pipeline_config,
)
from .event_integration import (
    EventLogger,
    get_event_logger,
    with_event_publishing,
)
from .factory import (
    get_cache,
    get_distributed_lock,
    get_event_bus,
    get_rate_limiter,
    get_task_queue,
)
from .google_events import (
    get_google_export_stats,
    get_security_events,
    log_abnormal_activity,
    log_admin_action,
    log_google_export,
)
from .interfaces import (
    CacheInterface,
    DistributedLockInterface,
    EventConsumerInterface,
    EventProducerInterface,
    RateLimiterInterface,
    TaskQueueInterface,
)
from .lock_integration import (
    LockManager,
    get_lock_manager,
    with_distributed_lock,
)
from .messaging import (
    ConcurrencyController,
    DistributedErrorHandler,
    MessageProducer,
    RequestMonitor,
)
from .pipeline_decorators import (
    with_batch_processing,
    with_distributed_features,
)
from .task_processor import (
    BatchProcessor,
    TaskProcessor,
)
from .utils import (
    ConnectionError,
    DistributedError,
    LockAcquisitionError,
    check_kafka_health,
    check_redis_health,
)

# Streaming (선택적)
try:
    from beanllm.infrastructure.streaming import (
        ProgressTracker,
        ProgressUpdate,
        StreamingSession,
        WebSocketServer,
        get_websocket_server,
    )

    STREAMING_AVAILABLE = True
except ImportError:
    WebSocketServer = None  # type: ignore
    StreamingSession = None  # type: ignore
    ProgressTracker = None  # type: ignore
    ProgressUpdate = None  # type: ignore
    get_websocket_server = None  # type: ignore
    STREAMING_AVAILABLE = False

__all__ = [
    # Interfaces
    "RateLimiterInterface",
    "CacheInterface",
    "TaskQueueInterface",
    "EventProducerInterface",
    "EventConsumerInterface",
    "DistributedLockInterface",
    # Factory functions
    "get_rate_limiter",
    "get_cache",
    "get_task_queue",
    "get_event_bus",
    "get_distributed_lock",
    # Messaging components
    "MessageProducer",
    "ConcurrencyController",
    "DistributedErrorHandler",
    "RequestMonitor",
    # Task processing
    "TaskProcessor",
    "BatchProcessor",
    # Event integration
    "with_event_publishing",
    "EventLogger",
    "get_event_logger",
    # Lock integration
    "with_distributed_lock",
    "LockManager",
    "get_lock_manager",
    # Utilities
    "check_redis_health",
    "check_kafka_health",
    "DistributedError",
    "LockAcquisitionError",
    "ConnectionError",
    # Cache helpers
    "get_rag_search_cache",
    "set_rag_search_cache",
    "get_llm_response_cache",
    "set_llm_response_cache",
    "get_agent_result_cache",
    "set_agent_result_cache",
    "get_chain_result_cache",
    "set_chain_result_cache",
    # Configuration
    "DistributedConfig",
    "OCRDistributedConfig",
    "VisionRAGDistributedConfig",
    "MultiAgentDistributedConfig",
    "ChainDistributedConfig",
    "GraphDistributedConfig",
    "get_distributed_config",
    "set_distributed_config",
    "get_pipeline_config",
    "update_pipeline_config",
    "reset_pipeline_config",
    # Pipeline Decorators
    "with_distributed_features",
    "with_batch_processing",
    # Streaming
    "WebSocketServer",
    "StreamingSession",
    "ProgressTracker",
    "ProgressUpdate",
    "get_websocket_server",
    "STREAMING_AVAILABLE",
]
