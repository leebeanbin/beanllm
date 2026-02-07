"""
이벤트 통합 (Event Integration)

기존 코드에 이벤트 발행 기능 추가
기존 최적화 패턴 참고: Helper 메서드, 에러 처리
"""

from typing import Any, Callable, Dict

from .factory import get_event_bus
from .messaging import MessageProducer
from .utils import sanitize_error_message

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


def with_event_publishing(event_type: str, include_result: bool = True):
    """
    이벤트 발행 데코레이터

    함수 실행 전후로 이벤트를 자동 발행

    Args:
        event_type: 이벤트 타입 (예: "document.loaded", "embedding.completed")
        include_result: 결과를 이벤트에 포함할지 여부

    Example:
        ```python
        @with_event_publishing("document.loaded")
        async def load_document(path: str):
            # 문서 로드
            return document
        ```
    """

    def decorator(func: Callable) -> Callable:
        import asyncio
        import functools

        producer, _ = get_event_bus()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 시작 이벤트 발행
            await producer.publish(
                "llm.events",
                {
                    "event_type": f"{event_type}.started",
                    "function": func.__name__,
                    "args": str(args)[:100],  # 처음 100자만
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                },
            )

            try:
                result = await func(*args, **kwargs)

                # 완료 이벤트 발행
                event_data = {
                    "event_type": f"{event_type}.completed",
                    "function": func.__name__,
                }
                if include_result:
                    event_data["result"] = str(result)[:500]  # 처음 500자만

                await producer.publish("llm.events", event_data)
                return result
            except Exception as e:
                # 오류 이벤트 발행
                await producer.publish(
                    "llm.events",
                    {
                        "event_type": f"{event_type}.error",
                        "function": func.__name__,
                        "error": sanitize_error_message(str(e)),
                    },
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import asyncio

            # 동기 함수는 비동기로 래핑
            async def _async_wrapper():
                return await async_wrapper(*args, **kwargs)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 순차 처리
                    return func(*args, **kwargs)
                else:
                    return loop.run_until_complete(_async_wrapper())
            except RuntimeError:
                return asyncio.run(_async_wrapper())

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class EventLogger:
    """
    이벤트 로거 (기존 로깅과 통합)

    기존 로깅 시스템과 이벤트 발행을 통합
    """

    def __init__(self):
        self.message_producer = MessageProducer()

    async def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        level: str = "info",
    ):
        """
        이벤트 로깅 (로깅 + 이벤트 발행)

        Args:
            event_type: 이벤트 타입
            event_data: 이벤트 데이터
            level: 로그 레벨 (info, warning, error)
        """
        # 기존 로깅
        if level == "error":
            logger.error(f"{event_type}: {event_data}")
        elif level == "warning":
            logger.warning(f"{event_type}: {event_data}")
        else:
            logger.info(f"{event_type}: {event_data}")

        # 이벤트 발행 (Kafka 또는 인메모리)
        try:
            await self.message_producer.publish_event(event_type, event_data)
        except Exception as e:
            # 이벤트 발행 실패 시 로깅만 (fallback)
            logger.warning(
                f"Failed to publish event {event_type}: {sanitize_error_message(str(e))}"
            )

    def log_document_loaded(self, document_path: str, document_type: str):
        """문서 로드 이벤트"""
        import asyncio

        asyncio.create_task(
            self.log_event(
                "document.loaded",
                {
                    "path": document_path,
                    "type": document_type,
                },
            )
        )

    def log_embedding_completed(self, text_count: int, model: str):
        """임베딩 완료 이벤트"""
        import asyncio

        asyncio.create_task(
            self.log_event(
                "embedding.completed",
                {
                    "text_count": text_count,
                    "model": model,
                },
            )
        )

    def log_rag_query(self, question: str, result_count: int):
        """RAG 쿼리 이벤트"""
        import asyncio

        asyncio.create_task(
            self.log_event(
                "rag.query",
                {
                    "question": question[:100],  # 처음 100자만
                    "result_count": result_count,
                },
            )
        )


# 전역 이벤트 로거
_global_event_logger = EventLogger()


def get_event_logger() -> EventLogger:
    """전역 이벤트 로거 반환"""
    return _global_event_logger
