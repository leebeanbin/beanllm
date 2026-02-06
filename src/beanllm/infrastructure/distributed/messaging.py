"""
메시지 기반 아키텍처 핵심 컴포넌트

모든 요청/동작을 메시지로 발행하고 Redis로 동시성 제어
기존 최적화 패턴 참고: 에러 처리, 로깅, Helper 메서드
"""

import json
import time
import traceback
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from .factory import get_distributed_lock, get_event_bus, get_rate_limiter
from .utils import sanitize_error_message

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class MessageProducer:
    """
    메시지 발행자

    모든 요청/동작을 메시지로 발행하여 이벤트 로그 생성
    """

    def __init__(self):
        self.producer, _ = get_event_bus()
        self.redis = None  # Redis는 필요시 lazy loading
        try:
            from .redis.client import get_redis_client

            self.redis = get_redis_client()
        except Exception as e:
            logger.debug(f"Redis client not available (continuing without Redis): {e}")

    async def publish_request(self, request_type: str, request_data: Dict[str, Any]) -> str:
        """
        요청 메시지 발행

        Args:
            request_type: 요청 타입 (ocr.recognize, audio.transcribe, rag.query, etc.)
            request_data: 요청 데이터

        Returns:
            request_id: 요청 ID
        """
        request_id = str(uuid.uuid4())

        message = {
            "request_id": request_id,
            "request_type": request_type,
            "timestamp": time.time(),
            "data": request_data,
            "status": "pending",
            "metadata": {
                "worker_id": None,
                "started_at": None,
                "completed_at": None,
                "error": None,
            },
        }

        try:
            # Kafka에 메시지 발행 (영구 저장, 이벤트 로그)
            await self.producer.publish("llm.requests", message)

            # Redis에 요청 상태 저장 (빠른 조회용)
            if self.redis:
                try:
                    await self.redis.setex(
                        f"request:status:{request_id}",
                        3600,  # 1시간 TTL
                        json.dumps(
                            {
                                "status": "pending",
                                "request_type": request_type,
                                "created_at": message["timestamp"],
                            }
                        ).encode("utf-8"),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to save request status to Redis: {sanitize_error_message(str(e))}"
                    )

            # Redis에 요청 큐 추가 (빠른 처리용)
            if self.redis:
                try:
                    await self.redis.lpush(
                        f"queue:{request_type}",
                        json.dumps(
                            {
                                "request_id": request_id,
                                "data": request_data,
                            }
                        ).encode("utf-8"),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to add request to Redis queue: {sanitize_error_message(str(e))}"
                    )

        except Exception as e:
            logger.error(
                f"Failed to publish request: {sanitize_error_message(str(e))}", exc_info=True
            )

        return request_id

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        이벤트 메시지 발행 (로깅, 모니터링)

        Args:
            event_type: 이벤트 타입 (ocr.request.started, document.processed, etc.)
            event_data: 이벤트 데이터
        """
        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "timestamp": time.time(),
                "data": event_data,
            }

            await self.producer.publish("llm.events", event)
        except Exception as e:
            logger.error(
                f"Failed to publish event: {sanitize_error_message(str(e))}", exc_info=True
            )


class ConcurrencyController:
    """
    동시성 제어자

    Redis 기반 동시성 제어 (Rate Limiting, Lock, 동시 실행 제한)
    기존 최적화 패턴 참고: Helper 메서드로 중복 제거
    """

    def __init__(self):
        self.rate_limiter = get_rate_limiter()
        self.lock = get_distributed_lock()
        self.redis = None
        try:
            from .redis.client import get_redis_client

            self.redis = get_redis_client()
        except Exception as e:
            logger.debug(f"Redis client not available (continuing without Redis): {e}")

    async def acquire_slot(self, resource_type: str, max_concurrent: int = 10) -> bool:
        """
        동시 실행 슬롯 획득

        Args:
            resource_type: 리소스 타입 (ocr, audio, embedding, etc.)
            max_concurrent: 최대 동시 실행 수

        Returns:
            True: 슬롯 획득 성공, False: 슬롯 부족
        """
        if not self.redis:
            # Redis 없으면 무제한 허용 (fallback)
            return True

        try:
            key = f"concurrency:{resource_type}"

            # 현재 실행 중인 작업 수 확인
            current = await self.redis.get(key)
            current_count = int(current) if current else 0

            if current_count >= max_concurrent:
                return False

            # 슬롯 획득 (원자적 연산)
            new_count = await self.redis.incr(key)
            await self.redis.expire(key, 300)  # 5분 TTL

            return new_count <= max_concurrent
        except Exception as e:
            logger.warning(f"Failed to acquire concurrency slot: {sanitize_error_message(str(e))}")
            return True  # 오류 시 허용 (fallback)

    async def release_slot(self, resource_type: str):
        """동시 실행 슬롯 해제"""
        if not self.redis:
            return

        try:
            key = f"concurrency:{resource_type}"
            await self.redis.decr(key)
        except Exception as e:
            logger.warning(f"Failed to release concurrency slot: {sanitize_error_message(str(e))}")

    async def with_concurrency_control(
        self,
        resource_type: str,
        max_concurrent: int = 10,
        rate_limit_key: Optional[str] = None,
    ):
        """
        동시성 제어 컨텍스트 매니저

        Usage:
            async with concurrency_controller.with_concurrency_control("ocr", max_concurrent=5):
                # OCR 처리
                result = await process_ocr(image)
        """
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _context():
            # Rate Limiting
            if rate_limit_key:
                await self.rate_limiter.wait(rate_limit_key)

            # 동시성 제어
            acquired = await self.acquire_slot(resource_type, max_concurrent)
            if not acquired:
                from .utils import DistributedError

                raise DistributedError(
                    f"Max concurrent {resource_type} operations reached ({max_concurrent})"
                )

            try:
                yield
            finally:
                await self.release_slot(resource_type)

        return _context()


class DistributedErrorHandler:
    """
    분산 환경 오류 처리

    기존 최적화 패턴 참고: 에러 처리, 로깅, 이벤트 발행
    """

    def __init__(self):
        self.message_producer = MessageProducer()
        self.redis = None
        try:
            from .redis.client import get_redis_client

            self.redis = get_redis_client()
        except Exception as e:
            logger.debug(f"Redis client not available (continuing without Redis): {e}")

    async def handle_error(
        self,
        request_id: str,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        오류 처리 (로깅, 이벤트 발행, 상태 저장)

        Args:
            request_id: 요청 ID
            error: 발생한 오류
            operation: 수행 중이던 작업
            context: 추가 컨텍스트
        """
        error_data = {
            "request_id": request_id,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": sanitize_error_message(str(error)),
            "traceback": sanitize_error_message(traceback.format_exc()),
            "context": context or {},
            "timestamp": time.time(),
        }

        # 1. 구조화된 로깅
        logger.error(
            f"Error in {operation}",
            exc_info=True,
            extra={
                "request_id": request_id,
                "operation": operation,
                "error_type": error_data["error_type"],
                "error_message": error_data["error_message"],
                **(context or {}),
            },
        )

        # 2. Kafka에 오류 이벤트 발행 (영구 저장)
        await self.message_producer.publish_event(f"{operation}.error", error_data)

        # 3. Redis에 오류 상태 저장 (빠른 조회)
        if self.redis:
            try:
                await self.redis.setex(
                    f"error:{request_id}",
                    86400,  # 24시간 TTL
                    json.dumps(error_data).encode("utf-8"),
                )

                # 오류 통계 업데이트
                await self.redis.incr(f"error:count:{operation}")
                await self.redis.expire(f"error:count:{operation}", 86400)

                # 요청 상태 업데이트
                await self.redis.setex(
                    f"request:status:{request_id}",
                    3600,
                    json.dumps(
                        {
                            "status": "error",
                            "error": error_data,
                            "updated_at": time.time(),
                        }
                    ).encode("utf-8"),
                )
            except Exception as e:
                logger.warning(f"Failed to save error to Redis: {sanitize_error_message(str(e))}")

    async def get_error_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """오류 로그 조회"""
        if not self.redis:
            return None

        try:
            error_data = await self.redis.get(f"error:{request_id}")
            if error_data:
                if isinstance(error_data, bytes):
                    error_data = error_data.decode("utf-8")
                return json.loads(error_data)
        except Exception as e:
            logger.warning(f"Failed to get error log: {sanitize_error_message(str(e))}")

        return None

    async def get_error_stats(self, operation: str, time_window: int = 3600) -> Dict[str, Any]:
        """오류 통계 조회"""
        if not self.redis:
            return {"operation": operation, "error_count": 0, "time_window": time_window}

        try:
            error_count = await self.redis.get(f"error:count:{operation}")
            return {
                "operation": operation,
                "error_count": int(error_count) if error_count else 0,
                "time_window": time_window,
            }
        except Exception as e:
            logger.warning(f"Failed to get error stats: {sanitize_error_message(str(e))}")
            return {"operation": operation, "error_count": 0, "time_window": time_window}


class RequestMonitor:
    """
    요청 모니터링 및 로그 확인

    기존 최적화 패턴 참고: Helper 메서드로 중복 제거
    """

    def __init__(self):
        self.redis = None
        self.consumer = None
        try:
            from .redis.client import get_redis_client

            self.redis = get_redis_client()
            if self.redis:
                logger.info("RequestMonitor: Redis client successfully obtained.")
            else:
                logger.warning("RequestMonitor: Redis client is None after get_redis_client().")
        except ImportError as e:
            logger.warning(f"RequestMonitor: Failed to import Redis client: {e}")
        except Exception as e:
            logger.warning(f"RequestMonitor: Failed to get Redis client: {e}", exc_info=True)
        try:
            _, self.consumer = get_event_bus()
        except Exception as e:
            logger.debug(
                f"RequestMonitor: Kafka consumer not available (continuing without Kafka): {e}"
            )

    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """요청 상태 조회 (Redis - 빠른 조회)"""
        if not self.redis:
            return None

        try:
            status = await self.redis.get(f"request:status:{request_id}")
            if status:
                if isinstance(status, bytes):
                    status = status.decode("utf-8")
                return json.loads(status)
        except Exception as e:
            logger.warning(f"Failed to get request status: {sanitize_error_message(str(e))}")

        return None

    async def get_request_logs(self, request_id: str) -> AsyncIterator[Dict[str, Any]]:
        """요청 로그 조회 (Kafka - 영구 저장)"""
        if not self.consumer:
            return

        # Kafka에서 해당 request_id의 모든 이벤트 조회
        # 실제 구현은 복잡하므로 여기서는 기본 구조만 제공
        async for event in self.consumer.subscribe("llm.events", lambda e: None):
            if event.get("data", {}).get("request_id") == request_id:
                yield event

    async def get_error_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """오류 로그 조회"""
        if not self.redis:
            return None

        try:
            error_data = await self.redis.get(f"error:{request_id}")
            if error_data:
                if isinstance(error_data, bytes):
                    error_data = error_data.decode("utf-8")
                return json.loads(error_data)
        except Exception as e:
            logger.warning(f"Failed to get error log: {sanitize_error_message(str(e))}")

        return None
