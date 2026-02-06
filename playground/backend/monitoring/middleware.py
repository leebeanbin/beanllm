"""
분산 모니터링 시스템

Kafka + Redis + 실시간 대시보드를 활용한 상세 로깅 및 모니터링
§0 메시징 처리 적극 활용: RECORD_METRICS_FOR_ALL_API=true 시 전체 /api/* 메트릭 기록
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from beanllm.infrastructure.distributed import get_event_logger
    from beanllm.infrastructure.distributed.messaging import RequestMonitor
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)

    def get_event_logger():
        return None

    def RequestMonitor():
        return None


logger = get_logger(__name__)


def _is_chat_api_path(path: str) -> bool:
    """채팅 API 경로 여부. 대시보드 집계는 채팅 전용."""
    return path == "/api/chat" or path.startswith("/api/chat/")


# 메트릭에 절대 넣지 않을 경로(대시보드 리프레시·헬스·설정 조회 등)
_METRICS_EXCLUDED_PREFIXES = ("/api/monitoring", "/health", "/api/config", "/api/models", "/")


def _should_record_metrics(path: str) -> bool:
    """
    이 요청에 대해 Redis 메트릭(requests/errors/response_time/endpoint)을 기록할지 여부.
    - 대시보드/모니터링·헬스·설정 조회는 항상 제외(리프레시가 total requests에 잡히지 않도록).
    - 채팅 API(/api/chat, /api/chat/*)만 기록. RECORD_METRICS_FOR_ALL_API=true여도 제외 경로는 안 넣음.
    """
    if not path or path == "/":
        return False
    for prefix in _METRICS_EXCLUDED_PREFIXES:
        if path == prefix or (len(prefix) > 1 and path.startswith(prefix)):
            return False
    if _is_chat_api_path(path):
        return True
    if os.getenv("RECORD_METRICS_FOR_ALL_API", "false").lower() == "true":
        return path.startswith("/api/")
    return False


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    상세 로깅 및 모니터링 미들웨어

    - Request ID 생성 및 전파
    - 요청/응답 로깅
    - 성능 메트릭 수집
    - Kafka를 통한 이벤트 스트리밍
    - Redis를 통한 실시간 메트릭 저장
    """

    def __init__(self, app, enable_kafka: bool = True, enable_redis: bool = True):
        super().__init__(app)
        self.enable_kafka = enable_kafka
        self.enable_redis = enable_redis
        self.event_logger = None
        self.request_monitor = None

        # 이벤트 로거 초기화
        if self.enable_kafka:
            try:
                self.event_logger = get_event_logger()
            except Exception as e:
                logger.warning(f"Failed to initialize event logger: {e}")
                self.enable_kafka = False

        # Request Monitor 초기화
        if self.enable_redis:
            try:
                # 직접 get_redis_client를 호출해서 테스트
                try:
                    from beanllm.infrastructure.distributed.redis.client import get_redis_client

                    logger.info("🔍 Testing Redis client creation...")
                    test_redis = get_redis_client()
                    if test_redis:
                        logger.info(f"✅ Redis client created successfully: {type(test_redis)}")
                    else:
                        logger.error("❌ get_redis_client() returned None")
                        self.enable_redis = False
                except ImportError as e:
                    logger.error(f"❌ Failed to import Redis client module: {e}", exc_info=True)
                    self.enable_redis = False
                except Exception as e:
                    logger.error(f"❌ Failed to create Redis client: {e}", exc_info=True)
                    logger.error("   This might be a connection issue or import error")
                    self.enable_redis = False

                # RequestMonitor 초기화
                if self.enable_redis:
                    try:
                        logger.info("🔍 Initializing RequestMonitor...")
                        self.request_monitor = RequestMonitor()
                        if self.request_monitor and self.request_monitor.redis:
                            logger.info("✅ Redis monitoring initialized successfully")
                            logger.info(f"   Redis client type: {type(self.request_monitor.redis)}")
                        else:
                            logger.error("❌ RequestMonitor.redis is None after initialization")
                            logger.error("   RequestMonitor.__init__() completed but redis is None")
                            logger.error(
                                "   This means get_redis_client() failed silently in RequestMonitor"
                            )
                            logger.error(
                                "   Check RequestMonitor.__init__() logs (DEBUG level) for details"
                            )
                            self.enable_redis = False
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize RequestMonitor: {e}", exc_info=True)
                        logger.error("   Redis monitoring will be disabled.")
                        self.enable_redis = False
            except Exception as e:
                logger.error(
                    f"❌ Unexpected error during Redis monitoring setup: {e}", exc_info=True
                )
                self.enable_redis = False

    async def dispatch(self, request: Request, call_next):
        """요청 처리 및 모니터링"""
        # Request ID 생성 (헤더에 있으면 사용, 없으면 생성)
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # 시작 시간
        start_time = time.time()

        # 요청 정보 수집
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.now().isoformat(),
        }

        # 요청 본문 읽기 (가능한 경우)
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    try:
                        request_info["body"] = json.loads(body.decode())
                    except:
                        request_info["body_size"] = len(body)
        except Exception as e:
            logger.debug(f"Failed to read request body: {e}")

        # 요청 로깅
        logger.info(
            f"[REQUEST] {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "request_info": request_info,
            },
        )

        # Kafka 이벤트 발행 (요청 시작)
        if self.enable_kafka and self.event_logger:
            try:
                await self.event_logger.log_event(
                    "api.request.started",
                    {
                        **request_info,
                        "event_type": "request_started",
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to publish request event: {e}")

        # Redis에 요청 상태 저장
        if self.enable_redis and self.request_monitor and self.request_monitor.redis:
            try:
                status_data = {
                    "request_id": request_id,
                    "status": "processing",
                    "started_at": start_time,
                    "path": str(request.url.path),
                    "method": request.method,
                }
                await self.request_monitor.redis.setex(
                    f"request:status:{request_id}",
                    3600,
                    json.dumps(status_data),  # 1시간 TTL
                )
                logger.debug(f"✅ Saved request status to Redis: {request_id}")
            except Exception as e:
                logger.warning(f"❌ Failed to save request status to Redis: {e}", exc_info=True)
        elif self.enable_redis:
            # Redis가 활성화되어 있지만 연결이 안 된 경우 (첫 요청에서만 로그)
            if not hasattr(self, "_redis_warning_logged"):
                if not self.request_monitor:
                    logger.error("❌ RequestMonitor not initialized - Redis monitoring disabled")
                elif not self.request_monitor.redis:
                    logger.error("❌ RequestMonitor.redis is None - Redis connection failed")
                    logger.error("   Check Redis server and connection settings")
                self._redis_warning_logged = True

        # 요청 처리
        error_occurred = False
        error_message = None
        status_code = 200
        response = None

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            status_code = 500
            logger.error(
                f"[ERROR] {request.method} {request.url.path}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "error": error_message,
                },
            )
            raise
        finally:
            # 처리 시간 계산
            duration_ms = (time.time() - start_time) * 1000

            # 응답 정보 수집
            response_info = {
                "request_id": request_id,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "error": error_message if error_occurred else None,
            }

            # 응답 로깅
            log_level = (
                "error"
                if error_occurred or status_code >= 500
                else "warning"
                if status_code >= 400
                else "info"
            )
            logger.log(
                getattr(logging, log_level.upper()),
                f"[RESPONSE] {request.method} {request.url.path} - {status_code} ({duration_ms:.2f}ms)",
                extra={
                    "request_id": request_id,
                    "response_info": response_info,
                },
            )

            # Kafka 이벤트 발행 (요청 완료)
            if self.enable_kafka and self.event_logger:
                try:
                    await self.event_logger.log_event(
                        "api.request.completed",
                        {
                            **request_info,
                            **response_info,
                            "event_type": "request_completed",
                        },
                        level="error" if error_occurred else "info",
                    )
                except Exception as e:
                    logger.debug(f"Failed to publish response event: {e}")

            # Redis에 응답 상태 업데이트 및 메트릭 저장
            if self.enable_redis and self.request_monitor and self.request_monitor.redis:
                try:
                    logger.debug(f"Saving metrics to Redis for request {request_id}")
                    # 상태 업데이트
                    status_data = {
                        "request_id": request_id,
                        "status": "completed" if not error_occurred else "error",
                        "started_at": start_time,
                        "completed_at": time.time(),
                        "duration_ms": duration_ms,
                        "status_code": status_code,
                        "path": str(request.url.path),
                        "method": request.method,
                    }
                    await self.request_monitor.redis.setex(
                        f"request:status:{request_id}", 3600, json.dumps(status_data)
                    )

                    # 추적 메트릭 기준: docs/CACHE_AND_METRICS_POLICY.md §2 — requests/errors/response_time/endpoint만 수집
                    # 대상 경로: 채팅 또는 RECORD_METRICS_FOR_ALL_API 시 /api/*
                    path = str(request.url.path)
                    if _should_record_metrics(path):
                        # 1. 응답 시간 메트릭
                        await self.request_monitor.redis.zadd(
                            "metrics:response_time", {request_id: duration_ms}
                        )
                        await self.request_monitor.redis.expire("metrics:response_time", 3600)
                        # 2. 요청 수 카운터 (시간대별)
                        minute_key = f"metrics:requests:{int(time.time() // 60)}"
                        await self.request_monitor.redis.incr(minute_key)
                        await self.request_monitor.redis.expire(minute_key, 3600)
                        # 3. 에러 카운터
                        if error_occurred or status_code >= 500:
                            error_key = f"metrics:errors:{int(time.time() // 60)}"
                            await self.request_monitor.redis.incr(error_key)
                            await self.request_monitor.redis.expire(error_key, 3600)
                        # 4. 엔드포인트별 통계
                        endpoint_key = f"metrics:endpoint:{request.method}:{path}"
                        await self.request_monitor.redis.hincrby(endpoint_key, "count", 1)
                        await self.request_monitor.redis.hincrby(
                            endpoint_key, "total_time_ms", int(duration_ms)
                        )
                        if error_occurred or status_code >= 500:
                            await self.request_monitor.redis.hincrby(endpoint_key, "errors", 1)
                        await self.request_monitor.redis.expire(endpoint_key, 3600)

                except Exception as e:
                    logger.warning(f"Failed to update metrics in Redis: {e}", exc_info=True)

            # Response 헤더에 Request ID 추가 (예외 시 response가 없으므로 생략)
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class ChatMonitoringMixin:
    """
    Chat API 전용 모니터링 믹스인

    LLM 호출에 대한 상세 로깅
    """

    @staticmethod
    def _chat_request_preview(messages: list, max_len: int = 100) -> str:
        """마지막 user 메시지 일부만 추출. CHAT_HISTORY_MASK_PREVIEW=true면 고정문."""
        if os.getenv("CHAT_HISTORY_MASK_PREVIEW", "false").lower() == "true":
            return "(마스킹)"
        if not messages:
            return ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return (part.get("text") or "")[:max_len]
                    return ""
                return (c or "")[:max_len]
        return ""

    @staticmethod
    async def log_chat_request(
        request_id: str,
        model: str,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Chat 요청 로깅"""
        event_logger = None
        try:
            event_logger = get_event_logger()
        except:
            pass

        if event_logger:
            try:
                await event_logger.log_event(
                    "chat.request",
                    {
                        "request_id": request_id,
                        "model": model,
                        "message_count": len(messages),
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "first_message_preview": (
                            messages[0].get("content", "")[:100] if messages else ""
                        ),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to log chat request: {e}")

        # 챗 상세 이력: 상시 수집 (env 없음, CHAT_HISTORY_METRICS.md)
        try:
            monitor = RequestMonitor()
            if not (monitor and monitor.redis):
                return
            now = time.time()
            at_minute = int(now // 60)
            preview = ChatMonitoringMixin._chat_request_preview(messages)
            await monitor.redis.hset(
                f"chat:record:{request_id}",
                mapping={
                    "model": model,
                    "at_ts": str(int(now)),
                    "at_minute": str(at_minute),
                    "request_preview": preview[:500],
                },
            )
            await monitor.redis.expire(f"chat:record:{request_id}", 86400)
            await monitor.redis.zadd("chat:history", {request_id: now})
            await monitor.redis.expire("chat:history", 86400)
        except Exception as e:
            logger.debug(f"Failed to write chat history (request): {e}")

    @staticmethod
    async def log_chat_response(
        request_id: str,
        model: str,
        response_content: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ):
        """Chat 응답 로깅"""
        event_logger = None
        try:
            event_logger = get_event_logger()
        except:
            pass

        if event_logger:
            try:
                await event_logger.log_event(
                    "chat.response",
                    {
                        "request_id": request_id,
                        "model": model,
                        "response_length": len(response_content),
                        "response_preview": response_content[:200],
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                        "duration_ms": duration_ms,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to log chat response: {e}")

        # Redis에 토큰 사용량 저장
        request_monitor = None
        try:
            request_monitor = RequestMonitor()
        except:
            pass

        if request_monitor and request_monitor.redis:
            try:
                if input_tokens and output_tokens:
                    token_key = f"metrics:tokens:{model}"
                    await request_monitor.redis.hincrby(token_key, "input_tokens", input_tokens)
                    await request_monitor.redis.hincrby(token_key, "output_tokens", output_tokens)
                    await request_monitor.redis.hincrby(
                        token_key, "total_tokens", input_tokens + output_tokens
                    )
                    await request_monitor.redis.hincrby(token_key, "request_count", 1)
                    await request_monitor.redis.expire(token_key, 86400)  # 24시간
            except Exception as e:
                logger.debug(f"Failed to save token metrics: {e}")

        # 챗 상세 이력: chat:record 갱신, 상시 수집 (CHAT_HISTORY_METRICS.md)
        if request_monitor and request_monitor.redis:
            try:
                now = time.time()
                at_minute = int(now // 60)
                total = (input_tokens or 0) + (output_tokens or 0)
                resp_preview = (
                    "(마스킹)"
                    if os.getenv("CHAT_HISTORY_MASK_PREVIEW", "false").lower() == "true"
                    else (response_content or "")[:200]
                )
                await request_monitor.redis.hset(
                    f"chat:record:{request_id}",
                    mapping={
                        "response_preview": resp_preview[:500],
                        "input_tokens": str(input_tokens or 0),
                        "output_tokens": str(output_tokens or 0),
                        "total_tokens": str(total),
                        "duration_ms": str(int(duration_ms or 0)),
                        "at_ts": str(int(now)),
                        "at_minute": str(at_minute),
                    },
                )
                await request_monitor.redis.expire(f"chat:record:{request_id}", 86400)
                await request_monitor.redis.zadd("chat:history", {request_id: now})
                await request_monitor.redis.expire("chat:history", 86400)
            except Exception as e:
                logger.debug(f"Failed to write chat history (response): {e}")
