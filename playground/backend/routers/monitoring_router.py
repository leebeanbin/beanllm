"""
Monitoring Router

챗(AI 활용) 메트릭 전용. 분산 시스템(Redis) 기반, /api/chat·/api/chat/* 만 집계.
CACHE_AND_METRICS_POLICY §3 — “모니터링은 챗에 대한 것만 제대로 보여주면 됨”, 세부 수치 전부 노출.

- Request metrics / Response time (avg, min, max, p50, p95, p99)
- Error rates (total_errors, error_rate)
- Token usage per model (input/output/total, request_count, avg_per_request)
- Endpoint performance (count, errors, avg_time_ms, error_rate)
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["Monitoring"])


# ===========================================
# Response Models
# ===========================================


class MetricsSummary(BaseModel):
    """Overall metrics summary"""

    total_requests: int = Field(default=0, description="Total requests in the last hour")
    total_errors: int = Field(default=0, description="Total errors in the last hour")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    avg_response_time_ms: float = Field(default=0.0, description="Average response time in ms")
    min_response_time_ms: float = Field(default=0.0, description="Minimum response time")
    max_response_time_ms: float = Field(default=0.0, description="Maximum response time")
    p50_response_time_ms: float = Field(default=0.0, description="50th percentile response time")
    p95_response_time_ms: float = Field(default=0.0, description="95th percentile response time")
    p99_response_time_ms: float = Field(default=0.0, description="99th percentile response time")
    last_updated: str = Field(default="", description="Last update timestamp")


class RequestTrend(BaseModel):
    """Request trend data point"""

    minute: int = Field(..., description="Unix timestamp (minute)")
    requests: int = Field(default=0, description="Request count")
    errors: int = Field(default=0, description="Error count")


class EndpointStats(BaseModel):
    """Per-endpoint statistics"""

    endpoint: str = Field(..., description="Endpoint path")
    method: str = Field(..., description="HTTP method")
    count: int = Field(default=0, description="Total request count")
    errors: int = Field(default=0, description="Error count")
    avg_time_ms: float = Field(default=0.0, description="Average response time")
    error_rate: float = Field(default=0.0, description="Error rate percentage")


class TokenUsage(BaseModel):
    """Token usage per model"""

    model: str = Field(..., description="Model name")
    input_tokens: int = Field(default=0, description="Total input tokens")
    output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    request_count: int = Field(default=0, description="Number of requests")
    avg_tokens_per_request: float = Field(default=0.0, description="Average tokens per request")


class SystemHealth(BaseModel):
    """System health status"""

    status: str = Field(default="healthy", description="Overall health status")
    redis_connected: bool = Field(default=False, description="Redis connection status")
    kafka_connected: bool = Field(default=False, description="Kafka connection status")
    uptime_seconds: float = Field(default=0.0, description="Server uptime in seconds")
    timestamp: str = Field(default="", description="Current timestamp")


class ChatHistoryItem(BaseModel):
    """챗 1회 단위 상세 이력 (CHAT_HISTORY_METRICS.md)"""

    request_id: str = Field(..., description="요청 식별자")
    model: str = Field(default="", description="사용 모델")
    at_ts: int = Field(default=0, description="완료 시각(Unix sec)")
    at_minute: int = Field(default=0, description="분 버킷")
    request_preview: str = Field(default="", description="요청 요약")
    response_preview: str = Field(default="", description="응답 요약")
    input_tokens: int = Field(default=0, description="입력 토큰")
    output_tokens: int = Field(default=0, description="출력 토큰")
    total_tokens: int = Field(default=0, description="총 토큰")
    duration_ms: float = Field(default=0.0, description="소요시간(ms)")


class MonitoringDashboard(BaseModel):
    """Complete monitoring dashboard data"""

    summary: MetricsSummary
    request_trend: List[RequestTrend]
    top_endpoints: List[EndpointStats]
    token_usage: List[TokenUsage]
    health: SystemHealth


# ===========================================
# Helper Functions
# ===========================================

# Server start time for uptime calculation
_server_start_time = time.time()


def _get_redis_client():
    """Get Redis client if available"""
    try:
        from beanllm.infrastructure.distributed.messaging import RequestMonitor

        monitor = RequestMonitor()
        if monitor and monitor.redis:
            return monitor.redis
    except Exception as e:
        logger.debug(f"Redis not available: {e}")
    return None


async def _get_metrics_summary(redis) -> MetricsSummary:
    """Collect metrics summary from Redis"""
    try:
        current_minute = int(time.time() // 60)
        total_requests = 0
        total_errors = 0

        # Get request/error counts for the last 60 minutes
        for i in range(60):
            minute_key = current_minute - i
            req_count = await redis.get(f"metrics:requests:{minute_key}")
            err_count = await redis.get(f"metrics:errors:{minute_key}")
            if req_count:
                total_requests += int(req_count)
            if err_count:
                total_errors += int(err_count)

        # Calculate error rate
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0

        # Get response time statistics
        response_times = await redis.zrange("metrics:response_time", 0, -1, withscores=True)

        if response_times:
            times = [score for _, score in response_times]
            times.sort()
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            # Calculate percentiles
            p50_idx = int(len(times) * 0.5)
            p95_idx = int(len(times) * 0.95)
            p99_idx = int(len(times) * 0.99)

            p50 = times[min(p50_idx, len(times) - 1)]
            p95 = times[min(p95_idx, len(times) - 1)]
            p99 = times[min(p99_idx, len(times) - 1)]
        else:
            avg_time = min_time = max_time = p50 = p95 = p99 = 0.0

        return MetricsSummary(
            total_requests=total_requests,
            total_errors=total_errors,
            error_rate=round(error_rate, 2),
            avg_response_time_ms=round(avg_time, 2),
            min_response_time_ms=round(min_time, 2),
            max_response_time_ms=round(max_time, 2),
            p50_response_time_ms=round(p50, 2),
            p95_response_time_ms=round(p95, 2),
            p99_response_time_ms=round(p99, 2),
            last_updated=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return MetricsSummary(last_updated=datetime.now().isoformat())


async def _get_request_trend(redis, minutes: int = 60) -> List[RequestTrend]:
    """Get request trend for the last N minutes"""
    try:
        current_minute = int(time.time() // 60)
        trend = []

        for i in range(minutes):
            minute_key = current_minute - (minutes - 1 - i)
            req_count = await redis.get(f"metrics:requests:{minute_key}")
            err_count = await redis.get(f"metrics:errors:{minute_key}")

            trend.append(
                RequestTrend(
                    minute=minute_key * 60,
                    requests=int(req_count) if req_count else 0,
                    errors=int(err_count) if err_count else 0,
                )
            )

        return trend
    except Exception as e:
        logger.error(f"Failed to get request trend: {e}")
        return []


def _int_val(data: dict, k: str, default: int = 0) -> int:
    """Read int from Redis hash; keys/values may be bytes (decode_responses=False)."""
    if not data:
        return default
    use_bytes = isinstance(next(iter(data), None), bytes)
    key = k.encode("utf-8") if (use_bytes and isinstance(k, str)) else k
    v = data.get(key, default)
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, bytes):
        return int(v.decode("utf-8", errors="replace"))
    return int(v)


def _scan_cursor_int(cursor: Union[int, bytes, str, None]) -> int:
    """Normalize SCAN cursor to int (Redis with decode_responses=False returns bytes)."""
    if cursor is None:
        return 0
    if isinstance(cursor, int):
        return cursor
    if isinstance(cursor, bytes):
        s = cursor.decode("utf-8", errors="replace").strip() or "0"
        return int(s)
    return int(cursor)


async def _get_endpoint_stats(redis, limit: int = 10) -> List[EndpointStats]:
    """Get top endpoints by request count"""
    try:
        endpoints = []
        cursor = 0

        while True:
            cursor, keys = await redis.scan(cursor, match="metrics:endpoint:*", count=100)
            cursor = _scan_cursor_int(cursor)

            for key in keys:
                data = await redis.hgetall(key)
                if data:
                    # Key may be bytes (decode_responses=False)
                    key_str = (
                        key.decode("utf-8", errors="replace") if isinstance(key, bytes) else key
                    )
                    parts = key_str.split(":", 3)
                    if len(parts) >= 4:
                        method = parts[2]
                        path = parts[3]

                        count = _int_val(data, "count", 0)
                        errors = _int_val(data, "errors", 0)
                        total_time = _int_val(data, "total_time_ms", 0)

                        avg_time = total_time / count if count > 0 else 0.0
                        error_rate = (errors / count * 100) if count > 0 else 0.0

                        endpoints.append(
                            EndpointStats(
                                endpoint=path,
                                method=method,
                                count=count,
                                errors=errors,
                                avg_time_ms=round(avg_time, 2),
                                error_rate=round(error_rate, 2),
                            )
                        )

            if cursor == 0:
                break

        # Sort by count and return top N
        endpoints.sort(key=lambda x: x.count, reverse=True)
        return endpoints[:limit]
    except Exception as e:
        logger.error(f"Failed to get endpoint stats: {e}")
        return []


async def _get_token_usage(redis) -> List[TokenUsage]:
    """Get token usage per model"""
    try:
        usage = []
        cursor = 0

        while True:
            cursor, keys = await redis.scan(cursor, match="metrics:tokens:*", count=100)
            cursor = _scan_cursor_int(cursor)

            for key in keys:
                data = await redis.hgetall(key)
                if data:
                    key_str = (
                        key.decode("utf-8", errors="replace") if isinstance(key, bytes) else key
                    )
                    parts = key_str.split(":", 2)
                    model = parts[2] if len(parts) > 2 else "unknown"

                    input_tokens = _int_val(data, "input_tokens", 0)
                    output_tokens = _int_val(data, "output_tokens", 0)
                    total_tokens = _int_val(data, "total_tokens", 0)
                    request_count = _int_val(data, "request_count", 0)

                    avg_tokens = total_tokens / request_count if request_count > 0 else 0.0

                    usage.append(
                        TokenUsage(
                            model=model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            request_count=request_count,
                            avg_tokens_per_request=round(avg_tokens, 2),
                        )
                    )

            if cursor == 0:
                break

        # Sort by total tokens
        usage.sort(key=lambda x: x.total_tokens, reverse=True)
        return usage
    except Exception as e:
        logger.error(f"Failed to get token usage: {e}")
        return []


def _get_system_health(redis_connected: bool) -> SystemHealth:
    """Get system health status"""
    # Check Kafka connection
    kafka_connected = False
    try:
        from beanllm.infrastructure.distributed import get_event_logger

        event_logger = get_event_logger()
        kafka_connected = event_logger is not None
    except Exception:
        pass

    uptime = time.time() - _server_start_time

    # Determine overall status
    if redis_connected and kafka_connected:
        status = "healthy"
    elif redis_connected or kafka_connected:
        status = "degraded"
    else:
        status = "unhealthy"

    return SystemHealth(
        status=status,
        redis_connected=redis_connected,
        kafka_connected=kafka_connected,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now().isoformat(),
    )


# ===========================================
# Endpoints
# ===========================================


@router.get("/health", response_model=SystemHealth)
async def get_health():
    """
    Get system health status

    Returns health of monitoring infrastructure including Redis and Kafka.
    """
    redis = _get_redis_client()
    redis_connected = redis is not None

    return _get_system_health(redis_connected)


@router.get("/summary", response_model=MetricsSummary)
async def get_summary():
    """
    Get metrics summary

    Returns aggregated metrics for the last hour including:
    - Total requests and errors
    - Error rate
    - Response time statistics (avg, min, max, percentiles)
    """
    redis = _get_redis_client()
    if not redis:
        return MetricsSummary(last_updated=datetime.now().isoformat())

    return await _get_metrics_summary(redis)


@router.get("/trend", response_model=List[RequestTrend])
async def get_trend(minutes: int = 60):
    """
    Get request trend

    Returns request and error counts per minute for the specified period.
    Default is last 60 minutes.
    """
    if minutes < 1 or minutes > 1440:  # Max 24 hours
        raise HTTPException(status_code=400, detail="Minutes must be between 1 and 1440")

    redis = _get_redis_client()
    if not redis:
        return []

    return await _get_request_trend(redis, minutes)


@router.get("/endpoints", response_model=List[EndpointStats])
async def get_endpoints(limit: int = 10):
    """
    Get endpoint statistics

    Returns top endpoints by request count with:
    - Request count
    - Error count and rate
    - Average response time
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

    redis = _get_redis_client()
    if not redis:
        return []

    return await _get_endpoint_stats(redis, limit)


@router.get("/tokens", response_model=List[TokenUsage])
async def get_token_usage():
    """
    Get token usage per model

    Returns token consumption statistics for each LLM model:
    - Input/output/total tokens
    - Request count
    - Average tokens per request
    """
    redis = _get_redis_client()
    if not redis:
        return []

    return await _get_token_usage(redis)


@router.get("/dashboard", response_model=MonitoringDashboard)
async def get_dashboard():
    """
    Get complete monitoring dashboard data (Chat API only).

    Request/error/response-time/endpoint metrics are aggregated from
    /api/chat and /api/chat/* only. Token usage is from chat completions.

    Returns all monitoring data in a single response:
    - Metrics summary
    - Request trend (last 60 minutes)
    - Top 10 endpoints
    - Token usage per model
    - System health
    """
    redis = _get_redis_client()
    redis_connected = redis is not None

    if redis:
        summary = await _get_metrics_summary(redis)
        trend = await _get_request_trend(redis, 60)
        endpoints = await _get_endpoint_stats(redis, 10)
        tokens = await _get_token_usage(redis)
    else:
        summary = MetricsSummary(last_updated=datetime.now().isoformat())
        trend = []
        endpoints = []
        tokens = []

    health = _get_system_health(redis_connected)

    return MonitoringDashboard(
        summary=summary,
        request_trend=trend,
        top_endpoints=endpoints,
        token_usage=tokens,
        health=health,
    )


@router.post("/clear")
async def clear_metrics():
    """
    Clear all metrics data.

    삭제 대상: 추적 메트릭 기준(CACHE_AND_METRICS_POLICY §2)에 정의된 키만
    — metrics:response_time, metrics:requests:*, metrics:errors:*,
      metrics:endpoint:*, metrics:tokens:*
    request:status:* 는 TTL 3600s로 자동 만료되며, 여기서 삭제하지 않음.

    WARNING: This will delete all collected metrics. Use with caution.
    """
    redis = _get_redis_client()
    if not redis:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Clear response times
        await redis.delete("metrics:response_time")

        # Clear request/error counts
        current_minute = int(time.time() // 60)
        for i in range(1440):  # Last 24 hours
            minute_key = current_minute - i
            await redis.delete(f"metrics:requests:{minute_key}")
            await redis.delete(f"metrics:errors:{minute_key}")

        # Clear endpoint stats
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="metrics:endpoint:*", count=100)
            cursor = _scan_cursor_int(cursor)
            for key in keys:
                await redis.delete(key)
            if cursor == 0:
                break

        # Clear token usage
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="metrics:tokens:*", count=100)
            cursor = _scan_cursor_int(cursor)
            for key in keys:
                await redis.delete(key)
            if cursor == 0:
                break

        logger.info("All metrics cleared")
        return {"status": "success", "message": "All metrics cleared"}
    except Exception as e:
        logger.error(f"Failed to clear metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _decode_hash(data: dict) -> Dict[str, str]:
    """Redis hash(bytes 키/값) → str 키/값 dict. decode_responses=False 대응."""
    if not data:
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        key = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
        val = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        out[key] = val
    return out


@router.get("/chat-history", response_model=List[ChatHistoryItem])
async def get_chat_history(minutes: int = 60, limit: int = 50):
    """
    챗 상세 이력 (CHAT_HISTORY_METRICS.md).

    어떤 챗·결과·모델·시간대·토큰 등 1회 단위 이력을 반환.
    상시 수집(env 없음). Redis에 chat:record/chat:history 가 있으면 반환.
    """
    if minutes < 1 or minutes > 10080:  # 최대 7일
        raise HTTPException(status_code=400, detail="Minutes must be between 1 and 10080")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 200")

    redis = _get_redis_client()
    if not redis:
        return []

    try:
        # ZREVRANGE returns members newest-first; values are request_id (bytes when decode_responses=False)
        now = time.time()
        cutoff = now - minutes * 60
        raw_ids = await redis.zrevrange("chat:history", 0, limit - 1)
        out: List[ChatHistoryItem] = []
        for rid in raw_ids:
            req_id = rid.decode("utf-8", errors="replace") if isinstance(rid, bytes) else str(rid)
            data = await redis.hgetall(f"chat:record:{req_id}")
            decoded = _decode_hash(data) if data else {}
            at_ts = int(decoded.get("at_ts") or 0)
            if at_ts < cutoff:
                continue
            out.append(
                ChatHistoryItem(
                    request_id=req_id,
                    model=decoded.get("model") or "",
                    at_ts=at_ts,
                    at_minute=int(decoded.get("at_minute") or 0),
                    request_preview=decoded.get("request_preview") or "",
                    response_preview=decoded.get("response_preview") or "",
                    input_tokens=int(decoded.get("input_tokens") or 0),
                    output_tokens=int(decoded.get("output_tokens") or 0),
                    total_tokens=int(decoded.get("total_tokens") or 0),
                    duration_ms=float(decoded.get("duration_ms") or 0),
                )
            )
        return out
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return []
