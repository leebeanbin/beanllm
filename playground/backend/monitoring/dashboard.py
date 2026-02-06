"""
Streamlit 기반 실시간 모니터링 대시보드

Kafka + Redis를 활용한 분산 모니터링 시스템의 시각화
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ plotly가 설치되지 않았습니다. 차트 기능이 제한됩니다. 설치: `pip install plotly`")

# Redis 모듈 초기화
redis_sync_module = None
redis_async_module = None
REDIS_AVAILABLE = False
REDIS_ASYNC = False

try:
    import redis as redis_sync_module
    import redis.asyncio as redis_async_module

    REDIS_AVAILABLE = True
    REDIS_ASYNC = True
except ImportError:
    try:
        import redis as redis_sync_module

        REDIS_AVAILABLE = True
        REDIS_ASYNC = False
    except ImportError:
        REDIS_AVAILABLE = False
        REDIS_ASYNC = False
        redis_sync_module = None
        redis_async_module = None

try:
    from kafka import KafkaConsumer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Redis 연결
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Kafka 설정
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_PREFIX = os.getenv("KAFKA_TOPIC_PREFIX", "llmkit")


@st.cache_resource
def get_redis_client():
    """Redis 클라이언트 생성"""
    if not REDIS_AVAILABLE or redis_sync_module is None:
        return None

    try:
        # 동기 버전 사용 (Streamlit은 동기 환경)
        client = redis_sync_module.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        # 연결 테스트
        client.ping()
        return client
    except redis_sync_module.ConnectionError as e:
        st.error(f"❌ Redis 연결 실패: {e}")
        st.info(f"Redis 서버가 {REDIS_HOST}:{REDIS_PORT}에서 실행 중인지 확인하세요.")
        return None
    except Exception as e:
        st.error(f"❌ Redis 오류: {e}")
        return None


def get_metrics_from_redis(redis_client, time_window_minutes: int = 60) -> Dict:
    """Redis에서 메트릭 데이터 가져오기"""
    if not redis_client:
        return {}

    try:
        current_time = int(time.time())
        start_time = current_time - (time_window_minutes * 60)

        metrics = {}

        # 1. 응답 시간 메트릭 (Sorted Set에서 모든 값 가져오기)
        try:
            # zrangebyscore는 score 범위로 필터링하는데, 우리는 request_id를 key로 사용하고 있음
            # 모든 항목을 가져온 후 최근 것만 필터링
            response_times_raw = redis_client.zrange(
                "metrics:response_time",
                0,
                -1,
                withscores=True,  # 모든 항목
            )
            if response_times_raw:
                # 최근 time_window_minutes 내의 데이터만 필터링
                metrics["response_times"] = [
                    float(score)
                    for _, score in response_times_raw
                    if float(score) > 0  # 응답 시간은 항상 양수
                ]
        except Exception:
            # logger가 없으면 pass
            pass

        # 2. 요청 수 (시간대별)
        request_counts = {}
        for minute in range(time_window_minutes):
            minute_key = f"metrics:requests:{int((current_time - minute * 60) // 60)}"
            count = redis_client.get(minute_key)
            if count:
                request_counts[minute] = int(count)
        metrics["request_counts"] = request_counts

        # 3. 에러 수 (시간대별)
        error_counts = {}
        for minute in range(time_window_minutes):
            minute_key = f"metrics:errors:{int((current_time - minute * 60) // 60)}"
            count = redis_client.get(minute_key)
            if count:
                error_counts[minute] = int(count)
        metrics["error_counts"] = error_counts

        # 4. 엔드포인트별 통계
        endpoint_stats = {}
        try:
            keys = redis_client.keys("metrics:endpoint:*")
            for key in keys:
                endpoint = key.replace("metrics:endpoint:", "")
                stats = redis_client.hgetall(key)
                if stats:
                    endpoint_stats[endpoint] = {
                        "count": int(stats.get("count", 0)),
                        "total_time_ms": int(stats.get("total_time_ms", 0)),
                        "errors": int(stats.get("errors", 0)),
                    }
        except:
            pass
        metrics["endpoint_stats"] = endpoint_stats

        # 5. 토큰 사용량 (모델별)
        token_stats = {}
        try:
            keys = redis_client.keys("metrics:tokens:*")
            for key in keys:
                model = key.replace("metrics:tokens:", "")
                stats = redis_client.hgetall(key)
                if stats:
                    token_stats[model] = {
                        "input_tokens": int(stats.get("input_tokens", 0)),
                        "output_tokens": int(stats.get("output_tokens", 0)),
                        "total_tokens": int(stats.get("total_tokens", 0)),
                        "request_count": int(stats.get("request_count", 0)),
                    }
        except:
            pass
        metrics["token_stats"] = token_stats

        return metrics
    except Exception as e:
        st.error(f"메트릭 가져오기 실패: {e}")
        return {}


def get_recent_requests(redis_client, limit: int = 50) -> List[Dict]:
    """최근 요청 목록 가져오기"""
    if not redis_client:
        return []

    try:
        keys = redis_client.keys("request:status:*")
        requests = []

        for key in keys[-limit:]:  # 최근 N개만
            try:
                data = redis_client.get(key)
                if data:
                    request_data = json.loads(data)
                    requests.append(request_data)
            except:
                pass

        # 시간순 정렬 (최신순)
        requests.sort(key=lambda x: x.get("started_at", 0), reverse=True)
        return requests[:limit]
    except Exception as e:
        st.error(f"요청 목록 가져오기 실패: {e}")
        return []


def main():
    """메인 대시보드"""
    st.set_page_config(
        page_title="beanllm 모니터링 대시보드",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 beanllm 실시간 모니터링 대시보드")
    st.markdown("Kafka + Redis 기반 분산 모니터링 시스템")

    # Redis 연결
    if not REDIS_AVAILABLE:
        st.error("⚠️ redis 패키지가 설치되지 않았습니다.")
        st.info("설치 방법:")
        st.code("pip install redis")
        st.info("또는 requirements.txt에서 설치:")
        st.code("pip install -r requirements.txt")
        return

    redis_client = get_redis_client()

    if not redis_client:
        st.error("Redis에 연결할 수 없습니다. Redis 서버가 실행 중인지 확인하세요.")
        st.info("환경 변수 설정:")
        st.code(
            """
        REDIS_HOST=localhost
        REDIS_PORT=6379
        REDIS_DB=0
        REDIS_PASSWORD=None
        """
        )
        st.info("Redis 서버 실행:")
        st.code("redis-server")
        return

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        time_window = st.slider("시간 범위 (분)", 5, 120, 60)
        auto_refresh = st.checkbox("자동 새로고침", value=True)
        refresh_interval = st.slider("새로고침 간격 (초)", 1, 60, 5)

    # 자동 새로고침
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

    # 메트릭 가져오기
    metrics = get_metrics_from_redis(redis_client, time_window)

    # 디버깅 정보 (접을 수 있게)
    with st.expander("🔍 디버깅 정보", expanded=True):
        if redis_client:
            try:
                # Redis 연결 테스트
                redis_client.ping()
                redis_status = "✅ 연결됨"
            except Exception as e:
                redis_status = f"❌ 연결 실패: {e}"

            try:
                all_keys = redis_client.keys("*")
                metrics_keys = redis_client.keys("metrics:*")
                request_keys = redis_client.keys("request:status:*")
            except Exception as e:
                all_keys = []
                metrics_keys = []
                request_keys = []
                st.error(f"Redis 키 조회 실패: {e}")
        else:
            redis_status = "❌ Redis 클라이언트 없음"
            all_keys = []
            metrics_keys = []
            request_keys = []

        st.json(
            {
                "Redis 연결": redis_status,
                "전체 키 개수": len(all_keys),
                "메트릭 키 개수": len(metrics_keys),
                "요청 상태 키 개수": len(request_keys),
                "수집된 메트릭": {
                    "request_counts": len(metrics.get("request_counts", {})),
                    "error_counts": len(metrics.get("error_counts", {})),
                    "response_times": len(metrics.get("response_times", [])),
                    "endpoint_stats": len(metrics.get("endpoint_stats", {})),
                    "token_stats": len(metrics.get("token_stats", {})),
                },
                "Redis 키 샘플 (metrics)": [
                    k.decode() if isinstance(k, bytes) else k for k in metrics_keys[:10]
                ],
                "Redis 키 샘플 (request)": [
                    k.decode() if isinstance(k, bytes) else k for k in request_keys[:10]
                ],
            }
        )

        if redis_client and len(all_keys) == 0:
            st.warning("⚠️ Redis에 데이터가 없습니다.")
            st.info("**해결 방법:**")
            st.markdown("""
            1. **백엔드가 실행 중인지 확인**: `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
            2. **백엔드 로그 확인**: 다음 메시지가 보여야 합니다:
               - `✅ Redis monitoring initialized successfully` (정상)
               - `❌ RequestMonitor.redis is None` (Redis 연결 실패)
            3. **프론트엔드에서 채팅 요청 보내기**: 실제 요청이 있어야 데이터가 저장됩니다
            4. **환경 변수 확인**: `.env` 파일에 `USE_REDIS_MONITORING=true`가 설정되어 있는지 확인
            """)
        elif redis_client and len(metrics_keys) == 0 and len(request_keys) == 0:
            st.warning("⚠️ Redis에는 데이터가 있지만 메트릭 키가 없습니다.")
            st.info("백엔드가 요청을 받았지만 메트릭을 저장하지 못했을 수 있습니다.")
            st.info(
                "백엔드 로그에서 `Failed to save request status to Redis` 또는 `Failed to update metrics in Redis` 메시지를 확인하세요."
            )

    # 대시보드 레이아웃
    col1, col2, col3, col4 = st.columns(4)

    # 1. 총 요청 수
    total_requests = sum(metrics.get("request_counts", {}).values())
    col1.metric("총 요청 수", f"{total_requests:,}")

    # 2. 총 에러 수
    total_errors = sum(metrics.get("error_counts", {}).values())
    col2.metric("총 에러 수", f"{total_errors:,}")

    # 3. 평균 응답 시간
    response_times = metrics.get("response_times", [])
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    col3.metric("평균 응답 시간", f"{avg_response_time:.2f}ms")

    # 4. 에러율
    error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
    col4.metric("에러율", f"{error_rate:.2f}%")

    # 차트 섹션
    st.header("📈 실시간 차트")

    col1, col2 = st.columns(2)

    # 1. 요청 수 추이
    with col1:
        st.subheader("요청 수 추이")
        request_counts = metrics.get("request_counts", {})
        if request_counts:
            df_requests = pd.DataFrame(
                [
                    {"시간": f"-{minute}분 전", "요청 수": count}
                    for minute, count in sorted(request_counts.items())
                ]
            )
            if PLOTLY_AVAILABLE:
                fig = px.line(df_requests, x="시간", y="요청 수", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df_requests.set_index("시간"))
        else:
            st.info("데이터가 없습니다.")

    # 2. 응답 시간 분포
    with col2:
        st.subheader("응답 시간 분포")
        response_times = metrics.get("response_times", [])
        if response_times:
            df_response = pd.DataFrame({"응답 시간 (ms)": response_times})
            if PLOTLY_AVAILABLE:
                fig = px.histogram(df_response, x="응답 시간 (ms)", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df_response)
        else:
            st.info("데이터가 없습니다.")

    # 엔드포인트별 통계
    st.header("🔍 엔드포인트별 통계")
    endpoint_stats = metrics.get("endpoint_stats", {})
    if endpoint_stats:
        df_endpoints = pd.DataFrame(
            [
                {
                    "엔드포인트": endpoint,
                    "요청 수": stats["count"],
                    "평균 응답 시간 (ms)": (
                        stats["total_time_ms"] / stats["count"] if stats["count"] > 0 else 0
                    ),
                    "에러 수": stats["errors"],
                    "에러율 (%)": (
                        (stats["errors"] / stats["count"] * 100) if stats["count"] > 0 else 0
                    ),
                }
                for endpoint, stats in endpoint_stats.items()
            ]
        )
        st.dataframe(df_endpoints, use_container_width=True)
    else:
        st.info("엔드포인트 통계 데이터가 없습니다.")

    # 토큰 사용량 통계
    st.header("💬 토큰 사용량 (모델별)")
    token_stats = metrics.get("token_stats", {})
    if token_stats:
        col1, col2 = st.columns(2)

        with col1:
            df_tokens = pd.DataFrame(
                [
                    {
                        "모델": model,
                        "입력 토큰": stats["input_tokens"],
                        "출력 토큰": stats["output_tokens"],
                        "총 토큰": stats["total_tokens"],
                        "요청 수": stats["request_count"],
                    }
                    for model, stats in token_stats.items()
                ]
            )
            st.dataframe(df_tokens, use_container_width=True)

        with col2:
            if token_stats:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    models = list(token_stats.keys())
                    input_tokens = [stats["input_tokens"] for stats in token_stats.values()]
                    output_tokens = [stats["output_tokens"] for stats in token_stats.values()]

                    fig.add_trace(go.Bar(name="입력 토큰", x=models, y=input_tokens))
                    fig.add_trace(go.Bar(name="출력 토큰", x=models, y=output_tokens))
                    fig.update_layout(barmode="stack", title="토큰 사용량 (모델별)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    df_chart = pd.DataFrame(
                        [
                            {
                                "모델": model,
                                "입력 토큰": stats["input_tokens"],
                                "출력 토큰": stats["output_tokens"],
                            }
                            for model, stats in token_stats.items()
                        ]
                    )
                    st.bar_chart(df_chart.set_index("모델"))
    else:
        st.info("토큰 사용량 데이터가 없습니다.")

    # 최근 요청 목록
    st.header("📋 최근 요청 목록")
    recent_requests = get_recent_requests(redis_client, limit=20)
    if recent_requests:
        df_requests = pd.DataFrame(
            [
                {
                    "Request ID": req.get("request_id", "N/A")[:8] + "...",
                    "메서드": req.get("method", "N/A"),
                    "경로": req.get("path", "N/A"),
                    "상태": req.get("status", "N/A"),
                    "응답 시간 (ms)": f"{req.get('duration_ms', 0):.2f}",
                    "시작 시간": (
                        datetime.fromtimestamp(req.get("started_at", 0)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if req.get("started_at")
                        else "N/A"
                    ),
                }
                for req in recent_requests
            ]
        )
        st.dataframe(df_requests, use_container_width=True)
    else:
        st.info("최근 요청이 없습니다.")

    # 수동 새로고침 버튼
    if st.button("🔄 새로고침"):
        st.rerun()


if __name__ == "__main__":
    main()
