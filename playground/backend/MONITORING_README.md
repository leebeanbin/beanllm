# 분산 모니터링 시스템

Kafka + Redis + Streamlit을 활용한 실시간 모니터링 시스템

## 개요

채팅 과정에서 상세한 로그가 백엔드에 표시되지 않는 문제를 해결하기 위해, 분산 모니터링 시스템을 구축했습니다.

## 아키텍처

```
클라이언트 요청
    ↓
FastAPI 백엔드 (MonitoringMiddleware)
    ↓
┌─────────────────┬─────────────────┐
│   Kafka         │   Redis          │
│   (이벤트 스트림) │   (실시간 메트릭) │
└─────────────────┴─────────────────┘
    ↓
Streamlit 대시보드 (시각화)
```

## 구성 요소

### 1. MonitoringMiddleware (`monitoring.py`)

FastAPI 미들웨어로 모든 요청을 가로채서 상세 로깅 및 모니터링을 수행합니다.

**기능:**
- Request ID 생성 및 전파
- 요청/응답 로깅 (메서드, 경로, 쿼리 파라미터, 본문 등)
- 성능 메트릭 수집 (응답 시간, 상태 코드 등)
- Kafka를 통한 이벤트 스트리밍
- Redis를 통한 실시간 메트릭 저장

**수집하는 메트릭:**
- 응답 시간 (response_time)
- 요청 수 (requests per minute)
- 에러 수 (errors per minute)
- 엔드포인트별 통계 (count, total_time, errors)

### 2. ChatMonitoringMixin (`monitoring.py`)

Chat API 전용 모니터링 믹스인으로 LLM 호출에 대한 상세 로깅을 제공합니다.

**기능:**
- Chat 요청 로깅 (모델, 메시지 수, 파라미터 등)
- Chat 응답 로깅 (응답 길이, 토큰 사용량, 응답 시간 등)
- 모델별 토큰 사용량 추적

**수집하는 메트릭:**
- 입력/출력 토큰 수
- 총 토큰 수
- 요청 수 (모델별)

### 3. Streamlit 대시보드 (`monitoring_dashboard.py`)

실시간 모니터링 대시보드로 Redis에서 메트릭을 읽어 시각화합니다.

**기능:**
- 실시간 메트릭 표시 (요청 수, 에러 수, 응답 시간, 에러율)
- 요청 수 추이 차트
- 응답 시간 분포 히스토그램
- 엔드포인트별 통계 테이블
- 토큰 사용량 (모델별) 차트 및 테이블
- 최근 요청 목록
- 자동 새로고침

## 설정

### 환경 변수

```bash
# 분산 모드 활성화
USE_DISTRIBUTED=true

# Redis 설정
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=None  # 필요시 설정

# Kafka 설정
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=llmkit
```

### 백엔드 설정

`main.py`에 모니터링 미들웨어가 자동으로 추가됩니다:

```python
# 모니터링 미들웨어 추가 (상세 로깅 및 이벤트 스트리밍)
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"
app.add_middleware(
    MonitoringMiddleware,
    enable_kafka=USE_DISTRIBUTED,
    enable_redis=USE_DISTRIBUTED,
)
```

### Chat 엔드포인트 모니터링

Chat 엔드포인트에 `ChatMonitoringMixin`을 사용하여 상세 로깅이 자동으로 수행됩니다.

## 사용 방법

### 1. 백엔드 실행

```bash
cd playground/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 모니터링 대시보드 실행

```bash
cd playground/backend
streamlit run monitoring_dashboard.py
```

대시보드는 `http://localhost:8501`에서 접근할 수 있습니다.

### 3. 요청 시 Request ID 확인

모든 응답 헤더에 `X-Request-ID`가 포함됩니다:

```bash
curl -v http://localhost:8000/api/chat
# 응답 헤더에 X-Request-ID: <uuid> 포함
```

## 수집되는 데이터

### Redis에 저장되는 메트릭

1. **응답 시간**: `metrics:response_time` (Sorted Set)
2. **요청 수**: `metrics:requests:<minute>` (String, TTL: 1시간)
3. **에러 수**: `metrics:errors:<minute>` (String, TTL: 1시간)
4. **엔드포인트 통계**: `metrics:endpoint:<method>:<path>` (Hash, TTL: 1시간)
5. **토큰 사용량**: `metrics:tokens:<model>` (Hash, TTL: 24시간)
6. **요청 상태**: `request:status:<request_id>` (String, TTL: 1시간)

### Kafka에 발행되는 이벤트

1. **API 요청 시작**: `api.request.started`
2. **API 요청 완료**: `api.request.completed`
3. **Chat 요청**: `chat.request`
4. **Chat 응답**: `chat.response`

## 로그 포맷

구조화된 로깅이 사용되며, 모든 로그에 Request ID가 포함됩니다:

```
2024-01-20 15:34:12 - __main__ - INFO - [abc123...] - [REQUEST] POST /api/chat
2024-01-20 15:34:13 - __main__ - INFO - [abc123...] - [RESPONSE] POST /api/chat - 200 (1234.56ms)
```

## 추가 기능

### Prometheus 메트릭 (선택적)

Prometheus를 사용하려면 `prometheus_client`를 설치하고 메트릭을 노출하는 엔드포인트를 추가할 수 있습니다:

```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana 대시보드 (선택적)

Grafana를 사용하여 더 고급 시각화를 구축할 수 있습니다:

1. Prometheus를 데이터 소스로 추가
2. Redis를 데이터 소스로 추가 (Redis Data Source 플러그인 사용)
3. Kafka를 데이터 소스로 추가 (Kafka Data Source 플러그인 사용)

## 문제 해결

### Redis 연결 실패

```
Redis 연결 실패: Connection refused
```

**해결 방법:**
1. Redis 서버가 실행 중인지 확인: `redis-cli ping`
2. 환경 변수 `REDIS_HOST`, `REDIS_PORT` 확인
3. 방화벽 설정 확인

### Kafka 연결 실패

```
Kafka not connected, event publish skipped
```

**해결 방법:**
1. Kafka 서버가 실행 중인지 확인
2. 환경 변수 `KAFKA_BOOTSTRAP_SERVERS` 확인
3. `USE_DISTRIBUTED=false`로 설정하여 Kafka 없이 실행 가능 (Redis만 사용)

### 대시보드에 데이터가 표시되지 않음

**해결 방법:**
1. 백엔드가 요청을 처리하고 있는지 확인
2. Redis에 데이터가 저장되는지 확인: `redis-cli KEYS "metrics:*"`
3. 시간 범위를 늘려서 확인

## 향후 개선 사항

- [ ] OpenTelemetry 통합
- [ ] 분산 트레이싱 (Jaeger/Zipkin)
- [ ] 알림 시스템 (Slack, Email)
- [ ] 로그 집계 (ELK Stack)
- [ ] 성능 프로파일링
- [ ] 비용 추적 (모델별 API 비용)

## 참고 자료

- [Kafka 공식 문서](https://kafka.apache.org/documentation/)
- [Redis 공식 문서](https://redis.io/docs/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
