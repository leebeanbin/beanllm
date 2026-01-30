# 메시징 처리 구현 계획

§0 “메시징 처리 적극 활용” 및 CURRENT_STACK_AND_SCOPE.md §6·§8을 구현할 때, **이미 있는 beanllm·playground 코드를 최대한 재사용**한다.

- **캐시·메트릭 정책**: [`CACHE_AND_METRICS_POLICY.md`](./CACHE_AND_METRICS_POLICY.md) — 캐시 만료·즉시 비우기, **추적 메트릭 기준**(측정할 필요가 있는 것만 수집) 준수.

---

## 1. 재사용할 기존 구현

### 1.1 beanllm (`src/beanllm/infrastructure/distributed`)

| 구성요소 | 경로/용도 | playground·MCP 활용 |
|----------|------------|---------------------|
| **get_redis_client** | `redis/client.py` — REDIS_HOST/PORT/DB/PASSWORD 등 환경변수, 싱글톤, decode_responses=False | MonitoringMiddleware(RequestMonitor), session_cache, MCP 세션 캐시(동일 키 `session:{id}`) |
| **RequestMonitor** | `messaging.py` — request:status, Redis setex | MonitoringMiddleware가 이미 사용 중 |
| **MessageProducer** | `messaging.py` — publish_request → Kafka "llm.requests" + Redis request:status + **Redis queue:{request_type}** (lpush) | 실행 큐 파이프라인 단계에서 요청 적재용 |
| **TaskProcessor** | `task_processor.py` — enqueue_task / process_task, get_task_queue(topic) | 실행 큐 파이프라인: chat/agentic 요청 enqueue, 워커 dequeue 후 client.chat() 호출 |
| **get_task_queue** | `factory.py` — USE_DISTRIBUTED ? KafkaTaskQueue : InMemoryTaskQueue | TaskProcessor 내부 사용 |
| **ConcurrencyController** | `messaging.py` — acquire_slot, Redis concurrency:{resource_type} | 대용량 시 동시 실행 제한 |
| **get_rate_limiter / get_cache / get_distributed_lock** | `factory.py` | pipeline_decorators, 서비스 레이어에서 이미 사용 |
| **with_distributed_features** | `pipeline_decorators.py` — cache, rate_limit, event, lock | RAG/KG/Vision 등 서비스에서 이미 사용 |
| **get_event_logger** | `event_integration.py` — Kafka/InMemory | ChatMonitoringMixin, 미들웨어 이벤트 |

### 1.2 playground

| 구성요소 | 경로/용도 | 확장 시 재사용 |
|----------|------------|----------------|
| **MonitoringMiddleware** | `monitoring/middleware.py` — RequestMonitor(beanllm), request:status, metrics:* (채팅 경로만) | 조건만 확장 → 전체 API 메트릭 |
| **ChatMonitoringMixin** | `monitoring/middleware.py` — log_chat_request/response, Kafka·Redis 토큰 | 그대로 유지 |
| **session_cache** | `services/session_cache.py` — get_redis_client(beanllm), session:{id}, sessions:list:* | MCP가 동일 Redis·동일 키 사용 시 캐시 일원화 |
| **monitoring_router** | `routers/monitoring_router.py` — Redis metrics:* 조회, bytes/str 처리 | 전체 API 메트릭 시 기존 키로 이미 “전체” 집계됨 |

### 1.3 Redis 키·역할 (이미 사용 중)

- `request:status:{id}` — 요청 상태 (모든 HTTP)
- `metrics:response_time` — Sorted Set (채팅만 기록 중)
- `metrics:requests:{minute}`, `metrics:errors:{minute}` — 채팅만
- `metrics:endpoint:{method}:{path}` — 채팅만
- `metrics:tokens:{model}` — log_chat_response에서 기록
- `session:{session_id}`, `sessions:list:{...}` — session_cache

beanllm MessageProducer 추가 키: `queue:{request_type}` (Redis list, lpush).

---

## 2. 구현 단계

### Phase 1: 전체 API 메트릭 확장 (우선 구현)

**목표**: 채팅뿐 아니라 **모든 /api/* 요청**에 대해 Redis 메트릭(requests, errors, response_time, endpoint) 기록.

**재사용**:
- `MonitoringMiddleware` 로직 그대로 두고, “메트릭 기록 대상” 조건만 확장.
- 기존 키(`metrics:requests:{minute}` 등)를 그대로 사용하면, 대시보드는 별도 수정 없이 “전체 API” 기준 집계를 보게 됨.

**변경**:
1. 환경 변수 `RECORD_METRICS_FOR_ALL_API` (기본 `false`).
2. `record_metrics = _is_chat_api_path(path) or (RECORD_METRICS_FOR_ALL_API and path.startswith("/api/"))`
   - `true`이면 `/api/*` 전부 메트릭 기록.
3. 필요 시 대시보드/모니터링 API에 “채팅만 / 전체” 필터용 쿼리 파라미 추가(선택).

**위치**: `playground/backend/monitoring/middleware.py`.

---

### Phase 2: MCP Redis 세션 캐시 경유

**목표**: MCP `get_session_messages()` 호출 시 **백엔드와 동일한 Redis `session:{id}`**를 1차로 조회하고, hit이면 MongoDB 조회 생략.

**재사용**:
- 백엔드 `session_cache`가 이미 `session:{id}`에 세션 문서(JSON) 저장.
- MCP는 같은 REDIS_HOST/PORT 등으로 접속해 동일 키를 읽기만 하면 됨.

**선택**:
- **A) beanllm 경유**: MCP 실행 환경에 beanllm이 있으면 `from beanllm.infrastructure.distributed.redis.client import get_redis_client` 사용.
- **B) 직접 Redis**: `redis.asyncio.Redis(host=os.getenv("REDIS_HOST","localhost"), ...)` 로 연결, `key="session:"+session_id`, 값을 JSON 파싱 후 `doc.get("messages", [])` 반환.

**동작**:
1. `get_session_messages(session_id)` 시작 시 `Redis.get("session:"+session_id)` 시도.
2. 있으면: JSON 파싱 → `payload.get("messages", [])` 반환 (필요 시 timestamp 등 직렬화).
3. 없으면: 기존대로 MongoDB `chat_sessions.find_one({"session_id": session_id})` → `session.get("messages", [])`.  
   (선택) 이때 Redis에 세션 문서를 set해 두면 다음 요청부터 캐시 hit. 단, TTL·무효화 정책은 백엔드와 맞출 것.

**추가**: MongoDB 클라이언트는 “호출마다 새로 생성” 대신 **모듈 레벨 싱글톤** 사용 권장(BACKEND_MCP_DISTRIBUTED_REVIEW.md 반영).

**위치**: `mcp_server/services/session_manager.py`.

---

### Phase 3: 실행을 메시지 큐 파이프라인으로 올리기 (선택, 별도 설계)

**목표**: LLM/채팅 요청을 즉시 실행하지 않고, **Redis/Kafka 큐에 메시지로 넣고**, 워커가 dequeue 후 `client.chat()` 등 실행.  
→ 대용량·트래픽·스로틀·우선순위 제어를 메시징 중심으로 통일.

**재사용**:
- beanllm **MessageProducer.publish_request** → 이미 Redis `queue:{request_type}` + request:status + Kafka.
- **TaskProcessor** + **get_task_queue** → enqueue/dequeue·handler 실행·이벤트 발행 이미 구현됨.
- 실행 “핸들러”만 playground 쪽 `client.chat()`/agentic 흐름을 인자로 넘기면 됨.

**설계 포인트**:
- “즉시 응답” vs “큐 적재 후 task_id 반환, 클라이언트는 폴링/SSE로 결과 수신” 여부.
- `request_type` 예: `chat.request`, `chat.agentic.request`.
- 워커 프로세스: 기존 FastAPI 프로세스 내 백그라운드 태스크로 dequeue 루프를 둘지, 별도 워커 프로세스로 둘지.

Phase 3은 CURRENT_STACK_AND_SCOPE §8 “메시지 처리 파이프라인으로 실행 올리기”와 함께 별도 상세 설계 후 진행.

---

## 3. 구현 순서 요약

1. **Phase 1**  
   - `RECORD_METRICS_FOR_ALL_API` 도입 및 `MonitoringMiddleware`에서 `/api/*` 전부 메트릭 기록하도록 조건 확장.
2. **Phase 2**  
   - MCP `get_session_messages()`에 Redis `session:{id}` 조회 추가(및 선택적으로 캐시 적재).  
   - MongoDB 클라이언트 싱글톤화.
3. **Phase 3**  
   - 실행 큐 파이프라인은 상세 설계 후, MessageProducer/TaskProcessor/get_task_queue를 활용해 구현.

이 계획은 **기존 beanllm·playground 코드를 최대한 활용**하고, §0 “메시징 처리 적극 활용”과 §6(채팅 외·MCP 분산) 방향에 맞춘다.
