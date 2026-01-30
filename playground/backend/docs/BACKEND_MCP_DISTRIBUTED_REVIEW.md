# 백엔드·MCP 분산 시스템·DB 설계 점검

`SCHEMA_AND_FLOWS.md` 기준으로 백엔드·MCP가 분산 시스템(Redis) 및 DB(MongoDB) 설계 하에 잘 동작하는지 점검한 결과입니다.

- **현재 스택·구현 범위**: [`CURRENT_STACK_AND_SCOPE.md`](./CURRENT_STACK_AND_SCOPE.md)에서 구성요소·버전·의존 관계, **DB별 저장 정보**, **API별 제공 정보**, “이미 되어 있는 범위”, **채팅 외·MCP 서버 분산 처리 계획**(§6)을 정리함. **메시징 처리 적극 활용 정책**(§0: Redis·메시지 큐로 대용량 처리·트래픽 관리·AI 활용 용이)과 “실행을 메시지 처리 파이프라인으로 올리는” 설계 및 “채팅 외·MCP 분산” 설계 전 참고.

---

## 1. 수정 사항 (이번 적용)

### 1.1 모니터링 Router – Redis bytes/str 오류

- **증상**: `Failed to get endpoint stats: a bytes-like object is required, not 'str'`
- **원인**: beanllm Redis 클라이언트는 `decode_responses=False`라 `scan`/`hgetall` 결과가 **bytes**인데, 라우터는 str 기준으로 처리함.
- **조치**: `monitoring_router.py`에서
  - `_int_val(data, k, default)` 추가: Redis hash의 키/값이 bytes여도 안전히 int 추출
  - `_get_endpoint_stats` / `_get_token_usage`에서 키는 `key.decode("utf-8")` 등으로 str 변환 후 파싱, hash 필드는 `_int_val`로 조회

### 1.2 모니터링 집계 범위 (이전 반영 사항)

- 요청/에러/응답시간/엔드포인트 메트릭은 **채팅 API**(`/api/chat`, `/api/chat/*`)만 집계하도록 `MonitoringMiddleware`에서 경로 필터 적용됨.

### 1.3 경고 완화

- **startup MongoDB**: `MONGODB_URI` 미설정 시 `logger.warning` 대신 `logger.info`로 "MongoDB not configured" 로그만 남기도록 변경. URI는 있는데 ping 실패한 경우만 warning 유지.
- **create_session_indexes**: MongoDB 없을 때 `logger.warning` → `logger.debug`로 변경(이미 startup에서 MongoDB 상태 로그 완료).

---

## 2. 백엔드 ↔ 분산·DB 설계 정합성

| 항목 | 설계(SCHEMA_AND_FLOWS) | 백엔드 구현 | 일치 여부 |
|------|-------------------------|-------------|-----------|
| MongoDB | `get_mongodb_database()` → `chat_sessions` 등, MONGODB_URI/DATABASE | `database.get_mongodb_database()`, history_router·config·google_auth·orchestrator에서 사용 | ✅ |
| Redis 세션 캐시 | `session:{id}`, `sessions:list:{...}`, TTL 60/300 | `session_cache`(SessionCacheService)가 `get_redis_client()` 사용, history_router에서 get/set/invalidate | ✅ |
| Redis 메트릭 | `metrics:requests|errors|response_time|endpoint:*|tokens:*` | MonitoringMiddleware(채팅 경로만) · ChatMonitoringMixin 토큰 기록, monitoring_router가 조회 | ✅ |
| Redis 클라이언트 통일 | 세션 캐시·모니터링 동일 인스턴스 | `beanllm.infrastructure.distributed.redis.client.get_redis_client()` 한 곳에서 사용 | ✅ |
| 세션 쓰기 플로우 | MongoDB insert/update + Redis 캐시 갱신 + 목록 캐시 무효화 | history_router에서 db + `session_cache.set_session` + `invalidate_session_lists` | ✅ |
| 세션 읽기 플로우 | Redis → miss 시 MongoDB → 캐시 적재 | history_router에서 `session_cache.get_session` → miss 시 db 조회 후 `set_session` | ✅ |
| 세션 검색 | Vector 우선, Fallback 시 `$regex`는 `re.escape(query)` 적용 | session_search_service에서 이스케이프 후 사용 | ✅ |
| Human-in-the-loop | run_id·Redis `run:approval:{id}`·재개 시 복원 | chat_router·orchestrator에서 Redis get/set 및 `start_from_tool_index` 재개 | ✅ |

**요약**: 백엔드는 MongoDB·Redis 역할 분리, 키 패턴, 세션/목록 캐시 무효화, 검색 시 `$regex` 이스케이프까지 설계와 일치하게 동작한다.

---

## 3. MCP ↔ DB·분산 설계 정합성

| 항목 | 설계/기대 | MCP 구현 | 일치 여부 / 권장 |
|------|-----------|----------|-------------------|
| DB 저장소 | 세션·메시지 메타는 MongoDB `chat_sessions` | `session_manager.get_session_messages()`에서 `MONGODB_URI`·`MONGODB_DATABASE` 사용, `db.chat_sessions.find_one({"session_id": ...})` | ✅ 같은 DB 설계 |
| Redis 사용 | 세션 캐시는 백엔드 전용(Redis → Mongo) | MCP는 Redis 미사용. 도구 호출 시 필요한 만큼 MongoDB에서 직접 조회 | ✅ 의도적 설계로 보는 것이 타당(캐시는 백엔드 담당) |
| MongoDB 접근 방식 | 백엔드는 싱글톤 `get_mongodb_database()` | MCP는 `get_session_messages()` 호출 시마다 `AsyncIOMotorClient(mongodb_uri)` 새로 생성 | ⚠️ 연결 재사용 권장 |

**MCP 권장 사항**

- `get_session_messages()` 내부에서 매번 `AsyncIOMotorClient(mongodb_uri)`를 만들지 말고, **모듈 레벨 싱글톤**(예: `get_mongodb_client()` / `get_mongodb_database()`)을 두고 재사용하는 편이 좋다.
- 같은 `MONGODB_URI`·`MONGODB_DATABASE`를 쓰므로, 백엔드와 동일한 MongoDB 인스턴스·DB를 바라보는 설계는 유지된다.

---

## 4. 분산 시스템 “경유” 여부

| 플로우 | 경유 여부 | 비고 |
|--------|-----------|------|
| Playground HTTP 요청 | ✅ | MonitoringMiddleware를 통해 Redis에 요청/에러/응답시간/엔드포인트(채팅만) 기록 |
| 세션 CRUD | ✅ | history_router → session_cache(Redis) → MongoDB, 캐시 무효화 포함 |
| Agentic 스트리밍 | ✅ | 동일 미들웨어 + 채팅 경로 필터로 Redis 메트릭 기록 |
| MCP 도구 호출 | N/A | 별도 프로세스일 때는 HTTP/stdio로 백엔드와 연동. **DB**는 MongoDB 직접 접근(같은 URI/DB). Redis는 사용하지 않음. |

---

## 5. 체크리스트 요약

- **백엔드**: MongoDB·Redis 역할, 세션 쓰기/읽기/목록/검색, 메트릭(채팅만), human-in-the-loop용 Redis 사용까지 설계와 정합성 있음.
- **MCP**: 동일 MongoDB·`chat_sessions` 사용으로 DB 설계와 일치. Redis는 도입하지 않았고, “캐시는 백엔드만”인 현재 구조와도 잘 맞음.
- **모니터링**: Redis가 bytes를 주는 환경에서도 endpoint/token 집계가 깨지지 않도록 `monitoring_router`에서 bytes/str 처리 보완 완료.

위 내용은 `SCHEMA_AND_FLOWS.md`의 “DB·분산 시스템 개요” 및 “분산 시스템 경유·모니터링 점검” 절을 기준으로 한 점검 결과이다.

---

## 6. 챗/모델/LLM ↔ 메시지 처리·분산 시스템 조율

"챗·모델·LLM 처리를 모두 메시지 처리와 분산 시스템으로 조율·조절한다"는 목표 기준으로 현재 구성을 정리한 것이다.

### 6.1 현재 잘 되어 있는 부분

| 구간 | 메시지·분산 시스템 경유 | 내용 |
|------|--------------------------|------|
| **모든 HTTP 요청** | ✅ | `MonitoringMiddleware`가 최상위에 있어, 모든 요청이 `request:status:{id}`(Redis) 저장·갱신을 거친다. |
| **채팅 API 요청/응답** | ✅ | `/api/chat`, `/api/chat/*` 경로는 동일 미들웨어에서 `metrics:requests`, `metrics:errors`, `metrics:response_time`, `metrics:endpoint:*`(Redis)에 기록된다. |
| **챗 요청 시작** | ✅ | `POST /api/chat`에서 `ChatMonitoringMixin.log_chat_request` → Kafka `chat.request` 이벤트(USE_DISTRIBUTED 시), 요청 메타 로깅. |
| **챗 응답 완료** | ✅ | `POST /api/chat`에서 `client.chat()` 직후 `ChatMonitoringMixin.log_chat_response` → Redis `metrics:tokens:{model}`, Kafka `chat.response`(USE_DISTRIBUTED 시). |
| **Agentic 스트리밍** | ✅ | `POST /api/chat/agentic`도 동일 미들웨어 통과 → 채팅 경로이므로 Redis 메트릭 기록. DONE 이벤트에 usage가 있으면 `ChatMonitoringMixin.log_chat_response`로 Redis 토큰 메트릭 기록. |
| **세션·메시지** | ✅ | 세션/메시지 CRUD는 `session_cache`(Redis) → MongoDB, 캐시 무효화까지 설계대로 동작. |

정리하면, **챗·모델·LLM 호출이 나가는 모든 경로**는 이미 다음을 거친다.

- **분산 시스템(Redis)**: 요청 상태, 채팅 메트릭(요청 수/에러/응답시간/엔드포인트/토큰) 기록
- **메시지·이벤트(Kafka, USE_DISTRIBUTED 시)**: `log_chat_request` / `log_chat_response`로 `chat.request`, `chat.response` 이벤트 발행

즉, "챗·모델·LLM 처리"가 **메시지(이벤트) + 분산(Redis 메트릭/상태)** 로 관측·기록되게끔 잘 묶여 있다.

### 6.2 "메시지 처리" 레이어의 의미

- **현재**: "메시지 처리"라는 **단일 추상 레이어 이름**은 없다. 대신 HTTP 요청 → `MonitoringMiddleware`(상태·메트릭), 챗 요청/응답 → `ChatMonitoringMixin.log_chat_request` / `log_chat_response`(이벤트·토큰) 로, **챗/모델/LLM이 나가는 모든 진입로**가 분산(Redis) + 선택적 Kafka를 거치도록 되어 있다.
- **추가로 "조율·조절"**을 넣고 싶다면(예: 큐 기반 스로틀, 우선순위, 중앙 메시지 버스에서 실행 제어): 모든 LLM 호출을 **한 레이어**(예: `MessageProcessor` 또는 Kafka consumer)에서 받아서 Redis/Kafka로 "요청 메시지"를 넣고, 워커가 꺼내서 `client.chat()` 등을 실행하게 하는 설계를 **추가**하면 된다. 그때에도 지금 구조(미들웨어 + ChatMonitoringMixin)는 "관측·메트릭·이벤트" 용도로 그대로 두고, "실행 제어"만 새 레이어에 두면 된다.

### 6.3 요약

- **조율·조절이 "관측 + 메트릭 + 이벤트 기록" 수준**이라면: 챗/모델/LLM 쪽은 이미 메시지(이벤트)와 분산 시스템(Redis·Kafka)으로 **잘 조율되어 있다**.
- **조율·조절이 "실행 경로 자체를 메시지 큐/버스로 통일"**하는 것이라면: 현재는 "모든 요청이 분산 시스템을 **경유해서 관측**되는" 단계까지 되어 있고, "실행을 메시지 처리 파이프라인으로 완전히 올리는" 단계는 별도 설계·구현이 필요하다. 그 설계 전에 **현재 스택과 이미 구현된 범위**는 [`CURRENT_STACK_AND_SCOPE.md`](./CURRENT_STACK_AND_SCOPE.md)를 참고한다.
