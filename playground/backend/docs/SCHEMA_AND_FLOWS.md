# Playground 스키마·플로우 및 제안 단계 정책

현재 코드베이스의 **DB·분산 시스템**과 **메시지/쿼리 처리** 관계를 정리하고,  
**제안 단계**(그래프·노드·쿼리 분산 등)는 **오로지 챗으로만** 이루어지도록 하는 정책을 명시한다.

---

## 1. DB·분산 시스템 개요

### 1.1 MongoDB

- **역할**: 채팅 세션·메시지 메타·API 키·Google OAuth 토큰 등 영구 저장
- **접근**: `database.get_mongodb_database()` → `db.chat_sessions` 등
- **환경**: `MONGODB_URI`, `MONGODB_DATABASE`(기본 `beanllm`)

**주요 컬렉션·스키마**:

| 컬렉션 | 용도 | 대표 필드 |
|--------|------|-----------|
| `chat_sessions` | 세션 메타·요약 | `session_id`, `title`, `feature_mode`, `model`, `messages`(요약/빈 배열), `created_at`, `updated_at`, `total_tokens`, `message_count` |
| API 키·OAuth | `schemas.database` 참고 | provider, encrypted key, metadata 등 |

**인덱스** (`database.create_session_indexes()`):

- `session_id` (unique)
- `updated_at`, `created_at`, `feature_mode`, `total_tokens`, `message_count`, `title`
- 복합: `(feature_mode, updated_at)`, `(updated_at, total_tokens)`

**메시지 상세**  
- 세션별 메시지 본문은 **message_vector_store** (Chroma 등)에 따로 저장되고, MongoDB에는 세션 메타/요약만 둔다.

### 1.2 Redis

- **역할**: 세션 캐시, 요청/메트릭, 토큰 사용량
- **접근**: `beanllm.infrastructure.distributed.redis.client.get_redis_client()` 또는 `playground` 내 `session_cache` 등

**키 패턴**:

| 패턴 | 용도 | TTL 등 |
|------|------|--------|
| `session:{session_id}` | 개별 세션 캐시 | 60s |
| `sessions:list:{params}:{skip}:{limit}` | 세션 목록 캐시 | 300s |
| `request:status:{request_id}` | 요청 상태(모니터링) | - |
| `metrics:requests:{minute}`, `metrics:errors:{minute}` | 분별 요청/에러 수 | 3600s |
| `metrics:response_time` | 응답 시간 (sorted set) | 3600s |
| `metrics:endpoint:{path}` | 엔드포인트별 통계 (hash) | 3600s |
| `metrics:tokens:{model}` | 토큰 사용량 (hash) | 86400s |

세션·목록 캐시는 MongoDB 조회를 줄이기 위한 1차 캐시이며, 세션/목록 갱신 시 해당 키들이 무효화된다.

### 1.3 기타 스토어

- **Chroma**(또는 설정된 Vector Store): 메시지 임베딩·세션 검색용 (`message_vector_store`, `session_search_service`)
- **Neo4j**(선택): KG 쿼리 시 Cypher 실행. playground에서는 beanllm KG API를 통해 사용.

---

## 2. 메시지·쿼리 처리 플로우

### 2.1 Agentic 챗 (의도 → 도구 → 스트리밍)

1. **진입**: `POST /agentic` (또는 프론트에서 해당 스트리밍 엔드포인트 호출)
2. **입력**: `request.messages` 중 마지막 user 메시지 → `query`
3. **의도 분류**: `intent_classifier.classify(query)` → `IntentResult` (primary/secondary intents, confidence)
4. **도구 선택**: `tool_registry.get_best_tool_for_intent(...)` → `selected_tools`
5. **실행**: `OrchestratorContext(query=query, intent=..., selected_tools=..., session_id=...)`  
   → `orchestrator.execute(context)` 또는 `execute_parallel(context)`  
   → 도구별로 `mcp_client` 등으로 실제 쿼리 수행 (RAG/KG/웹검색 등)
6. **출력**: SSE 스트림 (`intent` → `tool_select` → `proposal` → `human_approval`(도구별) → `tool_start` / `tool_progress` / `tool_result` → `text` / `text_done` → `done`)

이 과정에서 **DB 직접 접근**은 하지 않는다.  
세션/히스토리 저장은 별도 플로우(아래 2.2)에서 처리된다.

### 2.2 세션·히스토리 (MongoDB + Redis)

- **쓰기**  
  - 세션 생성: `POST /api/chat/sessions` → MongoDB `chat_sessions.insert_one` + Redis `session:{id}` + 목록 캐시 무효화  
  - 메시지 추가: `POST /api/chat/sessions/{session_id}/messages` → MongoDB 업데이트 + `message_vector_store` 인덱싱 + Redis 세션/목록 캐시 무효화
- **읽기**  
  - 세션 단건: Redis `session:{session_id}` 조회 → 없으면 MongoDB `chat_sessions.find_one({"session_id": session_id})` 후 캐시 적재  
  - 세션 목록: Redis `sessions:list:{...}` 조회 → 없으면 MongoDB `find(...).sort(...).skip().limit()` 후 캐시 적재

`session_id`는 경로/쿼리 파라미터로만 들어오며, MongoDB/Redis에는 **딕셔너리 필터/키**로 그대로 사용된다. (파라미터 바인딩 형태라 SQL/NoSQL 인젝션과는 무관.)

### 2.3 세션 검색 (MongoDB + Vector + Redis)

- **진입**: `GET /api/chat/sessions?query=...&...` (필터/정렬 파라미터 포함)
- **검색 경로**  
  1. `query`가 있으면 `session_search_service.search_sessions(query=query, db=db, ...)`  
  2. 우선 **Vector 검색**: `message_vector_store.search_messages(query=query, ...)` → session_id 리스트 구한 뒤, MongoDB에서 해당 `session_id`만 `find({"session_id": {"$in": ...}})`  
  3. Vector 미사용 또는 실패 시 **키워드 검색(Fallback)**:  
     - MongoDB `$regex` 사용 시 **반드시** `re.escape(query)` 한 `safe_pattern`으로만 쿼리 (ReDoS·정규식 인젝션 방지)

MongoDB에 넘기는 필터는 모두 서버에서 만든 딕셔너리이며, 사용자 입력이 들어가는 곳은 `$regex` 패턴 하나뿐이고, 그 부분은 이스케이프로 안전하게 처리한다.

---

## 3. 제안 단계 = 챗 전용 (Claude Code처럼)

그래프 사용 여부, **노드 개수·구성·쿼리 분산** 등은 다음 원칙으로만 다룬다.

- **제안**: Agent가 **챗 메시지**로 제안 (예: “이번에는 노드 3개, RAG → 요약 → 응답 순으로 가는 게 좋겠어요.”).
- **결정/수정**: 유저가 **챗으로만** 승인·수정·직접 입력 (예: “그대로 해줘”, “노드 4개로 늘려줘”, “두 번째는 KG 검색으로 바꿔줘”).
- **UI 폼/고정 입력란**으로 “노드 수”, “쿼리 분산” 등을 받지 않는다.  
  모든 제안·확정·수정은 **대화(챗) 스트림과 그에 대한 유저 메시지**로만 이루어진다.

구현 시에는:

- 제안 내용을 **챗 페이로드/이벤트**(예: `proposal` 타입 또는 assistant 메시지 본문)로 내려주고,
- 유저 입력을 파싱해 “승인 / 수정 지시 / 직접 스펙 입력”을 구분하는 방식이 필요하다.

---

## 4. 보안 (원초적·심각한 문제 위주)

### 4.1 MongoDB

- **일반 쿼리**: `find`, `insert_one`, `update_one`, `delete_one` 등은 모두 **딕셔너리 필터/문서**로만 사용. 사용자 문자열을 쿼리 문자열에 이어 붙이지 않음 → SQL/NoSQL 인젝션 해당 없음.
- **$regex**  
  - 사용자 입력 `query`를 그대로 `$regex`에 넣지 않는다.  
  - `session_search_service`에서는 `re.escape(query)`로 이스케이프한 `safe_pattern`만 사용 (ReDoS 및 정규식 메타문자 인젝션 방지).

### 4.2 Neo4j / Cypher

- `query_type="cypher"`이고 `request.query`를 그대로 `session.run(cypher_query, parameters)`에 넣는 경로가 있다.  
  이 경우 **사용자가 Cypher 문 전체**를 넣을 수 있으므로, “Cypher 인젝션”과 동일한 위험(예: `MATCH (n) DETACH DELETE n` 등)이 있다.
- 완화 방안:  
  - **파라미터화**: 쿼리 템플릿은 고정하고, 사용자 입력은 `parameters`로만 넘긴다.  
  - **권한**: Neo4j 사용자 권한을 읽기 전용 등으로 제한.  
  - **노출 제한**: 외부/비신뢰 사용자에게는 `query_type=cypher` + 원문 쿼리 입력을 막고, 자연어/사전 정의 쿼리 타입만 허용.

현재 “SQL injection 같은 원초적·심각한 문제”를 막는 선에서는, **MongoDB $regex 이스케이프**까지 반영된 상태이고, Cypher는 위 정책을 적용할 대상으로 문서에만 명시한다.

### 4.3 Redis

- 키/값은 서버가 만든 문자열·직렬화 결과만 사용.  
  `session_id`는 UUID 기반 또는 경로 파라미터로, 캐시 키 일부에만 들어가며, 사용자가 임의 문자열을 “쿼리”처럼 넣어 Redis 명령을 탈바꿈시키는 구조는 없음.

---

## 5. “지금 코드가 시스템을 잘 쓰는지” 체크리스트

| 구분 | 내용 | 상태 |
|------|------|------|
| 세션 쓰기 | MongoDB + Redis 캐시 갱신 + 목록 무효화 | 일치 (history_router, session_cache) |
| 세션 읽기 | Redis → Miss 시 MongoDB → 캐시 적재 | 일치 |
| 세션 목록 | Redis 목록 캐시 키에 필터 파라미터 반영 | 일치 (cache_key_params 등) |
| 검색 | Vector 우선, Fallback 시 MongoDB $regex는 이스케이프 후 사용 | 반영 완료 (session_search_service) |
| Agentic 쿼리 | Intent → 도구 선택 → MCP/도구 호출, DB 직접 사용 없음 | 일치 |
| 메트릭 | Redis `metrics:*` 키로 요청/에러/토큰 등 집계 | 모니터링 미들웨어/라우터에서 사용 |

---

## 6. 분산 시스템 경유·모니터링 점검

### 6.1 “기존 것들이 분산 시스템을 거치게 되어 있는지” 점검

| 플로우 | 분산 경유 여부 | 비고 |
|--------|----------------|------|
| **모든 HTTP 요청** | ✅ 경유 | `MonitoringMiddleware`가 최상위에 있어 **모든** 요청이 통과. `request:status:{id}`, `metrics:response_time`, `metrics:requests:{minute}`, `metrics:errors:{minute}`, `metrics:endpoint:{method}:{path}` 기록. |
| **Agentic 스트리밍** (`POST /api/chat/agentic`) | ✅ 경유 | HTTP 요청이므로 미들웨어 통과 → 요청 수·응답 시간·엔드포인트별 통계는 Redis에 적재됨. |
| **세션 API** (생성/조회/목록/메시지 추가/삭제) | ✅ 경유 | `history_router`가 `session_cache`(Redis) 사용. 세션 단건/목록 읽기 시 Redis → miss 시 MongoDB → 캐시 적재. 쓰기 시 MongoDB + Redis 캐시 갱신·무효화. |
| **세션 캐시 Redis 클라이언트** | ✅ 동일 인스턴스 | `session_cache`는 `beanllm.infrastructure.distributed.redis.client.get_redis_client()` 사용. 미들웨어의 `RequestMonitor`도 동일 `get_redis_client()` 사용 → 동일 Redis 인스턴스. |

### 6.2 모니터링 코드·커버리지

- **미들웨어** (`monitoring/middleware.py`): 요청 시작 시 `request:status:{id}` 저장, 응답 시 상태 업데이트 + `metrics:response_time`, `metrics:requests:{minute}`, `metrics:errors:{minute}`, `metrics:endpoint:{method}:{path}` 저장. **Redis 사용 여부**는 `USE_REDIS_MONITORING`(기본 `true`)으로 제어.
- **모니터링 API** (`routers/monitoring_router.py`): Redis에서 `metrics:*` 키를 읽어 요약/트렌드/엔드포인트별/토큰별 통계 제공. `_get_redis_client()`로 미들웨어와 동일한 Redis 풀(또는 RequestMonitor 내 Redis)에서 조회.
- **대시보드** (`monitoring/dashboard.py`): Redis에서 동일 키 패턴으로 메트릭·요청 상태 조회. Streamlit 기반.

**갭**:

- **토큰 메트릭** (`metrics:tokens:{model}`): **일반 Chat** (`POST /api/chat` in main.py)에서만 `ChatMonitoringMixin.log_chat_request` / `log_chat_response` 호출로 기록됨. **Agentic/스트리밍** 경로에서는 orchestrator·chat_router가 usage를 반환하지 않아 토큰 메트릭 미기록.  
  → 향후 orchestrator 종료 이벤트나 스트림 마지막에 usage가 포함되면, 그 시점에 `ChatMonitoringMixin.log_chat_response(..., input_tokens=..., output_tokens=...)` 호출을 넣으면 Redis 토큰 통계에 반영 가능.

### 6.3 요약

- 기존 **요청/응답/엔드포인트/에러** 메트릭은 분산(Redis)을 타고 있으며, 모니터링 코드도 해당 키들을 읽어 API·대시보드에 노출하고 있음.
- **세션** 읽기/쓰기는 Redis 캐시를 거쳐 MongoDB와 연동되어 있음.
- **토큰** 메트릭만 Agentic 경로에서 미기록인 상태이며, 필요 시 스트림 종료 시 usage 훅으로 보완하면 됨.

이 문서는 위 스키마·플로우·제안 정책·보안이 맞는지, 이후 그래프/노드 런 스키마를 설계할 때의 기준으로 사용하면 된다.
