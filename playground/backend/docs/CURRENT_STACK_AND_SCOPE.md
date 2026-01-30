# 현재 스택 및 구현 범위

"실행을 메시지 처리 파이프라인으로 완전히 올리는" 단계는 **별도 설계**로 둔다. 이 문서는 그 전에 **현재 스택**을 정확히 파악하고, **이미 구현된 범위**를 코드·문서 기준으로 정리한 것이다.

---

## 0. 메시징 처리 적극 활용 정책

**원칙**: Redis와 메시지 큐/버스(Kafka 등)를 **적극 활용**한다.  
요청·이벤트·상태·메트릭을 Redis·메시지 큐 중심으로 두어 다음을 목표로 한다.

| 목표 | 내용 |
|------|------|
| **대용량 처리** | 큐·버퍼 기반으로 피크 시 부하 분산, 비동기 처리·배치 구간 명확화, 스케일 아웃(워커 증설) 용이 |
| **트래픽 관리** | Redis 기반 Rate Limiting·우선순위·스로틀, 요청/에러/지연 메트릭 실시간 집계로 병목·이상 감지 |
| **AI 활용 용이** | 이벤트·메트릭·토큰 사용량이 메시지/Redis로 모이므로, 추론·채팅·RAG 등 AI 파이프라인 관측·튜닝·재시도·우선순위 제어가 쉬움 |

**적용 방향**  
- **관측·메트릭·이벤트**: 이미 Redis(`request:status`, `metrics:*`, `session:*`)·Kafka(USE_DISTRIBUTED 시)로 쌓고 있음. 채팅 외 API·MCP까지 같은 경로를 타도록 확장(§6).  
- **실행 제어**: “실행을 메시지 처리 파이프라인으로 올리기” 설계 시, Redis 큐(또는 Kafka 토픽)에 요청 메시지를 넣고 워커가 꺼내 실행하는 구조를 기본으로 두면, 대용량·트래픽·AI 제어를 한 흐름으로 맞추기 쉬움.

이 정책에 맞추어 **채팅 외·MCP 분산 처리**(§6)와 **메시지 파이프라인 설계**(§8)를 진행한다.

---

## 1. 현재 스택 (구성요소·버전·역할)

### 1.1 애플리케이션 계층

| 계층 | 기술 | 버전/환경 | 역할 |
|------|------|-----------|------|
| **프론트엔드** | Next.js | 15.x | 라우팅, SSR, 페이지 렌더링 |
| | React | 19.x | UI 컴포넌트 |
| | TypeScript | ~5.7 | 타입 체크 |
| | Tailwind CSS | 4.x | 스타일링 |
| | Framer Motion, Radix UI, Recharts 등 | package.json 기준 | UI/차트/애니메이션 |
| **백엔드** | FastAPI | (uvicorn 기반) | REST/SSE/WebSocket API |
| | Python | 3.10+ | 런타임 |
| **코어 라이브러리** | beanllm | 0.2.x (pyproject.toml) | Chat/RAG/KG/Agent/Chain/Vision/Audio 등 Facade·Handler·Service·Domain·Infrastructure |
| **MCP 서버** | Python (별도 프로세스) | - | 도구(RAG/KG/웹검색 등) 제공, HTTP/stdio로 백엔드와 연동 |

### 1.2 인프라·저장소

| 구성요소 | 용도 | 접근 경로 |
|----------|------|-----------|
| **MongoDB** | 채팅 세션·메시지 메타, API 키·OAuth 토큰 등 영구 저장 | `playground/backend/database.get_mongodb_database()` → `chat_sessions` 등 |
| **Redis** | 세션 캐시, 요청 상태, 메트릭(채팅 API만), 토큰 사용량 | `beanllm.infrastructure.distributed.redis.client.get_redis_client()` (decode_responses=False) |
| **Kafka** | 이벤트 스트리밍 (USE_DISTRIBUTED=true 시) | beanllm `get_event_logger()` → `chat.request` / `chat.response` 등 |
| **Chroma / Vector Store** | 메시지 임베딩·세션 검색 | `message_vector_store`, `session_search_service` |
| **Neo4j** | KG 쿼리 (선택) | beanllm KG API 경유 |

docker-compose 기준 서비스: mongodb(27017), redis(6379), zookeeper(2181), kafka(9092), (선택) kafka-ui, mongo-express, redis-commander.

### 1.3 의존 관계 요약

```
[프론트 Next.js/React] ──HTTP/SSE──► [Playground 백엔드 FastAPI]
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            [beanllm Client/Facade]   [MonitoringMiddleware]   [history_router / chat_router / …]
                    │                         │                         │
                    │                         ├─ RequestMonitor(Redis)   ├─ session_cache(Redis)
                    │                         │   request:status,        │   session:{id}, sessions:list:*
                    │                         │   metrics:* (채팅 경로만)  └─ MongoDB (get_mongodb_database)
                    │                         └─ EventLogger(Kafka, 선택)
                    ▼
            [Handler → Service → Domain → Infrastructure]
                    │
                    ├─ Providers (OpenAI/Anthropic/Ollama/…)
                    ├─ Redis/Kafka (beanllm.infrastructure.distributed)
                    └─ Vector/Neo4j 등

[MCP 서버] ◄──HTTP/stdio── [Playground 백엔드]
     │
     └─ get_session_messages() → MongoDB 직접 (매 호출 시 AsyncIOMotorClient 새로 생성)
```

---

## 2. 이미 구현된 범위 (메시지·분산·DB 기준)

### 2.1 분산·메트릭·이벤트에 “이미 타는” 것

| 구간 | 내용 | 코드/위치 |
|------|------|-----------|
| **모든 HTTP 요청** | `request:status:{request_id}` Redis 저장·갱신 | `MonitoringMiddleware.dispatch()` → `RequestMonitor.redis.setex("request:status:...")` |
| **채팅 API만 메트릭 집계** | `metrics:response_time`, `metrics:requests:{minute}`, `metrics:errors:{minute}`, `metrics:endpoint:{method}:{path}` | `MonitoringMiddleware` 내부 `_is_chat_api_path(path)` → True일 때만 Redis 기록. 경로 조건: `path == "/api/chat" or path.startswith("/api/chat/")` |
| **POST /api/chat (일반 챗)** | `ChatMonitoringMixin.log_chat_request` → Kafka `chat.request`(USE_DISTRIBUTED 시) | `main.py` 829행 근처 |
| | `ChatMonitoringMixin.log_chat_response` → Redis `metrics:tokens:{model}`, Kafka `chat.response`(USE_DISTRIBUTED 시) | `main.py` 850행 근처 |
| **POST /api/chat/agentic (스트리밍)** | 동일 미들웨어 통과 → 채팅 경로이므로 요청/에러/응답시간/엔드포인트 메트릭 기록 | `chat_router` prefix `/api/chat` |
| | DONE 이벤트에 usage 있으면 `log_chat_response` 호출 | `chat_router.py` 267, 376행 근처 |
| **세션·메시지 CRUD** | Redis `session:{id}`, `sessions:list:{params}:{skip}:{limit}` → miss 시 MongoDB → 캐시 적재 | `history_router` + `session_cache`(SessionCacheService) |
| **세션 쓰기** | MongoDB upsert + Redis 세션 갱신 + 목록 캐시 무효화 | `history_router` |
| **Human-in-the-loop** | Redis `run:approval:{id}`, `start_from_tool_index` 등으로 재개 | `chat_router` / `orchestrator` |

즉, **챗·모델·LLM이 나가는 진입로**는 이미 다음을 거친다.

- **분산(Redis)**: 요청 상태, 채팅 메트릭(요청 수/에러/응답시간/엔드포인트/토큰)
- **이벤트(Kafka, USE_DISTRIBUTED 시)**: `log_chat_request` / `log_chat_response` → `chat.request`, `chat.response`

### 2.2 “메시지·분산을 타지 않는” 것

| 구간 | 내용 | 비고 |
|------|------|------|
| **채팅 외 API** | `/api/kg`, `/api/rag`, `/api/chain`, `/api/audio` 등 | `_is_chat_api_path`가 False → `metrics:requests|errors|response_time|endpoint:*` 미기록. `request:status`만 기록 |
| **토큰 메트릭** | Agentic 스트림에서 usage 미전달 구간 | DONE에 usage 있을 때만 `log_chat_response` 호출되므로, 그외 구간은 토큰 미집계 |
| **MCP 서버** | Redis 미사용. MongoDB는 `get_session_messages()`에서 직접 조회 | 호출 시마다 `AsyncIOMotorClient(mongodb_uri)` 새로 생성 (백엔드와 동일 DB·컬렉션) |
| **WebSocket /ws/{session_id}** | 메인 앱에서 정의한 WS 엔드포인트 | 미들웨어는 HTTP 기준. 별도 메트릭/이벤트 없음 |

### 2.3 Redis 클라이언트 통일

- **세션 캐시**: `playground/backend/services/session_cache.py` → `beanllm.infrastructure.distributed.redis.client.get_redis_client()`
- **모니터링**: `MonitoringMiddleware` → `RequestMonitor()` → 동일 `get_redis_client()`
- **주의**: beanllm Redis는 `decode_responses=False` → 키/값이 bytes. `monitoring_router`에서는 bytes/str 모두 처리하도록 `_int_val`, `decode` 등 적용됨.

---

## 3. API·라우터·진입로 정리

### 3.1 채팅·히스토리 (분산·메트릭 적용 구간)

| 메서드 | 경로 | 집계·이벤트 |
|--------|------|--------------|
| POST | `/api/chat` | 메트릭 ✅, request/response 상태 ✅, `log_chat_request`/`log_chat_response` ✅ (main.py) |
| POST | `/api/chat/agentic` | 메트릭 ✅, DONE 시 usage 있으면 `log_chat_response` ✅ (chat_router) |
| GET/POST/DELETE 등 | `/api/chat/sessions` 및 하위 | 메트릭 ✅ (`/api/chat/` 접두사), 세션 캐시(Redis)→MongoDB ✅ |

### 3.2 그 외 라우터 (요청 상태만 Redis 기록)

| prefix | 용도 | Redis 메트릭 |
|--------|------|--------------|
| `/api/config`, `/api/models` 등 | 설정·모델 | request:status 만 |
| `/api/kg` | Knowledge Graph | 동일 |
| `/api/rag` | RAG | 동일 |
| `/api/chain` | Chain | 동일 |
| `/api/vision_rag` | Vision RAG | 동일 |
| `/api/audio` | Audio | 동일 |
| `/api/evaluation`, `/api/finetuning` | 평가·파인튜닝 | 동일 |
| `/api/ocr`, `/api/web`, `/api/optimizer` | OCR·웹검색·옵티마이저 | 동일 |
| `/api/monitoring` | 대시보드·메트릭 조회 | 동일 |
| `/api/auth/google` | Google OAuth | 동일 |

### 3.3 main.py에 남아 있는 라우트

- `GET /`, `GET /health`, `GET /api/config/providers`, `GET /api/config/models`
- `POST /api/chat` (유일한 “일반 챗” 진입로, ChatMonitoringMixin 사용)
- `WebSocket /ws/{session_id}`

---

## 4. DB별 저장 정보

### 4.1 MongoDB (영구 저장)

| 컬렉션 | 저장 내용 | 대표 필드 | 접근 위치 |
|--------|------------|-----------|-----------|
| **chat_sessions** | 세션 메타·요약·메시지 배열 | `session_id`, `title`, `feature_mode`, `model`, `messages[]`, `feature_options`, `created_at`, `updated_at`, `total_tokens`, `message_count` | history_router, session_search_service, MCP session_manager |
| **api_keys** | Provider별 API 키(암호화) | `provider`, 암호화된 키, 메타데이터 | config_router, config_service |
| **google_oauth_config** | Google OAuth 클라이언트 설정 | `client_id_encrypted`, `client_secret_encrypted` 등 | config_router, google_oauth_service |
| **google_oauth_tokens** | 사용자별 Google OAuth 토큰 | `user_id`, `access_token`, `refresh_token`, `expires_at` 등 | google_oauth_service, orchestrator |

메시지 상세는 MongoDB `chat_sessions.messages`에 배열로 두고, 검색용 임베딩은 Chroma(또는 설정된 Vector Store)에 따로 둔다.

### 4.2 Redis (캐시·상태·메트릭)

| 키 패턴 | 저장 내용 | TTL | 쓰는 쪽 |
|---------|------------|-----|---------|
| **session:{session_id}** | 세션 문서 전체(JSON) | 60s | session_cache, history_router |
| **sessions:list:{params}:{skip}:{limit}** | 세션 목록 캐시(JSON) | 300s | session_cache, history_router |
| **request:status:{request_id}** | 요청 상태 JSON (`status`, `started_at`, `path`, `method`, `duration_ms`, `status_code` 등) | 3600s | MonitoringMiddleware |
| **metrics:requests:{minute}** | 분당 요청 수(카운터) | 3600s | MonitoringMiddleware (채팅 경로만) |
| **metrics:errors:{minute}** | 분당 에러 수(카운터) | 3600s | MonitoringMiddleware (채팅 경로만) |
| **metrics:response_time** | Sorted Set, request_id → duration_ms | 3600s | MonitoringMiddleware (채팅 경로만) |
| **metrics:endpoint:{method}:{path}** | Hash: `count`, `total_time_ms`, `errors` | 3600s | MonitoringMiddleware (채팅 경로만) |
| **metrics:tokens:{model}** | Hash: `input_tokens`, `output_tokens`, `total_tokens`, `request_count` | 86400s | ChatMonitoringMixin.log_chat_response |
| **run:approval:{run_id}** | Human-in-the-loop 승인/재개 상태 | (설정에 따름) | chat_router, orchestrator |

Redis 클라이언트는 `decode_responses=False`이므로 키/값이 bytes일 수 있음. monitoring_router 등에서는 bytes/str 모두 처리 필요.

### 4.3 Chroma / Vector Store (메시지 검색)

| 저장소 | 저장 내용 | 용도 |
|--------|------------|------|
| **chat_messages** (collection) | 메시지 본문 임베딩 + metadata: `session_id`, `message_id`, `role`, `model`, `timestamp` | 세션 검색 시 Vector 유사도 검색. session_search_service, message_vector_store 사용. persist: `./.chroma_messages` |

세션 목록 검색 시: Vector 검색으로 `session_id` 리스트를 구한 뒤, MongoDB `chat_sessions`에서 해당 session_id만 조회.

### 4.4 Neo4j (선택)

| 용도 | 저장·조회 내용 |
|------|----------------|
| KG 빌드/쿼리 | 노드·관계·Cypher 쿼리 결과. playground에서는 beanllm KG API 경유만 사용. |

---

## 5. API별 제공 정보 (요청·응답 요약)

### 5.1 채팅·세션 (history_router, main.py, chat_router)

| 메서드 | 경로 | 제공 정보 (응답) |
|--------|------|------------------|
| POST | `/api/chat` | `role`, `content`, `usage`, `model`, `provider` |
| POST | `/api/chat/stream` | SSE 스트림 (텍스트 청크) |
| POST | `/api/chat/agentic` | SSE: `intent`, `tool_select`, `proposal`, `human_approval`, `tool_*`, `text`/`text_done`, `done` |
| POST | `/api/chat/classify` | 의도 분류 결과 (primary/secondary intents, confidence) |
| GET | `/api/chat/tools`, `/api/chat/tools/{name}` | 도구 목록/상태 |
| GET | `/api/chat/intents` | 지원 의도 목록 |
| POST | `/api/chat/sessions` | 생성된 세션 `SessionResponse(session=...)` |
| GET | `/api/chat/sessions` | `SessionListResponse(sessions, total)` (필터: query, date_from/to, min_tokens, feature_mode 등) |
| GET | `/api/chat/sessions/{id}` | `SessionResponse(session)` |
| POST | `/api/chat/sessions/{id}/messages` | 업데이트된 `SessionResponse` |
| PATCH | `/api/chat/sessions/{id}/title` | 204 |
| DELETE | `/api/chat/sessions/{id}` | 204 |
| GET | `/api/chat/sessions/{id}/messages` | 메시지 배열 |

### 5.2 설정·모델 (config_router, main.py, models_router)

| 메서드 | 경로 | 제공 정보 (응답) |
|--------|------|------------------|
| GET | `/api/config/providers` | 활성 provider 목록, 마스킹된 설정 |
| GET | `/api/config/models` | 모델 목록(다운로드 여부 등) |
| GET/POST/DELETE | `/api/config/keys`, `/api/config/keys/{provider}` | API 키 목록/단건/저장/삭제 |
| POST | `/api/config/keys/{provider}/validate` | 검증 결과 |
| GET | `/api/config/providers/all` | 전체 provider 목록 |
| POST/GET | `/api/config/google-oauth`, `/api/config/google-oauth/status` | Google OAuth 설정 저장/상태 |
| POST | `/api/config/keys/load-all` | env 로드 결과 |
| GET | `/api/config/provider-sdks` | SDK 설치 여부 |
| POST | `/api/config/install-package` | 패키지 설치 결과 |
| GET | `/api/models`, `/api/models/{provider}` | 모델 목록 |
| GET | `/api/models/{model_name}/parameters` | 모델 파라미터 |
| POST | `/api/models/scan`, `/{model}/pull`, `/{model}/analyze` | 스캔/풀/분석 결과 |

### 5.3 기능 API (kg, rag, chain, agent, vision, audio, evaluation, finetuning, ocr, web, optimizer)

| prefix | 대표 엔드포인트 | 제공 정보 (응답) |
|--------|-----------------|------------------|
| `/api/kg` | POST build, query, graph_rag; GET visualize/{id} | 그래프 빌드/쿼리/GraphRAG 결과, 시각화 데이터 |
| `/api/rag` | POST build, build_from_files, query; GET collections; DELETE collections/{name}; POST debug | RAG 빌드/쿼리/컬렉션 목록·삭제·디버그 결과 |
| `/api/chain` | POST run, build; GET list; DELETE /{id} | 체인 실행/빌드/목록/삭제 결과 |
| `/api/agent` | POST agent/run, multi_agent/run, orchestrator/run | 에이전트/멀티에이전트/오케스트레이터 실행 결과 |
| `/api/vision_rag` | POST build, query; GET status | 비전 RAG 빌드/쿼리/상태 |
| `/api/audio` | POST transcribe, synthesize, rag | 음성 인식/합성/Audio RAG 결과 |
| `/api/evaluation` | POST evaluate | 평가 메트릭 결과 |
| `/api/finetuning` | POST create; GET status/{id}, list; POST cancel/{id} | 파인튜닝 생성/상태/목록/취소 |
| `/api/ocr` | POST recognize; GET engines | OCR 결과, 엔진 목록 |
| `/api/web` | POST search; GET engines | 웹검색 결과, 엔진 목록 |
| `/api/optimizer` | POST optimize; GET methods | 최적화 결과, 메서드 목록 |

### 5.4 모니터링·인증

| 메서드 | 경로 | 제공 정보 (응답) |
|--------|------|------------------|
| GET | `/api/monitoring/health` | Redis/Kafka 연결, uptime, timestamp |
| GET | `/api/monitoring/summary` | total_requests, total_errors, error_rate, 응답시간(avg/min/max/p50/p95/p99) |
| GET | `/api/monitoring/trend` | 분별 requests/errors 트렌드 |
| GET | `/api/monitoring/endpoints` | 엔드포인트별 count, errors, avg_time_ms, error_rate |
| GET | `/api/monitoring/tokens` | 모델별 input/output/total tokens, request_count |
| GET | `/api/monitoring/dashboard` | 위 항목 통합 (MonitoringDashboard) |
| POST | `/api/monitoring/clear` | 메트릭 초기화 |
| GET/POST | `/api/auth/google/*` | 서비스 목록, auth URL, callback, status, logout, token |

---

## 6. 채팅 외·MCP 서버 분산 처리 계획

**목표**: 채팅만이 아니라 **채팅 외 API**와 **MCP 서버**도 분산(Redis·메트릭·선택적 Kafka)을 경유하도록 한다.  
§0 “메시징 처리 적극 활용”에 따라, **대용량 처리·트래픽 관리·AI 활용**을 위해 Redis·메시지 큐를 더 넓게 쓴다.

### 6.1 현재와의 차이

| 구간 | 현재 | 계획 |
|------|------|------|
| 채팅 API (`/api/chat`, `/api/chat/*`) | 이미 Redis 메트릭·Kafka 이벤트·request:status 경유 | 유지 |
| 채팅 외 API (`/api/kg`, `/api/rag`, `/api/chain` 등) | `request:status`만 Redis 기록, `metrics:*` 미기록 | 요청/에러/응답시간/엔드포인트 메트릭까지 Redis에 기록 (또는 전 구간 메트릭 집계 정책 확장) |
| MCP 서버 | Redis 미사용. MongoDB 직접 조회(호출 시마다 새 클라이언트) | MCP→백엔드 세션/메시지 조회 시 Redis(또는 공용 캐시) 경유, 메트릭·이벤트 시 Redis/Kafka 연동 검토 |

### 6.2 채팅 외 API 분산 처리

- **메트릭 확장**: `_is_chat_api_path(path)`를 완화하거나, “전체 API”용 메트릭 키(예: `metrics:requests:all`, `metrics:endpoint:*` 전체)를 두어, 채팅 외 경로도 `metrics:requests`, `metrics:errors`, `metrics:response_time`, `metrics:endpoint:*`에 기록.
- **이벤트(선택)**: USE_DISTRIBUTED 시 채팅 외 요청도 Kafka로 `api.request.started` / `api.request.completed` 등 발행하도록 미들웨어 확장.
- **영향**: monitoring_router·대시보드는 “채팅만” / “전체” 탭 또는 필터를 둘 수 있음.

### 6.3 MCP 서버 분산 처리

- **세션·메시지 조회**:  
  - MCP가 세션/메시지를 읽을 때, **백엔드와 동일한 Redis**를 1차 캐시로 사용 (예: 백엔드 `session_cache`와 동일 키 `session:{id}`).  
  - MCP 전용 경로(예: 백엔드에 `/internal/sessions/{id}` 또는 gRPC/메시지 큐)를 두고, 그쪽에서 Redis → MongoDB 플로우를 타게 하거나, MCP가 직접 같은 `get_redis_client()`를 쓰도록 라이브러리/설정 공유.
- **메트릭·이벤트**:  
  - MCP 도구 호출 시 “도구 요청/완료”를 Redis/Kafka에 남기려면, MCP에서 Redis/Kafka 클라이언트를 두거나, 백엔드가 MCP 호출 전후에 메트릭/이벤트를 대신 기록하는 방식이 필요함.
- **MongoDB 연결**:  
  - Redis 경유로 대부분 처리할 경우에도, 캐시 미스·쓰기는 MongoDB가 필요하므로, MCP는 “매 호출 새 AsyncIOMotorClient” 대신 **싱글톤 클라이언트/DB**를 쓰는 것이 우선 권장됨 (BACKEND_MCP_DISTRIBUTED_REVIEW.md 반영).

### 6.4 설계 시 정리해 둘 것

- **메트릭 범위**: “채팅만” vs “전체 API” 집계 정책과, 키 접두사/필터 규칙.
- **MCP 연동 방식**: Redis/캐시 직접 접근 vs 백엔드 프록시 API vs 메시지 큐 기반.
- **이벤트 토픽**: 채팅 `chat.request`/`chat.response` 외에 `api.request.*` 등 공통 토픽 정의 여부.
- **기존 모니터링 대시보드**: “채팅 전용” 뷰와 “전체 API” 뷰 공존 시 필터/탭 설계.

위 내용은 “실행을 메시지 처리 파이프라인으로 올리는” 설계와 함께, **채팅 외·MCP 분산 처리** 단계에서 반영할 기준으로 둔다.

---

## 7. 문서·코드 기준 정리

| 문서 | 역할 |
|------|------|
| `SCHEMA_AND_FLOWS.md` | DB·분산 키 패턴, 세션/히스토리/검색 플로우, 제안 단계=챗 전용, 보안 |
| `BACKEND_MCP_DISTRIBUTED_REVIEW.md` | 백엔드·MCP와 분산/DB 설계 정합성, 수정 사항, 챗·메시지·분산 조율 현황, “메시지 처리 파이프라인으로 실행 올리기”는 별도 설계로 둔다는 요약 |
| **이 문서 (CURRENT_STACK_AND_SCOPE.md)** | 현재 스택, DB별 저장 정보, API별 제공 정보, “이미 되어 있는 범위”, 채팅 외·MCP 분산 처리 계획을 코드·경로 기준으로 정리 |
| [`IMPLEMENTATION_PLAN_MESSAGING.md`](./IMPLEMENTATION_PLAN_MESSAGING.md) | beanllm·playground 재사용 정리, Phase 1(전체 API 메트릭)·Phase 2(MCP Redis 세션 캐시)·Phase 3(실행 큐 파이프라인) 구현 계획 |
| [`CACHE_AND_METRICS_POLICY.md`](./CACHE_AND_METRICS_POLICY.md) | 캐시 무효화·만료 정책(쓰기 시 즉시 무효화/갱신, TTL), **추적 메트릭 기준**(측정할 필요가 있는 것만 정의) |

---

## 8. “메시지 처리 파이프라인으로 실행 올리기”와의 경계

- **현재 상태**: 모든 HTTP 요청이 분산(Redis)을 **경유해 관측**되고, 채팅 API는 메트릭·토큰·(선택)Kafka 이벤트까지 기록됨. 즉 “관측·메트릭·이벤트” 수준에서는 이미 메시지·분산과 잘 묶여 있음.
- **별도 설계 대상**: “실행 자체를 메시지 큐/버스 기반으로 옮기는” 단계. §0 메시징 처리 적극 활용에 맞춰, **Redis·Kafka(또는 Redis Queue)** 로 요청 메시지를 넣고 워커가 꺼내 `client.chat()` 등을 실행하는 구조를 기본으로 두면, **대용량 처리**(큐·버퍼·스케일 아웃), **트래픽 관리**(Rate Limiting·우선순위·스로틀), **AI 활용**(추론·채팅·RAG 관측·튜닝·재시도)가 한 흐름으로 가능해진다. 이때 현재 구조(미들웨어 + ChatMonitoringMixin)는 관측·메트릭·이벤트 용도로 유지하고, “실행 제어”만 새 파이프라인으로 두는 식의 설계가 필요함.
- **채팅 외·MCP 분산**: 위와 별도로, “채팅 외 API·MCP도 분산(메트릭·캐시·이벤트) 경유”하는 계획은 §6에 정리되어 있으며, §0과 맞춰 Redis·메시징을 적극 쓰는 방향으로 설계 시 §6.4 항목을 함께 반영한다.

이 문서는 그 설계를 하기 전에 **현재 스택·DB·API·구현 범위·분산 확장 계획·메시징 적극 활용 정책**을 고정해 두기 위한 것이다.
