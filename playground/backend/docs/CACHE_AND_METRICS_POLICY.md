# 캐시 무효화·만료 정책 및 추적 메트릭 기준

캐시는 **만료 시점·즉시 비우기**를 명확히 하고, 추적 메트릭은 **측정할 필요가 있는 것만** 정의된 기준으로 수집한다.

---

## 1. 캐시 무효화·만료 정책

### 1.1 원칙

- **쓰기 시 즉시 무효화**: 세션/목록이 바뀌는 쓰기 직후, 해당 캐시 키는 반드시 무효화(삭제)하거나 최신 값으로 갱신한다.
- **TTL 만료**: Redis `setex`/`expire`로 TTL을 두고, 만료 시 **키는 Redis가 자동 삭제**한다. 별도 “만료 시 비우는” 로직은 두지 않는다.
- **즉 비우기**: 수동/운영 목적의 비우기는 전용 API 또는 정책에 따라 실행한다.

### 1.2 세션 캐시 (session:{id}, sessions:list:*)

| 동작 | 즉시 처리 | TTL |
|------|-----------|-----|
| **세션 생성** | `invalidate_session_lists()` — 목록 캐시만 무효화 (새 세션은 아직 조회되지 않음) | - |
| **세션 삭제** | `invalidate_session(session_id)` + `invalidate_session_lists()` | - |
| **세션 제목 변경** | `invalidate_session(session_id)` + `invalidate_session_lists()` | - |
| **메시지 추가** | `set_session(session_id, 최신 result)`로 **갱신** + `invalidate_session_lists()`. 삭제 대신 덮어쓰기로 최신 유지 | - |
| **세션 단건 조회 (miss)** | MongoDB 조회 후 `set_session(session_id, ...)` 적재 | 60s |
| **세션 목록 조회 (miss)** | MongoDB 조회 후 `set_session_list(...)` 적재 | 300s |

**TTL**: `session:{id}` 60s, `sessions:list:*` 300s. 만료 후에는 Redis가 해당 키를 자동 삭제한다.

**즉 비우기(수동)**  
- 세션/목록 캐시는 “전체 비우기” API는 없다. 필요 시 Redis `SCAN sessions:*` 후 `DEL` 하거나, 운영에서 `FLUSHDB` 등 사용.
- 개별 무효화는 `session_cache.invalidate_session(id)`, `session_cache.invalidate_session_lists()` 로 이미 제공됨.

### 1.3 구현 위치

- **무효화 호출**: `playground/backend/routers/history_router.py`  
  - create → `invalidate_session_lists()`  
  - add_message → `set_session` + `invalidate_session_lists()`  
  - update_title → `invalidate_session(session_id)` + `invalidate_session_lists()`  
  - delete → `invalidate_session(session_id)` + `invalidate_session_lists()`  
- **캐시 로직**: `playground/backend/services/session_cache.py`  
  - `invalidate_session`, `invalidate_session_lists`, `invalidate_all`  
  - TTL 상수: `SESSION_TTL = 60`, `SESSION_LIST_TTL = 300`

### 1.4 MCP 세션 캐시

- MCP는 `session:{id}`를 **읽기 전용**으로 사용한다. 백엔드가 쓰기 시 무효화·갱신하므로, MCP는 만료·비우기 정책을 따로 두지 않는다.
- MCP 쪽에서 세션을 “비우는” 연산은 하지 않는다.

---

## 2. 추적 메트릭 기준

### 2.1 “측정할 필요가 있는 것”만 수집

다음만 Redis 등에 기록한다. 이외(요청 본문, 사용자 식별자, 비즈니스 데이터 등)는 메트릭으로 넣지 않는다.

| 메트릭 | 목적 | Redis 키(패턴) | TTL |
|--------|------|-----------------|-----|
| **요청 수** | 트래픽·부하 파악 | `metrics:requests:{minute}` | 3600s |
| **에러 수** | 안정성·장애 감지 | `metrics:errors:{minute}` | 3600s |
| **응답 시간** | 지연·품질·백분위 | `metrics:response_time` (Sorted Set) | 3600s |
| **엔드포인트별 통계** | 병목·이상 구간 파악 | `metrics:endpoint:{method}:{path}` (Hash: count, total_time_ms, errors) | 3600s |
| **토큰 사용량** | 비용·사용량(모델별) | `metrics:tokens:{model}` (Hash: input_tokens, output_tokens, total_tokens, request_count) | 86400s |
| **요청 상태** | 개별 요청 추적(디버깅·타임아웃 등) | `request:status:{request_id}` | 3600s |

이 목록에 없는 메트릭은 **기본적으로 수집하지 않는다**. 새로 넣을 때는 목적과 키·TTL을 이 기준에 맞춰 정한다.

### 2.2 수집 대상 경로

- **채팅만(기본·권장)** (`RECORD_METRICS_FOR_ALL_API` 없거나 false): `/api/chat`, `/api/chat/*` 에 대해서만 위 메트릭 기록. **모니터링은 챗(AI 활용)에 대한 것만 제대로 보여주면 되며, 다른 것은 필요 없다.**
- **전체 API** (`RECORD_METRICS_FOR_ALL_API=true`): `/api/*` 경로 전부. 운영·디버깅 목적일 때만 사용.

대상 경로만 바꿀 뿐, “무엇을 잴지”는 동일하다.

### 2.3 수집하지 않는 것

- 요청/응답 본문, 사용자 ID, 세션 내용, API 키·토큰 등 개인·비즈니스 데이터  
- “측정할 필요가 있는 것” 목록에 없는 새로운 메트릭(목적·기준 문서화 전에는 추가하지 않음)

### 2.4 메트릭 “즉 비우기”

- **수동 비우기**: `POST /api/monitoring/clear` 에서 `metrics:*`, `request:status:*` 등 정책에 정의된 메트릭 키만 삭제한다.
- **만료**: 각 키에 설정한 TTL이 지나면 Redis가 자동 삭제한다. 별도 “만료 시점에 비우는” 애플리케이션 로직은 두지 않는다.

---

## 3. 모니터링 = 챗(AI 활용) 전용

### 3.1 원칙

- **모니터링은 분산 시스템(Redis) 안에 있고, 챗(AI 활용)에 대한 것만 제대로 보여주면 된다.** 채팅 외 API 메트릭은 필요 없다.
- 수집·표시 모두 **챗(AI 활용) 구간**만 대상으로 한다. 기본값 `RECORD_METRICS_FOR_ALL_API=false` 유지.

### 3.2 대시보드에서 반드시 보여줄 세부 수치

다음 세부 수치는 모두 노출한다. “세부적으로 다 있어야 한다”는 이 목록을 의미한다.

| 구간 | 세부 수치 |
|------|-----------|
| **요약** | 총 요청 수(total_requests), 총 에러 수(total_errors), 에러율(error_rate %), 평균/최소/최대 응답시간(ms), P50/P95/P99 응답시간(ms), last_updated |
| **추세** | 분별 요청 수(requests), 분별 에러 수(errors) — 최근 60분 |
| **엔드포인트별** | 경로(endpoint), 메서드(method), 요청 수(count), 에러 수(errors), 평균 응답시간(avg_time_ms), 에러율(error_rate %) |
| **토큰(모델별)** | 모델명(model), 입력 토큰(input_tokens), 출력 토큰(output_tokens), 총 토큰(total_tokens), 요청 수(request_count), 요청당 평균 토큰(avg_tokens_per_request) |
| **응답시간 분포** | Min, P50, P95, P99, Max (ms) |
| **시스템** | Redis/Kafka 연결, Uptime |

구현 위치: `playground/frontend/src/app/monitoring/page.tsx`, `playground/backend/routers/monitoring_router.py`. API `GET /api/monitoring/dashboard` 는 위 세부 수치를 담은 `MonitoringDashboard` 를 반환한다.

**챗 상세 이력**(어떤 챗·결과·모델·시간대·토큰 등 자세한 1회 단위 이력)은 별도 설계 문서 `CHAT_HISTORY_METRICS.md`를 참고한다. 저장 형식(Redis `chat:record:*`, `chat:history`), 필드·보안, API `GET /api/monitoring/chat-history`, 대시보드 “챗 이력” 블록이 정의되어 있다.

---

## 4. 구현·설정과의 대응

| 항목 | 설정/코드 |
|------|-----------|
| 캐시 TTL | `SessionCacheService.SESSION_TTL`(60), `SESSION_LIST_TTL`(300) |
| 메트릭 대상 경로 | `_should_record_metrics(path)` — 기본 채팅만(`RECORD_METRICS_FOR_ALL_API` false) |
| 모니터링 대시보드 | 챗(AI 활용) 메트릭 전용, §3.2 세부 수치 전부 노출 |
| 메트릭 수동 비우기 | `routers/monitoring_router.py` → `POST /api/monitoring/clear` |
| 세션 캐시 무효화 | `session_cache.invalidate_session`, `invalidate_session_lists` (history_router에서 호출) |

이 문서는 캐시 “만료·즉 비우기”, “추적 메트릭 기준”, **모니터링 = 챗(AI 활용) 전용·세부 수치**를 코드와 맞추기 위한 정책 문서다.
