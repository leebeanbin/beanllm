# 챗 상세 이력 — 저장·표시 설계

모니터링 대시보드에서 **어떤 챗을 남겼는지, 어떤 결과를 받았는지, 어떤 모델을 썼는지, 어느 시간대에 많이 했는지, 토큰을 얼마나 써서 최종 몇 개인지**를 자세히 보여주기 위한 저장·API·UI 설계.

---

## 1. 목표

| 항목 | 설명 |
|------|------|
| **어떤 챗** | 요청 본문 요약(첫 메시지 100자 내외, 마스킹 옵션) |
| **어떤 결과** | 응답 요약(200자 내외, 마스킹 옵션) |
| **어떤 모델** | `model` 필드 |
| **어느 시간대** | 분 단위 버킷, 시간대별 집계 |
| **토큰** | 입력/출력/합계, 요청당 평균, **최종 누적 수** |
| **기타** | 소요시간(ms), 요청 시각, 에러 여부 |

---

## 2. 저장 대상·형식

### 2.1 선택지

| 방식 | 장점 | 단점 |
|------|------|------|
| **Redis List/Sorted Set** | 이미 Redis 사용 중, 조회 빠름, TTL로 자동 정리 | 메모리 제한, 영구 보관 아님 |
| **MongoDB 컬렉션** | 영구 보관, 복잡 쿼리·페이징 용이 | 채팅 세션 DB와 분리 관리 필요 |
| **기존 chat_sessions 확장** | 한 곳에 모음 | 세션 = 대화 스레드, “챗 1회” 단위가 아니라 혼동 가능 |
| **Kafka + Redis 캐시** | 이미 `chat.request`/`chat.response` 이벤트 있음, 재가공 가능 | 연동·구독 구조 필요 |

**권장**: **Redis 기반 “챗 1회” 단위 레코드**로 먼저 구현하고, 필요 시 MongoDB/이벤트 재가공으로 확장.

### 2.2 Redis 스키마 (권장)

- **챗 1회 레코드**  
  - 키: `chat:record:{request_id}`  
  - 타입: Hash  
  - 필드: `model`, `at_ts`(Unix sec), `at_minute`(분 버킷), `request_preview`, `response_preview`, `input_tokens`, `output_tokens`, `total_tokens`, `duration_ms`, `path`, `error`(있을 때만)  
  - TTL: 86400(1일) 또는 3600(1시간). 정책에 따라 설정.

- **시간순 인덱스**  
  - 키: `chat:history`  
  - 타입: Sorted Set  
  - 멤버: `request_id`, 점수: `at_ts`  
  - TTL: 86400.  
  - 조회: `ZREVRANGE chat:history 0 (N-1)` → 최근 N건 `request_id` 나온 뒤 각각 `chat:record:{id}` HGETALL.

---

## 3. 남길 필드·보안

### 3.1 필드 목록

| 필드 | 의미 | 저장 시점 | 비고 |
|------|------|-----------|------|
| `request_id` | 요청 식별자 | 요청 진입 시 | 키에 포함 |
| `model` | 사용 모델명 | 요청·응답 모두 가능 | log_chat_request/response 시점 |
| `at_ts` | 완료 시각(Unix sec) | 응답 완료 시 | 정렬·시간대 집계용 |
| `at_minute` | 분 버킷 (`ts // 60`) | 응답 완료 시 | “어느 시간대에 많이 했는지”용 |
| `request_preview` | 요청 요약 | log_chat_request 또는 응답 시 | 아래 보안 규칙 |
| `response_preview` | 응답 요약 | log_chat_response 시 | 아래 보안 규칙 |
| `input_tokens` | 입력 토큰 수 | log_chat_response 시 | |
| `output_tokens` | 출력 토큰 수 | log_chat_response 시 | |
| `total_tokens` | 합계 | log_chat_response 시 | “최종 몇 개”에 사용 |
| `duration_ms` | 소요 시간(ms) | 응답 완료 시 | |
| `path` | API 경로 | 미들웨어 이미 기록 | `/api/chat`, `/api/chat/agentic` 등 |
| `error` | 에러 시 한해 플래그/메시지 | 응답 완료 시 | |

### 3.2 보안(본문 노출·마스킹)

- **request_preview**  
  - 저장: 마지막 user 메시지 `content` 일부만 사용.  
  - 문자열이면 앞 100자, 배열(멀티모달)이면 첫 텍스트 블록 100자.  
  - 선택: 환경 변수 `CHAT_HISTORY_MASK_PREVIEW=true` 이면 `"(마스킹)"` 등 고정 문자열로 저장하거나 저장 생략.

- **response_preview**  
  - 저장: `response_content` 앞 200자.  
  - `CHAT_HISTORY_MASK_PREVIEW=true` 이면 저장 생략 또는 `"(마스킹)"`.

- **로그·API**  
  - 대시보드 API는 서버 내부/관리자 전용으로 두고, 본문이 나가는 응답은 마스킹 옵션과 동일하게 적용.

---

## 4. 기록 시점·연동

- **요청 진입** (미들웨어): `request:status:{request_id}` 는 기존대로.  
- **챗 요청 로깅** (`log_chat_request`):  
  - Redis에 `chat:record:{request_id}` 초기 생성(HSET): `model`, `at_ts`(현재 시각으로 잡거나 나중에 응답 시 덮어씀), `request_preview`.  
  - `chat:history` 에 `ZADD chat:history {at_ts} {request_id}` (이미 있으면 나중에 응답 시 한 번 더 ZADD로 점수만 갱신해도 됨).  
- **챗 응답 로깅** (`log_chat_response`):  
  - `chat:record:{request_id}` 에 `response_preview`, `input_tokens`, `output_tokens`, `total_tokens`, `duration_ms`, `at_ts`(완료 시각), `at_minute` 갱신.  
  - 기존 `metrics:tokens:{model}` 집계는 그대로 유지.

미들웨어는 “응답 완료” 시점에 path/status_code/error 여부를 이미 알고 있으므로, `log_chat_response` 호출 직전/직후에 같은 request_id에 대해 `duration_ms`·`path`·`error`를 채우는 방식으로 연동하면 됨.  
**구현 위치 제안**:  
- Redis 키 생성/갱신: `ChatMonitoringMixin.log_chat_request` / `log_chat_response` 내부, 또는 미들웨어의 “응답 완료” 블록에서 `log_chat_response` 호출하는 쪽과 동일한 레이어에 “챗 이력 한 건 쓰기” 함수를 두고, `log_chat_request`(또는 요청 직후)에서 초기 레코드, `log_chat_response`(또는 응답 직후)에서 나머지 필드 갱신.

---

## 5. API·UI

### 5.1 API

- **GET /api/monitoring/chat-history**  
  - Query: `minutes`(기본 60), `limit`(기본 50).  
  - 응답: 최근 `minutes` 분 내 완료된 챗 레코드 목록.  
  - 각 항목: `request_id`, `model`, `at_ts`, `at_minute`, `request_preview`, `response_preview`, `input_tokens`, `output_tokens`, `total_tokens`, `duration_ms`, `path`, `error`.  
  - 정렬: `at_ts` 내림차순, 상위 `limit` 건.

### 5.2 UI

- **모니터링 대시보드 내 “챗 이력” 블록**  
  - 표: 시간, 모델, 요청 요약, 응답 요약, 토큰(입력/출력/합계), 소요시간(ms).  
  - “어느 시간대에 많이 했는지”는 기존 추세 차트(분별 요청 수) + 이 표의 `at_minute`으로 확인 가능.  
  - “최종 토큰 몇 개”는 각 행의 `total_tokens` 및 요약 영역 “총 토큰” 합계로 표시.

- **선택**: “챗 이력”만 별도 탭/페이지로 두고, 상세 필터(모델·시간대·토큰 구간) 제공.

---

## 6. 구현·설정과의 대응

| 항목 | 설정/코드 |
|------|-----------|
| 챗 이력 저장 여부 | **상시 수집**. env 없음. Redis 있으면 항상 `chat:record:*`, `chat:history` 기록 |
| 미리보기 마스킹 | `CHAT_HISTORY_MASK_PREVIEW=true` 이면 request/response preview 저장 생략 또는 고정문 |
| TTL | `chat:record:*` 86400, `chat:history` 86400 (필요 시 3600 등으로 조정) |
| 기록 호출 | `log_chat_request` 에서 초기 레코드 + `chat:history` ZADD, `log_chat_response` 에서 나머지 필드 갱신 |
| 조회 API | `routers/monitoring_router.py` → `GET /api/monitoring/chat-history` |
| 대시보드 블록 | `playground/frontend/src/app/monitoring/page.tsx` 에 “챗 이력” 테이블·필터 |

---

이 문서는 **챗 상세 이력**의 저장 형식, 필드·보안, API·UI 방안을 코드와 맞추기 위한 설계 문서다.  
정책 보강 시 `CACHE_AND_METRICS_POLICY.md` §3(모니터링 = 챗 전용)과 함께 참고한다.
