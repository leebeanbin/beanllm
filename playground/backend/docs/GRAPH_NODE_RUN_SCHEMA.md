# 그래프·노드·런 스키마 초안

현재 **DB·분산 시스템**(MongoDB, Redis)을 전제로, Agentic 파이프라인에서 쓰는 **그래프 정의·노드·실행(런)·로그** 개념을 스키마 수준으로 정리한 초안이다.

---

## 1. 전제·용어

- **그래프**: 한 번의 Agentic 실행에서의 “노드 순서” = **파이프라인**(ordered list of tools). 노드는 **최대 6개**.
- **노드**: 도구 한 개와 1:1. 역할(예: rag, kg, summarize, respond)은 도구 이름으로 표현.
- **런(run)**: 그래프(파이프라인)를 한 번 실행한 것 = `POST /api/chat/agentic` 한 번에 대응.
- **로그**: 런의 요약·결과를 나중에 조회·감사할 때 쓸 수 있도록 남기는 메타데이터.

---

## 2. 그래프 정의 (현재 구현 기준)

코드에서는 **그래프를 별도 컬렉션에 저장하지 않고**, 매 요청마다 **intent → tool_select → proposal** 으로 파이프라인을 결정한다.

| 개념 | 현재 표현 | 비고 |
|------|-----------|------|
| 그래프 id | 없음 (요청 단위) | 필요 시 `run_id` 또는 `request_id` 로 구분 |
| 노드 목록 | `context.selected_tools` → `tool_names[]` | 최대 6개, `orchestrator.MAX_NODES` |
| 노드 역할 | 도구 이름 = 역할 (rag, kg, chat, …) | `tool_registry` 의 도구 이름과 동일 |
| 엣지 | 순서대로 실행 (선형) | 분기·병렬은 `execute_parallel` 등으로 확장 가능 |

**제안 단계**: `proposal` 이벤트에 `nodes`, `pipeline`, `reason` 을 담아 챗 말풍선으로만 전달. 유저는 챗으로 승인/수정/직접 스펙(`proposal_action`, `proposal_pipeline`) 보냄.

---

## 3. 노드 스키마 (논리 모델)

나중에 “그래프 정의”를 DB에 둘 때 쓸 수 있는 노드 필드 예시는 아래와 같다.

```
Node (논리)
  - id: str (또는 index 0..N-1)
  - role: str          # "rag" | "kg" | "summarize" | "respond" | ...
  - tool_name: str      # tool_registry와 동일
  - config: {}          # 선택 (top_k, collection_name 등)
```

현재는 **런 타임**에만 `OrchestratorContext.selected_tools` 로 표현하며, 상시 저장 스키마는 없다.

---

## 4. 런(run) 스키마 (논리 모델)

한 번의 Agentic 실행을 표현할 때 쓰일 수 있는 필드 예시다.  
**저장소**: 필요 시 MongoDB `chat_sessions` 보조 정보, 또는 별도 컬렉션 `agentic_runs` 등.

| 필드 | 타입 | 설명 |
|------|------|------|
| run_id | str | 요청/런 구분자 (예: `request_id` 또는 uuid) |
| session_id | str | 세션 (있을 때) |
| query | str | 사용자 쿼리 |
| intent | str | primary_intent |
| pipeline | [str] | 도구 이름 순서, 최대 6개 |
| started_at | datetime | 시작 시각 |
| finished_at | datetime | 종료 시각 (선택) |
| status | str | "running" \| "completed" \| "error" \| "cancelled" |
| proposal_action | str | "approved" \| "modified" \| "custom_spec" (유저가 챗으로 보낸 값) |

**Redis**: 런 중간 상태는 이미 `request:status:{request_id}` 로 모니터링용 저장 중. 상세 런 메타를 쓸 때는 TTL 부여해서 `run:meta:{run_id}` 같은 키에 넣을 수 있음.

---

## 5. 로그·메트릭과의 관계

- **요청/메트릭**: `request:status:*`, `metrics:*` (Redis) — 기존 모니터링 미들웨어와 동일.
- **세션/메시지**: MongoDB `chat_sessions` + Redis `session:*` — 기존 스키마 유지.
- **런 로그**: 위 “런 스키마”를 채워서 MongoDB 또는 Redis에 남기면, “언제 어떤 파이프라인으로 실행했는지”를 나중에 조회·통계 낼 수 있음.  
  → 구현 시 `orchestrator.execute()` 시작/끝에서 `run_id`, `pipeline`, `status` 등을 한 번씩 기록하면 된다.

---

## 6. 기존 플로우와의 정합성

| 항목 | 현재 구현 | 이 스키마 초안 |
|------|-----------|-----------------|
| 노드 개수 | `MAX_NODES = 6`, `selected_tools[:6]` | 동일. 검증은 orchestrator 진입 시 |
| 파이프라인 | `tool_names` = 선택된 도구 순서 | `pipeline` 필드와 동일 |
| 제안/승인 | `proposal` 이벤트 + `proposal_action` / `proposal_pipeline` (챗 전용) | 스키마에서 `proposal_action` 으로 표현 |
| 저장 | 런 단위 영구 저장 없음 | 필요 시 `agentic_runs` 또는 `run:meta:*` 추가 |

이 문서는 **그래프/노드/런**을 나중에 DB·Redis에 반영할 때의 기준으로 쓰면 된다. 실제 컬렉션/키 설계는 구현 단계에서 위 논리 모델을 따르면 된다.
