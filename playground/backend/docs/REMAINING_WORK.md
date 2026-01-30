# Playground 남은 계획 작업 (코드 → UI)

분산 시스템·모니터링 점검을 마친 뒤, 아래 순서로 진행할 작업 목록이다. **코드(백엔드)** 를 먼저, **UI(프론트)** 를 이어서 정리했다.

---

## 전제·정책 (이미 반영된 것)

- **제안 단계**: Agent가 제안하고 유저가 결정하거나, 유저가 직접 채우되, **방식은 오로지 챗으로만** (Claude Code처럼). UI 폼이 아닌 대화로만.
- **분산·모니터링**: 모든 HTTP 요청이 Redis(요청 상태·메트릭)를 거치며, 세션 API는 Redis 캐시 + MongoDB를 사용. 모니터링 라우터/대시보드는 Redis `metrics:*` 를 읽어 노출. (토큰 메트릭만 Agentic 경로에서 미기록 → 필요 시 스트림 종료 시 usage 훅으로 보완.)
- **그래프/노드**: 최대 6개. 정의·실행 단계 모두에서 검증.
- **Human-in-the-loop**: “도구 실행 전”, “코드 수정 같은 결정 직전”에 사용자 확인.

---

## 1단계: 백엔드 (코드)

### 1.1 제안 단계용 이벤트·페이로드

- [x] **제안 이벤트 타입**: SSE에 `proposal` 추가 (`orchestrator.EventType.PROPOSAL`, `nodes`/`pipeline`/`reason`).  
  - 예: `{"type": "proposal", "data": {"nodes": 3, "pipeline": ["rag","summarize","respond"], "reason": "..."}}`  
- [ ] **유저 입력 해석**: 챗 메시지로 “승인/수정/직접 스펙” 구분.  
  - 예: “그대로 해줘” → 승인, “노드 4개로” → 수정, “RAG → KG → 요약” → 직접 스펙.

### 1.2 Human-in-the-loop 훅

- [ ] **도구 실행 전 승인**: orchestrator에서 도구 실행 직전에 “실행/취소/다른 도구로” 선택을 기다리는 지점 추가.  
  - 이벤트 예: `{"type": "human_approval", "data": {"tool": "rag", "query": "...", "actions": ["run","cancel","change_tool"]}}`.  
- [ ] **코드/중요 결정 직전 확인**: 코드 수정·삭제 등 위험 작업 직전에도 동일 패턴으로 확인 요청.

### 1.3 그래프/노드 검증

- [x] **노드 최대 6개**: `MAX_NODES = 6`, `selected_tools[:MAX_NODES]` 검증.  
- [x] **쿼리 분산·역할**: 파이프라인=도구 순서, 역할=도구 이름. `GRAPH_NODE_RUN_SCHEMA.md` 참고.

### 1.4 Agentic 토큰 메트릭 (선택)

- [ ] **스트림 종료 시 usage 훅**: orchestrator 또는 chat_router에서 스트림 완료 시 `usage`(input_tokens, output_tokens)가 있으면 `ChatMonitoringMixin.log_chat_response(..., input_tokens=..., output_tokens=...)` 호출하여 Redis `metrics:tokens:{model}` 에 반영.

---

## 2단계: 프론트엔드 (UI)

### 2.1 의사결정 UI

- [x] **Intent/Confidence/선택 도구 표시**: SSE `intent` / `tool_select` 로 Decision 블록 UI 표시.  
- [x] **Human-in-the-loop 버튼**: `human_approval` 시 [Run][Cancel][Change tool] 버튼 노출.

### 2.2 제안 단계 = 챗 전용

- [ ] **제안 표시**: `proposal` 이벤트 또는 assistant 메시지 본문으로 “노드 N개, 파이프라인 […]” 같은 제안을 **챗 말풍선** 안에만 표시.  
- [ ] **유저 입력**: 폼/고정 입력란이 아니라 **챗 입력창**으로만 “승인/수정/직접 스펙” 입력.  
- [ ] **승인/수정/직접 스펙 구분**: 유저 메시지를 파싱해 백엔드에 `approved` / `modified` / `custom_spec` 등 플래그와 함께 전달.

### 2.3 기타 UI 정리

- [ ] **ToolCallDisplay / 스트리밍**: “Generating” + blink 등 이미 반영된 부분 유지.  
- [ ] **Feature 설명**: 문서가 아닌 **챗으로만** 설명 (이미 정책 반영).

---

## 3단계: 스키마·문서

- [ ] **그래프/노드/런 스키마**: 현재 DB·분산 시스템(MongoDB, Redis)을 전제로, “그래프 정의·노드·실행 로그” 등 스키마 초안 작성 및 기존 플로우와 정합성 검토.  
- [ ] **보안**: MongoDB `$regex` 이스케이프·Cypher 파라미터화 노트는 `SCHEMA_AND_FLOWS.md` 및 neo4j_adapter docstring에 반영 완료. 필요 시 추가 점검만.

---

## 진행 순서 요약

1. **백엔드**: 제안 이벤트·유저 입력 해석 → Human-in-the-loop 훅 → 노드 개수/역할 검증 → (선택) 토큰 메트릭 훅  
2. **프론트**: 의사결정 UI(intent/도구 표시 + 승인 버튼) → 제안 단계 챗 전용(제안 표시 + 챗 입력) → 나머지 UI 정리  
3. **문서/스키마**: 그래프·노드·런 스키마 초안 및 정합성 검토

이 목록은 `SCHEMA_AND_FLOWS.md`의 분산 시스템·모니터링 점검 결과를 바탕으로 작성되었으며, 제안 단계 = 챗 전용 정책과 human-in-the-loop 기준을 따른다.
