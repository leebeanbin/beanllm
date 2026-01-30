# 현재 상태 분석 및 테스트 계획

## 📊 현재 상태 분석

### 1. Orchestrator Handlers 현황

**총 Handlers**: 15개

| Handler | 구현 상태 | Facade 직접 호출 | MCP Tool 매핑 가능 여부 |
|---------|----------|-----------------|----------------------|
| `_handle_chat` | ✅ 구현됨 | ✅ `Client` 직접 호출 | ⚠️ MCP tool 없음 (필요시 생성) |
| `_handle_rag` | ✅ 구현됨 | ✅ `RAGChain` 직접 호출 | ✅ `query_rag_system` |
| `_handle_agent` | ❌ TODO | ❌ 없음 | ✅ `run_agent_task` (확인 필요) |
| `_handle_multi_agent` | ❌ TODO | ❌ 없음 | ✅ `run_multiagent_task` |
| `_handle_web_search` | ❌ TODO | ❌ 없음 | ⚠️ MCP tool 없음 |
| `_handle_code` | ✅ 구현됨 | ✅ `_handle_chat` 재사용 | ⚠️ MCP tool 없음 |
| `_handle_google_drive` | ✅ 구현됨 | ✅ MCP tool 사용 중 | ✅ `save_to_google_drive` |
| `_handle_google_docs` | ✅ 구현됨 | ✅ MCP tool 사용 중 | ✅ `export_to_google_docs` |
| `_handle_google_gmail` | ✅ 구현됨 | ✅ MCP tool 사용 중 | ✅ `share_via_gmail` |
| `_handle_google_calendar` | ⚠️ 부분 구현 | ❌ TODO | ⚠️ MCP tool 없음 |
| `_handle_google_sheets` | ⚠️ 부분 구현 | ❌ TODO | ⚠️ MCP tool 없음 |
| `_handle_audio` | ❌ TODO | ❌ 없음 | ✅ `transcribe_audio` |
| `_handle_vision` | ❌ TODO | ❌ 없음 | ⚠️ MCP tool 없음 |
| `_handle_ocr` | ❌ TODO | ❌ 없음 | ✅ `extract_text_from_image` |
| `_handle_knowledge_graph` | ❌ TODO | ❌ 없음 | ✅ `query_knowledge_graph` |
| `_handle_evaluation` | ❌ TODO | ❌ 없음 | ✅ `evaluate_model` |

**요약**:
- ✅ 구현됨: 5개 (chat, rag, code, google_drive, google_docs, google_gmail)
- ⚠️ 부분 구현: 2개 (google_calendar, google_sheets)
- ❌ 미구현: 8개 (agent, multi_agent, web_search, audio, vision, ocr, kg, evaluation)

---

### 2. MCP Tools 현황

**총 MCP Tools**: 28개

#### RAG Tools (5개)
- `build_rag_system` - RAG 시스템 구축
- `query_rag_system` - RAG 질의 ✅
- `add_documents_to_rag` - 문서 추가
- `delete_rag_system` - RAG 시스템 삭제
- `list_rag_systems` - RAG 시스템 목록

#### Agent Tools (5개)
- `create_multiagent_system` - 멀티 에이전트 시스템 생성
- `run_multiagent_task` - 멀티 에이전트 작업 실행 ✅
- `get_multiagent_history` - 대화 히스토리 조회
- `delete_multiagent_system` - 시스템 삭제
- `list_multiagent_systems` - 시스템 목록

#### Knowledge Graph Tools (6개)
- `build_knowledge_graph` - 지식 그래프 구축
- `query_knowledge_graph` - 지식 그래프 질의 ✅
- `add_documents_to_kg` - 문서 추가
- `get_entities` - 엔티티 조회
- `get_relations` - 관계 조회
- `delete_knowledge_graph` - 그래프 삭제

#### ML Tools (7개)
- `transcribe_audio` - 음성 전사 ✅
- `batch_transcribe_audio` - 배치 전사
- `extract_text_from_image` - OCR ✅
- `batch_extract_text` - 배치 OCR
- `evaluate_model` - 모델 평가 ✅
- `run_benchmark` - 벤치마크 실행
- `compare_models` - 모델 비교

#### Google Tools (5개)
- `export_to_google_docs` - Google Docs 내보내기 ✅
- `save_to_google_drive` - Google Drive 저장 ✅
- `share_via_gmail` - Gmail 공유 ✅
- `list_google_drive_files` - 파일 목록
- `get_google_drive_file` - 파일 조회

---

### 3. 중복 코드 분석

#### Facade 직접 호출 위치

**orchestrator.py**:
```python
# Line 247
from beanllm.facade.core import Client

# Line 324
from beanllm.facade.core import RAGChain
```

**mcp_server/tools/**:
- 모든 tools에서 beanllm Facade 사용 (정상, 중앙 관리 포인트)

**문제점**:
- `orchestrator._handle_chat`: `Client` 직접 호출 → MCP tool 없음
- `orchestrator._handle_rag`: `RAGChain` 직접 호출 → `query_rag_system` 사용 가능

---

### 4. 매핑 테이블

| Orchestrator Handler | MCP Tool | 상태 | 우선순위 |
|----------------------|----------|------|---------|
| `_handle_rag` | `query_rag_system` | ✅ 매핑 가능 | 높음 |
| `_handle_multi_agent` | `run_multiagent_task` | ✅ 매핑 가능 | 높음 |
| `_handle_agent` | `run_agent_task` | ⚠️ 확인 필요 | 중간 |
| `_handle_knowledge_graph` | `query_knowledge_graph` | ✅ 매핑 가능 | 중간 |
| `_handle_audio` | `transcribe_audio` | ✅ 매핑 가능 | 중간 |
| `_handle_ocr` | `extract_text_from_image` | ✅ 매핑 가능 | 중간 |
| `_handle_evaluation` | `evaluate_model` | ✅ 매핑 가능 | 낮음 |
| `_handle_google_drive` | `save_to_google_drive` | ✅ 이미 사용 중 | - |
| `_handle_google_docs` | `export_to_google_docs` | ✅ 이미 사용 중 | - |
| `_handle_google_gmail` | `share_via_gmail` | ✅ 이미 사용 중 | - |
| `_handle_chat` | - | ⚠️ MCP tool 없음 | 낮음 |
| `_handle_web_search` | - | ⚠️ MCP tool 없음 | 낮음 |
| `_handle_vision` | - | ⚠️ MCP tool 없음 | 낮음 |
| `_handle_code` | - | ⚠️ MCP tool 없음 | 낮음 |
| `_handle_google_calendar` | - | ⚠️ MCP tool 없음 | 낮음 |
| `_handle_google_sheets` | - | ⚠️ MCP tool 없음 | 낮음 |

---

## 🧪 테스트 계획

### Phase 1: 현재 상태 검증

#### 1.1 중복 코드 확인

**테스트**:
```bash
# Facade 직접 호출 확인
grep -r "from beanllm.facade" playground/backend/services/orchestrator.py

# 예상 결과: 2개 (Client, RAGChain)
```

**체크리스트**:
- [ ] Facade 직접 호출 위치 확인
- [ ] 중복 코드 라인 수 계산
- [ ] MCP tools와 비교

#### 1.2 MCP Tools 목록 확인

**테스트**:
```bash
# 모든 MCP tools 목록
grep -r "@mcp.tool()" mcp_server/tools/ | wc -l

# 각 파일별 tools 수
grep -r "@mcp.tool()" mcp_server/tools/rag_tools.py | wc -l
grep -r "@mcp.tool()" mcp_server/tools/agent_tools.py | wc -l
grep -r "@mcp.tool()" mcp_server/tools/kg_tools.py | wc -l
grep -r "@mcp.tool()" mcp_server/tools/ml_tools.py | wc -l
grep -r "@mcp.tool()" mcp_server/tools/google_tools.py | wc -l
```

**체크리스트**:
- [ ] 총 tools 수 확인 (예상: 28개)
- [ ] 각 카테고리별 tools 수 확인
- [ ] tools 이름 목록 작성

#### 1.3 Orchestrator Handlers 확인

**테스트**:
```bash
# Handlers 목록
grep -r "async def _handle_" playground/backend/services/orchestrator.py

# TODO/미구현 확인
grep -r "TODO\|not yet implemented\|not implemented" playground/backend/services/orchestrator.py -i
```

**체크리스트**:
- [ ] 총 handlers 수 확인 (예상: 15개)
- [ ] 구현 상태 확인
- [ ] TODO 항목 목록 작성

---

### Phase 2: 매핑 검증

#### 2.1 Handler → MCP Tool 매핑

**테스트**:
각 handler에 대해:
1. MCP tool 존재 여부 확인
2. 매핑 가능 여부 확인
3. 매핑 테이블 작성

**체크리스트**:
- [ ] `_handle_rag` → `query_rag_system` 매핑 확인
- [ ] `_handle_multi_agent` → `run_multiagent_task` 매핑 확인
- [ ] `_handle_agent` → MCP tool 확인
- [ ] `_handle_knowledge_graph` → `query_knowledge_graph` 매핑 확인
- [ ] `_handle_audio` → `transcribe_audio` 매핑 확인
- [ ] `_handle_ocr` → `extract_text_from_image` 매핑 확인
- [ ] `_handle_evaluation` → `evaluate_model` 매핑 확인

#### 2.2 MCP Tool 함수 시그니처 확인

**테스트**:
각 MCP tool의 함수 시그니처 확인:
- 파라미터 이름
- 파라미터 타입
- 반환 타입
- 세션 ID 지원 여부

**체크리스트**:
- [ ] `query_rag_system` 시그니처 확인
- [ ] `run_multiagent_task` 시그니처 확인
- [ ] `query_knowledge_graph` 시그니처 확인
- [ ] `transcribe_audio` 시그니처 확인
- [ ] `extract_text_from_image` 시그니처 확인
- [ ] `evaluate_model` 시그니처 확인

---

### Phase 3: 통합 테스트 계획

#### 3.1 MCP Client Service 테스트

**테스트 시나리오**:
1. MCP Client Service 생성
2. `call_tool` 메서드 테스트
3. 각 MCP tool 호출 테스트

**체크리스트**:
- [ ] `mcp_client_service.py` 파일 생성 가능 여부 확인
- [ ] `call_tool` 메서드 구현 가능 여부 확인
- [ ] `_get_tool_function` 메서드 구현 가능 여부 확인
- [ ] 세션 ID 전달 테스트

#### 3.2 Orchestrator 수정 테스트

**테스트 시나리오**:
1. 각 handler를 MCP Client 사용으로 수정
2. 기존 동작 유지 확인
3. 새로운 기능 동작 확인

**체크리스트**:
- [ ] `_handle_rag` 수정 테스트
- [ ] `_handle_multi_agent` 수정 테스트
- [ ] `_handle_knowledge_graph` 수정 테스트
- [ ] `_handle_audio` 수정 테스트
- [ ] `_handle_ocr` 수정 테스트
- [ ] `_handle_evaluation` 수정 테스트

---

## 📋 검증 체크리스트

### 구현 전 검증

- [ ] 현재 상태 분석 완료
- [ ] 중복 코드 위치 확인
- [ ] MCP tools 목록 확인
- [ ] Handler → MCP Tool 매핑 테이블 작성
- [ ] 테스트 계획 수립

### 구현 중 검증

- [ ] MCP Client Service 생성
- [ ] 각 handler 수정
- [ ] 기존 기능 동작 확인
- [ ] 새로운 기능 동작 확인

### 구현 후 검증

- [ ] Facade 직접 호출 제거 확인
- [ ] MCP Client 사용 확인
- [ ] 모든 handlers 동작 확인
- [ ] 통합 테스트 통과

---

## 🔍 상세 분석 결과

### 중복 코드 위치

**orchestrator.py**:
- Line 247-261: `_handle_chat` - `Client` 직접 호출
- Line 324-395: `_handle_rag` - `RAGChain` 직접 호출

**중복 코드 라인 수**: 약 100줄

### MCP Tools 상세 목록

**RAG Tools** (5개):
1. `build_rag_system`
2. `query_rag_system` ✅
3. `add_documents_to_rag`
4. `delete_rag_system`
5. `list_rag_systems`

**Agent Tools** (5개):
1. `create_multiagent_system`
2. `run_multiagent_task` ✅
3. `get_multiagent_history`
4. `delete_multiagent_system`
5. `list_multiagent_systems`

**KG Tools** (6개):
1. `build_knowledge_graph`
2. `query_knowledge_graph` ✅
3. `add_documents_to_kg`
4. `get_entities`
5. `get_relations`
6. `delete_knowledge_graph`

**ML Tools** (7개):
1. `transcribe_audio` ✅
2. `batch_transcribe_audio`
3. `extract_text_from_image` ✅
4. `batch_extract_text`
5. `evaluate_model` ✅
6. `run_benchmark`
7. `compare_models`

**Google Tools** (5개):
1. `export_to_google_docs` ✅
2. `save_to_google_drive` ✅
3. `share_via_gmail` ✅
4. `list_google_drive_files`
5. `get_google_drive_file`

**총계**: 28개 tools

---

## 🎯 우선순위 매트릭스

| Handler | 구현 상태 | MCP Tool | 우선순위 | 이유 |
|---------|----------|----------|---------|------|
| `_handle_rag` | ✅ | ✅ | 높음 | 이미 구현, 중복 코드 제거 |
| `_handle_multi_agent` | ❌ | ✅ | 높음 | 미구현, MCP tool 있음 |
| `_handle_agent` | ❌ | ⚠️ | 중간 | 미구현, MCP tool 확인 필요 |
| `_handle_knowledge_graph` | ❌ | ✅ | 중간 | 미구현, MCP tool 있음 |
| `_handle_audio` | ❌ | ✅ | 중간 | 미구현, MCP tool 있음 |
| `_handle_ocr` | ❌ | ✅ | 중간 | 미구현, MCP tool 있음 |
| `_handle_evaluation` | ❌ | ✅ | 낮음 | 미구현, MCP tool 있음 |
| `_handle_chat` | ✅ | ❌ | 낮음 | 이미 구현, MCP tool 없음 |
| `_handle_web_search` | ❌ | ❌ | 낮음 | 미구현, MCP tool 없음 |
| `_handle_vision` | ❌ | ❌ | 낮음 | 미구현, MCP tool 없음 |

---

## 📝 다음 단계

1. **MCP Client Service 생성** (우선순위 높음)
2. **핵심 handlers 수정** (rag, multi_agent)
3. **나머지 handlers 수정** (kg, audio, ocr, evaluation)
4. **테스트 및 검증**
