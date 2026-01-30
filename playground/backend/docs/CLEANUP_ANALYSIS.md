# Playground Backend 코드 정리 분석

## 📊 발견된 문제점

### 1. 레거시/중복 코드 (우선순위 높음)

#### 1.1 mcp_streaming.py (삭제됨 ✅ 2025-01-25)
~~**위치**: `playground/backend/mcp_streaming.py` (714줄)~~

**완료된 조치**:
- ✅ `mcp_streaming.py` 삭제됨
- ✅ `/api/chat/stream` 엔드포인트가 `chat_router.py`로 통합됨
- ✅ MCP Client Service (`services/mcp_client_service.py`)가 대체함
- ✅ `orchestrator.py`가 모든 Tool 실행 담당

**감소된 코드**: 714줄

---

#### 1.2 main.py의 중복 엔드포인트 (우선순위 높음) ⚠️

**현재 상태**: `main.py`에 24개 엔드포인트, Routers에도 동일한 엔드포인트 존재

**중복 엔드포인트 목록**:

| main.py 엔드포인트 | Router 엔드포인트 | 라인 | 상태 |
|-------------------|------------------|------|------|
| `/api/rag_debug/analyze` | `rag_router.py` `/debug` | 1342 | 중복 |
| `/api/optimizer/optimize` | `optimizer_router.py` | 1407 | 중복 |
| `/api/multi_agent/run` | `agent_router.py` | 1444 | 중복 |
| `/api/orchestrator/run` | `agent_router.py` | 1606 | 중복 |
| `/api/chain/run` | `chain_router.py` | 1670 | 중복 |
| `/api/chain/build` | `chain_router.py` | 1708 | 중복 |
| `/api/vision_rag/build` | `vision_router.py` | 1745 | 중복 |
| `/api/vision_rag/query` | `vision_router.py` | 1846 | 중복 |
| `/api/audio/transcribe` | `audio_router.py` | 1897 | 중복 |
| `/api/audio/synthesize` | `audio_router.py` | 1928 | 중복 |
| `/api/audio/rag` | `audio_router.py` | 1956 | 중복 |
| `/api/evaluation/evaluate` | `evaluation_router.py` | 1996 | 중복 |
| `/api/finetuning/create` | `finetuning_router.py` | 2069 | 중복 |
| `/api/finetuning/status/{job_id}` | `finetuning_router.py` | 2114 | 중복 |
| `/api/ocr/recognize` | `ocr_router.py` | 2205 | 중복 |
| `/api/chat/export/docs` | `google_auth_router.py` | 2356 | 중복 |
| `/api/chat/save/drive` | `google_auth_router.py` | 2443 | 중복 |
| `/api/chat/share/email` | `google_auth_router.py` | 2541 | 중복 |

**권장 조치**:
- ✅ `main.py`에서 중복 엔드포인트 제거
- ✅ Routers만 사용

**예상 감소**: 약 1,200줄 (2,704줄 → 1,500줄)

---

#### 1.3 common.py와 main.py의 중복 전역 상태

**문제점**:
- `_rag_debugger`: `common.py` Line 41, `main.py` Line 233
- `_downloaded_models`: `common.py` Line 51, `main.py` Line 84
- `get_rag_debugger()`: `common.py` Line 97, `main.py` Line 270 (시그니처 다름)

**권장 조치**:
- ✅ `common.py`로 통일
- ✅ `main.py`에서 중복 제거

**예상 감소**: 약 50줄

---

### 2. 미구현 코드 (TODO)

**위치**: `playground/backend/services/orchestrator.py`

**TODO 항목** (14개):
1. Line 417: `_handle_agent` - "beanllm Agent Facade 연동"
2. Line 443: `_handle_multi_agent` - "beanllm MultiAgent Facade 연동"
3. Line 469: `_handle_web_search` - "beanllm WebSearch Facade 연동"
4. Line 758: `_handle_google_calendar` - "Google Calendar API 연동"
5. Line 810: `_handle_google_sheets` - "Google Sheets API 연동"
6. Line 843: `_handle_audio` - "beanllm Audio Facade 연동"
7. Line 863: `_handle_vision` - "Vision tool not yet implemented"
8. Line 878: `_handle_ocr` - "OCR tool not yet implemented"
9. Line 893: `_handle_knowledge_graph` - "Knowledge Graph tool not yet implemented"
10. Line 908: `_handle_evaluation` - "Evaluation tool not yet implemented"

**권장 조치**:
- MCP Client Service 생성 후 MCP tools로 구현
- TODO 제거

---

### 3. 빈/미사용 파일

#### 3.1 ml_router.py
**위치**: `playground/backend/routers/ml_router.py`

**문제점**:
- `# TODO: Add endpoints here`만 있음
- 실제 구현 없음
- 다른 routers에 이미 구현됨 (audio_router, vision_router, evaluation_router 등)

**권장 조치**:
- ✅ 삭제

---

#### 3.2 notebooks/
**위치**: `playground/backend/notebooks/`

**문제점**:
- 테스트용 Jupyter notebooks
- 프로덕션 코드에 포함 불필요

**권장 조치**:
- `.gitignore`에 추가 또는 별도 디렉토리로 이동

---

### 4. 중복된 RAG Debug 코드

**문제점**:
- `common.py`에 `get_rag_debugger()` 함수 (Line 97)
- `main.py`에도 `get_rag_debugger()` 함수 (Line 270, 시그니처 다름)
- `rag_router.py`에도 RAG Debug 엔드포인트
- `main.py`에도 RAG Debug 엔드포인트 (중복)

**권장 조치**:
- ✅ `common.py`로 통일
- ✅ `main.py`의 중복 제거

---

### 5. 사용되지 않는 Import

**문제점**:
- `main.py`에 많은 import가 있지만 Routers로 이동하면서 불필요할 수 있음

**예시**:
```python
# main.py Line 136-151
from beanllm import Client
from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph
from beanllm.facade.core.rag_facade import RAGChain, RAGBuilder
# ... 등등
```

**권장 조치**:
- 중복 엔드포인트 제거 후 사용되지 않는 import 제거

---

### 6. 주석 처리된 섹션

**문제점**:
- `main.py`에 "Moved to routers/..." 주석만 있고 실제 코드는 남아있음

**예시**:
```python
# ============================================================================
# Knowledge Graph API - Moved to routers/kg_router.py
# ============================================================================
# (하지만 실제 코드는 없음, 주석만)
```

**권장 조치**:
- 주석은 유지 (참고용)
- 실제 중복 코드만 제거

---

## 📋 정리 우선순위

### 높음 (즉시 정리)

1. **mcp_streaming.py 삭제 또는 레거시 표시**
   - `orchestrator.py`로 대체 가능
   - `/api/chat/stream` 엔드포인트를 `chat_router.py`로 이동
   - **예상 감소**: 714줄

2. **main.py의 중복 엔드포인트 제거**
   - Routers로 이동한 엔드포인트 삭제
   - **예상 감소**: 약 1,200줄

3. **common.py와 main.py의 중복 전역 상태 통일**
   - `common.py`로 통일
   - **예상 감소**: 약 50줄

4. **ml_router.py 삭제**
   - 빈 파일, 다른 routers에 구현됨

### 중간 (구현 후 정리)

5. **orchestrator.py의 TODO 구현**
   - MCP Client Service 생성 후
   - MCP tools로 구현

### 낮음 (선택적)

6. **notebooks/ 디렉토리 정리**
   - `.gitignore` 추가 또는 별도 디렉토리

7. **사용되지 않는 import 제거**
   - 각 파일별로 확인 필요

---

## 🔍 상세 분석

### main.py 크기 분석

**현재**: 2,704줄

**중복 엔드포인트 라인 범위**:
- RAG Debug: 1342-1399 (58줄)
- Optimizer: 1407-1436 (30줄)
- Multi-Agent: 1444-1598 (155줄)
- Orchestrator: 1606-1662 (57줄)
- Chain: 1670-1738 (69줄)
- VisionRAG: 1745-1890 (146줄)
- Audio: 1897-1994 (98줄)
- Evaluation: 1996-2062 (67줄)
- Fine-tuning: 2069-2141 (73줄)
- OCR: 2205-2318 (114줄)
- Google Workspace: 2356-2626 (271줄)

**총 중복 코드**: 약 1,038줄

**예상 감소**:
- 중복 엔드포인트 제거: 약 1,038줄
- 중복 전역 상태 제거: 약 50줄
- **총 약 1,088줄 감소**

**목표**: 약 1,600줄 이하

---

### mcp_streaming.py 분석

**현재**: 714줄

**문제점**:
- `orchestrator.py`와 기능 중복
- 실제로는 MCP 서버와 통신하지 않음 (Facade 직접 호출)
- `_rag_instances` 전역 캐시 중복
- `/api/chat/stream` 엔드포인트만 사용

**권장 조치**:
- ✅ 삭제 또는 레거시로 표시
- ✅ 기능은 `chat_router.py`의 agentic 엔드포인트로 통합

---

### common.py 분석

**현재**: 207줄

**문제점**:
- `main.py`와 일부 중복
- `_rag_debugger` 중복
- `get_rag_debugger()` 시그니처 차이

**권장 조치**:
- ✅ `common.py`로 통일
- ✅ `main.py`에서 중복 제거

---

## ✅ 정리 체크리스트

### Phase 1: 레거시 코드 제거
- [x] `mcp_streaming.py` 삭제 또는 레거시 표시 ✅
- [ ] `/api/chat/stream` 엔드포인트를 `chat_router.py`로 이동 (향후 MCP 통합 시)
- [ ] `orchestrator.py`에서 `_rag_instances` import 제거 (향후 MCP 통합 시)

### Phase 2: 중복 엔드포인트 제거
- [x] `main.py`에서 RAG Debug API 제거 (rag_router에 있음) ✅
- [x] `main.py`에서 Optimizer API 제거 (optimizer_router에 있음) ✅
- [x] `main.py`에서 Multi-Agent API 제거 (agent_router에 있음) ✅
- [x] `main.py`에서 Orchestrator API 제거 (agent_router에 있음) ✅
- [x] `main.py`에서 Chain API 제거 (chain_router에 있음) ✅
- [x] `main.py`에서 VisionRAG API 제거 (vision_router에 있음) ✅
- [x] `main.py`에서 Audio API 제거 (audio_router에 있음) ✅
- [x] `main.py`에서 Evaluation API 제거 (evaluation_router에 있음) ✅
- [x] `main.py`에서 Fine-tuning API 제거 (finetuning_router에 있음) ✅
- [x] `main.py`에서 OCR API 제거 (ocr_router에 있음) ✅
- [x] `main.py`에서 Google Workspace API 제거 (google_auth_router에 있음) ✅

### Phase 3: 중복 전역 상태 통일
- [x] `_rag_debugger`를 `common.py`로 통일 ✅
- [x] `_downloaded_models`를 `common.py`로 통일 ✅
- [x] `main.py`에서 중복 제거 ✅

### Phase 4: 빈 파일 정리
- [x] `ml_router.py` 삭제 ✅
- [x] `notebooks/` 디렉토리 정리 (`.gitignore`에 추가) ✅

### Phase 5: 사용되지 않는 import 정리
- [x] `main.py`에서 사용되지 않는 beanllm facade import 제거 ✅
- [x] `main.py`에서 사용되지 않는 기타 import 제거 ✅

### Phase 6: TODO 구현 (MCP 통합 필수)
- [ ] MCP Client Service 생성 (최우선)
- [ ] `orchestrator.py`의 TODO 항목들 구현 (10개)
- [ ] `/api/chat/stream` 엔드포인트를 `chat_router.py`로 이동
- [ ] `mcp_streaming.py` 완전 제거 또는 통합
- [ ] `orchestrator.py`에서 `_rag_instances` import 제거

**상세 내용**: [REMAINING_TASKS.md](./REMAINING_TASKS.md) 참고

---

## 📊 실제 효과

### 코드 감소
- `main.py`: **1,543줄 감소** (2,704줄 → 1,161줄, **57% 감소**)
- `ml_router.py`: 삭제 완료
- **총 약 1,543줄 감소**

### 유지보수성 향상
- 중복 코드 제거
- 단일 진실의 원천 확보
- 코드 가독성 향상

---

## ⚠️ 주의사항

### 삭제 전 확인
1. **엔드포인트 사용 여부 확인**
   - Frontend에서 사용 중인지 확인
   - API 문서 확인

2. **의존성 확인**
   - 다른 파일에서 import하는지 확인
   - 테스트 코드에서 사용하는지 확인

3. **Git History 보존**
   - 삭제 전 커밋
   - 필요시 복구 가능

---

## 🎯 정리 순서

1. **mcp_streaming.py 정리** (우선순위 높음)
2. **main.py 중복 엔드포인트 제거** (우선순위 높음)
3. **전역 상태 통일** (우선순위 중간)
4. **빈 파일 정리** (우선순위 낮음)
5. **TODO 구현** (MCP Client Service 생성 후)
