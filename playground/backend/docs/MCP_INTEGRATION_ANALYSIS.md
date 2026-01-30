# MCP 통합 및 코드 정리 분석

## 🎯 사용자 의도

**핵심 목표**:
1. **MCP를 통한 능동적 도구 호출**: 모델이 능동적으로 함수/도구를 사용할 수 있게
2. **코드 통합**: 현재 산재된 코드를 MCP 서버 중심으로 통합

---

## 🔍 현재 구조 분석

### 1. MCP 서버 (중앙 관리 포인트) ✅

**위치**: `mcp_server/tools/`

**구조**:
```
mcp_server/tools/
├── rag_tools.py        # RAG 관련 tools
├── agent_tools.py      # Agent 관련 tools
├── kg_tools.py         # Knowledge Graph tools
├── ml_tools.py         # Audio, OCR, Evaluation tools
└── google_tools.py     # Google Workspace tools
```

**특징**:
- ✅ FastMCP 사용 (`@mcp.tool()` 데코레이터)
- ✅ beanllm Facade 직접 호출
- ✅ 세션 기반 인스턴스 관리 (`session_manager`)
- ✅ **33개 tools 완전 구현됨**

**예시**:
```python
# mcp_server/tools/rag_tools.py
@mcp.tool()
async def build_rag_system(...):
    # beanllm Facade 직접 호출
    rag = RAGChain.from_documents(...)
    return {"status": "success", ...}
```

---

### 2. Playground Backend (현재 문제점) ⚠️

#### 2.1 orchestrator.py
**위치**: `playground/backend/services/orchestrator.py`

**문제점**:
- ❌ beanllm Facade를 **직접 호출** (MCP tools 사용 안 함)
- ❌ `mcp_streaming.py`의 `_rag_instances`에 의존
- ❌ TODO 항목들 (하지만 MCP 서버에는 이미 구현됨)

**예시**:
```python
# orchestrator.py Line 324
from beanllm.facade.core import RAGChain  # 직접 호출
rag = RAGChain.from_documents(...)  # 중복 코드!

# orchestrator.py Line 341
from mcp_streaming import _rag_instances  # 의존성 문제
```

**TODO 항목들**:
- `_handle_agent`: TODO (하지만 `mcp_server/tools/agent_tools.py`에 구현됨)
- `_handle_multi_agent`: TODO (하지만 `mcp_server/tools/agent_tools.py`에 구현됨)
- `_handle_web_search`: TODO (하지만 `mcp_server/tools/ml_tools.py`에 구현됨)

---

#### 2.2 mcp_streaming.py
**위치**: `playground/backend/mcp_streaming.py`

**문제점**:
- ❌ **실제로는 MCP 서버와 통신하지 않음**
- ❌ 주석: "MCP Server가 아닌 beanllm Facade/Handler를 직접 호출합니다"
- ❌ `_rag_instances` 전역 캐시 (MCP 서버의 `session_manager`와 중복)
- ❌ 키워드 기반 Tool 감지 (LLM 기반이 아님)

**예시**:
```python
# mcp_streaming.py Line 5
# MCP Server가 아닌 beanllm Facade/Handler를 직접 호출합니다.

# mcp_streaming.py Line 132
async def _detect_tools(self, query: str):
    # 간단한 키워드 매칭 (LLM 기반 아님)
    if "rag" in query.lower():
        tool_calls.append({"name": "rag", ...})
```

---

#### 2.3 main.py의 중복 엔드포인트
**위치**: `playground/backend/main.py`

**문제점**:
- ❌ Routers로 이동했지만 여전히 중복 엔드포인트 존재
- ❌ 약 1,038줄의 중복 코드

---

## ✅ 해결 방안

### 방안: MCP 서버를 단일 진실의 원천으로 사용

**핵심 아이디어**:
1. **MCP 서버의 tools만 사용** (중복 코드 제거)
2. **orchestrator가 MCP tools를 직접 호출** (HTTP가 아닌 Python 함수 호출)
3. **LLM이 능동적으로 tools를 선택** (Intent Classifier + Tool Registry 활용)

---

## 📋 구현 계획

### Phase 1: MCP Client Service 생성

**목표**: MCP 서버의 tools를 Python 함수로 직접 호출

**파일**: `playground/backend/services/mcp_client_service.py` (신규)

**기능**:
```python
class MCPClientService:
    """MCP 서버의 tools를 직접 호출하는 서비스"""
    
    async def call_tool(self, tool_name: str, **kwargs):
        """MCP tool을 직접 함수 호출로 실행"""
        # mcp_server.tools에서 직접 import
        if tool_name == "build_rag_system":
            from mcp_server.tools.rag_tools import build_rag_system
            return await build_rag_system(**kwargs)
        # ...
```

**장점**:
- ✅ HTTP 통신 불필요 (같은 프로세스 내)
- ✅ 타입 안정성
- ✅ 빠른 실행

---

### Phase 2: orchestrator.py 수정

**목표**: MCP Client Service를 사용하도록 변경

**변경 사항**:
```python
# Before (현재)
async def _handle_rag(...):
    from beanllm.facade.core import RAGChain  # 직접 호출
    rag = RAGChain.from_documents(...)

# After (목표)
async def _handle_rag(...):
    from services.mcp_client_service import mcp_client
    result = await mcp_client.call_tool(
        "build_rag_system",
        documents_path=...,
        collection_name=...
    )
```

**효과**:
- ✅ 중복 코드 제거
- ✅ TODO 항목들 해결 (MCP tools 활용)
- ✅ 일관성 확보

---

### Phase 3: mcp_streaming.py 정리

**옵션 A: 삭제** (권장)
- `orchestrator.py`로 통합
- `/api/chat/stream` 엔드포인트는 `chat_router.py`로 이동

**옵션 B: MCP 통합**
- MCP Client Service 사용
- LLM 기반 Tool 감지 (Intent Classifier 활용)

---

### Phase 4: main.py 중복 엔드포인트 제거

**목표**: Routers만 사용

**제거 대상**:
- RAG Debug API (rag_router에 있음)
- Optimizer API (optimizer_router에 있음)
- Multi-Agent API (agent_router에 있음)
- 등등...

---

## 🎯 최종 구조

### 목표 아키텍처

```
beanllm 프로젝트
├── src/beanllm/          # Core 라이브러리 (변경 없음)
│
├── mcp_server/           # ⭐ 중앙 관리 포인트
│   └── tools/            # 단일 진실의 원천 (33개 tools)
│       ├── rag_tools.py
│       ├── agent_tools.py
│       ├── kg_tools.py
│       ├── ml_tools.py
│       └── google_tools.py
│
└── playground/
    └── backend/
        └── services/
            ├── mcp_client_service.py  # ✅ 신규: MCP tools 직접 호출
            ├── orchestrator.py        # ✅ 수정: MCP Client 사용
            ├── intent_classifier.py   # 의도 분류
            └── tool_registry.py      # Tool 관리
```

### 데이터 흐름

```
사용자 질의
    ↓
Intent Classifier (의도 분류)
    ↓
Tool Registry (도구 선택)
    ↓
Orchestrator (실행)
    ↓
MCP Client Service (MCP tools 호출)
    ↓
MCP Server Tools (beanllm Facade 호출)
    ↓
결과 반환 (SSE 스트리밍)
```

---

## 📊 예상 효과

### 코드 감소
- `main.py`: 약 1,038줄 감소 (중복 엔드포인트 제거)
- `mcp_streaming.py`: 714줄 삭제 (orchestrator로 통합)
- `orchestrator.py`: 중복 코드 제거, TODO 해결
- **총 약 1,800줄 감소**

### 기능 향상
- ✅ **능동적 도구 호출**: LLM이 Intent Classifier를 통해 tools 선택
- ✅ **일관성**: 모든 클라이언트가 같은 MCP tools 사용
- ✅ **유지보수성**: 한 곳에서만 관리 (MCP 서버)

---

## ✅ 구현 체크리스트

### Phase 1: MCP Client Service
- [ ] `playground/backend/services/mcp_client_service.py` 생성
- [ ] MCP tools 직접 호출 기능 구현
- [ ] 세션 관리 지원

### Phase 2: Orchestrator 수정
- [ ] `orchestrator.py`에서 MCP Client Service 사용
- [ ] Facade 직접 호출 제거
- [ ] TODO 항목들 해결 (MCP tools 활용)

### Phase 3: mcp_streaming.py 정리
- [ ] 삭제 또는 MCP 통합
- [ ] `/api/chat/stream` 엔드포인트를 `chat_router.py`로 이동

### Phase 4: main.py 정리
- [ ] 중복 엔드포인트 제거
- [ ] Routers만 사용

---

## 🚀 다음 단계

1. **MCP Client Service 구현** (최우선)
2. **Orchestrator 수정**
3. **코드 정리** (mcp_streaming, main.py)

이 방향으로 진행하면 **MCP를 통한 능동적 도구 호출**이 가능하고, **코드 통합**도 완료됩니다.
