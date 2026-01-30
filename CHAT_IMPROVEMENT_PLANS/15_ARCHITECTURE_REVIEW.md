# 아키텍처 재검토 및 최종 픽스 방안

## 📋 개요

아키텍처 문제점 분석 및 최종 픽스 방안

**관련 문서**: [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)

---

## 🔍 현재 구조 문제점

### 1. 중복 코드 및 관리 포인트 분산

**문제**:
- `orchestrator.py`: beanllm Facade 직접 호출
- `mcp_server/tools/`: beanllm Facade 직접 호출
- **같은 로직이 두 곳에 존재** → 유지보수 어려움

**예시**:
```python
# orchestrator.py
async def _handle_rag(...):
    from beanllm.facade.core import RAGChain
    rag = RAGChain.from_documents(...)  # 직접 호출

# mcp_server/tools/rag_tools.py
@mcp.tool()
async def build_rag_system(...):
    rag = RAGChain.from_documents(...)  # 같은 로직 중복!
```

### 2. 중앙 관리 부재

**현재 구조**:
```
Playground Backend
├── orchestrator.py (Facade 직접 호출)
└── mcp_streaming.py (Facade 직접 호출)

MCP Server
└── tools/ (Facade 직접 호출)
```

**문제점**:
- MCP 서버에 33개 tools가 잘 정의되어 있음
- 하지만 playground는 이를 사용하지 않음
- **단일 진실의 원천(Single Source of Truth) 부재**

### 3. 미구현 기능

**orchestrator.py에서**:
- `_handle_agent`: TODO (구현 안 됨)
- `_handle_multi_agent`: TODO (구현 안 됨)
- `_handle_web_search`: TODO (구현 안 됨)

**하지만 MCP 서버에는 이미 구현되어 있음**:
- `agent_tools.py`: Multi-Agent 완전 구현
- `ml_tools.py`: Audio, OCR, Evaluation 구현
- `kg_tools.py`: Knowledge Graph 구현

---

## ✅ 최종 픽스 방안

### 방안 1: MCP 서버를 중앙 관리 포인트로 사용 (권장 ⭐)

**핵심 아이디어**: 
- MCP 서버의 tools를 **단일 진실의 원천**으로 사용
- Playground orchestrator는 MCP 서버의 tools를 호출

**구조**:
```
MCP Server (중앙 관리)
└── tools/ (33개 tools, beanllm Facade 호출)
    ├── rag_tools.py
    ├── agent_tools.py
    ├── kg_tools.py
    ├── ml_tools.py
    └── google_tools.py

Playground Backend
└── orchestrator.py (MCP 서버 tools 호출)
    └── MCP Client를 통해 tools 실행
```

**장점**:
1. ✅ **중복 코드 제거**: 한 곳에서만 관리
2. ✅ **일관성**: 모든 클라이언트(Playground, Claude Desktop 등)가 같은 tools 사용
3. ✅ **유지보수 용이**: tools 수정 시 한 곳만 수정
4. ✅ **기능 완전성**: MCP 서버의 모든 기능 활용 가능

**구현 방법**:

#### 1. MCP Client 서비스 생성

```python
# playground/backend/services/mcp_client_service.py (신규)
"""
MCP Client Service

MCP 서버의 tools를 호출하는 클라이언트
"""
import httpx
import json
from typing import Dict, Any, Optional, AsyncGenerator
from fastmcp import FastMCP

class MCPClientService:
    """MCP 서버와 통신하는 클라이언트"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8765"):
        self.mcp_server_url = mcp_server_url
        self._mcp_instance = None
    
    async def _get_mcp_instance(self) -> FastMCP:
        """MCP 인스턴스 가져오기 (직접 import)"""
        if self._mcp_instance is None:
            # MCP 서버의 tools를 직접 import하여 사용
            from mcp_server.tools import rag_tools
            from mcp_server.tools import agent_tools
            from mcp_server.tools import kg_tools
            from mcp_server.tools import ml_tools
            from mcp_server.tools import google_tools
            
            # FastMCP 인스턴스는 각 tools 모듈에서 공유됨
            # rag_tools.mcp, agent_tools.mcp 등으로 접근 가능
            self._mcp_instance = rag_tools.mcp  # 또는 통합 인스턴스
        
        return self._mcp_instance
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP tool 호출
        
        Args:
            tool_name: tool 이름 (예: "build_rag_system")
            arguments: tool 인자
            session_id: 세션 ID (세션별 인스턴스 관리용)
        
        Returns:
            tool 실행 결과
        """
        # MCP tools를 직접 호출 (HTTP가 아닌 직접 함수 호출)
        mcp = await self._get_mcp_instance()
        
        # tool 함수 찾기
        tool_func = None
        for tool in mcp.list_tools():
            if tool.name == tool_name:
                # tool 함수 가져오기
                tool_func = getattr(mcp, f"_{tool_name}", None)
                if tool_func is None:
                    # tools 모듈에서 직접 찾기
                    from mcp_server.tools import rag_tools, agent_tools, kg_tools, ml_tools, google_tools
                    modules = [rag_tools, agent_tools, kg_tools, ml_tools, google_tools]
                    for module in modules:
                        if hasattr(module, tool_name):
                            tool_func = getattr(module, tool_name)
                            break
                break
        
        if tool_func is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # session_id 추가 (세션별 관리)
        if session_id and "session_id" not in arguments:
            arguments["session_id"] = session_id
        
        # tool 실행
        result = await tool_func(**arguments)
        return result
    
    async def call_tool_streaming(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        MCP tool 호출 (스트리밍)
        
        일부 tools는 스트리밍을 지원하지 않으므로
        결과를 청크로 나누어 yield
        """
        result = await self.call_tool(tool_name, arguments, session_id)
        
        # 결과를 청크로 나누어 스트리밍
        if isinstance(result, dict) and "success" in result:
            yield {
                "type": "tool_progress",
                "data": {
                    "tool": tool_name,
                    "step": "executing",
                    "message": f"Executing {tool_name}...",
                    "progress": 0.5,
                }
            }
            
            yield {
                "type": "tool_result",
                "data": {
                    "tool": tool_name,
                    "result": result,
                    "status": "completed" if result.get("success") else "failed",
                }
            }
        else:
            # 단순 결과
            yield {
                "type": "tool_result",
                "data": {
                    "tool": tool_name,
                    "result": result,
                    "status": "completed",
                }
            }

# 싱글톤 인스턴스
mcp_client_service = MCPClientService()
```

#### 2. Orchestrator 수정

```python
# playground/backend/services/orchestrator.py (수정)
class AgenticOrchestrator:
    """Agentic Orchestrator - MCP 서버 tools 사용"""
    
    def __init__(self, registry: ToolRegistry = None):
        self._registry = registry or tool_registry
        self._mcp_client = mcp_client_service  # MCP Client 사용
    
    async def _handle_rag(
        self,
        context: OrchestratorContext,
        tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """RAG 도구 핸들러 (MCP 서버 사용)"""
        try:
            # 진행 상황
            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "searching",
                    "message": "관련 문서 검색 중...",
                    "progress": 0.2,
                }
            )
            
            # ✅ MCP 서버의 query_rag_system 호출
            result = await self._mcp_client.call_tool(
                tool_name="query_rag_system",
                arguments={
                    "query": context.query,
                    "collection_name": context.extra_params.get("collection_name", "default"),
                    "top_k": context.extra_params.get("top_k", 5),
                    "model": context.model,
                    "temperature": context.temperature,
                },
                session_id=context.session_id
            )
            
            # 결과 포맷팅
            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": result,
                    "status": "completed" if result.get("success") else "failed",
                }
            )
            
            # 답변 스트리밍 (RAG 결과를 텍스트로)
            if result.get("success") and result.get("answer"):
                answer = result["answer"]
                # 청크로 나누어 스트리밍
                chunk_size = 50
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i+chunk_size]
                    yield AgenticEvent(
                        type=EventType.TEXT,
                        data={
                            "tool": tool.name,
                            "content": chunk,
                        }
                    )
        
        except Exception as e:
            logger.error(f"RAG handler error: {e}")
            yield AgenticEvent(
                type=EventType.ERROR,
                data={
                    "tool": tool.name,
                    "message": str(e),
                }
            )
    
    async def _handle_multi_agent(
        self,
        context: OrchestratorContext,
        tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Multi-Agent 도구 핸들러 (MCP 서버 사용)"""
        try:
            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "initializing",
                    "message": "멀티 에이전트 시스템 초기화 중...",
                    "progress": 0.2,
                }
            )
            
            # ✅ MCP 서버의 run_multiagent_task 호출
            result = await self._mcp_client.call_tool(
                tool_name="run_multiagent_task",
                arguments={
                    "system_name": context.extra_params.get("system_name", "default"),
                    "task": context.query,
                    "context": context.extra_params.get("context", {}),
                },
                session_id=context.session_id
            )
            
            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": result,
                    "status": "completed" if result.get("success") else "failed",
                }
            )
        
        except Exception as e:
            logger.error(f"Multi-agent handler error: {e}")
            yield AgenticEvent(
                type=EventType.ERROR,
                data={
                    "tool": tool.name,
                    "message": str(e),
                }
            )
    
    # 다른 handlers도 동일한 패턴으로 수정
    # _handle_agent, _handle_web_search, _handle_audio, _handle_ocr 등
```

#### 3. Tool Registry와 MCP Tools 매핑

```python
# playground/backend/services/tool_registry.py (수정)
# Tool 정의에 MCP tool 이름 추가

Tool(
    name="rag",
    description="Document retrieval and Q&A with RAG",
    description_ko="RAG 기반 문서 검색 및 질의응답",
    intent_types=[IntentType.RAG],
    mcp_tool_name="query_rag_system",  # ✅ MCP tool 이름 추가
    requirements=ToolRequirement(...),
    is_streaming=True,
    priority=90,
),

Tool(
    name="multi_agent",
    description="Multi-agent debate and collaboration",
    description_ko="멀티 에이전트 토론/협업",
    intent_types=[IntentType.MULTI_AGENT],
    mcp_tool_name="run_multiagent_task",  # ✅ MCP tool 이름 추가
    requirements=ToolRequirement(...),
    is_streaming=True,
    priority=85,
),
```

---

### 방안 2: MCP 서버를 HTTP API로 노출 (대안)

**핵심 아이디어**:
- MCP 서버를 HTTP API로 노출
- Playground는 HTTP로 MCP 서버 호출

**구조**:
```
MCP Server (HTTP API)
└── /api/tools/{tool_name} (POST)

Playground Backend
└── orchestrator.py (HTTP 클라이언트로 MCP 서버 호출)
```

**장점**:
- MCP 서버를 독립 프로세스로 실행 가능
- 여러 클라이언트가 동시에 사용 가능

**단점**:
- HTTP 오버헤드
- 네트워크 의존성

---

## 🎯 권장 방안: 방안 1 (직접 함수 호출)

### 이유

1. **성능**: HTTP 오버헤드 없음
2. **단순성**: 같은 프로세스에서 실행
3. **타입 안정성**: Python 함수 직접 호출
4. **디버깅 용이**: 스택 트레이스 명확

### 구현 단계

#### Phase 1: MCP Client Service 생성
- [ ] `mcp_client_service.py` 생성
- [ ] MCP tools 직접 호출 로직 구현
- [ ] 스트리밍 지원

#### Phase 2: Orchestrator 수정
- [ ] `_handle_rag`: MCP 서버 사용
- [ ] `_handle_multi_agent`: MCP 서버 사용
- [ ] `_handle_agent`: MCP 서버 사용
- [ ] `_handle_web_search`: MCP 서버 사용
- [ ] `_handle_audio`, `_handle_ocr`, `_handle_evaluation`: MCP 서버 사용
- [ ] `_handle_kg`: MCP 서버 사용

#### Phase 3: 중복 코드 제거
- [ ] `orchestrator.py`에서 Facade 직접 호출 제거
- [ ] `mcp_streaming.py`에서 Facade 직접 호출 제거 (또는 MCP Client 사용)
- [ ] 모든 beanllm 호출을 MCP 서버를 통해

#### Phase 4: 테스트 및 검증
- [ ] 모든 tools 동작 확인
- [ ] 스트리밍 동작 확인
- [ ] 세션 관리 동작 확인

---

## ✅ 완료된 작업 (2025-01-24)

### 코드 정리
- ✅ 중복 엔드포인트 제거 (11개)
- ✅ 중복 전역 상태 통일
- ✅ 사용되지 않는 import 제거 (15개)
- ✅ 빈 파일 정리 (ml_router.py 삭제)
- ✅ 레거시 코드 표시 (mcp_streaming.py)
- ✅ 불필요한 주석 제거
- ✅ `main.py` 크기 감소 (2,704줄 → 1,161줄, 57% 감소)

### 구조 개선
- ✅ 디렉토리 구조 정리 (scripts/, docs/ 생성)
- ✅ routers/__init__.py 완성 (17개 라우터 export)
- ✅ 파일 이동 (chat_history.py → routers/history_router.py)
- ✅ 파일 이동 (models.py → schemas/database.py)
- ✅ 의존성 관리 정리 (requirements.txt 삭제, pyproject.toml 통합)
- ✅ 문서화 (README.md 생성)

**상세 내용**: `playground/backend/docs/CLEANUP_ANALYSIS.md`, `playground/backend/docs/STRUCTURE_ANALYSIS.md` 참고

---

## 📊 비교: 현재 vs 개선 후

### 현재 구조 (2025-01-24 업데이트)

```
Playground Backend
├── main.py                    # 1,161줄 (57% 감소) ✅
├── common.py                  # 공통 유틸리티
├── database.py                # DB 연결
├── mcp_streaming.py           # 레거시 (향후 제거) ⚠️
├── routers/                   # 18개 라우터 (정리 완료) ✅
│   ├── __init__.py            # 모든 라우터 export ✅
│   ├── history_router.py      # 이동됨 ✅
│   └── ...
├── schemas/                   # 스키마 (정리 완료) ✅
│   ├── database.py            # 이동됨 ✅
│   └── ...
├── services/
│   ├── orchestrator.py
│   │   ├── _handle_rag (Facade 직접 호출) ❌
│   │   ├── _handle_multi_agent (TODO) ❌
│   │   └── _handle_agent (TODO) ❌
│   └── ...
├── scripts/                   # 스크립트 정리 완료 ✅
└── docs/                      # 문서 정리 완료 ✅

MCP Server
└── tools/ (33개 tools, 사용 안 됨) ⚠️
```

**완료된 개선 (2025-01-24)**:
- ✅ 중복 엔드포인트 제거 (11개)
- ✅ 파일 구조 정리 (scripts/, docs/ 생성)
- ✅ routers/__init__.py 완성
- ✅ 파일 이동 및 정리
- ✅ 의존성 관리 정리 (Poetry 사용)

**남은 문제점**:
- 중복 코드 (orchestrator.py에서 Facade 직접 호출)
- 미구현 기능 (10개 TODO)
- 관리 포인트 분산 (MCP tools 미사용)

### 개선 후 구조

```
MCP Server (중앙 관리)
└── tools/ (33개 tools, 단일 진실의 원천) ✅
    ├── rag_tools.py
    ├── agent_tools.py
    ├── kg_tools.py
    ├── ml_tools.py
    └── google_tools.py

Playground Backend
└── orchestrator.py
    └── MCP Client를 통해 tools 호출 ✅
        ├── _handle_rag → query_rag_system
        ├── _handle_multi_agent → run_multiagent_task
        ├── _handle_agent → (MCP tool)
        └── 모든 handlers가 MCP tools 사용
```

**장점**:
- ✅ 중복 코드 제거
- ✅ 모든 기능 활용 가능
- ✅ 단일 진실의 원천
- ✅ 유지보수 용이

---

## 💡 핵심 원칙

1. **Single Source of Truth**: MCP 서버의 tools가 유일한 구현
2. **DRY (Don't Repeat Yourself)**: 중복 코드 제거
3. **Separation of Concerns**: 
   - MCP Server: beanllm 기능 wrapping
   - Playground Orchestrator: Intent 분류 및 도구 선택
4. **Consistency**: 모든 클라이언트가 같은 tools 사용

---

## 🔗 관련 문서

- [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) ⭐: 구현 가이드 (Claude Code 위임용)
- [12_MCP_INTEGRATION.md](./12_MCP_INTEGRATION.md): MCP 통합 계획
- [14_SEARCH_ARCHITECTURE.md](./14_SEARCH_ARCHITECTURE.md): 현재 구조 분석

---

## 📝 Claude Code에게 위임 시

**필수 읽기**:
1. 이 문서 (15_ARCHITECTURE_REVIEW.md)
2. [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) ⭐

**시작 지점**:
- `playground/backend/services/mcp_client_service.py` 생성
- `playground/backend/services/orchestrator.py` 수정

**검증 방법**:
```bash
# Facade 직접 호출 확인 (없어야 함)
grep -r "from beanllm.facade" playground/backend/services/orchestrator.py

# MCP Client 사용 확인 (있어야 함)
grep -r "mcp_client_service" playground/backend/services/orchestrator.py
```
