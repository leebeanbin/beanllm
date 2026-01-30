# 구현 가이드 (Implementation Guide)

## 🎯 목적

이 문서는 Claude Code에게 작업을 위임할 때 참고할 **마스터 가이드**입니다.

**읽기 순서**:
1. [QUICK_START.md](./QUICK_START.md) (5분) ⚡
2. [CURRENT_STATE_ANALYSIS.md](./CURRENT_STATE_ANALYSIS.md) (10분)
3. 이 문서 (30분) ⭐

---

## 📋 전체 구조 개요

### 현재 상태 (2025-01-24 업데이트)

```
beanllm 프로젝트
├── src/beanllm/          # Core 라이브러리 (Facade, Service, Domain, Infrastructure)
├── mcp_server/           # MCP 서버 (33개 tools, 중앙 관리 포인트) ⭐
│   └── tools/
│       ├── rag_tools.py
│       ├── agent_tools.py
│       ├── kg_tools.py
│       ├── ml_tools.py
│       └── google_tools.py
└── playground/            # Playground (Frontend + Backend)
    ├── frontend/          # Next.js 15 + React 19
    └── backend/           # FastAPI (정리 완료 ✅)
        ├── main.py        # 1,161줄 (57% 감소) ✅
        ├── routers/       # 18개 라우터 (정리 완료) ✅
        │   ├── __init__.py # 모든 라우터 export ✅
        │   ├── history_router.py # 이동됨 ✅
        │   └── ...
        ├── schemas/       # 스키마 (정리 완료) ✅
        │   ├── database.py # 이동됨 ✅
        │   └── ...
        ├── services/
        │   ├── orchestrator.py      # ⚠️ 현재: Facade 직접 호출 (MCP 통합 필요)
        │   ├── tool_registry.py
        │   └── intent_classifier.py
        ├── scripts/       # 스크립트 정리 완료 ✅
        └── docs/         # 문서 정리 완료 ✅
```

### 목표 구조

```
beanllm 프로젝트
├── src/beanllm/          # Core 라이브러리 (변경 없음)
├── mcp_server/           # MCP 서버 (중앙 관리 포인트) ⭐
│   └── tools/            # 단일 진실의 원천 (Single Source of Truth)
│       └── [33개 tools]
└── playground/
    └── backend/
        └── services/
            ├── mcp_client_service.py  # ✅ 신규: MCP tools 호출
            └── orchestrator.py        # ✅ 수정: MCP Client 사용
```

---

## 🚨 핵심 문제점

### 1. 중복 코드
- **현재**: `orchestrator.py`와 `mcp_server/tools/`에서 같은 로직 중복
- **해결**: MCP 서버의 tools만 사용

### 2. 관리 포인트 분산
- **현재**: 두 곳에서 beanllm Facade 직접 호출
- **해결**: MCP 서버를 단일 진실의 원천으로

### 3. 미구현 기능
- **현재**: `orchestrator._handle_agent`, `_handle_multi_agent` 등 TODO
- **해결**: MCP 서버의 이미 구현된 tools 활용

---

## ✅ 구현 우선순위

### Phase 1: MCP 서버를 통한 중앙 관리 (최우선 ⭐)

**목표**: MCP 서버를 단일 진실의 원천으로 사용

**작업**:
1. `playground/backend/services/mcp_client_service.py` 생성
2. `playground/backend/services/orchestrator.py` 수정
3. 중복 코드 제거

**참고 문서**: `15_ARCHITECTURE_REVIEW.md`

---

## 📝 단계별 구현 가이드

### Step 1: MCP Client Service 생성

**파일**: `playground/backend/services/mcp_client_service.py`

**기능**:
- MCP 서버의 tools를 직접 함수 호출로 실행
- 세션 관리 지원
- 스트리밍 지원 (선택적)

**구현 예시**:
```python
"""
MCP Client Service

MCP 서버의 tools를 호출하는 클라이언트
중앙 관리 포인트: MCP 서버의 tools만 사용
"""
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class MCPClientService:
    """MCP 서버와 통신하는 클라이언트 (직접 함수 호출)"""
    
    def __init__(self):
        self._tools_cache = {}
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP tool 호출
        
        Args:
            tool_name: tool 이름 (예: "query_rag_system", "run_multiagent_task")
            arguments: tool 인자
            session_id: 세션 ID (세션별 인스턴스 관리용)
        
        Returns:
            tool 실행 결과
        """
        # MCP tools를 직접 import하여 호출
        # HTTP가 아닌 Python 함수 직접 호출
        
        # session_id 추가
        if session_id and "session_id" not in arguments:
            arguments["session_id"] = session_id
        
        # tool 함수 찾기 및 실행
        tool_func = self._get_tool_function(tool_name)
        if tool_func is None:
            raise ValueError(f"Tool '{tool_name}' not found in MCP server")
        
        # tool 실행
        result = await tool_func(**arguments)
        return result
    
    def _get_tool_function(self, tool_name: str):
        """MCP tool 함수 가져오기"""
        # tools 모듈에서 직접 찾기
        from mcp_server.tools import rag_tools, agent_tools, kg_tools, ml_tools, google_tools
        
        modules = [
            ("rag", rag_tools),
            ("agent", agent_tools),
            ("kg", kg_tools),
            ("ml", ml_tools),
            ("google", google_tools),
        ]
        
        for module_name, module in modules:
            if hasattr(module, tool_name):
                return getattr(module, tool_name)
        
        return None

# 싱글톤 인스턴스
mcp_client_service = MCPClientService()
```

**체크리스트**:
- [ ] 파일 생성
- [ ] `call_tool` 메서드 구현
- [ ] `_get_tool_function` 메서드 구현
- [ ] 세션 ID 지원
- [ ] 에러 처리

---

### Step 2: Orchestrator 수정

**파일**: `playground/backend/services/orchestrator.py`

**변경 사항**:
1. MCP Client Service import
2. 모든 handlers가 MCP Client 사용
3. Facade 직접 호출 제거

**구현 예시**:

```python
# orchestrator.py 상단에 추가
from services.mcp_client_service import mcp_client_service

class AgenticOrchestrator:
    def __init__(self, registry: ToolRegistry = None):
        self._registry = registry or tool_registry
        self._mcp_client = mcp_client_service  # ✅ MCP Client 사용
    
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
            
            # 결과 처리
            if result.get("success"):
                # 답변 스트리밍
                answer = result.get("answer", "")
                if answer:
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
                
                # 결과 이벤트
                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": result,
                        "status": "completed",
                    }
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "RAG query failed"),
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
            
            # 결과 처리
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
```

**수정할 handlers**:
- [ ] `_handle_rag` → `query_rag_system`
- [ ] `_handle_multi_agent` → `run_multiagent_task`
- [ ] `_handle_agent` → MCP tool 찾아서 연결
- [ ] `_handle_kg` → MCP kg_tools 사용
- [ ] `_handle_audio` → MCP ml_tools 사용
- [ ] `_handle_ocr` → MCP ml_tools 사용
- [ ] `_handle_evaluation` → MCP ml_tools 사용
- [ ] `_handle_google_drive` → MCP google_tools 사용
- [ ] `_handle_google_docs` → MCP google_tools 사용
- [ ] `_handle_gmail` → MCP google_tools 사용

**제거할 코드**:
- [ ] `from beanllm.facade.core import Client, RAGChain` (직접 호출 제거)
- [ ] Facade 직접 호출 로직 제거

---

### Step 3: Tool Registry 업데이트

**파일**: `playground/backend/services/tool_registry.py`

**변경 사항**:
- Tool 정의에 MCP tool 이름 추가 (선택적, 매핑용)

**구현 예시**:
```python
Tool(
    name="rag",
    description="Document retrieval and Q&A with RAG",
    description_ko="RAG 기반 문서 검색 및 질의응답",
    intent_types=[IntentType.RAG],
    mcp_tool_name="query_rag_system",  # ✅ MCP tool 이름 (선택적)
    requirements=ToolRequirement(...),
    is_streaming=True,
    priority=90,
),
```

**체크리스트**:
- [ ] 각 Tool에 `mcp_tool_name` 추가 (선택적)
- [ ] Tool → MCP tool 매핑 확인

---

### Step 4: 중복 코드 제거

**파일**: `playground/backend/mcp_streaming.py`

**변경 사항**:
- Facade 직접 호출 제거
- MCP Client Service 사용 (선택적)

**참고**: `mcp_streaming.py`는 레거시일 수 있으므로, `orchestrator.py` 사용을 권장

**체크리스트**:
- [ ] `mcp_streaming.py` 검토
- [ ] 필요시 MCP Client 사용으로 변경
- [ ] 또는 레거시로 표시

---

## 🔍 MCP Tools 매핑 테이블

| Orchestrator Handler | MCP Tool | 파일 |
|----------------------|----------|------|
| `_handle_rag` | `query_rag_system` | `mcp_server/tools/rag_tools.py` |
| `_handle_multi_agent` | `run_multiagent_task` | `mcp_server/tools/agent_tools.py` |
| `_handle_agent` | `run_agent_task` (확인 필요) | `mcp_server/tools/agent_tools.py` |
| `_handle_kg` | `query_knowledge_graph` | `mcp_server/tools/kg_tools.py` |
| `_handle_audio` | `transcribe_audio` | `mcp_server/tools/ml_tools.py` |
| `_handle_ocr` | `extract_text_from_image` | `mcp_server/tools/ml_tools.py` |
| `_handle_evaluation` | `evaluate_model` | `mcp_server/tools/ml_tools.py` |
| `_handle_google_drive` | `save_to_google_drive` | `mcp_server/tools/google_tools.py` |
| `_handle_google_docs` | `export_to_google_docs` | `mcp_server/tools/google_tools.py` |
| `_handle_gmail` | `share_via_gmail` | `mcp_server/tools/google_tools.py` |

**확인 방법**:
```bash
# MCP 서버의 모든 tools 확인
grep -r "@mcp.tool()" mcp_server/tools/
```

---

## 🧪 테스트 체크리스트

### 기능 테스트
- [ ] RAG 질의 동작 확인
- [ ] Multi-Agent 실행 확인
- [ ] Agent 실행 확인
- [ ] Knowledge Graph 질의 확인
- [ ] Audio 전사 확인
- [ ] OCR 동작 확인
- [ ] Evaluation 동작 확인
- [ ] Google Drive 저장 확인
- [ ] Google Docs 내보내기 확인
- [ ] Gmail 공유 확인

### 통합 테스트
- [ ] Intent 분류 → Tool 선택 → MCP tool 실행 전체 플로우
- [ ] 스트리밍 동작 확인
- [ ] 세션 관리 동작 확인
- [ ] 에러 처리 확인

### 성능 테스트
- [ ] MCP tool 호출 지연 시간 측정
- [ ] 동시 요청 처리 확인

---

## 📚 참고 문서

### 필수 읽기
1. **`15_ARCHITECTURE_REVIEW.md`**: 아키텍처 재검토 및 최종 픽스 방안
2. **`14_SEARCH_ARCHITECTURE.md`**: 검색 시스템 구조
3. **`00_INDEX.md`**: 전체 문서 인덱스

### MCP 서버 관련
- `mcp_server/run.py`: MCP 서버 메인
- `mcp_server/tools/`: 모든 tools 정의

### Playground 관련
- `playground/backend/services/orchestrator.py`: 현재 구조
- `playground/backend/services/tool_registry.py`: Tool 정의

---

## ⚠️ 주의사항

### 1. Clean Architecture 준수
- MCP Client Service는 `mcp_server/tools/`만 import
- beanllm Facade 직접 호출 금지 (MCP tools를 통해)

### 2. 세션 관리
- 모든 MCP tool 호출 시 `session_id` 전달
- MCP tools는 세션별 인스턴스 관리 지원

### 3. 에러 처리
- MCP tool 호출 실패 시 적절한 에러 메시지
- 사용자에게 명확한 피드백

### 4. 스트리밍
- 일부 tools는 스트리밍을 지원하지 않음
- 결과를 청크로 나누어 스트리밍 (필요시)

---

## 🎯 성공 기준

### 완료 조건
1. ✅ 모든 orchestrator handlers가 MCP Client 사용
2. ✅ Facade 직접 호출 제거
3. ✅ 모든 기능 동작 확인
4. ✅ 테스트 통과
5. ✅ 중복 코드 제거

### 검증 방법
```bash
# Facade 직접 호출 확인 (없어야 함)
grep -r "from beanllm.facade" playground/backend/services/orchestrator.py

# MCP Client 사용 확인 (있어야 함)
grep -r "mcp_client_service" playground/backend/services/orchestrator.py
```

---

## ✅ 완료된 개선 사항 (2025-01-24)

### 1. 코드 정리 ✅
- ✅ 중복 엔드포인트 제거 (11개)
- ✅ 사용되지 않는 import 제거 (15개)
- ✅ 레거시 코드 표시 (mcp_streaming.py)
- ✅ 불필요한 주석 제거
- ✅ `main.py` 크기 감소 (57% 감소)

### 2. 구조 개선 ✅
- ✅ 디렉토리 구조 정리 (scripts/, docs/)
- ✅ 파일 이동 및 정리
- ✅ 의존성 관리 정리 (Poetry)
- ✅ 문서화 (README.md)

**상세 내용**: `playground/backend/docs/CLEANUP_ANALYSIS.md`, `playground/backend/docs/STRUCTURE_ANALYSIS.md` 참고

---

## 💡 추가 개선 사항 (선택)

### 1. Tool 자동 매핑
- Tool Registry에서 MCP tool 이름 자동 매핑
- 동적 tool 발견

### 2. 스트리밍 최적화
- MCP tools의 스트리밍 지원 확인
- 필요시 청크 분할 로직 개선

### 3. 캐싱
- MCP tool 결과 캐싱 (선택적)
- 세션별 캐시 관리

---

## 📞 문제 해결

### 문제: MCP tool을 찾을 수 없음
**해결**: `_get_tool_function`에서 모든 tools 모듈 확인

### 문제: 세션 관리가 동작하지 않음
**해결**: `session_id`가 arguments에 포함되는지 확인

### 문제: 스트리밍이 동작하지 않음
**해결**: 결과를 수동으로 청크로 나누어 스트리밍

---

## ✅ 최종 체크리스트

### 구현 전
- [ ] `15_ARCHITECTURE_REVIEW.md` 읽기
- [ ] 현재 구조 이해
- [ ] MCP tools 목록 확인

### 구현 중
- [ ] MCP Client Service 생성
- [ ] Orchestrator 수정
- [ ] 각 handler 테스트

### 구현 후
- [ ] 전체 기능 테스트
- [ ] 중복 코드 제거 확인
- [ ] 문서 업데이트

---

## 🎉 완료 후

구현이 완료되면:
1. `DEVELOPMENT_LOG.md` 업데이트
2. 변경 사항 커밋
3. 테스트 결과 문서화
