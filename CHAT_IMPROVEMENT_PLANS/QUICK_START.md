# 빠른 시작 가이드 (Quick Start Guide)

## 📋 개요

이 문서는 5분 안에 전체 구조를 파악하기 위한 빠른 시작 가이드입니다.

**읽기 순서**:
1. 이 문서 (5분) ⚡
2. [CURRENT_STATE_ANALYSIS.md](./CURRENT_STATE_ANALYSIS.md) (10분)
3. [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) (30분) ⭐

---

## 🎯 Claude Code에게 작업 위임 시

### Step 1: 필수 문서 읽기 (5분)

1. **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** ⭐
   - 구현 가이드 (마스터 문서)
   - 단계별 구현 방법
   - 체크리스트

2. **[15_ARCHITECTURE_REVIEW.md](./15_ARCHITECTURE_REVIEW.md)**
   - 현재 구조 문제점
   - 최종 픽스 방안
   - 비교 분석

### Step 2: 현재 구조 이해 (10분)

**핵심 문제**:
- `orchestrator.py`와 `mcp_server/tools/`에서 중복 코드
- MCP 서버의 33개 tools를 사용하지 않음
- 미구현 기능 (orchestrator handlers)

**목표**:
- MCP 서버를 중앙 관리 포인트로 사용
- 모든 handlers가 MCP Client를 통해 tools 호출

### Step 3: 구현 시작 (1-2시간)

#### 3.1 MCP Client Service 생성

**파일**: `playground/backend/services/mcp_client_service.py`

**기능**:
```python
class MCPClientService:
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        # MCP tools를 직접 함수 호출로 실행
```

**참고**: [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)의 Step 1

#### 3.2 Orchestrator 수정

**파일**: `playground/backend/services/orchestrator.py`

**변경 사항**:
- MCP Client Service import
- 모든 handlers가 MCP Client 사용
- Facade 직접 호출 제거

**예시**:
```python
# Before
from beanllm.facade.core import RAGChain
rag = RAGChain.from_documents(...)

# After
result = await self._mcp_client.call_tool(
    tool_name="query_rag_system",
    arguments={...},
    session_id=context.session_id
)
```

**참고**: [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)의 Step 2

### Step 4: 테스트 (30분)

**체크리스트**:
- [ ] RAG 질의 동작 확인
- [ ] Multi-Agent 실행 확인
- [ ] 모든 handlers 동작 확인
- [ ] 스트리밍 동작 확인
- [ ] 세션 관리 동작 확인

**검증 명령**:
```bash
# Facade 직접 호출 확인 (없어야 함)
grep -r "from beanllm.facade" playground/backend/services/orchestrator.py

# MCP Client 사용 확인 (있어야 함)
grep -r "mcp_client_service" playground/backend/services/orchestrator.py
```

---

## 📋 핵심 체크리스트

### 구현 전
- [ ] `IMPLEMENTATION_GUIDE.md` 읽기
- [ ] `15_ARCHITECTURE_REVIEW.md` 읽기
- [ ] MCP tools 목록 확인 (`mcp_server/tools/`)

### 구현 중
- [ ] `mcp_client_service.py` 생성
- [ ] `orchestrator.py` 수정
- [ ] 각 handler 테스트

### 구현 후
- [ ] 전체 기능 테스트
- [ ] 중복 코드 제거 확인
- [ ] 문서 업데이트

---

## 🔍 MCP Tools 매핑

| Orchestrator Handler | MCP Tool | 파일 |
|----------------------|----------|------|
| `_handle_rag` | `query_rag_system` | `rag_tools.py` |
| `_handle_multi_agent` | `run_multiagent_task` | `agent_tools.py` |
| `_handle_kg` | `query_knowledge_graph` | `kg_tools.py` |
| `_handle_audio` | `transcribe_audio` | `ml_tools.py` |
| `_handle_ocr` | `extract_text_from_image` | `ml_tools.py` |
| `_handle_google_drive` | `save_to_google_drive` | `google_tools.py` |

**전체 목록**: `grep -r "@mcp.tool()" mcp_server/tools/`

---

## ⚠️ 주의사항

1. **Clean Architecture**: Facade 직접 호출 금지
2. **세션 관리**: 모든 tool 호출 시 `session_id` 전달
3. **에러 처리**: 적절한 에러 메시지 및 피드백

---

## 📞 문제 해결

### MCP tool을 찾을 수 없음
→ `_get_tool_function`에서 모든 tools 모듈 확인

### 세션 관리가 동작하지 않음
→ `session_id`가 arguments에 포함되는지 확인

### 스트리밍이 동작하지 않음
→ 결과를 수동으로 청크로 나누어 스트리밍

---

## 🎉 완료 후

1. `DEVELOPMENT_LOG.md` 업데이트
2. 변경 사항 커밋
3. 테스트 결과 문서화
