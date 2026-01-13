# Backend API 수정/구현 체크리스트

## ✅ 완료된 작업

1. **RAG API** - asyncio.run() 문제 해결 및 Ollama 테스트 완료
   - `_safe_log_event()` 헬퍼 함수 생성
   - 여러 파일에서 asyncio.run() 호출 수정
   - RAG 빌드/쿼리 테스트 성공

## 🔧 수정이 필요한 부분

### 1. Chat API (Line 263-292)
**현재 상태**: Client를 직접 사용 (Handler 미사용)
**문제점**: 
- Handler를 사용하지 않아 일관성 부족
- 하지만 현재 구현은 작동함

**수정 필요도**: 낮음 (선택사항)

---

### 2. RAG Debug API (Line 562-592)
**현재 상태**: `get_rag_debugger()`가 vector_store 없이 생성 시도
**문제점**:
```python
# 현재 (Line 97)
_rag_debugger = RAGDebug()  # ❌ vector_store 필수 파라미터 누락

# RAGDebug.__init__ 시그니처
def __init__(self, vector_store: "BaseVectorStore", ...)
```

**수정 방법**:
- Option 1: RAG Debug API에서 vector_store를 요청에서 받아서 전달
- Option 2: 기본 vector_store를 생성하거나 None 체크 추가

**수정 필요도**: 높음 (필수)

---

### 3. Multi-Agent API (Line 627-675)
**현재 상태**: 시뮬레이션 코드 사용 (실제 MultiAgentCoordinator 미사용)
**문제점**:
```python
# 현재 (Line 111)
_multi_agent = MultiAgentCoordinator()  # ❌ agents 필수 파라미터 누락

# MultiAgentCoordinator.__init__ 시그니처
def __init__(self, agents: Dict[str, Any], ...)
```

**수정 방법**:
- 요청에서 받은 정보로 Agent들을 생성하고 MultiAgentCoordinator에 전달
- 실제 `execute_sequential`, `execute_parallel`, `execute_hierarchical`, `execute_debate` 메서드 사용

**수정 필요도**: 높음 (필수)

---

### 4. Agent API (Line 494-524)
**현재 상태**: Agent facade 사용
**확인 필요**:
- `agent.run()` 메서드가 올바르게 작동하는지 테스트 필요
- 응답 형식 확인 필요

**수정 필요도**: 중간 (테스트 후 결정)

---

### 5. Knowledge Graph API (Line 298-400)
**현재 상태**: KnowledgeGraph facade 사용
**확인 필요**:
- `quick_build()`, `query_graph()`, `ask()`, `visualize_graph()` 메서드 테스트 필요
- 응답 형식 확인 필요

**수정 필요도**: 중간 (테스트 후 결정)

---

### 6. Orchestrator API (Line 681-715)
**현재 상태**: Orchestrator facade 사용
**확인 필요**:
- `quick_research_write()`, `quick_parallel_consensus()`, `quick_debate()`, `run_full_workflow()` 메서드 테스트 필요
- 응답 형식 확인 필요

**수정 필요도**: 중간 (테스트 후 결정)

---

### 7. Optimizer API (Line 598-621)
**현재 상태**: Optimizer facade 사용
**확인 필요**:
- `quick_optimize()` 메서드 시그니처 확인
- 응답 형식 확인 필요

**수정 필요도**: 중간 (테스트 후 결정)

---

### 8. Web Search API (Line 530-556)
**현재 상태**: WebSearch facade 사용
**확인 필요**:
- `search_async()` 메서드 테스트 필요
- 응답 형식 확인 필요

**수정 필요도**: 중간 (테스트 후 결정)

---

## 우선순위

### 높음 (즉시 수정 필요)
1. **RAG Debug API** - vector_store 필수 파라미터 누락
2. **Multi-Agent API** - 시뮬레이션 코드를 실제 구현으로 교체

### 중간 (테스트 후 수정)
3. **Agent API** - 테스트 및 응답 형식 확인
4. **Knowledge Graph API** - 테스트 및 응답 형식 확인
5. **Orchestrator API** - 테스트 및 응답 형식 확인
6. **Optimizer API** - 테스트 및 응답 형식 확인
7. **Web Search API** - 테스트 및 응답 형식 확인

### 낮음 (선택사항)
8. **Chat API** - Handler 사용으로 변경 (선택사항)

---

## 수정 계획

### Phase 1: 필수 수정 (높음 우선순위)
1. RAG Debug API 수정
2. Multi-Agent API 수정

### Phase 2: 테스트 및 검증 (중간 우선순위)
3. 각 API 엔드포인트 테스트
4. 응답 형식 확인 및 수정
5. 에러 처리 개선

### Phase 3: 선택적 개선 (낮음 우선순위)
6. Chat API Handler 사용으로 변경 (선택사항)
