# 세션별 RAG 자동 관리

## 🎯 목표

세션 생성 시 자동으로 RAG 컬렉션을 생성하고, 문서 업로드 시 자동으로 인덱싱

---

## 📊 현재 문제점

- ❌ RAG 시스템을 먼저 구축해야 함
- ❌ 세션별 RAG 컬렉션이 없음 (전역 "default"만 사용)
- ❌ 문서 업로드가 별도 엔드포인트

---

## ✅ 개선 방안

### 1. 세션 생성 시 RAG 컬렉션 자동 생성

```python
# playground/backend/chat_history.py
@router.post("")
async def create_session(request: CreateSessionRequest):
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    # 세션 생성
    await db.chat_sessions.insert_one(session_data)
    
    # ✅ 세션별 RAG 컬렉션 자동 생성
    from services.session_rag_service import session_rag_service
    await session_rag_service.create_session_rag(session_id)
    
    return SessionResponse(...)
```

### 2. Orchestrator에서 세션 RAG 자동 사용

```python
# playground/backend/services/orchestrator.py
async def _handle_rag(context, tool):
    # 세션 ID가 있으면 세션별 RAG 사용
    if context.session_id:
        collection_name = f"session_{context.session_id}"
    else:
        collection_name = context.extra_params.get("collection_name", "default")
    
    # RAG 인스턴스 가져오기 또는 생성
    rag = await self._get_or_create_session_rag(collection_name)
    
    # RAG 질의
    result = await rag.query(context.query)
```

### 3. 세션 삭제 시 RAG 컬렉션 삭제

```python
@router.delete("/{session_id}")
async def delete_session(session_id: str):
    # 세션 삭제
    await db.chat_sessions.delete_one({"session_id": session_id})
    
    # ✅ RAG 컬렉션도 삭제
    from services.session_rag_service import session_rag_service
    await session_rag_service.delete_session_rag(session_id)
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] 세션 생성 (`routers/history_router.py`의 `create_session`)
- [x] 세션 Vector DB 인덱싱 (`session_search_service.py`의 `index_session`)

### ❌ 미구현
- [ ] **SessionRAGService 생성**
  - **파일**: `playground/backend/services/session_rag_service.py` (신규 생성 필요)
  - **구현 방향**:
    1. 세션별 RAG 컬렉션 관리 (`session_{session_id}` 형식)
    2. MCP Client Service 활용 (`mcp_client_service.call_rag_build`)
    3. 세션별 RAG 인스턴스 캐싱 (메모리 또는 Redis)
  - **구현 방법**:
    ```python
    class SessionRAGService:
        async def create_session_rag(session_id: str) -> None:
            """세션별 RAG 컬렉션 생성 (빈 컬렉션)"""
            collection_name = f"session_{session_id}"
            # MCP tool로 빈 RAG 시스템 생성
            await mcp_client.call_rag_build(
                documents_path="",  # 빈 경로
                collection_name=collection_name
            )
        
        async def add_documents_to_session(
            session_id: str, 
            files: List[UploadFile]
        ) -> Dict[str, Any]:
            """세션에 문서 추가 및 자동 인덱싱"""
            # 1. 파일 저장
            # 2. MCP tool로 문서 추가
            # 3. 진행 상황 스트리밍
    ```
- [ ] **세션 생성 시 RAG 자동 생성**
  - **통합 위치**: `routers/history_router.py`의 `create_session` 함수
  - **방법**: 세션 생성 후 `SessionRAGService.create_session_rag()` 호출
  - **주의**: 비동기로 실행하여 세션 생성 속도에 영향 없도록
- [ ] **Orchestrator 세션 RAG 자동 사용**
  - **통합 위치**: `services/orchestrator.py`의 `_handle_rag` 메서드
  - **방법**: 
    ```python
    if context.session_id:
        collection_name = f"session_{context.session_id}"
        # 세션 RAG가 없으면 자동 생성
        await session_rag_service.ensure_session_rag(context.session_id)
    ```
  - **MCP 통합**: `mcp_client.call_rag_query(collection_name=collection_name)`
- [ ] **세션 삭제 시 RAG 컬렉션 삭제**
  - **통합 위치**: `routers/history_router.py`의 `delete_session` 함수
  - **방법**: MCP tool `delete_rag_system` 호출
  - **주의**: 세션 삭제 실패해도 RAG 삭제는 시도 (에러 로그만)
- [ ] Vector DB 인덱싱 최적화 - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조
- [ ] RAG 컬렉션 최적화 파이프라인 - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조

---

## 🎯 우선순위

**높음**: 문서 기반 챗봇의 핵심 기능

---

## 🔗 관련 문서

- [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md): Vector DB 인덱싱 및 최적화 파이프라인
