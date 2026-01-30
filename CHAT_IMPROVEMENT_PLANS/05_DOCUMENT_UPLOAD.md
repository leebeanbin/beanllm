# 문서 업로드 자동 처리

## 🎯 목표

채팅 인터페이스에서 파일 드래그 앤 드롭으로 문서를 업로드하고 즉시 RAG에 인덱싱

---

## 📊 현재 문제점

- ❌ 문서 업로드가 별도 엔드포인트
- ❌ 업로드 후 수동으로 RAG 구축 필요
- ❌ 진행 상황 표시 없음

---

## ✅ 개선 방안

### 1. 프론트엔드: 파일 업로드 UI

```typescript
// playground/frontend/src/app/chat/page.tsx
const handleFileUpload = async (files: FileList) => {
  const formData = new FormData();
  Array.from(files).forEach(file => {
    formData.append('files', file);
  });
  
  // 세션에 문서 추가
  const response = await fetch(
    `/api/chat/sessions/${sessionId}/documents`,
    {
      method: 'POST',
      body: formData
    }
  );
  
  // SSE로 진행 상황 스트리밍
  const reader = response.body.getReader();
  // ...
};
```

### 2. 백엔드: 세션 문서 추가 엔드포인트

```python
# playground/backend/routers/chat_router.py
@router.post("/sessions/{session_id}/documents")
async def add_session_documents(
    session_id: str,
    files: List[UploadFile] = File(...)
):
    """세션에 문서 추가 및 자동 RAG 구축"""
    from services.session_rag_service import session_rag_service
    
    async def generate():
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={"step": "uploading", "progress": 0.1}
        )
        
        # 문서 추가
        result = await session_rag_service.add_documents_to_session(
            session_id, files
        )
        
        yield AgenticEvent(
            type=EventType.TOOL_RESULT,
            data={"result": result}
        )
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] 파일 업로드 엔드포인트 (`routers/rag_router.py`의 `rag_build_from_files`)
- [x] SSE 스트리밍 기본 구조 (`orchestrator.py`의 `AgenticEvent`)

### ❌ 미구현
- [ ] **파일 업로드 UI 추가**
  - **위치**: `playground/frontend/src/app/chat/page.tsx`
  - **구현 방향**:
    1. 드래그 앤 드롭 영역 추가
    2. 파일 선택 버튼 추가
    3. 업로드 진행 상황 표시 (SSE 이벤트 수신)
  - **방법**:
    ```typescript
    // 파일 드롭 핸들러
    const handleDrop = async (e: DragEvent) => {
      const files = Array.from(e.dataTransfer.files);
      await uploadFiles(files);
    };
    
    // SSE로 진행 상황 수신
    const eventSource = new EventSource(`/api/chat/sessions/${sessionId}/documents`);
    eventSource.addEventListener("tool_progress", (e) => {
      const data = JSON.parse(e.data);
      setUploadProgress(data.progress);
    });
    ```
- [ ] **세션 문서 추가 엔드포인트**
  - **위치**: `routers/chat_router.py` 또는 `routers/rag_router.py`
  - **구현 방향**:
    1. `SessionRAGService` 활용 (04_SESSION_RAG.md 참조)
    2. SSE 스트리밍으로 진행 상황 전달
    3. 파일 저장 → 문서 로드 → RAG 인덱싱 단계별 진행
  - **방법**:
    ```python
    @router.post("/sessions/{session_id}/documents")
    async def add_session_documents(
        session_id: str,
        files: List[UploadFile] = File(...)
    ) -> StreamingResponse:
        async def generate():
            # 1. 파일 저장
            yield AgenticEvent(type=EventType.TOOL_PROGRESS, 
                             data={"step": "uploading", "progress": 0.1})
            
            # 2. SessionRAGService로 문서 추가
            result = await session_rag_service.add_documents_to_session(
                session_id, files
            )
            
            yield AgenticEvent(type=EventType.TOOL_RESULT, data={"result": result})
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    ```
- [ ] **업로드 즉시 RAG 인덱싱**
  - **통합 위치**: `SessionRAGService.add_documents_to_session()`
  - **방법**: MCP tool `add_documents_to_rag` 호출
  - **진행 상황**: 
    - 파일 저장 (10%)
    - 문서 로드 (30%)
    - 청크 분할 (50%)
    - 임베딩 생성 (70%)
    - Vector DB 저장 (90%)
    - 완료 (100%)
- [ ] **진행 상황 SSE 스트리밍**
  - **현재**: `AgenticEvent` 구조는 있음
  - **필요**: 문서 업로드 전용 이벤트 타입 추가
  - **방법**: `EventType.DOCUMENT_UPLOAD`, `EventType.DOCUMENT_INDEXING` 추가

---

## 🎯 우선순위

**높음**: 사용자 경험 개선
