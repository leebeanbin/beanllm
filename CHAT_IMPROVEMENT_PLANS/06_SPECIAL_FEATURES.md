# 특화 기능 처리 전략

## 🎯 목표

일반 기능은 자동 감지, 특화 기능은 사용자 선택 또는 LLM 분석

---

## 📊 기능 분류

### 자동 감지 가능 (일반 기능)
- `chat`: 기본 대화
- `rag`: 문서 검색 및 질의응답
- `web_search`: 웹 검색
- `code`: 코드 생성/분석
- `ocr`: 이미지 텍스트 추출
- `vision`: 이미지 분석

### 사용자 선택 또는 LLM 분석 필요 (특화 기능)
- `multi_agent`: 멀티 에이전트 토론/협업
- `knowledge_graph`: 지식 그래프 구축 및 탐색
- `audio_transcribe`: 음성 전사
- `evaluation`: 모델/RAG 평가

---

## ✅ 처리 전략

### 1. 일반 기능: 자동 감지
```
사용자 입력: "문서에서 AI 찾아줘"
    ↓
Intent Classifier (Rule-based + LLM)
    ↓
자동으로 RAG 도구 선택 및 실행
```

### 2. 특화 기능: 3가지 방식

**방식 1: 사용자 명시적 선택**
```
사용자가 특화 기능 버튼 클릭
    ↓
"멀티 에이전트로 토론해줘" 입력
    ↓
force_intent="multi_agent"로 전달
```

**방식 2: LLM 자동 분석**
```
사용자 입력: "이미지에서 텍스트 추출해줘"
    ↓
Intent Classifier (LLM fallback)
    ↓
LLM이 "ocr" intent로 분류
```

**방식 3: 컨텍스트 기반 자동 선택**
```
대화 히스토리: "이미지 업로드했어"
현재 입력: "텍스트 추출해줘"
    ↓
Intent Classifier가 컨텍스트 고려
    ↓
OCR 도구 자동 선택
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] Intent Classifier 기본 구현 (`intent_classifier.py`)
- [x] Rule-based 분류
- [x] LLM fallback 기본 구현

### ⚠️ 부분 구현
- [ ] Intent Classifier LLM fallback 강화
  - **현재**: 기본 LLM 분류만
  - **필요**: 쿼리 재구성, Ensemble Prompting (07_INTENT_CLASSIFIER.md 참조)
  - **통합 위치**: `intent_classifier.py`의 `_classify_by_llm` 메서드 확장

### ❌ 미구현
- [ ] **파일 타입 기반 자동 선택**
  - **통합 위치**: `intent_classifier.py`의 `classify()` 메서드
  - **구현 방향**:
    1. 세션에 업로드된 파일 정보 조회 (MongoDB `chat_sessions` 컬렉션)
    2. 파일 타입에 따라 Intent 우선순위 조정
    3. 이미지 파일 + "텍스트" 키워드 → OCR 우선
  - **방법**:
    ```python
    async def _get_uploaded_files(self, session_id: str) -> List[Dict]:
        """세션에 업로드된 파일 정보 조회"""
        from database import get_mongodb_database
        db = get_mongodb_database()
        session = await db.chat_sessions.find_one({"session_id": session_id})
        return session.get("uploaded_files", []) if session else []
    
    # classify() 메서드에 추가
    if session_id:
        files = await self._get_uploaded_files(session_id)
        if files and any(f['type'].startswith('image/') for f in files):
            if any(kw in query.lower() for kw in ["텍스트", "글자", "ocr"]):
                return IntentResult(primary_intent=IntentType.OCR, confidence=0.95)
    ```
- [ ] **컨텍스트 기반 분류**
  - **통합 위치**: `intent_classifier.py`의 `classify()` 메서드
  - **구현 방향**:
    1. 이전 대화 메시지 조회 (ContextManager 활용)
    2. 세션에 문서 존재 여부 확인 (SessionRAGService 활용)
    3. 컨텍스트 키워드로 Intent 조정
  - **방법**:
    ```python
    from services.context_manager import context_manager
    
    if session_id:
        # 이전 메시지 확인
        previous_messages = context_manager.get_context(session_id, as_dict=True)
        if previous_messages:
            last_message = previous_messages[-1].get("content", "")
            # "이미지 업로드" + "텍스트 추출" → OCR
            if "이미지" in last_message.lower() and "텍스트" in query.lower():
                return IntentResult(primary_intent=IntentType.OCR, confidence=0.9)
        
        # 세션에 문서 확인
        from services.session_rag_service import session_rag_service
        has_docs = await session_rag_service.has_documents(session_id)
        if has_docs and any(kw in query.lower() for kw in ["찾아", "검색"]):
            return IntentResult(primary_intent=IntentType.RAG, confidence=0.9)
    ```
- [ ] **특화 기능 버튼 UI (선택적)**
  - **위치**: `playground/frontend/src/app/chat/page.tsx`
  - **구현 방향**: 02_AGENTIC_MODE.md의 "옵션 B" 참조
  - **방법**: FeatureSelector를 특화 기능만 선택 가능하도록 변경

---

## 🎯 우선순위

**중간**: 사용자 경험 개선
