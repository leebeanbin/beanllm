# 컨텍스트 정리 및 메모리 관리

## 🎯 목표

일정량의 대화가 쌓이면 자동으로 컨텍스트를 정리하여:
1. 토큰 제한 내에서 최대한 많은 컨텍스트 유지
2. 중요한 정보는 보존하면서 오래된 정보는 요약
3. 사용자에게 요약 과정을 명확히 표시

---

## 📊 현재 상태

### 구현된 기능
- ✅ `SummaryMemory`: 오래된 대화 자동 요약 (beanllm.domain.memory)
- ✅ `TokenMemory`: 토큰 수 기준 컨텍스트 관리
- ✅ `WindowMemory`: 최근 N개 메시지만 유지
- ✅ `BufferMemory`: 모든 메시지 저장

### 문제점
- ❌ playground에서 활용 안 됨
- ❌ 요약 전략이 구체적이지 않음
- ❌ 저장 및 전달 방식이 명확하지 않음

---

## ✅ 개선 방안

### 1. 요약 전략 (어떻게 요약할지)

#### A. 요약 트리거 조건

**옵션 1: 메시지 수 기반 (권장)**
```python
# 20개 메시지 초과 시 요약
max_messages = 20
summary_trigger = 15  # 15개 초과 시 요약 시작
```

**옵션 2: 토큰 수 기반**
```python
# 4000 토큰 초과 시 요약
max_tokens = 4000
token_threshold = 3500  # 3500 토큰 초과 시 요약 시작
```

**옵션 3: 하이브리드 (권장)**
```python
# 메시지 수와 토큰 수 모두 고려
if message_count > 20 or estimated_tokens > 3500:
    trigger_summarization()
```

#### B. 요약 프롬프트 전략 (동적 구성)

**전략 1: 핵심 정보 보존 (권장) - 동적 구성**
```python
from services.prompt_builder import PromptBuilder

prompt_builder = PromptBuilder()

def build_summarization_prompt(
    conversation_history: str,
    session_context: Dict[str, Any],
    previous_summaries: Optional[List[str]] = None
) -> str:
    """
    요약 프롬프트 동적 구성
    
    컨텍스트에 따라 프롬프트를 개선
    """
    base_template = PromptTemplate(
        template="""다음 대화 내용을 요약해주세요. 다음 정보는 반드시 포함해야 합니다:

1. 주요 주제 및 목적
2. 중요한 결정 사항
3. 사용자가 언급한 특별한 요구사항
4. 해결된 문제와 해결 방법
5. 미완료된 작업이나 다음 단계

대화 내용:
{conversation_history}

요약 (200-300자):""",
        input_variables=["conversation_history"]
    )
    
    # 세션 컨텍스트 반영
    if session_context.get("uploaded_files"):
        base_template = prompt_builder.optimizer.add_instructions(
            base_template.format(conversation_history=conversation_history),
            [f"세션에 업로드된 파일: {', '.join(session_context['uploaded_files'])}"]
        )
    
    # 이전 요약이 있으면 연속성 유지
    if previous_summaries:
        base_template = prompt_builder.optimizer.add_instructions(
            base_template,
            [f"이전 요약: {previous_summaries[-1]}", "연속성을 유지하며 요약하세요."]
        )
    
    return base_template
```

**전략 2: 구조화된 요약 - 동적 구성**
```python
def build_structured_summarization_prompt(
    conversation_history: str,
    intent_history: List[str],
    tool_usage_history: List[str]
) -> str:
    """
    구조화된 요약 프롬프트 (컨텍스트 기반)
    
    사용된 도구와 의도를 반영하여 요약
    """
    composer = PromptComposer()
    
    # 기본 구조
    composer.add_text("""
다음 대화를 구조화된 형식으로 요약해주세요:

주제: [대화의 주요 주제]
목적: [사용자의 목적]
주요 내용:
- [핵심 포인트 1]
- [핵심 포인트 2]
- [핵심 포인트 3]
중요 정보: [보존해야 할 특별한 정보]
다음 단계: [미완료 작업이나 다음 단계]
""")
    
    # 사용된 도구 정보 추가
    if tool_usage_history:
        composer.add_text(f"사용된 도구: {', '.join(set(tool_usage_history))}")
    
    # 의도 변화 추적
    if len(intent_history) > 1:
        composer.add_text(f"의도 변화: {' → '.join(intent_history[-3:])}")
    
    # 대화 내용
    composer.add_template(
        PromptTemplate(
            template="대화 내용:\n{conversation_history}",
            input_variables=["conversation_history"]
        )
    )
    
    return composer.compose(conversation_history=conversation_history)
```

**전략 3: 계층적 요약 (긴 대화용)**
```python
# 1단계: 대화를 청크로 나누어 각각 요약
# 2단계: 요약된 청크들을 다시 요약
HIERARCHICAL_SUMMARIZATION = """
1단계: 대화를 5-10개 메시지 단위로 나누어 각각 요약
2단계: 요약된 내용들을 통합하여 최종 요약
"""
```

#### C. 요약 모델 선택

**옵션 1: 동일 모델 사용**
```python
# 사용자가 사용 중인 모델로 요약
summarizer = Client(model=context.model)  # 예: qwen2.5:0.5b
```

**옵션 2: 전용 요약 모델**
```python
# 요약 전용 경량 모델 사용 (비용 절감)
summarizer = Client(model="qwen2.5:0.5b")  # 빠르고 저렴
```

**옵션 3: 사용자 선택**
```python
# 사용자가 요약 모델 선택 가능
summarizer = Client(model=user_settings.get("summary_model", "qwen2.5:0.5b"))
```

---

### 2. 저장 전략 (어떻게 저장할지)

#### A. 하이브리드 저장 구조

**구조**:
```
MongoDB (메타데이터)
├─ session_id
├─ summary (요약된 내용)
├─ summary_created_at
├─ message_count
└─ recent_messages (최근 10개 메시지)

Vector DB (전체 메시지)
├─ 모든 메시지 내용 (임베딩)
├─ 요약도 별도 문서로 저장
└─ 세션별 컬렉션
```

#### B. 저장 구현

```python
# playground/backend/services/context_manager.py
class ContextManager:
    """
    컨텍스트 관리 서비스
    
    요약, 저장, 전달을 통합 관리
    """
    
    def __init__(self, session_id: str):
        from beanllm.domain.memory import create_memory
        from services.message_vector_store import message_vector_store
        
        self.session_id = session_id
        self.memory = create_memory(
            "summary",
            max_messages=20,
            summary_trigger=15
        )
        self.message_vector_store = message_vector_store
    
    async def add_message(
        self,
        role: str,
        content: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """메시지 추가 및 자동 요약"""
        # 1. 메모리에 추가
        self.memory.add_message(role, content, model=model, **metadata)
        
        # 2. Vector DB에 저장
        await self.message_vector_store.save_message(
            session_id=self.session_id,
            message_id=f"{self.session_id}_{uuid.uuid4().hex[:8]}",
            role=role,
            content=content,
            model=model,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        
        # 3. 요약 트리거 확인
        if len(self.memory.messages) > self.memory.summary_trigger:
            await self._summarize_if_needed()
    
    async def _summarize_if_needed(self):
        """필요 시 요약 실행 (프롬프트 동적 구성)"""
        # 요약이 이미 생성되었는지 확인
        if self.memory.summary:
            return
        
        # 요약할 메시지 선택 (오래된 메시지들)
        messages_to_summarize = self.memory.messages[:-10]  # 최근 10개 제외
        
        # 프롬프트 동적 구성
        from services.prompt_builder import PromptBuilder
        prompt_builder = PromptBuilder()
        
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in messages_to_summarize
        ])
        
        # 세션 컨텍스트 수집
        session_context = {
            "uploaded_files": await self._get_session_files(),
            "intent_history": await self._get_intent_history(),
            "tool_usage": await self._get_tool_usage_history()
        }
        
        # 동적 프롬프트 구성
        summarization_prompt = prompt_builder.build_summarization_prompt(
            conversation_history=conversation_text,
            session_context=session_context,
            previous_summaries=[self.memory.summary] if self.memory.summary else None
        )
        messages_to_summarize = self.memory.messages[:-10]  # 최근 10개 제외
        
        if len(messages_to_summarize) < 5:
            return  # 요약할 메시지가 너무 적음
        
        # 요약 생성
        summary = await self._generate_summary(messages_to_summarize)
        
        # 메모리에 저장
        self.memory.summary = summary
        
        # MongoDB에 요약 저장
        await self._save_summary_to_mongodb(summary)
        
        # Vector DB에도 요약을 별도 문서로 저장
        await self._save_summary_to_vector_db(summary)
    
    async def _generate_summary(self, messages: List[Message]) -> str:
        """LLM을 사용하여 요약 생성"""
        from beanllm.facade.core import Client
        
        # 대화 내용 구성
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])
        
        # 요약 프롬프트 (핵심 정보 보존)
        SUMMARIZATION_PROMPT = """
다음 대화 내용을 요약해주세요. 다음 정보는 반드시 포함해야 합니다:

1. 주요 주제 및 목적
2. 중요한 결정 사항
3. 사용자가 언급한 특별한 요구사항
4. 해결된 문제와 해결 방법
5. 미완료된 작업이나 다음 단계

대화 내용:
{conversation_history}

요약 (200-300자):
"""
        
        prompt = SUMMARIZATION_PROMPT.format(
            conversation_history=conversation_text
        )
        
        # 요약 모델 선택 (경량 모델 사용)
        summarizer = Client(model="qwen2.5:0.5b")
        
        # 요약 생성
        response = await summarizer.chat([
            {"role": "user", "content": prompt}
        ])
        
        return response.content
    
    async def _save_summary_to_mongodb(self, summary: str):
        """MongoDB에 요약 저장"""
        from database import get_mongodb_database
        db = get_mongodb_database()
        
        await db.chat_sessions.update_one(
            {"session_id": self.session_id},
            {
                "$set": {
                    "summary": summary,
                    "summary_created_at": datetime.now(timezone.utc),
                    "summary_message_count": len(self.memory.messages)
                }
            }
        )
    
    async def _save_summary_to_vector_db(self, summary: str):
        """Vector DB에 요약 저장 (검색 가능하도록)"""
        await self.message_vector_store.save_message(
            session_id=self.session_id,
            message_id=f"{self.session_id}_summary_{uuid.uuid4().hex[:8]}",
            role="system",
            content=f"[요약] {summary}",
            model="summary",
            timestamp=datetime.now(timezone.utc),
            metadata={"type": "summary"}
        )
    
    async def get_context_for_llm(
        self,
        query: Optional[str] = None,
        use_query_refinement: bool = True
    ) -> List[Dict[str, str]]:
        """
        LLM에 전달할 컨텍스트 가져오기 (쿼리 재구성 포함)
        
        Args:
            query: 현재 쿼리 (재구성 대상)
            use_query_refinement: 쿼리 재구성 사용 여부
        """
        messages = []
        
        # 1. 요약이 있으면 system 메시지로 추가 (프롬프트 동적 구성)
        if self.memory.summary:
            from services.prompt_builder import PromptBuilder
            prompt_builder = PromptBuilder()
            
            summary_prompt = prompt_builder.build_context_prompt(
                summary=self.memory.summary,
                session_context=await self._get_session_context()
            )
            
            messages.append({
                "role": "system",
                "content": summary_prompt
            })
        
        # 2. 쿼리 재구성 (필요 시)
        if query and use_query_refinement:
            from services.query_refiner import QueryRefiner
            refiner = QueryRefiner()
            
            # 이전 쿼리 재구성 히스토리 가져오기
            previous_refinements = await self._get_query_refinement_history()
            
            refined_query = await refiner.refine_query(
                original_query=query,
                session_id=self.session_id,
                previous_results=previous_refinements
            )
            
            # 재구성된 쿼리 사용
            query = refined_query
        
        # 3. 최근 메시지 추가
        recent_messages = self.memory.get_messages()
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # 4. 현재 쿼리 추가 (재구성된 경우)
        if query:
            messages.append({
                "role": "user",
                "content": query
            })
        
        return messages
```

---

### 3. 전달 전략 (어떻게 전달할지)

#### A. SSE 스트리밍으로 요약 과정 표시

```python
# playground/backend/services/orchestrator.py (업데이트)
async def _handle_chat(
    self,
    context: OrchestratorContext,
    tool: Tool
) -> AsyncGenerator[AgenticEvent, None]:
    """Chat 도구 핸들러 (컨텍스트 관리 통합)"""
    from services.context_manager import ContextManager
    
    # 컨텍스트 매니저 가져오기
    context_manager = await get_context_manager(context.session_id)
    
    # 요약이 필요한지 확인
    if context_manager.memory.needs_summarization():
        # 요약 시작 이벤트
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "step": "summarizing",
                "message": "대화 내용을 요약 중입니다...",
                "progress": 0.1
            }
        )
        
        # 요약 실행
        await context_manager._summarize_if_needed()
        
        # 요약 완료 이벤트
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "step": "summarized",
                "message": f"요약 완료: {context_manager.memory.summary[:100]}...",
                "progress": 0.2
            }
        )
    
    # 컨텍스트 가져오기
    messages = await context_manager.get_context_for_llm()
    
    # 사용자 메시지 추가
    messages.append({"role": "user", "content": context.query})
    
    # LLM 호출
    # ...
    
    # 응답을 컨텍스트에 추가
    await context_manager.add_message("assistant", response.content, context.model)
```

#### B. 프론트엔드에서 요약 표시

```typescript
// playground/frontend/src/app/chat/page.tsx
const handleSSEEvent = (event: MessageEvent) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "tool_progress") {
    if (data.data.step === "summarizing") {
      // 요약 중 표시
      setStatusMessage("대화 내용을 요약 중입니다...");
    } else if (data.data.step === "summarized") {
      // 요약 완료 표시
      setStatusMessage(`요약 완료: ${data.data.message}`);
      // 요약 내용을 사이드바에 표시
      setSummary(data.data.message);
    }
  }
};
```

#### C. 요약 캐싱 및 전달

**1. Redis 캐싱**
```python
# Redis에 요약 캐시
async def get_cached_summary(session_id: str) -> Optional[str]:
    """캐시된 요약 가져오기"""
    from services.session_cache import session_cache
    
    cached = await session_cache.get(f"summary:{session_id}")
    if cached:
        return cached
    
    # MongoDB에서 가져오기
    db = get_mongodb_database()
    session = await db.chat_sessions.find_one({"session_id": session_id})
    if session and session.get("summary"):
        # Redis에 캐시
        await session_cache.set(
            f"summary:{session_id}",
            session["summary"],
            ttl=3600  # 1시간
        )
        return session["summary"]
    
    return None
```

**2. LLM에 전달하는 방식**
```python
async def get_context_for_llm(self) -> List[Dict[str, str]]:
    """LLM에 전달할 컨텍스트 가져오기"""
    messages = []
    
    # 1. 요약이 있으면 system 메시지로 추가
    if self.memory.summary:
        messages.append({
            "role": "system",
            "content": f"""이전 대화 요약:
{self.memory.summary}

최근 대화:"""
        })
    
    # 2. 최근 메시지 추가 (요약 제외)
    recent_messages = self.memory.get_messages()
    for msg in recent_messages:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    return messages
```

**3. 요약 업데이트 시 전달**
```python
# 요약이 새로 생성되면 사용자에게 알림
if new_summary_created:
    yield AgenticEvent(
        type=EventType.TOOL_PROGRESS,
        data={
            "step": "context_summarized",
            "message": "대화 내용을 요약했습니다. 이전 대화의 핵심 내용은 기억하고 있습니다.",
            "summary_preview": summary[:100] + "..."
        }
    )
```

---

## 📋 구현 체크리스트

### 요약 전략
- [x] 요약 트리거 조건 결정 (메시지 수/토큰 수) ✅ (2025-01-26)
- [x] 요약 프롬프트 작성 (핵심 정보 보존) ✅ (2025-01-26)
- [x] 요약 모델 선택 (경량 모델) ✅ (2025-01-26)
- [x] 요약 생성 구현 (`context_manager.py`의 `summarize_if_needed`) ✅
- [ ] **프롬프트 동적 구성** (컨텍스트 기반)
  - **현재**: 고정된 요약 프롬프트 사용
  - **필요**: 세션 컨텍스트, 이전 요약, 도구 이력 반영
  - **통합 위치**: `context_manager.py`의 `_generate_summary()` 메서드
  - **방법**: `PromptBuilder` 서비스 활용 (07_INTENT_CLASSIFIER.md 참조)
  - [ ] 세션 컨텍스트 반영
  - [ ] 이전 요약 연속성 유지
  - [ ] 도구 사용 이력 반영
- [ ] 요약 품질 검증

### 저장 전략
- [x] ContextManager 생성 ✅ (2025-01-25)
- [x] 요약 생성 및 메모리 저장 ✅ (`context_manager.py`)
- [ ] **MongoDB에 요약 저장**
  - **현재**: 메모리에만 저장 (`_session_summaries` dict)
  - **필요**: MongoDB `chat_sessions` 컬렉션에 `summary` 필드 저장
  - **통합 위치**: `context_manager.py`의 `summarize_if_needed()` 메서드
  - **방법**:
    ```python
    # summarize_if_needed() 메서드에 추가
    from database import get_mongodb_database
    db = get_mongodb_database()
    await db.chat_sessions.update_one(
        {"session_id": session_id},
        {"$set": {
            "summary": summary,
            "summary_created_at": datetime.now(timezone.utc),
            "summary_message_count": len(messages_to_summarize)
        }}
    )
    ```
- [ ] **Vector DB에 요약 저장 (검색 가능)**
  - **통합 위치**: `message_vector_store.py`
  - **방법**: 요약을 별도 메시지로 저장 (role="system", type="summary")
- [ ] 하이브리드 저장 구조 구현
- [ ] MongoDB 인덱싱 (요약 검색 최적화) - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조
- [ ] Vector DB 인덱싱 (요약 검색 최적화) - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조
- [ ] **쿼리 재구성 히스토리 저장**
  - **파일**: MongoDB `query_refinements` 컬렉션 (신규)
  - **구조**:
    ```python
    {
        "refinement_id": str,
        "session_id": str,
        "original_query": str,
        "refined_query": str,
        "refinement_type": "feedback" | "relevance" | "hyde",
        "user_feedback": Optional[str],
        "success": bool,  # 재구성 후 검색 성공 여부
        "improvement_score": Optional[float],  # 개선도 (0.0-1.0)
        "created_at": datetime
    }
    ```
  - **통합 위치**: `QueryRefiner.refine_query()` 메서드 (07_INTENT_CLASSIFIER.md 참조)
  - [ ] 원본 쿼리 → 재구성된 쿼리 매핑
  - [ ] 피드백 기반 재구성 기록
  - [ ] 재구성 효과 추적 (성공률, 개선도)
- [ ] **프롬프트 구성 히스토리 저장**
  - **파일**: MongoDB `prompt_history` 컬렉션 (신규)
  - **구조**:
    ```python
    {
        "prompt_id": str,
        "session_id": str,
        "prompt_type": "rag" | "intent" | "summary" | "chat",
        "base_template": str,
        "final_prompt": str,
        "modifications": Dict[str, Any],  # 추가된 instructions, constraints 등
        "effectiveness": Optional[float],  # 효과성 점수
        "created_at": datetime
    }
    ```
  - **통합 위치**: `PromptBuilder` 각 메서드에서 (07_INTENT_CLASSIFIER.md 참조)
  - [ ] 사용된 프롬프트 템플릿 기록
  - [ ] 프롬프트 변형 이력
  - [ ] 프롬프트 효과 분석

### 전달 전략
- [x] SSE로 요약 과정 표시 ✅ (2025-01-26)
- [x] 요약된 컨텍스트를 LLM에 전달 ✅ (`get_context_with_summary()` 구현됨)
- [ ] **프론트엔드에서 요약 표시**
  - **위치**: `playground/frontend/src/app/chat/page.tsx`
  - **방법**: SSE 이벤트 `tool_progress`에서 `step="summarized"` 처리
  - **UI**: 사이드바 또는 상단에 요약 표시
- [ ] **요약 캐싱 (Redis)**
  - **통합 위치**: `context_manager.py`의 `summarize_if_needed()`
  - **방법**: 요약 생성 전 Redis 캐시 확인, 생성 후 캐시 저장
  - **캐시 키**: `summary:{session_id}:{message_count}`
- [ ] **쿼리 재구성 통합** (전달 시)
  - **통합 위치**: `context_manager.py`의 `get_context_for_llm()` 메서드
  - **방법**: 쿼리 재구성 옵션이 있으면 `QueryRefiner` 활용
  - **주의**: 요약 생성에는 영향 없도록 (요약은 원본 메시지 기반)
- [ ] Redis 인덱싱 (요약 캐시 최적화) - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조

---

## 🎯 최종 구조

```
메시지 추가
    ↓
메모리에 저장 (SummaryMemory)
    ↓
20개 초과? → 요약 트리거
    ↓
LLM으로 요약 생성
    ↓
MongoDB에 요약 저장
    ↓
Vector DB에 요약 저장 (검색 가능)
    ↓
최근 10개 메시지 + 요약을 LLM에 전달
```

---

## 💡 핵심 원칙

1. **자동 요약**: 20개 메시지 초과 시 자동 요약
2. **핵심 보존**: 중요한 정보는 반드시 요약에 포함
3. **검색 가능**: 요약도 Vector DB에 저장하여 검색 가능
4. **사용자 인지**: 요약 과정을 SSE로 실시간 표시
