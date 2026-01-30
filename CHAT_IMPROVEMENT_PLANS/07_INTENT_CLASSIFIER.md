# Intent Classifier 개선

## 🎯 목표

LLM fallback 강화 및 컨텍스트 기반 분류로 특화 기능 정확도 향상

---

## 📊 현재 상태

- ✅ Rule-based 분류 구현됨
- ✅ LLM fallback 기본 구현됨
- ⚠️ 컨텍스트 활용 부족

---

## ✅ 개선 방안

### 1. LLM Fallback 강화 (쿼리 재구성 포함)

```python
# playground/backend/services/intent_classifier.py
async def _classify_by_llm_enhanced(
    self,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_feedback: Optional[str] = None  # 사용자 피드백 추가
) -> IntentResult:
    """강화된 LLM 분류 (쿼리 재구성 포함)"""
    
    # 1. 쿼리 재구성 (사용자 피드백 기반)
    reformulated_query = await self._reformulate_query(query, user_feedback, context)
    
    # 2. 세션 컨텍스트 포함
    if session_id:
        session_info = await self._get_session_info(session_id)
        # - 업로드된 파일 타입
        # - 이전 대화 내용
        # - 사용된 도구 이력
        # - 이전 쿼리 재구성 히스토리
    
    # 3. Ensemble Prompting (GenQREnsemble 방식)
    # 여러 프롬프트 변형으로 의도 분류
    classification_prompts = self._generate_ensemble_prompts(reformulated_query, context)
    
    # 4. LLM 호출 (여러 프롬프트로 분류)
    results = []
    for prompt in classification_prompts:
        result = await self._llm_classify(prompt)
        results.append(result)
    
    # 5. 결과 통합 (다수결 또는 가중 평균)
    final_result = self._aggregate_classification_results(results)
    
    return final_result

async def _reformulate_query(
    self,
    query: str,
    user_feedback: Optional[str],
    context: Optional[Dict[str, Any]]
) -> str:
    """
    쿼리 재구성 (사용자 피드백 기반)
    
    참고 기법:
    - GenQREnsemble: Ensemble Prompting으로 여러 쿼리 생성
    - GenQRFusion: Document Fusion과 Relevance Feedback 활용
    - QueryGym: 표준화된 쿼리 재구성 프레임워크
    """
    if not user_feedback:
        return query
    
    # 사용자 피드백을 반영한 쿼리 재구성
    reformulation_prompt = f"""
    사용자 원본 쿼리: {query}
    사용자 피드백: {user_feedback}
    컨텍스트: {context.get('previous_messages', [])[-3:] if context else []}
    
    사용자의 피드백을 반영하여 쿼리를 재구성해주세요.
    의도를 더 명확하게 표현하되, 원본 의미는 유지하세요.
    
    재구성된 쿼리:
    """
    
    from beanllm import Client
    client = Client(model="gpt-4o-mini")
    response = await client.chat([{"role": "user", "content": reformulation_prompt}])
    
    return response.content.strip()

def _generate_ensemble_prompts(
    self,
    query: str,
    context: Optional[Dict[str, Any]]
) -> List[str]:
    """
    Ensemble Prompting (GenQREnsemble 방식)
    
    여러 프롬프트 변형을 생성하여 분류 정확도 향상
    """
    base_prompt = f"""
    사용자 쿼리: {query}
    컨텍스트: {context}
    
    다음 의도 중 하나로 분류하세요:
    - chat, rag, agent, multi_agent, kg, web_search, audio, vision, ocr, code, evaluation
    """
    
    # 프롬프트 변형 생성 (paraphrase)
    prompts = [
        base_prompt,  # 원본
        f"분석: {query} → 의도는?",  # 간결형
        f"다음 쿼리의 의도를 분류: {query}",  # 명시형
        f"사용자가 '{query}'라고 했을 때 필요한 기능은?",  # 기능 중심
    ]
    
    return prompts
```

### 2. 컨텍스트 기반 분류

```python
async def classify(
    self,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> IntentResult:
    """컨텍스트 기반 Intent 분류"""
    
    # 파일 타입 기반 자동 선택
    if session_id:
        uploaded_files = await self._get_uploaded_files(session_id)
        if uploaded_files:
            # 이미지 파일이 있으면 OCR/Vision 우선
            if any(f['type'].startswith('image/') for f in uploaded_files):
                if any(kw in query.lower() for kw in ["텍스트", "글자", "ocr"]):
                    return IntentResult(
                        primary_intent=IntentType.OCR,
                        confidence=0.95
                    )
    
    # 세션에 문서가 있으면 RAG 우선
    if session_id and await self._has_documents(session_id):
        if any(kw in query.lower() for kw in ["찾아", "검색", "알려"]):
            return IntentResult(
                primary_intent=IntentType.RAG,
                confidence=0.9
            )
    
    # 기존 분류 로직
    return await self._classify_by_rules(query)
```

---

### 3. 쿼리 재구성 서비스 (신규)

**파일**: `playground/backend/services/query_refiner.py` (신규)

```python
from beanllm.domain.retrieval.query_expansion import HyDEExpander, MultiQueryExpander
from beanllm.domain.evaluation.human_feedback import HumanFeedbackCollector

class QueryRefiner:
    """
    쿼리 재구성 서비스
    
    사용자 피드백을 받아 쿼리를 개선하고 프롬프트를 재구성
    """
    
    def __init__(self):
        self.feedback_collector = HumanFeedbackCollector()
        self.query_history: Dict[str, List[str]] = {}  # session_id -> [queries]
    
    async def refine_query(
        self,
        original_query: str,
        user_feedback: Optional[str] = None,
        session_id: Optional[str] = None,
        previous_results: Optional[List[Any]] = None
    ) -> str:
        """
        쿼리 재구성 (피드백 기반)
        
        기법:
        1. 사용자 피드백 반영
        2. 이전 결과 기반 Relevance Feedback
        3. HyDE로 가상 문서 생성 후 재구성
        """
        # 1. 피드백이 있으면 직접 반영
        if user_feedback:
            refined = await self._apply_feedback(original_query, user_feedback)
            return refined
        
        # 2. 이전 결과 기반 재구성 (Relevance Feedback)
        if previous_results:
            refined = await self._apply_relevance_feedback(
                original_query, previous_results
            )
            return refined
        
        # 3. HyDE로 가상 문서 생성 후 재구성
        from beanllm import Client
        client = Client(model="gpt-4o-mini")
        
        hyde_expander = HyDEExpander(
            llm_function=lambda p: client.chat([{"role": "user", "content": p}]).content
        )
        hypothetical_doc = hyde_expander.expand(original_query)
        
        # 가상 문서를 기반으로 쿼리 재구성
        reformulation_prompt = f"""
        원본 쿼리: {original_query}
        가상 답변: {hypothetical_doc[:500]}
        
        가상 답변을 기반으로 검색에 더 적합한 쿼리로 재구성해주세요.
        """
        
        response = await client.chat([{"role": "user", "content": reformulation_prompt}])
        return response.content.strip()
    
    async def _apply_feedback(
        self,
        query: str,
        feedback: str
    ) -> str:
        """사용자 피드백 반영"""
        from beanllm import Client
        client = Client(model="gpt-4o-mini")
        
        prompt = f"""
        원본 쿼리: {query}
        사용자 피드백: {feedback}
        
        피드백을 반영하여 쿼리를 개선해주세요.
        """
        
        response = await client.chat([{"role": "user", "content": prompt}])
        return response.content.strip()
    
    async def _apply_relevance_feedback(
        self,
        query: str,
        results: List[Any]
    ) -> str:
        """
        Relevance Feedback 기반 재구성 (GenQRFusion 방식)
        
        검색 결과 중 관련성 높은 문서를 활용하여 쿼리 재구성
        """
        # 상위 3개 결과 추출
        top_results = results[:3]
        relevant_context = "\n".join([
            f"[{i+1}] {r.document.content[:200]}" 
            for i, r in enumerate(top_results)
        ])
        
        from beanllm import Client
        client = Client(model="gpt-4o-mini")
        
        prompt = f"""
        원본 쿼리: {query}
        관련 문서:
        {relevant_context}
        
        관련 문서를 참고하여 검색 품질을 개선할 수 있도록 쿼리를 재구성해주세요.
        """
        
        response = await client.chat([{"role": "user", "content": prompt}])
        return response.content.strip()
```

### 4. 프롬프트 재구성 서비스 (신규)

**파일**: `playground/backend/services/prompt_builder.py` (신규)

```python
from beanllm.domain.prompts.templates import PromptTemplate
from beanllm.domain.prompts.composer import PromptComposer
from beanllm.domain.prompts.optimizer import PromptOptimizer

class PromptBuilder:
    """
    프롬프트 동적 구성 서비스
    
    컨텍스트, 사용자 피드백, 이전 결과를 기반으로 프롬프트 재구성
    """
    
    def __init__(self):
        self.optimizer = PromptOptimizer()
        self.composer = PromptComposer()
    
    def build_rag_prompt(
        self,
        query: str,
        context: str,
        user_feedback: Optional[str] = None,
        previous_attempts: Optional[List[Dict]] = None
    ) -> str:
        """
        RAG 프롬프트 동적 구성
        
        Args:
            query: 사용자 쿼리
            context: 검색된 문서 컨텍스트
            user_feedback: 사용자 피드백 (이전 답변에 대한)
            previous_attempts: 이전 시도 기록
        """
        # 기본 프롬프트
        base_template = PromptTemplate(
            template="""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # 사용자 피드백이 있으면 프롬프트 개선
        if user_feedback:
            base_template = self.optimizer.add_instructions(
                base_template.format(context=context, question=query),
                [
                    f"사용자 피드백: {user_feedback}",
                    "이 피드백을 반영하여 더 정확하고 유용한 답변을 제공하세요."
                ]
            )
        
        # 이전 시도가 있으면 학습 반영
        if previous_attempts:
            lessons = self._extract_lessons(previous_attempts)
            base_template = self.optimizer.add_instructions(
                base_template,
                [f"이전 시도에서 배운 점: {lessons}"]
            )
        
        return base_template
    
    def build_intent_classification_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        available_tools: List[str]
    ) -> str:
        """Intent 분류 프롬프트 구성"""
        composer = PromptComposer()
        
        # 시스템 프롬프트
        composer.add_text(
            "You are an intent classifier. Classify the user's query into one of the available intents."
        )
        
        # 사용 가능한 도구 목록
        tools_text = "\n".join([f"- {tool}" for tool in available_tools])
        composer.add_template(
            PromptTemplate(
                template="Available intents: {tools}",
                input_variables=["tools"]
            )
        )
        
        # 컨텍스트 정보
        if context.get("session_files"):
            composer.add_text(
                f"Session has files: {', '.join(context['session_files'])}"
            )
        
        # 사용자 쿼리
        composer.add_template(
            PromptTemplate(
                template="User query: {query}",
                input_variables=["query"]
            )
        )
        
        return composer.compose(tools=tools_text, query=query)
    
    def _extract_lessons(self, previous_attempts: List[Dict]) -> str:
        """이전 시도에서 교훈 추출"""
        # 실패한 시도 분석
        failed = [a for a in previous_attempts if not a.get("success")]
        if not failed:
            return "이전 시도가 모두 성공적이었습니다."
        
        # 공통 실패 패턴 추출
        patterns = []
        for attempt in failed:
            if "error" in attempt:
                patterns.append(attempt["error"])
        
        return "; ".join(set(patterns[:3]))  # 상위 3개 패턴
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] Rule-based 분류 (`intent_classifier.py`의 `_classify_by_rules`)
- [x] LLM fallback 기본 구현 (`_classify_by_llm`)
- [x] 컨텍스트 추가 (이전 메시지 3개)

### ⚠️ 부분 구현
- [ ] LLM fallback 강화
  - **현재**: 기본 LLM 분류만 구현
  - **필요**: 쿼리 재구성, Ensemble Prompting 통합
  - **방향**: `_classify_by_llm` 메서드 확장
- [ ] 컨텍스트 기반 분류
  - **현재**: 이전 메시지 3개만 사용
  - **필요**: 세션 파일, 도구 이력, 쿼리 히스토리 활용
  - **방향**: `classify()` 메서드에 세션 정보 수집 로직 추가

### ❌ 미구현
- [ ] **쿼리 재구성 서비스** (`QueryRefiner`)
  - **파일**: `playground/backend/services/query_refiner.py` (신규 생성 필요)
  - **구현 방향**:
    1. `beanllm.domain.retrieval.query_expansion.HyDEExpander` 활용 (이미 구현됨)
    2. 사용자 피드백 수집: `HumanFeedbackCollector` 활용 (이미 구현됨)
    3. Relevance Feedback: 검색 결과 상위 3개 활용
    4. 쿼리 히스토리 저장: MongoDB에 `query_refinements` 컬렉션
  - [ ] 사용자 피드백 기반 재구성
    - **방법**: LLM에 피드백과 원본 쿼리를 함께 전달하여 재구성
    - **통합 위치**: `intent_classifier.py`의 `_classify_by_llm_enhanced`에서 호출
  - [ ] Relevance Feedback 기반 재구성
    - **방법**: 이전 검색 결과의 상위 문서를 컨텍스트로 활용
    - **통합 위치**: RAG 질의 후 결과가 부족할 때 자동 재구성
  - [ ] HyDE 활용 재구성
    - **방법**: `HyDEExpander`로 가상 문서 생성 후 쿼리 재구성
    - **통합 위치**: Intent 분류 전 또는 RAG 검색 전
- [ ] **프롬프트 재구성 서비스** (`PromptBuilder`)
  - **파일**: `playground/backend/services/prompt_builder.py` (신규 생성 필요)
  - **구현 방향**:
    1. `beanllm.domain.prompts` 모듈 활용 (PromptTemplate, PromptComposer, PromptOptimizer)
    2. 프롬프트 템플릿을 동적으로 조합
    3. 피드백 히스토리를 프롬프트에 반영
  - [ ] 동적 프롬프트 구성
    - **방법**: `PromptComposer`로 여러 템플릿 조합
    - **통합 위치**: `orchestrator.py`의 각 핸들러에서 사용
  - [ ] 피드백 반영 프롬프트
    - **방법**: `PromptOptimizer.add_instructions()`로 피드백 추가
    - **통합 위치**: RAG 질의, Chat 응답 생성 시
  - [ ] 이전 시도 학습 반영
    - **방법**: 실패한 시도 패턴을 분석하여 프롬프트에 경고 추가
    - **통합 위치**: Orchestrator 실행 전 프롬프트 구성 단계
- [ ] **Ensemble Prompting** (GenQREnsemble 방식)
  - **구현 방향**:
    1. 여러 프롬프트 변형 생성 (paraphrase)
    2. 각 변형으로 LLM 호출
    3. 결과 통합 (다수결 또는 가중 평균)
  - [ ] 여러 프롬프트 변형 생성
    - **방법**: `_generate_ensemble_prompts()` 메서드 구현
    - **통합 위치**: `intent_classifier.py`의 `_classify_by_llm_enhanced`
  - [ ] 결과 통합 로직
    - **방법**: `_aggregate_classification_results()` 메서드 구현
    - **알고리즘**: 다수결 또는 confidence 가중 평균

---

## 🎯 우선순위

**높음**: 정확도 개선, 사용자 피드백 활용

---

## 📚 참고 기법

### 쿼리 재구성
- **GenQREnsemble**: Ensemble Prompting으로 여러 쿼리 생성 후 통합
- **GenQRFusion**: Document Fusion과 Relevance Feedback 활용
- **HyDE**: 가상 문서 생성 후 재구성 (이미 구현됨: `src/beanllm/domain/retrieval/query_expansion.py`)
- **QueryGym**: 표준화된 쿼리 재구성 프레임워크 (참고용)

### 프롬프트 구성
- **Ensemble Prompting**: 여러 프롬프트 변형으로 정확도 향상
- **Relevance Feedback**: 검색 결과 기반 프롬프트 개선
- **User Feedback Loop**: 사용자 피드백을 다음 프롬프트에 반영
