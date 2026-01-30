# Plan Mode (Claude 스타일 계획 검토)

## 🎯 목표

사용자가 AI의 실행 계획을 검토하고 승인/수정할 수 있는 Plan 모드 구현

**Claude의 Plan 모드처럼**:
- AI가 계획을 제시
- 사용자가 검토 및 승인/수정
- 승인 후 자동 실행
- 내부적으로는 자동 모드도 지원

---

## 📊 현재 상태

### 구현된 기능
- ✅ Orchestrator: Intent 분류 및 Tool 선택
- ✅ WorkflowGraph: 노드 기반 워크플로우 실행
- ✅ SSE 스트리밍: 진행 상황 실시간 전달

### 없는 기능
- ❌ Plan 생성 및 제시
- ❌ 사용자 승인/수정 인터페이스
- ❌ Plan 모드 vs Auto 모드 선택

---

## ✅ 구현 방안

### 1. Plan 생성 서비스 (쿼리 재구성 포함)

**파일**: `playground/backend/services/plan_service.py`

```python
from services.query_refiner import QueryRefiner
from services.prompt_builder import PromptBuilder

class PlanService:
    """계획 생성 및 관리 (쿼리 재구성 포함)"""
    
    def __init__(self):
        self.query_refiner = QueryRefiner()
        self.prompt_builder = PromptBuilder()
    
    async def generate_plan(
        self,
        query: str,
        intent: IntentResult,
        context: Dict[str, Any],
        user_feedback: Optional[str] = None,
        previous_plan: Optional[Plan] = None
    ) -> Plan:
        """
        사용자 쿼리로부터 실행 계획 생성 (쿼리 재구성 포함)
        
        Process:
        1. 쿼리 재구성 (피드백 기반)
        2. 프롬프트 구성 (컨텍스트 기반)
        3. 계획 생성 (재구성된 쿼리로)
        
        Returns:
            Plan: 실행 계획 (단계별 작업 목록)
        """
        # 1. 쿼리 재구성 (사용자 피드백 반영)
        refined_query = await self.query_refiner.refine_query(
            original_query=query,
            user_feedback=user_feedback,
            session_id=context.get("session_id"),
            previous_results=context.get("previous_results")
        )
        
        # 2. 이전 계획이 있으면 학습 반영
        if previous_plan:
            lessons = self._extract_plan_lessons(previous_plan)
            refined_query = await self._apply_lessons(refined_query, lessons)
        
        # 3. 계획 생성 프롬프트 구성 (동적)
        plan_prompt = self.prompt_builder.build_plan_generation_prompt(
            query=refined_query,
            intent=intent,
            context=context,
            previous_plan=previous_plan
        )
        
        # 4. LLM으로 계획 생성 (Ensemble 방식)
        plan_candidates = await self._generate_plan_ensemble(plan_prompt)
        
        # 5. 최적 계획 선택
        best_plan = self._select_best_plan(plan_candidates, context)
        
        return best_plan
    
    async def _generate_plan_ensemble(
        self,
        base_prompt: str
    ) -> List[Plan]:
        """
        Ensemble Prompting으로 여러 계획 생성 (GenQREnsemble 방식)
        
        여러 프롬프트 변형으로 계획을 생성하여 최적 선택
        """
        from beanllm import Client
        client = Client(model="gpt-4o-mini")
        
        # 프롬프트 변형 생성
        prompt_variants = [
            base_prompt,  # 원본
            f"간결하게: {base_prompt}",  # 간결형
            f"상세하게: {base_prompt}",  # 상세형
            f"단계별로: {base_prompt}",  # 단계 중심
        ]
        
        plans = []
        for variant in prompt_variants:
            response = await client.chat([
                {"role": "system", "content": "You are a planning assistant. Generate execution plans in JSON format."},
                {"role": "user", "content": variant}
            ])
            
            try:
                plan_json = json.loads(response.content)
                plans.append(Plan.from_dict(plan_json))
            except:
                continue
        
        return plans
    
    def _select_best_plan(
        self,
        candidates: List[Plan],
        context: Dict[str, Any]
    ) -> Plan:
        """
        최적 계획 선택
        
        기준:
        - 단계 수 (적절한 수준)
        - 예상 시간 (짧을수록 좋음)
        - 도구 다양성 (적절한 조합)
        """
        if not candidates:
            raise ValueError("No valid plans generated")
        
        # 간단한 스코어링
        scored = []
        for plan in candidates:
            score = 0
            score += len(plan.steps) * -0.1  # 단계 수가 적을수록 좋음
            score += len(set(s.tool for s in plan.steps)) * 0.2  # 도구 다양성
            scored.append((score, plan))
        
        # 최고 점수 계획 선택
        best = max(scored, key=lambda x: x[0])
        return best[1]
    
    def _extract_plan_lessons(self, previous_plan: Plan) -> List[str]:
        """이전 계획에서 교훈 추출"""
        lessons = []
        
        if previous_plan.status == "rejected":
            lessons.append("사용자가 거부한 계획 패턴을 피하세요")
        
        if previous_plan.user_modifications:
            lessons.append(f"사용자 수정 사항: {previous_plan.user_modifications}")
        
        return lessons
    
    async def _apply_lessons(
        self,
        query: str,
        lessons: List[str]
    ) -> str:
        """교훈을 쿼리에 반영"""
        if not lessons:
            return query
        
        from beanllm import Client
        client = Client(model="gpt-4o-mini")
        
        prompt = f"""
        원본 쿼리: {query}
        이전 경험에서 배운 점:
        {chr(10).join(f"- {lesson}" for lesson in lessons)}
        
        배운 점을 반영하여 쿼리를 개선해주세요.
        """
        
        response = await client.chat([{"role": "user", "content": prompt}])
        return response.content.strip()
```

### 2. Plan 모델

**파일**: `playground/backend/schemas/plan.py`

```python
class PlanStep(BaseModel):
    """계획 단계"""
    step_id: str
    tool: str  # 사용할 도구
    action: str  # 수행할 작업
    reason: str  # 이유
    dependencies: List[str] = []  # 의존하는 단계들
    estimated_time: Optional[str] = None
    estimated_cost: Optional[str] = None

class Plan(BaseModel):
    """실행 계획"""
    plan_id: str
    query: str
    steps: List[PlanStep]
    estimated_time: str
    estimated_cost: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected, executing, completed
    user_modifications: Optional[Dict[str, Any]] = None
    created_at: datetime
    approved_at: Optional[datetime] = None
```

### 3. Plan 모드 선택 (Playground UI)

**프론트엔드**: `playground/frontend/src/components/PlanModeSelector.tsx`

```typescript
interface PlanModeSelectorProps {
  onModeChange: (mode: "auto" | "plan") => void;
  currentMode: "auto" | "plan";
}

export function PlanModeSelector({ onModeChange, currentMode }: PlanModeSelectorProps) {
  return (
    <div className="flex gap-2">
      <button
        onClick={() => onModeChange("auto")}
        className={currentMode === "auto" ? "active" : ""}
      >
        🤖 자동 모드
      </button>
      <button
        onClick={() => onModeChange("plan")}
        className={currentMode === "plan" ? "active" : ""}
      >
        📋 계획 모드
      </button>
    </div>
  );
}
```

### 4. Plan 검토 UI

**프론트엔드**: `playground/frontend/src/components/PlanReview.tsx`

```typescript
interface PlanReviewProps {
  plan: Plan;
  onApprove: (plan: Plan) => void;
  onModify: (plan: Plan, modifications: Dict) => void;
  onReject: () => void;
}

export function PlanReview({ plan, onApprove, onModify, onReject }: PlanReviewProps) {
  return (
    <div className="plan-review">
      <h3>실행 계획 검토</h3>
      
      {/* 계획 단계 목록 */}
      {plan.steps.map((step, idx) => (
        <PlanStepCard
          key={step.step_id}
          step={step}
          index={idx}
          onEdit={(modified) => onModify(plan, { step_id: step.step_id, ...modified })}
        />
      ))}
      
      {/* 예상 시간/비용 */}
      <div className="plan-summary">
        <p>예상 시간: {plan.estimated_time}</p>
        {plan.estimated_cost && <p>예상 비용: {plan.estimated_cost}</p>}
      </div>
      
      {/* 승인/수정/거부 버튼 */}
      <div className="plan-actions">
        <button onClick={() => onApprove(plan)}>✅ 승인</button>
        <button onClick={() => onModify(plan, {})}>✏️ 수정</button>
        <button onClick={onReject}>❌ 거부</button>
      </div>
    </div>
  );
}
```

### 5. Orchestrator 통합

**파일**: `playground/backend/services/orchestrator.py` (수정)

```python
class AgenticOrchestrator:
    async def execute_with_plan(
        self,
        context: OrchestratorContext,
        plan: Optional[Plan] = None,
        mode: str = "auto"  # "auto" or "plan"
    ) -> AsyncGenerator[AgenticEvent, None]:
        """
        Plan 모드 또는 Auto 모드로 실행
        
        Args:
            context: Orchestrator 컨텍스트
            plan: 실행 계획 (Plan 모드일 때 필수)
            mode: 실행 모드 ("auto" or "plan")
        """
        if mode == "plan" and plan:
            # Plan 모드: 계획에 따라 단계별 실행
            yield AgenticEvent(
                type=EventType.INTENT,
                data={"intent": context.intent.to_dict(), "mode": "plan"}
            )
            
            for step in plan.steps:
                yield AgenticEvent(
                    type=EventType.TOOL_START,
                    data={"step": step.step_id, "tool": step.tool}
                )
                
                # 단계 실행
                result = await self._execute_step(step, context)
                
                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={"step": step.step_id, "result": result}
                )
        else:
            # Auto 모드: 기존 로직 (자동 실행)
            async for event in self.execute(context):
                yield event
```

### 6. API 엔드포인트

**파일**: `playground/backend/routers/chat_router.py` (추가)

```python
@router.post("/api/chat/plan")
async def generate_plan(request: ChatRequest) -> Plan:
    """계획 생성"""
    plan_service = PlanService()
    intent = await intent_classifier.classify(request.query)
    plan = await plan_service.generate_plan(
        query=request.query,
        intent=intent,
        context={"session_id": request.session_id}
    )
    return plan

@router.post("/api/chat/execute-plan")
async def execute_plan(request: ExecutePlanRequest) -> StreamingResponse:
    """승인된 계획 실행"""
    orchestrator = AgenticOrchestrator()
    
    async def stream():
        async for event in orchestrator.execute_with_plan(
            context=request.context,
            plan=request.plan,
            mode="plan"
        ):
            yield event.to_sse()
    
    return StreamingResponse(stream(), media_type="text/event-stream")
```

---

## 📋 구현 체크리스트

### 백엔드
- [ ] **`PlanService` 생성 (계획 생성)**
  - **파일**: `playground/backend/services/plan_service.py` (신규 생성 필요)
  - **의존성**: `QueryRefiner`, `PromptBuilder` (07_INTENT_CLASSIFIER.md 참조)
  - **구현 방향**:
    1. 쿼리 재구성 통합 (`QueryRefiner` 사용)
    2. 프롬프트 동적 구성 (`PromptBuilder` 사용)
    3. Ensemble Prompting으로 여러 계획 생성 후 최적 선택
    4. 이전 계획 학습 반영 (MongoDB에 저장)
  - [ ] 쿼리 재구성 통합 (`QueryRefiner` 사용)
    - **방법**: `PlanService.generate_plan()`에서 `QueryRefiner.refine_query()` 호출
  - [ ] 프롬프트 동적 구성 (`PromptBuilder` 사용)
    - **방법**: `PromptBuilder.build_plan_generation_prompt()` 메서드 활용
  - [ ] Ensemble Prompting 구현
    - **방법**: 여러 프롬프트 변형으로 계획 생성 후 스코어링으로 최적 선택
  - [ ] 이전 계획 학습 반영
    - **방법**: MongoDB `plans` 컬렉션에서 이전 계획 조회 및 패턴 분석
- [ ] **`QueryRefiner` 서비스 생성** (07_INTENT_CLASSIFIER.md 참조)
- [ ] **`PromptBuilder` 서비스 생성** (07_INTENT_CLASSIFIER.md 참조)
- [ ] **`Plan`, `PlanStep` 모델 생성**
  - **파일**: `playground/backend/schemas/plan.py` (신규 생성 필요)
  - **구조**: 문서의 "2. Plan 모델" 섹션 참조
- [ ] **`orchestrator.py`에 Plan 모드 통합**
  - **통합 위치**: `AgenticOrchestrator` 클래스에 `execute_with_plan()` 메서드 추가
  - **방법**: 문서의 "5. Orchestrator 통합" 섹션 참조
- [ ] **`/api/chat/plan` 엔드포인트 (계획 생성)**
  - **위치**: `routers/chat_router.py`
  - **방법**: `PlanService.generate_plan()` 호출
- [ ] **`/api/chat/execute-plan` 엔드포인트 (계획 실행)**
  - **위치**: `routers/chat_router.py`
  - **방법**: `orchestrator.execute_with_plan(mode="plan")` 호출

### 프론트엔드
- [ ] `PlanModeSelector` 컴포넌트 (모드 선택)
- [ ] `PlanReview` 컴포넌트 (계획 검토)
- [ ] `PlanStepCard` 컴포넌트 (단계 카드)
- [ ] Chat UI에 Plan 모드 통합
- [ ] Plan 수정 UI (단계 편집)

### 통합
- [ ] Plan 모드에서 Auto 모드로 전환 가능
- [ ] Auto 모드에서 Plan 모드로 전환 가능
- [ ] Plan 히스토리 저장 (MongoDB)
- [ ] Plan 재사용 기능

---

## 🎯 우선순위

**높음**: 사용자 경험 개선, 투명성 향상

---

## 💡 추가 기능 (선택)

### 1. Plan 템플릿
- 자주 사용하는 계획을 템플릿으로 저장
- 템플릿에서 빠르게 계획 생성

### 2. Plan 비교
- 여러 계획을 비교하여 최적 선택
- 비용/시간/정확도 비교

### 3. Plan 학습
- 사용자 승인/수정 패턴 학습
- 다음 계획 생성 시 개선

---

## 🔗 관련 문서

- [02_AGENTIC_MODE.md](./02_AGENTIC_MODE.md): Agentic 모드 기본 구조
- [07_INTENT_CLASSIFIER.md](./07_INTENT_CLASSIFIER.md): 쿼리 재구성 및 프롬프트 구성
- [17_VISUAL_WORKFLOW.md](./17_VISUAL_WORKFLOW.md): 시각적 워크플로우 구성

---

## 📚 쿼리 재구성 및 프롬프트 구성 기법

### 현재 코드베이스 활용
- ✅ **Query Expansion**: `src/beanllm/domain/retrieval/query_expansion.py`
  - HyDE (Hypothetical Document Embeddings)
  - Multi-Query Expansion
  - Step-back Prompting
- ✅ **Prompt Templates**: `src/beanllm/domain/prompts/`
  - PromptTemplate, PromptComposer, PromptOptimizer

### 추가 구현 필요
- **QueryRefiner**: 사용자 피드백 기반 쿼리 재구성
- **PromptBuilder**: 동적 프롬프트 구성
- **Ensemble Prompting**: GenQREnsemble 방식 (여러 프롬프트 변형)
- **Relevance Feedback**: 검색 결과 기반 쿼리/프롬프트 개선

### 참고 기법 (2024-2025 최신)
- **GenQREnsemble**: Ensemble Prompting으로 nDCG@10 18% 향상
- **GenQRFusion**: Document Fusion + Relevance Feedback
- **QueryGym**: 표준화된 쿼리 재구성 프레임워크
