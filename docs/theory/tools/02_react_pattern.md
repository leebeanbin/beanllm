# ReAct Pattern: 추론과 행동의 결합

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit Agent 실제 구현 분석

---

## 목차

1. [ReAct 패턴의 정의](#1-react-패턴의-정의)
2. [Thought-Action-Observation 사이클](#2-thought-action-observation-사이클)
3. [도구 선택의 확률 모델](#3-도구-선택의-확률-모델)
4. [수렴 조건](#4-수렴-조건)
5. [CS 관점: 구현과 최적화](#5-cs-관점-구현과-최적화)

---

## 1. ReAct 패턴의 정의

### 1.1 ReAct의 구성

#### 정의 1.1.1: ReAct (Reasoning + Acting)

**ReAct 패턴**은 추론과 행동을 결합합니다:

$$
\text{ReAct} = \text{Reasoning} + \text{Acting}
$$

**단계:**
1. **Thought**: 추론 (어떤 도구 사용할지)
2. **Action**: 행동 (도구 실행)
3. **Observation**: 관찰 (결과 확인)

#### 시각적 표현: ReAct 사이클

```
ReAct 사이클:

Task: "서울의 날씨는?"
    │
    ▼
┌─────────────────────────────────────┐
│ Step 1: Thought                     │
│ "날씨 정보가 필요하므로             │
│  웹 검색 도구를 사용해야 한다"      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Step 2: Action                      │
│ Action: search_weather              │
│ Action Input: {"city": "Seoul"}     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Step 3: Observation                 │
│ "서울: 맑음, 15°C"                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Step 4: Thought                     │
│ "정보를 얻었으므로 최종 답변 가능"   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Final Answer:                       │
│ "서울의 현재 날씨는 맑고 15°C입니다"│
└─────────────────────────────────────┘
```

---

## 2. Thought-Action-Observation 사이클

### 2.1 사이클의 수학적 모델

#### 정의 2.1.1: T-A-O 사이클

**T-A-O 사이클:**

$$
\text{State}_{t+1} = f(\text{State}_t, \text{Thought}_t, \text{Action}_t, \text{Observation}_t)
$$

**Thought:**
$$
\text{Thought}_t = \text{LLM}(\text{State}_t, \text{Task})
$$

**Action:**
$$
\text{Action}_t = \text{SelectTool}(\text{Thought}_t)
$$

**Observation:**
$$
\text{Observation}_t = \text{ExecuteTool}(\text{Action}_t)
$$

### 2.2 llmkit 구현

#### 구현 2.2.1: ReAct 실행

**llmkit 구현:**
```python
# service/impl/agent_service_impl.py: AgentServiceImpl.run()
# handler/agent_handler.py: AgentHandler.handle_run()
# facade/agent_facade.py: Agent.run()
class AgentServiceImpl(IAgentService):
    """
    에이전트 서비스 구현체: ReAct 패턴 실행
    
    수학적 표현:
    - State_{t+1} = f(State_t, Thought_t, Action_t, Observation_t)
    - Thought_t = LLM(State_t, Task)
    - Action_t = SelectTool(Thought_t)
    - Observation_t = ExecuteTool(Action_t)
    
    실제 구현:
    - service/impl/agent_service_impl.py: AgentServiceImpl.run()
    - handler/agent_handler.py: AgentHandler.handle_run() (입력 검증)
    - facade/agent_facade.py: Agent.run() (사용자 API)
    """
    REACT_PROMPT = """You are a helpful AI assistant with access to tools.

To solve the task, you should follow the ReAct (Reasoning + Acting) pattern:
1. **Thought**: Think about what to do next
2. **Action**: Choose a tool to use
3. **Observation**: See the result
4. Repeat until you have the final answer

Available tools:
{tools_description}

Format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: [tool result]
... (repeat as needed)
Thought: I now know the final answer
Final Answer: [your final answer]

Task: {task}"""
    
    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        ReAct 패턴 실행
        
        Process:
        for t in range(max_steps):
            1. Thought_t = LLM(State_t, Task)
            2. Action_t = Parse(Thought_t)
            3. Observation_t = ExecuteTool(Action_t)
            4. State_{t+1} = Update(State_t, Thought_t, Action_t, Observation_t)
            5. if Final Answer: break
        
        시간 복잡도: O(max_steps · (T_LLM + T_tool))
        
        실제 구현:
        - service/impl/agent_service_impl.py: AgentServiceImpl.run()
        """
        steps: List[Dict[str, Any]] = []
        step_number = 0
        
        # 도구 설명 생성
        tools_description = self._format_tools(request.tool_registry or self._tool_registry)
        
        # 초기 프롬프트 (ReAct 패턴)
        prompt = self.REACT_PROMPT.format(
            tools_description=tools_description,
            task=request.task
        )
        
        messages = [{"role": "user", "content": prompt}]
        conversation_history = prompt
        
        # ReAct 사이클
        while step_number < request.max_steps:
            step_number += 1
            
            # 1. Thought: LLM 추론
            chat_request = ChatRequest(
                messages=messages,
                model=request.model,
                temperature=request.temperature or 0.0,
            )
            response = await self._chat_service.chat(chat_request)
            content = response.content
            
            # 2. Action: 파싱
            parsed_step = self._parse_response(content, step_number)
            steps.append(parsed_step)
            
            # 3. 최종 답변 확인
            if parsed_step.get("is_final") and parsed_step.get("final_answer"):
                return AgentResponse(
                    answer=parsed_step["final_answer"],
                    steps=steps,
                    total_steps=step_number,
                    success=True,
                )
            
            # 4. Observation: 도구 실행
            action_name = parsed_step.get("action")
            action_input = parsed_step.get("action_input")
            if action_name and action_input:
                observation = self._execute_tool(
                    action_name,
                    action_input,
                    request.tool_registry or self._tool_registry
                )
                parsed_step["observation"] = observation
                
                # 대화 히스토리 업데이트
                conversation_history += f"\n\n{content}\nObservation: {observation}"
                messages = [{"role": "user", "content": conversation_history + "\n\nContinue..."}]
        
        # 최대 반복 도달
        return AgentResponse(
            answer="Maximum iterations reached",
            steps=steps,
            total_steps=step_number,
            success=False,
        )
    
    def _parse_response(self, content: str, step_number: int) -> Dict[str, Any]:
        """
        LLM 응답 파싱: Action, Action Input, Final Answer 추출
        
        실제 구현:
        - service/impl/agent_service_impl.py: AgentServiceImpl._parse_response()
        - 정규표현식으로 "Action:", "Action Input:", "Final Answer:" 추출
        """
        import re
        
        # Action 추출
        action_match = re.search(r"Action:\s*(\w+)", content)
        action = action_match.group(1) if action_match else None
        
        # Action Input 추출 (JSON)
        action_input_match = re.search(r"Action Input:\s*(\{.*?\})", content, re.DOTALL)
        action_input = None
        if action_input_match:
            try:
                action_input = json.loads(action_input_match.group(1))
            except:
                action_input = {}
        
        # Final Answer 추출
        final_answer_match = re.search(r"Final Answer:\s*(.+)", content, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else None
        
        return {
            "step": step_number,
            "thought": content,
            "action": action,
            "action_input": action_input,
            "final_answer": final_answer,
            "is_final": final_answer is not None,
        }
    
    def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_registry: Optional["ToolRegistryProtocol"]
    ) -> str:
        """
        도구 실행: Observation = ExecuteTool(Action)
        
        실제 구현:
        - service/impl/agent_service_impl.py: AgentServiceImpl._execute_tool()
        - tool_registry에서 도구 조회 및 실행
        """
        if not tool_registry:
            return "Tool registry not available"
        
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return f"Tool {tool_name} not found"
        
        try:
            result = tool.execute(tool_input)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
```

---

## 3. 도구 선택의 확률 모델

### 3.1 도구 선택 확률

#### 정의 3.1.1: 도구 선택

**LLM이 도구를 선택할 확률:**

$$
P(\text{tool}_i | \text{query}) = \text{softmax}([\text{score}_1, \text{score}_2, \ldots, \text{score}_n])_i
$$

**구체적 수치 예시:**

**예시 3.1.1: 도구 선택 확률**

쿼리: "서울의 날씨는?"

가능한 도구:
- `search_weather`: score = 8.5
- `search_web`: score = 6.2
- `calculator`: score = 0.1

**Softmax 계산:**

$$
P(\text{search\_weather}) = \frac{\exp(8.5)}{\exp(8.5) + \exp(6.2) + \exp(0.1)} = 0.909
$$

$$
P(\text{search\_web}) = 0.091
$$

$$
P(\text{calculator}) = 0.0002
$$

**결과:** `search_weather` 선택 (90.9%)

---

## 4. 수렴 조건

### 4.1 종료 조건

#### 정의 4.1.1: 종료 조건

**종료 조건:**

1. **최대 반복:**
   $$
   \text{step} \geq \text{max\_iterations}
   $$

2. **최종 답변:**
   $$
   \text{Action} = \text{None} \implies \text{종료}
   $$

3. **에러:**
   $$
   \text{Error} \implies \text{종료}
   $$

---

## 5. CS 관점: 구현과 성능

### 5.1 반복 제한

#### CS 관점 5.1.1: 무한 루프 방지

**최대 반복 설정:**

```python
max_iterations = 10  # 무한 루프 방지

for step in range(max_iterations):
    # ReAct 사이클
    if should_stop(state):
        break
```

**효과:**
- 무한 루프 방지
- 리소스 보호

---

## 질문과 답변 (Q&A)

### Q1: ReAct는 언제 사용하나요?

**A:** 사용 시기:

1. **복잡한 작업:**
   - 여러 단계 필요
   - 도구 체이닝

2. **동적 결정:**
   - 상황에 따라 다른 도구
   - 조건부 실행

3. **에러 처리:**
   - 실패 시 재시도
   - 대안 경로

---

## 참고 문헌

1. **Yao et al. (2022)**: "ReAct: Synergizing Reasoning and Acting in Language Models"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

