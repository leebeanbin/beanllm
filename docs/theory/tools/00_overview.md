# Tool Calling Theory: 함수 호출의 수학적 모델

**석사 수준 이론 문서**  
**기반**: llmkit Tool, Agent 실제 구현 분석

---

## 목차

### Part I: 함수 호출 이론
1. [도구의 형식적 정의](#part-i-함수-호출-이론)
2. [스키마와 타입 시스템](#12-스키마와-타입-시스템)
3. [도구 선택의 확률 모델](#13-도구-선택의-확률-모델)

### Part II: 실행 모델
4. [도구 실행의 수학적 모델](#part-ii-실행-모델)
5. [오류 처리와 재시도](#42-오류-처리와-재시도)
6. [도구 체이닝](#43-도구-체이닝)

---

## Part I: 함수 호출 이론

### 1.1 도구의 형식적 정의

#### 정의 1.1.1: 도구 (Tool)

**도구**는 다음 튜플로 정의됩니다:

$$
\text{Tool} = (name, description, parameters, function)
$$

여기서:
- $name$: 도구 이름
- $description$: 설명
- $parameters$: 파라미터 스키마
- $function$: 실행 함수

**llmkit 구현:**
```python
# domain/tools/tool.py: Tool
# domain/tools/advanced/decorator.py: @tool 데코레이터
@dataclass
class Tool:
    """
    도구: Tool = (name, description, parameters, function)
    
    수학적 정의:
    - name: 도구 식별자
    - description: 도구 설명 (LLM이 선택할 때 사용)
    - parameters: 파라미터 스키마 (JSON Schema 형식)
    - function: 실행 함수 f: Parameters → Result
    
    실제 구현:
    - domain/tools/tool.py: Tool (기본 도구 클래스)
    - domain/tools/advanced/decorator.py: @tool 데코레이터 (함수 → Tool 변환)
    - facade/agent_facade.py: Agent (도구 사용 에이전트)
    """
    name: str
    description: str
    parameters: List[ToolParameter]  # 또는 Dict[str, Any] (JSON Schema)
    function: Callable
    
    def execute(self, params: Dict[str, Any]) -> Any:
        """
        도구 실행: f(params)
        
        수학적 표현: result = f(params)
        
        실제 구현:
        - domain/tools/tool.py: Tool.execute()
        - 파라미터 검증 후 함수 실행
        - 오류 처리 및 재시도 지원
        """
        # 파라미터 검증
        validated_params = self._validate_params(params)
        
        # 함수 실행
        return self.function(**validated_params)
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON Schema 형식으로 변환 (LLM에 전달)"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }
```

---

### 1.2 스키마와 타입 시스템

#### 정의 1.2.1: 파라미터 스키마

**파라미터 스키마**는 JSON Schema 형식입니다:

$$
\text{Schema} = \{type, properties, required\}
$$

**llmkit 구현:**
```python
# domain/tools/tool.py: ToolParameter
@dataclass
class ToolParameter:
    """
    파라미터 스키마: (name, type, description, required)
    
    실제 구현:
    - domain/tools/tool.py: ToolParameter
    """
    name: str
    type: str  # string, number, boolean, object, array
    description: str
    required: bool = True
```

---

### 1.3 도구 선택의 확률 모델

#### 정의 1.3.1: 도구 선택 확률

**LLM이 도구를 선택할 확률:**

$$
P(\text{tool}_i | \text{query}) = \text{softmax}([\text{score}_1, \text{score}_2, \ldots, \text{score}_n])_i
$$

#### 시각적 표현: ReAct 패턴

```
┌─────────────────────────────────────────────────────────┐
│                  ReAct 패턴 실행                         │
└─────────────────────────────────────────────────────────┘

Task: "서울의 현재 날씨는?"
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

#### 구체적 수치 예시

**예시 1.3.1: 도구 선택 확률 계산**

**쿼리:** "서울의 날씨는?"

**가능한 도구:**
- `search_weather`: score = 8.5
- `search_web`: score = 6.2
- `calculator`: score = 0.1

**Softmax 계산:**

$$
P(\text{search\_weather}) = \frac{\exp(8.5)}{\exp(8.5) + \exp(6.2) + \exp(0.1)}
$$

$$
= \frac{4914.77}{4914.77 + 492.75 + 1.11} = \frac{4914.77}{5408.63} \approx 0.909
$$

$$
P(\text{search\_web}) = \frac{492.75}{5408.63} \approx 0.091
$$

$$
P(\text{calculator}) = \frac{1.11}{5408.63} \approx 0.0002
$$

**결과:** `search_weather` 선택 (확률 90.9%)

**llmkit 구현:**
```python
# service/impl/agent_service_impl.py: AgentServiceImpl.run()
# facade/agent_facade.py: Agent.run()
async def run(self, task: str) -> AgentResult:
    """
    ReAct 패턴:
    1. Thought: 어떤 도구를 사용할지 생각
    2. Action: 도구 선택 P(tool | query)
    3. Observation: 도구 실행 결과
    
    실제 구현:
    - service/impl/agent_service_impl.py: AgentServiceImpl.run()
    - facade/agent_facade.py: Agent.run() (사용자 API)
    """
    # LLM이 도구 선택
    response = await self.client.chat(messages)
    
    # 파싱: Action: tool_name
    step = self._parse_response(response.content)
    
    if step.action:
        # 도구 실행
        observation = self._execute_tool(step.action, step.action_input)
```

**실제 실행 예시:**
```python
from llmkit import Agent, Tool

# 도구 정의
def search_weather(city: str) -> str:
    return f"{city}: 맑음, 15°C"

# 에이전트 생성
agent = Agent(
    model="gpt-4o-mini",
    tools=[Tool.from_function(search_weather)]
)

# 실행
result = await agent.run("서울의 날씨는?")

# 출력:
# Step 1: Thought: "날씨 정보가 필요..."
# Step 2: Action: search_weather({"city": "Seoul"})
# Step 3: Observation: "Seoul: 맑음, 15°C"
# Final Answer: "서울의 현재 날씨는 맑고 15°C입니다"
```

---

## Part II: 실행 모델

### 2.1 도구 실행의 수학적 모델

#### 정의 2.1.1: 도구 실행 함수

**도구 실행:**

$$
\text{result} = f(\text{params})
$$

**타입 안전성:**

$$
f: \text{Params} \rightarrow \text{Result}
$$

**llmkit 구현:**
```python
# domain/tools/tool.py: Tool.execute()
def execute(self, params: Dict[str, Any]) -> Any:
    """
    도구 실행: result = f(params)
    
    타입 검증 포함
    
    실제 구현:
    - domain/tools/tool.py: Tool.execute()
    - 파라미터 검증 후 함수 실행
    """
    # 파라미터 검증
    self._validate_params(params)
    
    # 함수 실행
    return self.function(**params)
```

---

### 2.2 오류 처리와 재시도

#### 정의 2.2.1: 재시도 전략

**지수 백오프:**

$$
\text{delay}_n = \min(2^n \cdot \text{base}, \text{max\_delay})
$$

**llmkit 구현:**
```python
# service/impl/agent_service_impl.py: AgentServiceImpl._execute_tool()
# utils/error_handling.py: RetryHandler
def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
    """
    도구 실행 + 재시도 (지수 백오프)
    
    실제 구현:
    - service/impl/agent_service_impl.py: AgentServiceImpl._execute_tool()
    - utils/error_handling.py: RetryHandler (재시도 로직)
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            tool = self.registry.get_tool(tool_name)
            return str(tool.execute(params))
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(2 ** attempt * 0.1, 1.0)  # 지수 백오프
                await asyncio.sleep(delay)
            else:
                return f"Error: {str(e)}"
```

---

### 2.3 도구 체이닝

#### 정의 2.3.1: 도구 체이닝

**도구 체이닝**은 여러 도구를 순차 실행합니다:

$$
\text{result} = f_n \circ f_{n-1} \circ \cdots \circ f_1(\text{input})
$$

**llmkit 구현:**
```python
# service/impl/agent_service_impl.py: AgentServiceImpl.run()
# facade/agent_facade.py: Agent.run()
async def run(self, task: str) -> AgentResult:
    """
    도구 체이닝: fₙ ∘ fₙ₋₁ ∘ ... ∘ f₁(task)
    
    ReAct 패턴에서 순차적 도구 실행
    
    실제 구현:
    - service/impl/agent_service_impl.py: AgentServiceImpl.run()
    - facade/agent_facade.py: Agent.run() (사용자 API)
    """
    while step_number < self.max_iterations:
        # 도구 실행
        observation = self._execute_tool(step.action, step.action_input)
        
        # 다음 도구 선택 (이전 결과를 입력으로)
        current_input = observation
```

---

## 참고 문헌

1. **Yao et al. (2022)**: "ReAct: Synergizing Reasoning and Acting in Language Models"
2. **OpenAI (2023)**: "Function Calling" - OpenAI API 문서

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
