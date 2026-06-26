# Agent

`beanllm.Agent` — ReAct 패턴 기반 도구 호출 에이전트 (Facade 패턴)

Agent는 Reasoning + Acting(ReAct) 루프로 복잡한 멀티스텝 작업을 수행합니다.
각 단계에서 생각(Thought) → 행동(Action) → 관찰(Observation)을 반복하여 최종 답변에 도달합니다.

## Import

```python
from beanllm import Agent
from beanllm.domain.tools import Tool, tool
```

---

## ReAct 패턴

```
Thought:  "I need to find the GDP of Japan."
Action:   web_search("Japan GDP 2024")
Observation: "Japan GDP: $4.21 trillion (2024)"

Thought:  "Now I'll divide by 10."
Action:   calculator("4210000000000 / 10")
Observation: "421000000000"

Thought:  "I have the answer."
Final:    "10% of Japan's GDP is approximately $421 billion."
```

---

## `__init__`

```python
Agent(
    model: str,
    provider: Optional[str] = None,
    tools: List[Tool] = [],
    max_steps: int = 10,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
)
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model` | `str` | — (필수) | 사용할 모델 ID |
| `provider` | `str \| None` | `None` | 프로바이더. `None`이면 자동 감지 |
| `tools` | `List[Tool]` | `[]` | 에이전트가 사용할 도구 목록 |
| `max_steps` | `int` | `10` | 최대 ReAct 반복 수. 초과 시 중단 |
| `system_prompt` | `str \| None` | `None` | 커스텀 시스템 프롬프트 |

---

## `run`

```python
async def run(
    task: str,
    tools: Optional[List[Tool]] = None,
    **kwargs: Any,
) -> AgentResult
```

에이전트를 실행하고 결과를 반환합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `task` | `str` | — (필수) | 수행할 작업 설명 |
| `tools` | `List[Tool] \| None` | `None` | 이 실행에만 적용할 추가 도구 |

**반환:** `AgentResult`

---

## `AgentResult`

```python
@dataclass(frozen=True)
class AgentResult:
    answer: str               # 최종 답변
    steps: list[AgentStep]    # 실행된 모든 단계
    total_steps: int          # 총 단계 수
    success: bool             # 성공 여부
    error: Optional[str]      # 실패 시 오류 메시지
```

---

## `AgentStep`

```python
@dataclass(frozen=True)
class AgentStep:
    step_number: int              # 단계 번호 (1부터)
    thought: str                  # 에이전트의 추론
    action: Optional[str]         # 호출한 도구 이름
    action_input: Optional[dict]  # 도구에 전달한 인자
    observation: Optional[str]    # 도구 실행 결과
    is_final: bool                # 마지막 단계 여부
    final_answer: Optional[str]   # 최종 답변 (is_final=True일 때)
```

---

## 도구 정의

### `@tool` 데코레이터

```python
from beanllm.domain.tools import tool

@tool(description="Search the web for up-to-date information")
def web_search(query: str) -> str:
    """
    Args:
        query: Search query string
    Returns:
        Search results as text
    """
    # 실제 구현
    import requests
    resp = requests.get(f"https://duckduckgo.com/?q={query}&format=json")
    return resp.text[:1000]

@tool
def read_file(path: str) -> str:
    """Read file contents. Args: path (str): File path."""
    with open(path) as f:
        return f.read()
```

### `Tool.from_function`

```python
from beanllm.domain.tools import Tool

def calculator(expression: str) -> str:
    """Evaluate a safe math expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

tool = Tool.from_function(calculator, description="Evaluate a math expression safely")
```

---

## 전체 예시

```python
import asyncio
from beanllm import Agent
from beanllm.domain.tools import tool

@tool(description="Get current temperature in Celsius for a city")
def get_temperature(city: str) -> str:
    # 실제 날씨 API 호출
    return f"{city}: 22°C"

@tool(description="Convert Celsius to Fahrenheit")
def celsius_to_fahrenheit(celsius: float) -> str:
    f = celsius * 9 / 5 + 32
    return f"{celsius}°C = {f}°F"

async def main():
    agent = Agent(
        model="gpt-4o-mini",
        tools=[get_temperature, celsius_to_fahrenheit],
        max_steps=6,
    )

    result = await agent.run("What is the temperature in Seoul in Fahrenheit?")

    print(f"Answer: {result.answer}")
    print(f"Steps: {result.total_steps}")
    print(f"Success: {result.success}")

    # 단계별 추적
    for step in result.steps:
        print(f"\nStep {step.step_number}:")
        print(f"  Thought: {step.thought}")
        if step.action:
            print(f"  Action: {step.action}({step.action_input})")
            print(f"  Observation: {step.observation}")
        if step.is_final:
            print(f"  Final Answer: {step.final_answer}")

asyncio.run(main())
```

---

## ToolRegistry

여러 에이전트에서 공유할 도구를 등록·관리합니다.

```python
from beanllm.domain.tools import ToolRegistry

registry = ToolRegistry()
registry.register(web_search)
registry.register(calculator)

# 등록된 도구로 에이전트 생성
agent = Agent(model="gpt-4o-mini", tools=registry.get_all())
```

---

## 오류 처리

```python
result = await agent.run("Impossible task")
if not result.success:
    print(f"Agent failed: {result.error}")
    # max_steps 초과, 도구 오류 등

# 도구 실행 오류는 observation에 포함됨
for step in result.steps:
    if step.observation and "Error" in step.observation:
        print(f"Step {step.step_number} tool error: {step.observation}")
```

---

## 관련 문서

- [wiki/facade.md](../../wiki/facade.md#agent--react-도구-호출) — Agent 고수준 가이드
- [client.md](client.md) — Client 레퍼런스 (Agent가 내부적으로 사용)
