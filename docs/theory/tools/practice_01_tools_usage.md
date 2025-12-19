# Tool Calling 실무 가이드: LLM 함수 호출 활용

**실무 적용 문서**

---

## 목차

1. [도구 정의](#1-도구-정의)
2. [에이전트에 도구 연결](#2-에이전트에-도구-연결)
3. [도구 실행](#3-도구-실행)
4. [실무 패턴](#4-실무-패턴)

---

## 1. 도구 정의

### 1.1 기본 도구

```python
from llmkit.tools import Tool, ToolParameter

def get_weather(city: str) -> str:
    """날씨 정보 조회"""
    return f"{city}의 날씨는 맑음입니다."

weather_tool = Tool(
    name="get_weather",
    description="도시의 날씨를 조회합니다",
    parameters=[
        ToolParameter(
            name="city",
            type="string",
            description="도시 이름",
            required=True
        )
    ],
    function=get_weather
)
```

### 1.2 함수에서 자동 생성

```python
from llmkit.tools import Tool

@Tool.from_function
def calculate(expression: str) -> str:
    """수식 계산"""
    return str(eval(expression))
```

---

## 2. 에이전트에 도구 연결

```python
from llmkit import Agent

agent = Agent(
    model="gpt-4o-mini",
    tools=[weather_tool, calculate]
)
```

---

## 3. 도구 실행

```python
result = await agent.run("서울의 날씨는?")
# 에이전트가 자동으로 get_weather 도구 사용
```

---

## 4. 실무 패턴

### 4.1 웹 검색 도구

```python
def web_search(query: str) -> str:
    # 웹 검색 구현
    pass

search_tool = Tool.from_function(web_search)
```

### 4.2 데이터베이스 쿼리

```python
def db_query(sql: str) -> str:
    # DB 쿼리 실행
    pass

db_tool = Tool.from_function(db_query)
```

---

**작성일**: 2025-01-XX  
**버전**: 1.0

