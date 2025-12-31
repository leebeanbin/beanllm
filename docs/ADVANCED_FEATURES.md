# Advanced LLM Features (2024-2025)

Latest LLM features supported in beanLLM.

## Contents
- [Structured Outputs](#structured-outputs)
- [Prompt Caching](#prompt-caching)
- [Parallel Tool Calling](#parallel-tool-calling)

---

## Structured Outputs

**100% 스키마 정확도 보장** (OpenAI 2024년 8월 출시)

### 지원 모델
- **OpenAI**: `gpt-4o-2024-08-06`, `gpt-4o-mini`
- **Anthropic**: Claude Sonnet 4.5, Opus 4.1

### OpenAI Structured Outputs 사용법

```python
from openai import AsyncOpenAI
import json

client = AsyncOpenAI(api_key="your-api-key")

# JSON 스키마 정의
schema = {
    "name": "user_info",
    "strict": True,  # 100% 정확도 보장
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name", "age", "email"],
        "additionalProperties": False
    }
}

# Structured Output 사용
response = await client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "Extract: John Doe, 30, john@example.com"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": schema
    }
)

# 결과는 항상 스키마를 준수
result = json.loads(response.choices[0].message.content)
print(result)  # {"name": "John Doe", "age": 30, "email": "john@example.com"}
```

### Anthropic Structured Outputs 사용법

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="your-api-key")

response = await client.messages.create(
    model="claude-sonnet-4-5-20250514",
    messages=[
        {"role": "user", "content": "Extract: John Doe, 30, john@example.com"}
    ],
    extra_headers={
        "anthropic-beta": "structured-outputs-2025-11-13"
    },
    response_format={
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            "required": ["name", "age", "email"]
        }
    }
)
```

### 장점
- **100% 스키마 준수** (OpenAI strict mode)
- **JSON 파싱 실패 없음** (기존 14-20% 실패율 → 0%)
- **타입 안정성** 보장
- **복잡한 중첩 구조** 지원

### 주의사항
- `strict: true`는 서버 측 검증 활성화
- `additionalProperties: False` 권장
- 재귀 스키마는 제한적 지원

---

## Prompt Caching

**최대 85% 지연시간 감소, 10배 비용 절감** (Anthropic)

### 지원 Provider
- **Anthropic**: Claude 전 모델 (200K 토큰 캐싱)
- **OpenAI**: GPT-5.1, GPT-4.1 (자동 캐싱, 24시간 유지)

### Anthropic Prompt Caching 사용법

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="your-api-key")

# 긴 시스템 프롬프트 (캐싱 대상)
long_system_prompt = "Your are a helpful assistant..." * 1000  # 긴 프롬프트

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": long_system_prompt,
            "cache_control": {"type": "ephemeral"}  # 캐싱 활성화
        }
    ],
    messages=[
        {"role": "user", "content": "What can you do?"}
    ],
    extra_headers={
        "anthropic-beta": "prompt-caching-2024-07-31"
    }
)

# 캐시 히트 정보 확인
print(response.usage.cache_creation_input_tokens)  # 처음 캐시 생성
print(response.usage.cache_read_input_tokens)  # 캐시에서 읽음
```

### 캐싱 전략

#### 1. 시스템 프롬프트 캐싱
```python
# 긴 시스템 프롬프트는 항상 캐싱
system = [
    {
        "type": "text",
        "text": "Long instruction...",
        "cache_control": {"type": "ephemeral"}
    }
]
```

#### 2. 대화 기록 캐싱
```python
# 이전 대화를 캐싱하여 재사용
messages = [
    {"role": "user", "content": "Previous message 1"},
    {
        "role": "assistant",
        "content": "Previous response 1",
        "cache_control": {"type": "ephemeral"}  # 캐싱
    },
    # ... more history
    {"role": "user", "content": "New question"}
]
```

#### 3. 문서/컨텍스트 캐싱
```python
# 긴 문서를 캐싱
system = [
    {"type": "text", "text": "Instructions..."},
    {
        "type": "text",
        "text": long_document,  # 200K 토큰까지
        "cache_control": {"type": "ephemeral"}
    }
]
```

### 비용 절감
- **캐시된 토큰**: 일반 입력 토큰의 **10%** 비용
- **캐시 TTL**:
  - 기본 5분
  - 1시간 캐시 (추가 비용)
  - OpenAI: 24시간 (자동)

### 최적화 팁
1. **긴 프롬프트 우선 캐싱** (1024+ 토큰)
2. **캐시 브레이크포인트 최대 4개**
3. **변하지 않는 부분만 캐싱**
4. **5분 이내 재사용 확실할 때만 캐싱**

---

## Parallel Tool Calling

**여러 도구를 동시에 호출하여 성능 향상**

### 지원 Provider
- **OpenAI**: 모든 GPT-4 시리즈 (기본 활성화)
- **Anthropic**: Claude 전 모델 (기본 비활성화, 안전 중시)

### OpenAI Parallel Tool Calling

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="your-api-key")

# 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": ["timezone"]
            }
        }
    }
]

# Parallel Tool Calling (기본 활성화)
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in Seoul and what time is it in Tokyo?"}
    ],
    tools=tools,
    parallel_tool_calls=True  # 병렬 호출 활성화 (기본값)
)

# 결과: 2개 도구 동시 호출
for tool_call in response.choices[0].message.tool_calls:
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

### 병렬 호출 비활성화

```python
# 순차적 호출 (한 번에 하나씩)
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    parallel_tool_calls=False  # 순차 호출
)
```

### Tool Choice 전략

#### 1. **"auto"** (기본값)
모델이 도구 호출 여부/선택 결정
```python
tool_choice="auto"
```

#### 2. **"required"**
항상 하나 이상의 도구 호출 강제
```python
tool_choice="required"
```

#### 3. **"none"**
도구 호출 비활성화
```python
tool_choice="none"
```

#### 4. **특정 도구 지정**
```python
tool_choice={
    "type": "function",
    "function": {"name": "get_weather"}
}
```

### 안전성 고려사항

#### OpenAI
- 기본적으로 병렬 호출 활성화
- 도구 간 의존성 없을 때 사용
- 병렬 실행이 안전한지 확인 필요

#### Anthropic
- 기본적으로 순차 호출 (안전)
- 엔터프라이즈 사용자: 더 안정적인 병렬 체인
- 도구 호출 순서 중요할 때 순차 사용

### 사용 예시

```python
# 안전한 병렬 호출 (독립적인 도구들)
tools = [
    get_weather,  # 날씨 조회
    get_stock_price,  # 주가 조회
    get_news  # 뉴스 조회
]
parallel_tool_calls=True  # OK

# 위험한 병렬 호출 (의존성 있음)
tools = [
    create_user,  # 사용자 생성
    assign_role,  # 역할 할당 (create_user에 의존)
    send_email  # 이메일 전송 (create_user에 의존)
]
parallel_tool_calls=False  # 순차 실행 권장
```

---

## 통합 사용 예시

### 모든 고급 기능 조합

```python
from openai import AsyncOpenAI
import json

client = AsyncOpenAI(api_key="your-api-key")

# 도구 정의
tools = [...]

# 긴 시스템 프롬프트 (캐싱될 예정)
system_prompt = "You are an expert assistant..." * 500

response = await client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": system_prompt},  # 자동 캐싱
        {"role": "user", "content": "Extract data and call tools"}
    ],
    tools=tools,
    parallel_tool_calls=True,  # 병렬 도구 호출
    response_format={  # Structured Output
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "strict": True,
            "schema": {...}
        }
    }
)

# 100% 스키마 준수 + 병렬 도구 호출 + 캐시 비용 절감
```

---

## 모범 사례

### 1. Structured Outputs
- 복잡한 데이터 추출 작업에 사용
- `strict: true`로 100% 정확도 보장
- Pydantic 모델과 통합 권장

### 2. Prompt Caching
- 1024+ 토큰 프롬프트에만 적용
- 5분 이내 재사용 확실할 때만 사용
- 시스템 프롬프트와 문서 우선 캐싱

### 3. Parallel Tool Calling
- 독립적인 도구만 병렬 호출
- 의존성 있으면 순차 실행
- 안전성 우선 고려

---

## 참고 자료

- [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [Anthropic Prompt Caching](https://www.anthropic.com/news/prompt-caching)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
