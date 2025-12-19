# Tool Schemas and Type Systems: 도구 스키마와 타입 시스템

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit Tool 실제 구현 분석

---

## 목차

1. [도구의 형식적 정의](#1-도구의-형식적-정의)
2. [스키마와 타입 시스템](#2-스키마와-타입-시스템)
3. [JSON Schema 표현](#3-json-schema-표현)
4. [타입 검증](#4-타입-검증)
5. [CS 관점: 구현과 최적화](#5-cs-관점-구현과-최적화)

---

## 1. 도구의 형식적 정의

### 1.1 도구 튜플

#### 정의 1.1.1: 도구 (Tool)

**도구**는 다음 튜플로 정의됩니다:

$$
\text{Tool} = (name, description, parameters, function)
$$

**llmkit 구현:**
```python
# tools.py: Line 25-46
@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
```

---

## 2. 스키마와 타입 시스템

### 2.1 파라미터 스키마

#### 정의 2.1.1: 파라미터 스키마

**파라미터 스키마:**

$$
\text{Schema} = \{type, properties, required\}
$$

**타입:**
- `string`: 문자열
- `number`: 숫자
- `boolean`: 불린
- `object`: 객체
- `array`: 배열

---

## 3. JSON Schema 표현

### 3.1 OpenAI 형식

#### 정의 3.1.1: OpenAI Tool Schema

**OpenAI 형식:**

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "날씨 조회",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "도시 이름"
        }
      },
      "required": ["city"]
    }
  }
}
```

---

## 4. 타입 검증

### 4.1 런타임 검증

#### 알고리즘 4.1.1: 타입 검증

```
Algorithm: ValidateParameters(params, schema)
1. for param in schema.required:
2.     if param not in params:
3.         raise ValidationError
4. 
5. for param, value in params.items():
6.     expected_type = schema.properties[param].type
7.     if not isinstance(value, expected_type):
8.         raise TypeError
9. 
10. return True
```

---

## 5. CS 관점: 구현과 최적화

### 5.1 스키마 컴파일

#### CS 관점 5.1.1: 스키마 최적화

**스키마 사전 컴파일:**

```python
# 스키마를 파싱하여 검증 함수 생성
def compile_schema(schema):
    validators = {}
    for param, spec in schema["properties"].items():
        validators[param] = create_validator(spec)
    return validators
```

**효과:**
- 런타임 검증 속도 향상
- 타입 체크 최적화

---

## 질문과 답변 (Q&A)

### Q1: 타입 검증은 왜 필요한가요?

**A:** 필요성:

1. **타입 안전성:**
   - 런타임 에러 방지
   - 예상치 못한 동작 방지

2. **LLM 가이드:**
   - 올바른 파라미터 형식 제공
   - 에러 감소

---

## 참고 문헌

1. **OpenAI (2023)**: "Function Calling" - Tool Schema

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

