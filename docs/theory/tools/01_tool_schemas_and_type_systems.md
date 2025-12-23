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
# domain/tools/tool.py: Tool
# domain/tools/advanced/decorator.py: @tool 데코레이터
# domain/tools/advanced/schema.py: SchemaGenerator
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any

@dataclass
class ToolParameter:
    """
    도구 파라미터: (name, type, description, required)
    
    실제 구현:
    - domain/tools/tool.py: ToolParameter
    """
    name: str
    type: str  # string, number, boolean, object, array
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

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
    parameters: List[ToolParameter]
    function: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """
        OpenAI Function Calling 형식으로 변환
        
        실제 구현:
        - domain/tools/tool.py: Tool.to_openai_format()
        - OpenAI API에 전달할 JSON Schema 생성
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        도구 실행: f(params)
        
        수학적 표현: result = f(params)
        
        실제 구현:
        - domain/tools/tool.py: Tool.execute()
        - 파라미터 검증 후 함수 실행
        - 오류 처리 및 재시도 지원
        """
        # 파라미터 검증
        validated_params = self._validate_params(arguments)
        
        # 함수 실행
        return self.function(**validated_params)
```

**@tool 데코레이터:**
```python
# domain/tools/advanced/decorator.py: @tool
from typing import Optional, Dict, Any, Callable

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    retry: int = 1,
    cache: bool = False,
    cache_ttl: int = 300,
):
    """
    도구 데코레이터: 함수를 Tool로 변환
    
    실제 구현:
    - domain/tools/advanced/decorator.py: @tool 데코레이터
    - 함수 시그니처에서 자동으로 스키마 생성
    - SchemaGenerator.from_function() 사용
    """
    def decorator(func: Callable) -> Callable:
        # 함수에서 스키마 자동 생성
        from .schema import SchemaGenerator
        tool_schema = schema or SchemaGenerator.from_function(func)
        
        # Tool 객체 생성 및 메타데이터 저장
        func.tool_name = name or func.__name__
        func.tool_description = description or func.__doc__ or ""
        func.schema = tool_schema
        func.validate = validate
        func.retry = retry
        func.cache = cache
        func.cache_ttl = cache_ttl
        
        return func
    
    return decorator
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

