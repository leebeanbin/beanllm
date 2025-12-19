"""
Advanced Tool Calling System

동적 스키마 생성, 외부 API 통합, 도구 조합 및 체이닝 등
고급 Tool Calling 기능을 제공합니다.

Mathematical Foundations:
=======================

1. Function Typing (Type Theory):
   Γ ⊢ f: A → B
   where Γ is type context, f is function, A is input type, B is output type

   For tool with multiple parameters:
   f: A₁ × A₂ × ... × Aₙ → B

2. Schema Validation (Formal Language Theory):
   Schema S = (Σ, G, s₀)
   where Σ is alphabet, G is grammar rules, s₀ is start symbol

   Valid input: x ∈ L(S) where L(S) is language accepted by schema

3. API Rate Limiting (Token Bucket Algorithm):
   Tokens(t) = min(capacity, Tokens(t-1) + rate × Δt)

   Request allowed if: Tokens(t) ≥ cost

4. Retry Strategy (Exponential Backoff):
   Wait_time(n) = min(max_wait, base_wait × 2^n + jitter)
   where n is retry attempt number

5. Tool Composition (Category Theory):
   (g ∘ f)(x) = g(f(x))

   Associativity: h ∘ (g ∘ f) = (h ∘ g) ∘ f
   Identity: id ∘ f = f ∘ id = f

References:
----------
- Pierce, B. C. (2002). Types and Programming Languages
- JSON Schema Specification: https://json-schema.org/
- RESTful API Design: Fielding's dissertation (2000)
- GraphQL Specification: https://spec.graphql.org/

Author: LLMKit Team
"""

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import httpx
import requests
from pydantic import BaseModel

# ============================================================================
# Part 1: Dynamic Schema Generation
# ============================================================================


class SchemaGenerator:
    """
    동적 스키마 생성기

    Python 함수의 타입 힌트로부터 JSON Schema를 자동 생성합니다.

    Mathematical Foundation:
        Type Inference: Γ ⊢ e: τ
        where Γ is type environment, e is expression, τ is type

        For function f with signature f: T₁ × T₂ × ... × Tₙ → R:
        Schema(f) = {
            "type": "object",
            "properties": {pᵢ: Schema(Tᵢ) for i in 1..n},
            "required": [pᵢ for i in 1..n if pᵢ has no default]
        }
    """

    _type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    @classmethod
    def from_function(cls, func: Callable) -> Dict[str, Any]:
        """
        함수로부터 JSON Schema 생성

        Args:
            func: Python 함수

        Returns:
            JSON Schema dict

        Example:
            >>> def greet(name: str, age: int = 25) -> str:
            ...     return f"Hello {name}, age {age}"
            >>> schema = SchemaGenerator.from_function(greet)
            >>> schema['properties']['name']
            {'type': 'string'}
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in type_hints:
                param_type = type_hints[param_name]
                properties[param_name] = cls._type_to_schema(param_type)

                # Add description from docstring if available
                if func.__doc__:
                    # Simple parsing - can be enhanced
                    properties[param_name]["description"] = f"Parameter {param_name}"

                # Required if no default value
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "description": func.__doc__ or f"Schema for {func.__name__}",
        }

    @classmethod
    def _type_to_schema(cls, type_hint: Type) -> Dict[str, Any]:
        """타입 힌트를 JSON Schema로 변환"""
        origin = get_origin(type_hint)

        # Handle Optional[T] -> Union[T, None]
        if origin is Union:
            args = get_args(type_hint)
            # Filter out NoneType
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return cls._type_to_schema(non_none_args[0])

        # Handle List[T]
        if origin is list:
            args = get_args(type_hint)
            if args:
                return {"type": "array", "items": cls._type_to_schema(args[0])}
            return {"type": "array"}

        # Handle Dict[K, V]
        if origin is dict:
            return {"type": "object"}

        # Base types
        if type_hint in cls._type_mapping:
            return cls._type_mapping[type_hint].copy()

        # Enum
        if isinstance(type_hint, type) and issubclass(type_hint, Enum):
            return {"type": "string", "enum": [e.value for e in type_hint]}

        # Fallback
        return {"type": "object"}

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Pydantic 모델로부터 JSON Schema 생성

        Args:
            model: Pydantic BaseModel 클래스

        Returns:
            JSON Schema dict
        """
        return model.schema()


# ============================================================================
# Part 2: Tool Validator
# ============================================================================


class ToolValidator:
    """
    도구 입력 검증기

    Mathematical Foundation:
        Schema Validation as Language Acceptance:

        Given schema S and input x:
        Valid(x, S) ⟺ x ∈ L(S)

        where L(S) is the language defined by schema S

        Validation Rules:
        - Type checking: typeof(x) = T where T is expected type
        - Range checking: x ∈ [min, max] for numeric types
        - Pattern matching: x matches regex pattern
        - Required fields: ∀f ∈ required. f ∈ keys(x)
    """

    @staticmethod
    def validate(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        데이터가 스키마를 만족하는지 검증

        Args:
            data: 검증할 데이터
            schema: JSON Schema

        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Check properties
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                field_schema = properties[key]
                is_valid, error = ToolValidator._validate_field(value, field_schema, key)
                if not is_valid:
                    return False, error

        return True, None

    @staticmethod
    def _validate_field(
        value: Any, schema: Dict[str, Any], field_name: str
    ) -> tuple[bool, Optional[str]]:
        """개별 필드 검증"""
        expected_type = schema.get("type")

        type_check_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type in type_check_map:
            expected_python_type = type_check_map[expected_type]
            if not isinstance(value, expected_python_type):
                return (
                    False,
                    f"Field '{field_name}' must be of type {expected_type}, got {type(value).__name__}",
                )

        # Enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                return False, f"Field '{field_name}' must be one of {schema['enum']}, got {value}"

        # Range validation for numbers
        if expected_type in ("integer", "number"):
            if "minimum" in schema and value < schema["minimum"]:
                return False, f"Field '{field_name}' must be >= {schema['minimum']}"
            if "maximum" in schema and value > schema["maximum"]:
                return False, f"Field '{field_name}' must be <= {schema['maximum']}"

        # Array items validation
        if expected_type == "array" and "items" in schema:
            for i, item in enumerate(value):
                is_valid, error = ToolValidator._validate_field(
                    item, schema["items"], f"{field_name}[{i}]"
                )
                if not is_valid:
                    return False, error

        return True, None


# ============================================================================
# Part 3: External API Integration
# ============================================================================


class APIProtocol(Enum):
    """API 프로토콜"""

    REST = "rest"
    GRAPHQL = "graphql"


@dataclass
class APIConfig:
    """API 설정"""

    base_url: str
    protocol: APIProtocol = APIProtocol.REST
    auth_type: Optional[str] = None  # "bearer", "api_key", "basic"
    auth_value: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    rate_limit: Optional[int] = None  # requests per minute


class ExternalAPITool:
    """
    외부 API 통합 도구

    Mathematical Foundation:
        API Call as Function Composition:

        API_call(endpoint, params) = parse ∘ send ∘ validate ∘ prepare

        where:
        - prepare: params → request
        - validate: request → validated_request
        - send: validated_request → response
        - parse: response → result

        Error Handling with Retry:
        Result = try_with_exponential_backoff(API_call, max_retries)

        where wait_time(n) = min(max_wait, base × 2^n)
    """

    def __init__(self, config: APIConfig):
        """
        Args:
            config: API 설정
        """
        self.config = config
        self.session = requests.Session()
        self._setup_auth()
        self._last_request_time = 0

    def _setup_auth(self):
        """인증 설정"""
        if self.config.auth_type == "bearer":
            self.session.headers["Authorization"] = f"Bearer {self.config.auth_value}"
        elif self.config.auth_type == "api_key":
            self.session.headers["X-API-Key"] = self.config.auth_value
        elif self.config.auth_type == "basic":
            from requests.auth import HTTPBasicAuth

            username, password = self.config.auth_value.split(":", 1)
            self.session.auth = HTTPBasicAuth(username, password)

        # Add custom headers
        self.session.headers.update(self.config.headers)

    def _rate_limit_check(self):
        """Rate limiting (Token Bucket Algorithm)"""
        if self.config.rate_limit is None:
            return

        # Simple implementation: ensure minimum time between requests
        min_interval = 60.0 / self.config.rate_limit  # seconds per request
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    def call(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        API 호출 (동기)

        Args:
            endpoint: API 엔드포인트 (예: "/users/123")
            method: HTTP 메서드
            params: URL 쿼리 파라미터
            data: 요청 본문 데이터
            **kwargs: 추가 requests 옵션

        Returns:
            API 응답 (JSON)

        Raises:
            requests.RequestException: API 호출 실패
        """
        self._rate_limit_check()

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Exponential backoff retry
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.config.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()

            except requests.RequestException:
                if attempt == self.config.max_retries - 1:
                    raise

                # Exponential backoff: 2^attempt seconds
                wait_time = min(30, 2**attempt)
                time.sleep(wait_time)

        raise RuntimeError("Unexpected error in retry logic")

    async def call_async(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        API 호출 (비동기)

        Args:
            endpoint: API 엔드포인트
            method: HTTP 메서드
            params: URL 쿼리 파라미터
            data: 요청 본문 데이터
            **kwargs: 추가 httpx 옵션

        Returns:
            API 응답 (JSON)
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            # Setup auth headers
            headers = self.session.headers.copy()

            for attempt in range(self.config.max_retries):
                try:
                    response = await client.request(
                        method=method, url=url, params=params, json=data, headers=headers, **kwargs
                    )
                    response.raise_for_status()
                    return response.json()

                except httpx.HTTPError:
                    if attempt == self.config.max_retries - 1:
                        raise

                    wait_time = min(30, 2**attempt)
                    await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected error in retry logic")

    def call_graphql(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        GraphQL 쿼리 실행

        Args:
            query: GraphQL 쿼리 문자열
            variables: 쿼리 변수

        Returns:
            GraphQL 응답
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        return self.call(endpoint="/graphql", method="POST", data=payload)


# ============================================================================
# Part 4: Tool Composition and Chaining
# ============================================================================


class ToolChain:
    """
    도구 체이닝 및 조합

    Mathematical Foundation:
        Function Composition in Category Theory:

        Given tools f: A → B and g: B → C:
        (g ∘ f): A → C
        (g ∘ f)(x) = g(f(x))

        Properties:
        1. Associativity: h ∘ (g ∘ f) = (h ∘ g) ∘ f
        2. Identity: id_B ∘ f = f ∘ id_A = f

        Sequential Execution:
        result = fₙ(fₙ₋₁(...f₂(f₁(input))))

        Parallel Execution:
        results = (f₁(input), f₂(input), ..., fₙ(input))
    """

    def __init__(self, tools: List[Callable]):
        """
        Args:
            tools: 체이닝할 도구 함수 리스트
        """
        self.tools = tools

    def execute(self, initial_input: Any) -> Any:
        """
        순차적 도구 실행 (Composition)

        Args:
            initial_input: 첫 번째 도구의 입력

        Returns:
            마지막 도구의 출력

        Example:
            >>> chain = ToolChain([str.lower, str.strip, str.title])
            >>> chain.execute("  HELLO WORLD  ")
            'Hello World'
        """
        result = initial_input
        for tool in self.tools:
            result = tool(result)
        return result

    async def execute_async(self, initial_input: Any) -> Any:
        """비동기 순차 실행"""
        result = initial_input
        for tool in self.tools:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(result)
            else:
                result = tool(result)
        return result

    @staticmethod
    async def execute_parallel(
        tools: List[Callable], inputs: Union[Any, List[Any]], aggregate: Optional[Callable] = None
    ) -> Union[List[Any], Any]:
        """
        병렬 도구 실행

        Args:
            tools: 실행할 도구 리스트
            inputs: 각 도구의 입력 (단일 값이면 모든 도구에 동일하게 적용)
            aggregate: 결과 집계 함수 (선택)

        Returns:
            각 도구의 결과 리스트 (aggregate가 있으면 집계된 결과)

        Example:
            >>> async def f1(x): return x + 1
            >>> async def f2(x): return x * 2
            >>> results = await ToolChain.execute_parallel([f1, f2], 5)
            >>> results
            [6, 10]
        """
        # Prepare inputs
        if not isinstance(inputs, list):
            inputs = [inputs] * len(tools)

        # Execute in parallel
        tasks = []
        for tool, input_val in zip(tools, inputs):
            if asyncio.iscoroutinefunction(tool):
                tasks.append(tool(input_val))
            else:
                tasks.append(asyncio.to_thread(tool, input_val))

        results = await asyncio.gather(*tasks)

        # Aggregate if needed
        if aggregate:
            return aggregate(results)

        return list(results)


# ============================================================================
# Part 5: Advanced Tool Decorator
# ============================================================================


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
    고급 도구 데코레이터

    기능:
    - 자동 스키마 생성
    - 입력 검증
    - 재시도 로직
    - 결과 캐싱

    Args:
        name: 도구 이름 (기본값: 함수 이름)
        description: 도구 설명
        schema: 커스텀 JSON Schema (자동 생성 대신)
        validate: 입력 검증 활성화
        retry: 재시도 횟수
        cache: 결과 캐싱 활성화
        cache_ttl: 캐시 유효 시간 (초)

    Example:
        >>> @tool(description="Calculate sum", validate=True, retry=3)
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> add.schema
        {'type': 'object', 'properties': {...}, ...}
    """

    def decorator(func: Callable) -> Callable:
        # Generate schema
        func_schema = schema or SchemaGenerator.from_function(func)
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""

        # Cache storage
        _cache = {} if cache else None

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args to kwargs for validation
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments

            # Validate input
            if validate:
                is_valid, error = ToolValidator.validate(params, func_schema)
                if not is_valid:
                    raise ValueError(f"Tool validation failed: {error}")

            # Check cache
            if cache:
                cache_key = json.dumps(params, sort_keys=True)
                if cache_key in _cache:
                    cached_result, cached_time = _cache[cache_key]
                    if time.time() - cached_time < cache_ttl:
                        return cached_result

            # Execute with retry
            last_exception = None
            for attempt in range(retry):
                try:
                    result = func(**params)

                    # Store in cache
                    if cache:
                        _cache[cache_key] = (result, time.time())

                    return result

                except Exception as e:
                    last_exception = e
                    if attempt < retry - 1:
                        wait_time = 2**attempt
                        time.sleep(wait_time)

            raise last_exception

        # Attach metadata
        wrapper.schema = func_schema
        wrapper.tool_name = func_name
        wrapper.tool_description = func_description
        wrapper.is_tool = True

        return wrapper

    return decorator


# ============================================================================
# Part 6: Tool Registry
# ============================================================================


class ToolRegistry:
    """
    도구 레지스트리

    모든 도구를 중앙에서 관리하고, 이름으로 검색/실행할 수 있습니다.

    Mathematical Foundation:
        Registry as Mapping:
        R: ToolName → Tool

        where ToolName is string identifier
        and Tool is (function, schema, metadata)

        Lookup: R[name] → Tool or ∅ (empty if not found)
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **metadata,
    ):
        """
        도구 등록

        Args:
            func: 도구 함수
            name: 도구 이름 (기본값: 함수 이름)
            schema: JSON Schema
            **metadata: 추가 메타데이터
        """
        tool_name = name or getattr(func, "tool_name", func.__name__)
        tool_schema = schema or getattr(func, "schema", SchemaGenerator.from_function(func))

        self._tools[tool_name] = func
        self._schemas[tool_name] = tool_schema
        self._metadata[tool_name] = {
            "description": getattr(func, "tool_description", func.__doc__ or ""),
            **metadata,
        }

    def get(self, name: str) -> Optional[Callable]:
        """도구 조회"""
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """스키마 조회"""
        return self._schemas.get(name)

    def list_tools(self) -> List[str]:
        """등록된 모든 도구 이름 목록"""
        return list(self._tools.keys())

    def execute(self, name: str, **params) -> Any:
        """
        이름으로 도구 실행

        Args:
            name: 도구 이름
            **params: 도구 파라미터

        Returns:
            도구 실행 결과

        Raises:
            KeyError: 도구가 없는 경우
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        tool = self._tools[name]
        return tool(**params)

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        OpenAI function calling 형식으로 변환

        Returns:
            OpenAI tools 리스트
        """
        tools = []
        for name in self._tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": self._metadata[name].get("description", ""),
                        "parameters": self._schemas[name],
                    },
                }
            )
        return tools


# ============================================================================
# Global Registry Instance
# ============================================================================

# 전역 레지스트리
default_registry = ToolRegistry()
