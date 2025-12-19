"""
Tool Calling 실습 튜토리얼

이 튜토리얼에서는 동적 스키마 생성, 외부 API 통합, 도구 조합 등
고급 Tool Calling 기능을 실습합니다.

Prerequisites:
- Python 3.10+
- requests, httpx, pydantic 설치
- API keys (optional, for real API examples)

Author: LLMKit Team
"""

import asyncio
import time
import json
from typing import List, Optional
from enum import Enum

# ============================================================================
# Part 1: Dynamic Schema Generation
# ============================================================================

print("=" * 80)
print("Part 1: Dynamic Schema Generation")
print("=" * 80)

from llmkit.tools_advanced import SchemaGenerator

# Example 1.1: Simple function schema
def greet(name: str, age: int = 25) -> str:
    """
    Greet a person with their name and age.

    Args:
        name: Person's name
        age: Person's age (default: 25)
    """
    return f"Hello {name}, you are {age} years old!"

schema = SchemaGenerator.from_function(greet)
print("\n1.1 Schema for greet function:")
print(json.dumps(schema, indent=2))

# Output:
# {
#   "type": "object",
#   "properties": {
#     "name": {"type": "string", "description": "Parameter name"},
#     "age": {"type": "integer", "description": "Parameter age"}
#   },
#   "required": ["name"],
#   "description": "Greet a person..."
# }


# Example 1.2: Complex types with Optional
def search_users(
    query: str,
    limit: int = 10,
    tags: Optional[List[str]] = None
) -> List[dict]:
    """Search users by query"""
    return []

schema = SchemaGenerator.from_function(search_users)
print("\n1.2 Schema for search_users:")
print(json.dumps(schema, indent=2))


# Example 1.3: Enum support
class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

def create_task(title: str, priority: Priority = Priority.MEDIUM) -> dict:
    """Create a new task"""
    return {"title": title, "priority": priority.value}

schema = SchemaGenerator.from_function(create_task)
print("\n1.3 Schema with Enum:")
print(json.dumps(schema, indent=2))


# Example 1.4: Pydantic model schema
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int
    name: str
    email: str = Field(..., description="User's email address")
    age: Optional[int] = Field(None, ge=0, le=150)

schema = SchemaGenerator.from_pydantic(User)
print("\n1.4 Pydantic schema:")
print(json.dumps(schema, indent=2))


# ============================================================================
# Part 2: Input Validation
# ============================================================================

print("\n" + "=" * 80)
print("Part 2: Input Validation")
print("=" * 80)

from llmkit.tools_advanced import ToolValidator

# Example 2.1: Valid input
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    },
    "required": ["name"]
}

data = {"name": "Alice", "age": 30}
is_valid, error = ToolValidator.validate(data, schema)
print(f"\n2.1 Valid data: {is_valid}, Error: {error}")
# Output: Valid data: True, Error: None


# Example 2.2: Missing required field
data = {"age": 30}
is_valid, error = ToolValidator.validate(data, schema)
print(f"\n2.2 Missing required field: {is_valid}, Error: {error}")
# Output: Missing required field: False, Error: Missing required field: name


# Example 2.3: Type mismatch
data = {"name": "Alice", "age": "thirty"}
is_valid, error = ToolValidator.validate(data, schema)
print(f"\n2.3 Type mismatch: {is_valid}, Error: {error}")
# Output: Type mismatch: False, Error: Field 'age' must be of type integer...


# Example 2.4: Range validation
data = {"name": "Alice", "age": 200}
is_valid, error = ToolValidator.validate(data, schema)
print(f"\n2.4 Range violation: {is_valid}, Error: {error}")
# Output: Range violation: False, Error: Field 'age' must be <= 150


# Example 2.5: Enum validation
schema_with_enum = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pending", "active", "completed"]}
    },
    "required": ["status"]
}

data = {"status": "invalid"}
is_valid, error = ToolValidator.validate(data, schema_with_enum)
print(f"\n2.5 Enum validation: {is_valid}, Error: {error}")
# Output: Enum validation: False, Error: Field 'status' must be one of...


# ============================================================================
# Part 3: External REST API Integration
# ============================================================================

print("\n" + "=" * 80)
print("Part 3: External REST API Integration")
print("=" * 80)

from llmkit.tools_advanced import ExternalAPITool, APIConfig, APIProtocol

# Example 3.1: JSONPlaceholder API (Public test API)
config = APIConfig(
    base_url="https://jsonplaceholder.typicode.com",
    protocol=APIProtocol.REST,
    timeout=10
)

api = ExternalAPITool(config)

try:
    # GET request
    user = api.call(endpoint="/users/1", method="GET")
    print("\n3.1 GET user:")
    print(json.dumps(user, indent=2)[:200] + "...")

    # POST request
    new_post = api.call(
        endpoint="/posts",
        method="POST",
        data={
            "title": "My First Post",
            "body": "This is the content",
            "userId": 1
        }
    )
    print("\n3.1 POST new post:")
    print(json.dumps(new_post, indent=2))

except Exception as e:
    print(f"API call failed: {e}")


# Example 3.2: API with authentication
config_auth = APIConfig(
    base_url="https://api.example.com",
    auth_type="bearer",
    auth_value="your_api_token_here",
    headers={"User-Agent": "LLMKit/1.0"}
)

# api_auth = ExternalAPITool(config_auth)
# response = api_auth.call("/protected/endpoint")


# Example 3.3: Rate limiting
config_rate_limited = APIConfig(
    base_url="https://jsonplaceholder.typicode.com",
    rate_limit=5,  # 5 requests per minute
    max_retries=3
)

api_limited = ExternalAPITool(config_rate_limited)

print("\n3.3 Testing rate limiting (5 req/min):")
start_time = time.time()

for i in range(3):
    try:
        result = api_limited.call(f"/users/{i+1}")
        elapsed = time.time() - start_time
        print(f"  Request {i+1} completed at {elapsed:.2f}s")
    except Exception as e:
        print(f"  Request {i+1} failed: {e}")


# Example 3.4: Async API calls
async def async_api_example():
    print("\n3.4 Async API calls:")
    api = ExternalAPITool(APIConfig(
        base_url="https://jsonplaceholder.typicode.com"
    ))

    # Parallel requests
    tasks = [
        api.call_async(f"/users/{i}")
        for i in range(1, 4)
    ]

    start = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"  Fetched {len(results)} users in {elapsed:.2f}s (parallel)")
    for i, user in enumerate(results, 1):
        print(f"  User {i}: {user['name']}")

# asyncio.run(async_api_example())


# ============================================================================
# Part 4: GraphQL Integration
# ============================================================================

print("\n" + "=" * 80)
print("Part 4: GraphQL Integration")
print("=" * 80)

# Example 4.1: GraphQL query (using a public GraphQL API)
# Note: This example uses SpaceX GraphQL API

config_graphql = APIConfig(
    base_url="https://spacex-production.up.railway.app",
    protocol=APIProtocol.GRAPHQL
)

api_graphql = ExternalAPITool(config_graphql)

query = """
query {
  launches(limit: 3) {
    mission_name
    launch_date_local
    rocket {
      rocket_name
    }
  }
}
"""

try:
    result = api_graphql.call_graphql(query)
    print("\n4.1 SpaceX launches:")
    for launch in result.get('data', {}).get('launches', []):
        print(f"  - {launch['mission_name']} ({launch['rocket']['rocket_name']})")
except Exception as e:
    print(f"GraphQL query failed: {e}")
    print("  (This is expected if the API is down or changed)")


# Example 4.2: GraphQL with variables
query_with_vars = """
query GetLaunches($limit: Int!) {
  launches(limit: $limit) {
    mission_name
    launch_year
  }
}
"""

variables = {"limit": 5}

try:
    result = api_graphql.call_graphql(query_with_vars, variables)
    print("\n4.2 Launches with variables (limit=5):")
    for launch in result.get('data', {}).get('launches', []):
        print(f"  - {launch['mission_name']} ({launch.get('launch_year', 'N/A')})")
except Exception as e:
    print(f"GraphQL query failed: {e}")


# ============================================================================
# Part 5: Tool Composition and Chaining
# ============================================================================

print("\n" + "=" * 80)
print("Part 5: Tool Composition and Chaining")
print("=" * 80)

from llmkit.tools_advanced import ToolChain

# Example 5.1: Sequential tool chain
def extract_text(data: dict) -> str:
    """Extract text from data"""
    return data.get("text", "")

def to_lowercase(text: str) -> str:
    """Convert to lowercase"""
    return text.lower()

def remove_punctuation(text: str) -> str:
    """Remove punctuation"""
    import string
    return text.translate(str.maketrans("", "", string.punctuation))

def tokenize(text: str) -> List[str]:
    """Split into words"""
    return text.split()

# Create chain
chain = ToolChain([
    extract_text,
    to_lowercase,
    remove_punctuation,
    tokenize
])

# Execute
input_data = {"text": "Hello, World! How are YOU?"}
result = chain.execute(input_data)
print(f"\n5.1 Sequential chain result: {result}")
# Output: ['hello', 'world', 'how', 'are', 'you']


# Example 5.2: Parallel tool execution
async def parallel_example():
    async def fetch_weather(city: str) -> dict:
        """Simulate weather API"""
        await asyncio.sleep(0.5)
        return {"city": city, "temp": 20 + hash(city) % 10}

    async def fetch_news(topic: str) -> dict:
        """Simulate news API"""
        await asyncio.sleep(0.5)
        return {"topic": topic, "count": 10 + hash(topic) % 5}

    async def fetch_stock(symbol: str) -> dict:
        """Simulate stock API"""
        await asyncio.sleep(0.5)
        return {"symbol": symbol, "price": 100 + hash(symbol) % 50}

    # Execute in parallel
    start = time.time()
    results = await ToolChain.execute_parallel(
        tools=[fetch_weather, fetch_news, fetch_stock],
        inputs=["Seoul", "AI", "AAPL"]
    )
    elapsed = time.time() - start

    print(f"\n5.2 Parallel execution ({elapsed:.2f}s):")
    print(f"  Weather: {results[0]}")
    print(f"  News: {results[1]}")
    print(f"  Stock: {results[2]}")

# asyncio.run(parallel_example())


# Example 5.3: Aggregated parallel results
async def aggregated_example():
    async def sentiment_analyzer_1(text: str) -> float:
        await asyncio.sleep(0.1)
        return 0.8  # Positive

    async def sentiment_analyzer_2(text: str) -> float:
        await asyncio.sleep(0.1)
        return 0.6  # Somewhat positive

    async def sentiment_analyzer_3(text: str) -> float:
        await asyncio.sleep(0.1)
        return 0.7  # Positive

    # Aggregate using average
    def average(scores: List[float]) -> float:
        return sum(scores) / len(scores)

    text = "This product is great!"
    avg_score = await ToolChain.execute_parallel(
        tools=[sentiment_analyzer_1, sentiment_analyzer_2, sentiment_analyzer_3],
        inputs=text,
        aggregate=average
    )

    print(f"\n5.3 Ensemble sentiment score: {avg_score:.2f}")

# asyncio.run(aggregated_example())


# ============================================================================
# Part 6: Advanced Tool Decorator
# ============================================================================

print("\n" + "=" * 80)
print("Part 6: Advanced Tool Decorator")
print("=" * 80)

from llmkit.tools_advanced import tool

# Example 6.1: Basic tool decorator
@tool(description="Calculate the sum of two numbers", validate=True)
def add(a: int, b: int) -> int:
    """Add two integers"""
    return a + b

print(f"\n6.1 Tool metadata:")
print(f"  Name: {add.tool_name}")
print(f"  Description: {add.tool_description}")
print(f"  Schema: {json.dumps(add.schema, indent=2)[:200]}...")

result = add(10, 20)
print(f"  Result: add(10, 20) = {result}")


# Example 6.2: Validation with error
@tool(validate=True)
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

try:
    result = divide(10, 2)
    print(f"\n6.2 Valid call: divide(10, 2) = {result}")

    # This will fail validation (wrong type)
    result = divide("10", 2)
except ValueError as e:
    print(f"  Validation error: {e}")


# Example 6.3: Retry logic
attempt_count = 0

@tool(retry=3, validate=False)
def flaky_api_call(endpoint: str) -> dict:
    """Simulate flaky API that fails sometimes"""
    global attempt_count
    attempt_count += 1
    print(f"    Attempt {attempt_count}...")

    if attempt_count < 3:
        raise ConnectionError("Network error")

    return {"status": "success", "data": "Hello"}

print("\n6.3 Tool with retry:")
attempt_count = 0
try:
    result = flaky_api_call("https://api.example.com/data")
    print(f"  Result after retries: {result}")
except Exception as e:
    print(f"  Failed after retries: {e}")


# Example 6.4: Caching
@tool(cache=True, cache_ttl=5)  # 5 second cache
def expensive_computation(x: int) -> int:
    """Expensive computation (simulated)"""
    print(f"    Computing for x={x}...")
    time.sleep(1)  # Simulate expensive work
    return x * x

print("\n6.4 Tool with caching:")
start = time.time()
result1 = expensive_computation(10)
time1 = time.time() - start
print(f"  First call: {result1} (took {time1:.2f}s)")

start = time.time()
result2 = expensive_computation(10)  # Cache hit
time2 = time.time() - start
print(f"  Second call (cached): {result2} (took {time2:.4f}s)")

time.sleep(6)  # Wait for cache to expire

start = time.time()
result3 = expensive_computation(10)  # Cache expired
time3 = time.time() - start
print(f"  Third call (expired): {result3} (took {time3:.2f}s)")


# ============================================================================
# Part 7: Tool Registry
# ============================================================================

print("\n" + "=" * 80)
print("Part 7: Tool Registry")
print("=" * 80)

from llmkit.tools_advanced import ToolRegistry

# Example 7.1: Register tools
registry = ToolRegistry()

@tool(description="Get weather for a city")
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get weather information"""
    # Simulated data
    return {
        "city": city,
        "temperature": 20,
        "units": units,
        "condition": "sunny"
    }

@tool(description="Search for news articles")
def search_news(query: str, limit: int = 10) -> List[dict]:
    """Search news articles"""
    # Simulated data
    return [
        {"title": f"Article about {query}", "source": "News Site"}
        for _ in range(limit)
    ]

@tool(description="Get stock price")
def get_stock_price(symbol: str) -> dict:
    """Get current stock price"""
    # Simulated data
    return {
        "symbol": symbol,
        "price": 150.25,
        "change": +2.5
    }

# Register all tools
registry.register(get_weather)
registry.register(search_news)
registry.register(get_stock_price)

print("\n7.1 Registered tools:")
for tool_name in registry.list_tools():
    print(f"  - {tool_name}")


# Example 7.2: Execute by name
print("\n7.2 Execute tools by name:")
result = registry.execute("get_weather", city="Seoul", units="celsius")
print(f"  get_weather: {result}")

result = registry.execute("get_stock_price", symbol="AAPL")
print(f"  get_stock_price: {result}")


# Example 7.3: Get schema
print("\n7.3 Tool schema:")
schema = registry.get_schema("search_news")
print(json.dumps(schema, indent=2))


# Example 7.4: OpenAI function calling format
print("\n7.4 OpenAI format:")
openai_tools = registry.to_openai_format()
print(json.dumps(openai_tools[0], indent=2))  # First tool


# ============================================================================
# Part 8: Real-World Integration Example
# ============================================================================

print("\n" + "=" * 80)
print("Part 8: Real-World Integration - Multi-Tool Agent")
print("=" * 80)

# Example 8.1: Building a multi-tool agent
class ToolAgent:
    """Simple agent that can use multiple tools"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_plan(self, plan: List[tuple]) -> dict:
        """
        Execute a plan of tool calls

        Args:
            plan: List of (tool_name, params) tuples

        Returns:
            Dictionary of results
        """
        results = {}
        for tool_name, params in plan:
            print(f"  Executing: {tool_name}({params})")
            try:
                result = self.registry.execute(tool_name, **params)
                results[tool_name] = result
            except Exception as e:
                results[tool_name] = {"error": str(e)}

        return results


# Create agent
agent = ToolAgent(registry)

# Execute plan
print("\n8.1 Agent executing plan:")
plan = [
    ("get_weather", {"city": "Seoul"}),
    ("search_news", {"query": "AI", "limit": 3}),
    ("get_stock_price", {"symbol": "GOOGL"})
]

results = agent.execute_plan(plan)
print("\n  Results:")
for tool_name, result in results.items():
    print(f"    {tool_name}: {result}")


# Example 8.2: LLM + Tools integration pattern
def simulate_llm_tool_use():
    """Simulate how an LLM would use tools"""
    print("\n8.2 Simulated LLM + Tools:")

    # User query
    user_query = "What's the weather in Seoul and latest AI news?"
    print(f"  User: {user_query}")

    # LLM decides which tools to call (simulated)
    print("\n  LLM thinking: I need to call get_weather and search_news")

    # Tool calls
    tool_calls = [
        ("get_weather", {"city": "Seoul"}),
        ("search_news", {"query": "AI", "limit": 2})
    ]

    # Execute
    results = agent.execute_plan(tool_calls)

    # LLM generates response using tool results (simulated)
    weather = results["get_weather"]
    news = results["search_news"]

    response = f"""
  Assistant: Based on the latest information:

  Weather in Seoul: {weather['temperature']}°{weather['units']}, {weather['condition']}

  Latest AI News:
    - {news[0]['title']}
    - {news[1]['title']}
    """

    print(response)

simulate_llm_tool_use()


# Example 8.3: Error handling and fallback
@tool(retry=2)
def unreliable_api(endpoint: str) -> dict:
    """API that might fail"""
    import random
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError("API unavailable")
    return {"status": "ok", "data": "success"}

registry.register(unreliable_api)

print("\n8.3 Error handling:")
for i in range(3):
    try:
        result = registry.execute("unreliable_api", endpoint="/data")
        print(f"  Attempt {i+1}: {result}")
    except Exception as e:
        print(f"  Attempt {i+1}: Failed - {e}")


# ============================================================================
# Part 9: Performance Benchmarking
# ============================================================================

print("\n" + "=" * 80)
print("Part 9: Performance Benchmarking")
print("=" * 80)

# Example 9.1: Cache performance
@tool(cache=False)
def slow_function_nocache(x: int) -> int:
    time.sleep(0.1)
    return x * x

@tool(cache=True, cache_ttl=60)
def slow_function_cached(x: int) -> int:
    time.sleep(0.1)
    return x * x

print("\n9.1 Cache performance comparison:")

# Without cache
start = time.time()
for _ in range(10):
    slow_function_nocache(5)
time_nocache = time.time() - start

# With cache
start = time.time()
for _ in range(10):
    slow_function_cached(5)
time_cached = time.time() - start

print(f"  Without cache: {time_nocache:.2f}s")
print(f"  With cache: {time_cached:.2f}s")
print(f"  Speedup: {time_nocache / time_cached:.1f}x")


# Example 9.2: Parallel vs Sequential
async def benchmark_parallel_vs_sequential():
    async def api_call(i: int) -> int:
        await asyncio.sleep(0.2)
        return i * 2

    # Sequential
    start = time.time()
    results_seq = []
    for i in range(5):
        result = await api_call(i)
        results_seq.append(result)
    time_sequential = time.time() - start

    # Parallel
    start = time.time()
    results_par = await ToolChain.execute_parallel(
        tools=[api_call] * 5,
        inputs=list(range(5))
    )
    time_parallel = time.time() - start

    print("\n9.2 Parallel vs Sequential:")
    print(f"  Sequential: {time_sequential:.2f}s")
    print(f"  Parallel: {time_parallel:.2f}s")
    print(f"  Speedup: {time_sequential / time_parallel:.1f}x")

# asyncio.run(benchmark_parallel_vs_sequential())


# ============================================================================
# Part 10: Advanced Patterns
# ============================================================================

print("\n" + "=" * 80)
print("Part 10: Advanced Patterns")
print("=" * 80)

# Example 10.1: Tool with dependencies (DAG)
class DependentToolChain:
    """Execute tools with dependencies"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.results = {}

    def execute_dag(self, dag: dict):
        """
        Execute tools in DAG order

        Args:
            dag: {tool_name: (params, dependencies)}
        """
        executed = set()

        def execute_node(name):
            if name in executed:
                return self.results[name]

            params, deps = dag[name]

            # Execute dependencies first
            for dep in deps:
                execute_node(dep)

            # Resolve parameter references
            resolved_params = {}
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("$"):
                    # Reference to previous result
                    dep_name = value[1:]
                    resolved_params[key] = self.results[dep_name]
                else:
                    resolved_params[key] = value

            # Execute this tool
            result = self.registry.execute(name, **resolved_params)
            self.results[name] = result
            executed.add(name)
            return result

        # Execute all nodes
        for name in dag:
            execute_node(name)

        return self.results


# Register tools for DAG example
@tool()
def fetch_user(user_id: int) -> dict:
    return {"id": user_id, "name": "Alice", "city_id": 1}

@tool()
def fetch_city(city_id: int) -> dict:
    return {"id": city_id, "name": "Seoul", "country": "Korea"}

@tool()
def format_user_info(user: dict, city: dict) -> str:
    return f"{user['name']} lives in {city['name']}, {city['country']}"

dag_registry = ToolRegistry()
dag_registry.register(fetch_user)
dag_registry.register(fetch_city)
dag_registry.register(format_user_info)

# However, format_user_info expects dict inputs, not simple params
# This is a simplified example - real implementation would need more sophisticated parameter passing

print("\n10.1 DAG-based tool execution (simplified example):")
print("  (In production, use a proper workflow engine)")


# Example 10.2: Conditional tool execution
def execute_with_condition(
    registry: ToolRegistry,
    tool_name: str,
    params: dict,
    condition: callable
) -> Optional[dict]:
    """Execute tool only if condition is met"""
    if condition(params):
        print(f"  Condition met, executing {tool_name}")
        return registry.execute(tool_name, **params)
    else:
        print(f"  Condition not met, skipping {tool_name}")
        return None

print("\n10.2 Conditional execution:")
result = execute_with_condition(
    registry,
    "get_weather",
    {"city": "Seoul"},
    condition=lambda p: len(p.get("city", "")) > 0
)
print(f"  Result: {result}")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("TUTORIAL SUMMARY")
print("=" * 80)

summary = """
이 튜토리얼에서 다룬 내용:

1. Dynamic Schema Generation
   - 함수로부터 자동 스키마 생성
   - Pydantic 모델 지원
   - Enum 타입 처리

2. Input Validation
   - 타입 체크
   - 범위 검증
   - 필수 필드 확인

3. External API Integration
   - REST API 호출
   - 인증 (Bearer, API Key)
   - Rate limiting
   - Retry 로직

4. GraphQL
   - 쿼리 실행
   - 변수 전달

5. Tool Composition
   - Sequential chaining
   - Parallel execution
   - Result aggregation

6. Advanced Decorators
   - @tool decorator
   - Validation
   - Retry
   - Caching

7. Tool Registry
   - 도구 등록 및 관리
   - 이름으로 실행
   - OpenAI 형식 변환

8. Real-World Patterns
   - Multi-tool agents
   - LLM integration
   - Error handling

9. Performance
   - Caching 성능
   - Parallel vs Sequential

10. Advanced Patterns
    - DAG execution
    - Conditional execution

다음 단계:
- 실제 API 키로 프로덕션 통합 테스트
- 커스텀 도구 개발
- LLM 프레임워크 통합 (LangChain, LlamaIndex 등)
"""

print(summary)

print("\n" + "=" * 80)
print("튜토리얼 완료!")
print("=" * 80)
