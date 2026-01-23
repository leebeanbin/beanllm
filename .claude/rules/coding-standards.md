# Coding Standards

**우선순위**: HIGH
**적용 범위**: 모든 코드

## Python Standards

### PEP 8 준수

```python
# ✅ Good
def calculate_similarity(
    query_embedding: List[float],
    document_embedding: List[float],
    method: str = "cosine"
) -> float:
    """Calculate similarity between query and document."""
    pass

# ❌ Bad
def calc_sim(q,d,m="cosine"):
    pass
```

### 명명 규칙

```python
# ✅ Classes: PascalCase
class ChatServiceImpl:
    pass

class RAGChain:
    pass

# ✅ Functions/Variables: snake_case
def get_embedding(text: str) -> List[float]:
    pass

max_retries = 3
api_base_url = "https://api.openai.com"

# ✅ Constants: UPPER_SNAKE_CASE
MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
OPENAI_API_VERSION = "2024-02-01"

# ✅ Private: _leading_underscore
class Client:
    def __init__(self):
        self._handler = None  # Private
        self._cache = {}

    def _internal_method(self):  # Private
        pass

# ❌ Bad
class chatService:  # PascalCase 아님
    pass

def GetEmbedding(Text):  # camelCase, 첫 글자 대문자
    pass

maxRetries = 3  # UPPER_SNAKE_CASE 아님
```

### 줄 길이

```python
# 제한: 100자 (pyproject.toml 설정)
# Black이 자동으로 포매팅

# ✅ Good (자동 줄바꿈)
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=100,
)

# ❌ Bad (100자 초과)
response = await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}], temperature=0.7, max_tokens=100)
```

### Import 순서

```python
# 1. 표준 라이브러리
import os
import sys
from typing import List, Dict, Optional

# 2. 서드파티 라이브러리
import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# 3. 로컬 모듈 (절대 경로)
from beanllm.domain.loaders import DocumentLoader
from beanllm.service.chat_service import IChatService
from beanllm.utils.logger import get_logger

# ❌ Bad: 상대 경로
from ...domain.loaders import DocumentLoader
```

### 타입 힌트

```python
# ✅ Good: 모든 함수에 타입 힌트
from typing import List, Dict, Optional, Union

def embed_documents(
    documents: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 32
) -> List[List[float]]:
    """Embed multiple documents."""
    pass

async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Union[str, Dict[str, any]]:
    """Chat with LLM."""
    pass

# ❌ Bad: 타입 힌트 없음
def embed_documents(documents, model="text-embedding-3-small"):
    pass
```

### Docstrings

```python
# ✅ Good: Google 스타일
def similarity_search(
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, any]] = None
) -> List[Document]:
    """
    Search for similar documents.

    Args:
        query: Query text to search
        k: Number of documents to return (default: 5)
        filter: Metadata filter (optional)

    Returns:
        List of k most similar documents

    Raises:
        ValueError: If k <= 0
        VectorStoreError: If search fails

    Example:
        >>> store = VectorStore()
        >>> results = store.similarity_search("AI", k=3)
        >>> len(results)
        3
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # Implementation...
    pass

# ❌ Bad: Docstring 없음
def similarity_search(query, k=5):
    pass
```

### 에러 처리

```python
# ✅ Good: 구체적인 예외 처리
from beanllm.utils.exceptions import RateLimitError, APIError

try:
    response = await provider.chat(messages)
except RateLimitError as e:
    logger.warning(f"Rate limit hit, retrying... {e}")
    await asyncio.sleep(1)
    response = await provider.chat(messages)
except APIError as e:
    logger.error(f"API error: {e}")
    raise
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise

# ❌ Bad: 모든 예외를 동일하게 처리
try:
    response = await provider.chat(messages)
except:  # Bare except
    pass  # Silent failure
```

### 컨텍스트 매니저

```python
# ✅ Good: async with 사용
async with aiofiles.open("file.txt") as f:
    content = await f.read()

# 수동 리소스 관리 필요 시
client = Client(model="gpt-4o")
try:
    response = await client.chat(messages)
finally:
    await client.close()

# ❌ Bad: 리소스 누수
f = open("file.txt")
content = f.read()
# f.close() 누락!
```

### List Comprehension

```python
# ✅ Good: 간단한 경우 list comprehension
squares = [x**2 for x in range(10)]
even_numbers = [x for x in range(10) if x % 2 == 0]

# ✅ Good: 복잡한 경우 명시적 루프
results = []
for doc in documents:
    if doc.metadata.get("source") == "pdf":
        processed = preprocess(doc.content)
        if len(processed) > 100:
            results.append(processed)

# ❌ Bad: 너무 복잡한 list comprehension
results = [
    preprocess(doc.content)
    for doc in documents
    if doc.metadata.get("source") == "pdf"
    if len(preprocess(doc.content)) > 100
]
```

### f-strings

```python
# ✅ Good: f-strings 사용
name = "Alice"
age = 30
message = f"Hello, {name}! You are {age} years old."

# ✅ Good: 복잡한 표현식
total_cost = sum(item.price for item in cart)
summary = f"Total: ${total_cost:.2f} ({len(cart)} items)"

# ❌ Bad: % formatting (Python 2 스타일)
message = "Hello, %s! You are %d years old." % (name, age)

# ❌ Bad: .format() (불필요하게 장황)
message = "Hello, {}! You are {} years old.".format(name, age)
```

## TypeScript/JavaScript Standards (Playground)

### 명명 규칙

```typescript
// ✅ Classes: PascalCase
class ChatClient {
  constructor() {}
}

// ✅ Functions/Variables: camelCase
function calculateSimilarity(a: number[], b: number[]): number {
  return 0;
}

const maxRetries = 3;
const apiBaseUrl = "https://api.openai.com";

// ✅ Constants: UPPER_SNAKE_CASE
const MAX_TOKENS = 4096;
const DEFAULT_TEMPERATURE = 0.7;

// ✅ Interfaces: PascalCase with I prefix (optional)
interface ChatRequest {
  messages: Message[];
  model: string;
  temperature?: number;
}

// ✅ Types: PascalCase
type MessageRole = "system" | "user" | "assistant";
```

### TypeScript 타입

```typescript
// ✅ Good: 명시적 타입
interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatRequest {
  messages: Message[];
  model: string;
  temperature?: number;
  max_tokens?: number;
}

async function chat(request: ChatRequest): Promise<string> {
  // Implementation
  return "response";
}

// ❌ Bad: any 타입
async function chat(request: any): Promise<any> {
  return request;
}
```

### React Components

```typescript
// ✅ Good: Functional component with types
interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  return (
    <div className={`message message-${role}`}>
      <p>{content}</p>
      <time>{timestamp.toLocaleString()}</time>
    </div>
  );
}

// ✅ Good: Custom hooks
function useChatStream(model: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (content: string) => {
    setIsLoading(true);
    // Implementation
    setIsLoading(false);
  };

  return { messages, isLoading, sendMessage };
}

// ❌ Bad: Untyped props
export function ChatMessage({ role, content, timestamp }) {
  return <div>{content}</div>;
}
```

### Async/Await

```typescript
// ✅ Good: async/await
async function fetchModels(): Promise<Model[]> {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch models:", error);
    throw error;
  }
}

// ❌ Bad: Promise chains
function fetchModels() {
  return fetch("/api/models")
    .then((response) => response.json())
    .then((data) => data)
    .catch((error) => console.error(error));
}
```

### Optional Chaining & Nullish Coalescing

```typescript
// ✅ Good: Optional chaining
const userName = user?.profile?.name ?? "Anonymous";
const firstMessage = chat?.messages?.[0]?.content;

// ✅ Good: Nullish coalescing
const temperature = settings.temperature ?? 0.7;
const maxTokens = settings.maxTokens ?? 4096;

// ❌ Bad: Nested if checks
let userName = "Anonymous";
if (user && user.profile && user.profile.name) {
  userName = user.profile.name;
}
```

## 파일 구조

### Python 모듈 구조

```
src/beanllm/domain/retrieval/
├── __init__.py          # Public API exports
├── base.py              # Base classes & interfaces
├── vector_search.py     # Vector search implementation
├── hybrid_search.py     # Hybrid search implementation
└── rerankers.py         # Reranking implementations
```

```python
# __init__.py - Public API만 export
from beanllm.domain.retrieval.vector_search import VectorSearch
from beanllm.domain.retrieval.hybrid_search import HybridSearch
from beanllm.domain.retrieval.rerankers import CrossEncoderReranker

__all__ = [
    "VectorSearch",
    "HybridSearch",
    "CrossEncoderReranker",
]
```

### TypeScript 모듈 구조

```
src/components/chat/
├── index.ts             # Public API exports
├── ChatMessage.tsx
├── ChatInput.tsx
└── ChatHistory.tsx
```

```typescript
// index.ts
export { ChatMessage } from "./ChatMessage";
export { ChatInput } from "./ChatInput";
export { ChatHistory } from "./ChatHistory";
```

## 코드 품질 도구

### Python

```bash
# 자동 포매팅
black src/beanllm/

# 린팅
ruff check src/beanllm/
ruff check --fix src/beanllm/  # 자동 수정

# 타입 체크
mypy src/beanllm/

# 전체 품질 체크
make quick-fix  # Black + Ruff
make type-check  # MyPy
make lint        # Ruff only
```

### TypeScript

```bash
# 타입 체크
pnpm tsc --noEmit

# 린팅
pnpm eslint src/

# 포매팅
pnpm prettier --write src/
```

## 주석 규칙

### 코드 주석

```python
# ✅ Good: 복잡한 로직 설명
# Calculate cosine similarity using NumPy for performance
# similarity = (A · B) / (||A|| × ||B||)
similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ✅ Good: 비즈니스 로직 설명
# HyDE: Generate hypothetical answer to improve retrieval
# Instead of embedding the raw query, we generate a hypothetical
# document that would answer the query, then embed that instead.
hypothetical = await self._generate_hypothetical_answer(query)

# ❌ Bad: 자명한 코드 주석 (불필요)
# Increment counter
counter += 1

# ❌ Bad: 주석으로 코드 설명 (Docstring 사용)
# This function calculates similarity between two vectors
def calculate_similarity(a, b):
    pass
```

### TODO 주석

```python
# ✅ Good: TODO with context
# TODO(leebeanbin): Implement caching for embeddings (issue #123)
# TODO: Add support for async batch processing (performance)

# ❌ Bad: Vague TODO
# TODO: fix this
# TODO: improve
```

## 상수 관리

```python
# ✅ Good: 중앙 집중식 상수
# beanllm/utils/constants.py
MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
SUPPORTED_MODELS = ["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"]

# ❌ Bad: Magic numbers
if len(text) > 4096:  # 4096이 뭐지?
    truncate(text)

# ✅ Good: 상수 사용
from beanllm.utils.constants import MAX_TOKENS

if len(text) > MAX_TOKENS:
    truncate(text)
```

## 참고 문서

- **PEP 8**: https://pep8.org/
- **Google Python Style Guide**: https://google.github.io/styleguide/pyguide.html
- **Airbnb JavaScript Style Guide**: https://github.com/airbnb/javascript
- `pyproject.toml` - Black, Ruff, MyPy 설정
- `CLAUDE.md` - 프로젝트 컨텍스트
