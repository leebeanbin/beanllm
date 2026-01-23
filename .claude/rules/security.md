# 보안 규칙

**우선순위**: CRITICAL
**적용 범위**: 모든 코드 변경

## API 키 하드코딩 금지

### ❌ 절대 금지

```python
# ❌ Bad: API 키 하드코딩
OPENAI_API_KEY = "sk-1234567890abcdef"
client = OpenAI(api_key="sk-1234567890abcdef")

# ❌ Bad: .env 파일 커밋
# .env 파일은 .gitignore에 추가
```

### ✅ 환경변수 사용

```python
# ✅ Good: 환경변수에서 로드
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Good: 설정 클래스 사용
from beanllm.utils.config import Config

config = Config()
api_key = config.get("OPENAI_API_KEY")
```

## 민감한 정보 마스킹

### 로그에서 API 키 마스킹

```python
from beanllm.utils.integration.security import sanitize_error_message

# ✅ Good: API 키 마스킹
try:
    response = await provider.chat(messages)
except Exception as e:
    # "Error: sk-1234...abcdef" → "Error: sk-****...****"
    safe_message = sanitize_error_message(str(e))
    logger.error(f"Chat failed: {safe_message}")
```

### 사용자 입력 로깅 시 민감 정보 제거

```python
# ✅ Good: 민감 정보 제거
def log_request(request: ChatRequest):
    safe_request = {
        "model": request.model,
        "message_count": len(request.messages),
        # messages 내용은 로깅하지 않음
    }
    logger.info(f"Chat request: {safe_request}")
```

## 입력 검증

### LLM 입력 검증

```python
# ✅ Good: 길이 제한
MAX_MESSAGE_LENGTH = 100_000  # 100K chars

def validate_messages(messages: List[Dict[str, str]]):
    if not messages:
        raise ValueError("messages는 비어있을 수 없습니다")

    total_length = sum(len(msg.get("content", "")) for msg in messages)
    if total_length > MAX_MESSAGE_LENGTH:
        raise ValueError(f"메시지 길이가 최대 {MAX_MESSAGE_LENGTH}자를 초과했습니다")

    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError("각 메시지는 'role'과 'content'를 포함해야 합니다")

        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(f"유효하지 않은 role: {msg['role']}")
```

### 파일 업로드 검증

```python
# ✅ Good: 파일 크기 제한
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_file(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"파일 크기가 최대 {MAX_FILE_SIZE}바이트를 초과했습니다")

    # 허용된 확장자만
    allowed_extensions = [".pdf", ".txt", ".md", ".csv", ".json"]
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"허용되지 않은 파일 형식: {ext}")
```

## SQL Injection 방지

### ❌ 문자열 포매팅으로 쿼리 생성 금지

```python
# ❌ Bad: SQL Injection 취약
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)

# ✅ Good: Parameterized query
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
```

### Neo4j Cypher 쿼리도 동일

```python
# ❌ Bad: Cypher Injection 취약
def query_graph(entity_name):
    query = f"MATCH (n {{name: '{entity_name}'}}) RETURN n"
    session.run(query)

# ✅ Good: Parameterized query
def query_graph(entity_name):
    query = "MATCH (n {name: $name}) RETURN n"
    session.run(query, name=entity_name)
```

## XSS 방지 (Playground)

### HTML 이스케이프

```python
# ✅ Good: HTML 이스케이프
import html

def render_message(content: str) -> str:
    # 사용자 입력을 HTML로 렌더링 시 이스케이프
    return html.escape(content)
```

## CORS 설정 (Playground)

```python
# ✅ Good: 제한적인 CORS 설정
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # 개발 환경
        "https://your-domain.com"  # 프로덕션 환경
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ❌ Bad: 모든 origin 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 취약!
    allow_credentials=True,
)
```

## Rate Limiting

### Redis 기반 Rate Limiter

```python
# ✅ Good: Rate limiting 적용
from beanllm.infrastructure.distributed import with_distributed_features

@with_distributed_features(
    pipeline_type="chat",
    enable_rate_limiting=True,
    rate_limit_key="chat:api",
    rate_limit_max_requests=100,
    rate_limit_window_seconds=60,
)
async def chat(self, request: ChatRequest):
    # Rate limiting 자동 적용
    pass
```

## 의존성 보안

### 정기적인 보안 업데이트

```bash
# 보안 취약점 스캔
pip install safety
safety check

# 의존성 업데이트
pip install --upgrade <package>
```

### 버전 범위 제한

```toml
# pyproject.toml
[project]
dependencies = [
    "httpx>=0.24.0,<1.0.0",  # ✅ 버전 범위 제한
    "openai>=1.0.0,<3.0.0",
]
```

## 참고 문서

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- `beanllm/utils/integration/security.py` - 보안 유틸리티
- `CLAUDE.md` - 프로젝트 컨텍스트
