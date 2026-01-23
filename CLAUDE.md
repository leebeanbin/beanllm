# beanllm - LLM 통합 프레임워크

**프로젝트**: 프로덕션 레벨의 LLM 통합 툴킷
**아키텍처**: Clean Architecture
**언어**: Python 3.11+, TypeScript (Frontend)
**버전**: 0.2.2

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처](#아키텍처)
3. [기술 스택](#기술-스택)
4. [개발 워크플로우](#개발-워크플로우)
5. [Claude Code 활용](#claude-code-활용)
6. [핵심 패턴](#핵심-패턴)

---

## 프로젝트 개요

### 목표

**Production-Ready LLM Toolkit** with Clean Architecture

- ✅ 프로덕션 레벨 코드 품질 (테스트 커버리지 80%+)
- ✅ Clean Architecture로 유지보수성 극대화
- ✅ 10+ LLM 프로바이더 통합 (OpenAI, Anthropic, Google, Ollama 등)
- ✅ 고급 RAG, Multi-Agent, Knowledge Graph 지원
- ✅ Google Workspace 통합 (Docs, Drive, Gmail)
- ✅ MCP Server 제공 (Claude Desktop 연동)

### 핵심 기능

1. **Core Features**
   - 통합 Chat 인터페이스 (10+ 모델)
   - Chain 패턴 (Sequential, Parallel, Router)
   - Agent 패턴 (ReAct, Zero-shot, Tool-use)

2. **Advanced Features**
   - RAG (Retrieval-Augmented Generation)
   - Multi-Agent Collaboration (Debate, Hierarchical, Graph)
   - Knowledge Graph (Neo4j)
   - Vision RAG (이미지 + 텍스트)
   - Audio Transcription (Whisper, Google Speech)

3. **ML Features**
   - Model Evaluation & Benchmarking
   - Fine-tuning Support
   - OCR (Tesseract, Google Vision)

4. **Integrations**
   - Google Workspace (OAuth 2.0)
   - Web Search (Tavily, Google)
   - Vector Stores (Chroma, FAISS, Qdrant, Pinecone)

---

## 아키텍처

### Clean Architecture 레이어 구조

```
Facade → Handler → Service (인터페이스) → Domain (인터페이스) ← Infrastructure
```

#### 의존성 규칙 (CRITICAL)

- ✅ **Facade**: Handler 호출, DTO 사용
- ✅ **Handler**: Service **인터페이스**만 의존
- ✅ **Service 구현체**: Domain + Infrastructure **인터페이스** 의존
- ✅ **Domain**: 외부 의존성 없음 (표준 라이브러리만)
- ✅ **Infrastructure**: Domain 인터페이스 구현

❌ **금지**:
- Handler가 Service **구현체** 직접 사용
- Domain이 Service/Infrastructure 의존
- 순환 의존성
- 상대 경로 import

### 디렉토리 구조

```
src/beanllm/
├── facade/          # 사용자 인터페이스 (Client, RAGChain, AgentChain)
├── handler/         # 요청 처리 로직
├── service/         # 비즈니스 로직 인터페이스
│   └── impl/        # 비즈니스 로직 구현
├── domain/          # 핵심 도메인 로직 (Loaders, Embeddings, Tools)
├── infrastructure/  # 외부 시스템 연동 (Cache, Events, Queue)
├── providers/       # LLM 프로바이더 (OpenAI, Anthropic, Google, Ollama)
└── utils/           # 유틸리티 (Logger, Config, Exceptions)

playground/
├── backend/         # FastAPI 서버
│   └── main.py      # API endpoints
└── frontend/        # Next.js 15 + React 19
    └── src/
        ├── app/     # App Router pages
        ├── components/
        └── lib/

mcp_server/          # Model Context Protocol Server (Claude Desktop 연동)
├── tools/           # MCP tools (RAG, Multi-Agent, KG, ML, Google)
├── resources/       # MCP resources (세션, 상태)
└── prompts/         # MCP prompt templates
```

---

## 기술 스택

### Backend

| 카테고리 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| **언어** | Python | 3.11+ | Core framework |
| **웹 프레임워크** | FastAPI | latest | REST API |
| **LLM 프로바이더** | OpenAI | 1.x | GPT models |
| | Anthropic | latest | Claude models |
| | Google GenAI | latest | Gemini models |
| | Ollama | latest | Local models |
| **Vector DB** | Chroma | latest | Local vector store |
| | FAISS | latest | Facebook vector search |
| | Qdrant | latest | Cloud vector DB |
| | Pinecone | latest | Managed vector DB |
| **Graph DB** | Neo4j | latest | Knowledge graphs |
| **Embeddings** | Ollama | - | Local embeddings |
| | OpenAI | - | text-embedding-3 |
| **Speech** | Whisper | latest | Audio transcription |
| | Google Speech | latest | Cloud speech-to-text |
| **OCR** | Tesseract | latest | Local OCR |
| | Google Vision | latest | Cloud OCR |
| **Caching** | Redis | latest | Distributed cache |
| **Queue** | Kafka | latest | Message queue |
| **Testing** | pytest | latest | Unit/Integration tests |

### Frontend

| 카테고리 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| **프레임워크** | Next.js | 15.4.10 | React framework |
| **UI 라이브러리** | React | 19.0.0 | UI components |
| **스타일링** | Tailwind CSS | 4.0.13 | Utility-first CSS |
| **애니메이션** | Framer Motion | 12.4.9 | Animations |
| **컴포넌트** | shadcn/ui | latest | UI components |
| **언어** | TypeScript | 5.x | Type safety |
| **상태 관리** | React Hooks | - | Local state |

### Infrastructure

| 카테고리 | 기술 | 용도 |
|---------|------|------|
| **컨테이너** | Docker | Containerization |
| **오케스트레이션** | Docker Compose | Local development |
| **CI/CD** | GitHub Actions | Automated testing |
| **코드 품질** | Black, Ruff, MyPy | Python linting |
| **MCP** | FastMCP | Claude Desktop integration |

### 모델 선택 전략

**개발 (로컬)**:
- Chat: `qwen2.5:0.5b` (빠른 응답)
- Embedding: `mxbai-embed-large:335m` (높은 품질)

**프로덕션 (클라우드)**:
- Chat: `gpt-4o`, `claude-sonnet-4`, `gemini-2.5-pro`
- Embedding: `text-embedding-3-small` (OpenAI)

---

## 개발 워크플로우

### 1. 기능 추가 워크플로우

```bash
# 1️⃣ 계획 수립 (Sonnet)
/plan "HyDE query expansion 추가"

# 2️⃣ 테스트 작성 (TDD)
/test-gen --type unit --path src/beanllm/domain/retrieval/hyde.py

# 3️⃣ 구현 (Sonnet)
# (코드 작성)

# 4️⃣ 아키텍처 검증 (Sonnet)
/arch-check

# 5️⃣ 중복 코드 제거 (Sonnet)
/dedup

# 6️⃣ 테스트 실행
pytest --cov=src/beanllm --cov-report=html

# 7️⃣ 최종 리뷰 (Opus)
/code-review

# 8️⃣ 커밋 & PR
/commit
/pr
```

### 2. 버그 수정 워크플로우

```bash
# 1️⃣ 버그 분석 (Sonnet)
# (로그 확인, 재현)

# 2️⃣ 테스트 작성 (TDD)
/test-gen --type unit

# 3️⃣ 수정 (Sonnet)
# (에러 처리 추가)

# 4️⃣ 빌드 체크 (Haiku)
/build-fix

# 5️⃣ 커밋
/commit
```

### 3. 문서 업데이트 워크플로우

```bash
# 1️⃣ API 문서 업데이트 (Haiku)
/update-docs --path src/beanllm/facade/

# 2️⃣ README 업데이트 (Haiku)
# (새 기능 추가)

# 3️⃣ Changelog 생성 (Haiku)
# (git log 기반)
```

---

## Claude Code 활용

### 모델 선택 가이드

| 작업 | 모델 | 이유 | 예상 비용 |
|------|------|------|-----------|
| **코드 리뷰** | Opus | 보안/성능 심층 분석 | ~$0.42 |
| **리팩토링** | Sonnet | Clean Architecture 자동 수정 | ~$0.16 |
| **일반 코딩** | Sonnet | 복잡한 비즈니스 로직 | ~$0.08 |
| **테스트 작성** | Sonnet | 엣지 케이스 발견 | ~$0.12 |
| **문서 작성** | Haiku | 빠르고 저렴 | ~$0.013 |
| **간단한 수정** | Haiku | 오타, 포매팅 | ~$0.001 |

### 스킬 (Skills)

**사용 가능한 스킬:**

- `/arch-check` - Clean Architecture 검증
- `/plan` - 기능 계획 수립
- `/code-review` - 코드 리뷰 (Opus)
- `/dedup` - 중복 코드 제거
- `/test-gen` - 테스트 자동 생성
- `/build-fix` - 빌드 에러 수정
- `/update-docs` - 문서 자동 업데이트
- `/commit` - Intelligent Commit Splitter
- `/pr` - Smart Pull Request Creator
- `/tdd` - TDD 워크플로우

**스킬 사용 시 주의사항:**

1. **항상 스킬 먼저 사용**: 직접 코딩보다 스킬 활용
2. **모델 선택**: 스킬마다 적절한 모델 사용
3. **검증**: 각 단계마다 검증 (arch-check, test 등)

### Agents

**사용 가능한 에이전트:**

- `code-reviewer` - 코드 리뷰 전문 (Opus)
- `architecture-fixer` - 아키텍처 위반 자동 수정
- `performance-optimizer` - 성능 최적화

---

## 핵심 패턴

### 1. 데코레이터 패턴 (중복 코드 85% 감소)

```python
# ✅ Good: 데코레이터 패턴
@with_distributed_features(
    pipeline_type="rag",
    enable_cache=True,
    enable_rate_limiting=True,
    enable_event_streaming=True,
)
async def retrieve(self, request):
    # 실제 비즈니스 로직만
    results = self._vector_store.similarity_search(query, k=k)
    return results
```

### 2. Service 패턴 (Clean Architecture)

```python
# ✅ Handler → Service 인터페이스
class ChatHandler:
    def __init__(self, chat_service: IChatService):  # 인터페이스
        self._service = chat_service

# ✅ Service 구현체
class ChatServiceImpl(IChatService):
    async def chat(self, request: ChatRequest):
        return await self._provider.chat(request.messages)
```

### 3. Factory 패턴 (Provider 생성)

```python
# ✅ Factory로 Provider 생성
class ProviderFactory:
    @staticmethod
    def create(model: str) -> BaseProvider:
        if model.startswith("gpt"):
            return OpenAIProvider()
        elif model.startswith("claude"):
            return AnthropicProvider()
        # ...
```

### 4. 알고리즘 최적화

```python
# ✅ O(n) → O(1): 딕셔너리 캐싱
MODEL_REGISTRY = {model["name"]: model for model in all_models}

def get_model_info(model_name):
    return MODEL_REGISTRY.get(model_name)  # O(1)

# ✅ O(n log n) → O(n log k): heapq 활용
import heapq

def get_top_k(scores, k):
    return heapq.nlargest(k, scores)  # O(n log k)
```

---

## 프로젝트 구조 상세

### MCP Server (Claude Desktop 연동)

**목적**: Claude Desktop, Cursor, ChatGPT에서 beanllm 기능 사용

**구조**:
```
mcp_server/
├── run.py              # FastMCP 서버 엔트리포인트
├── config.py           # 설정 (모델, 임베딩, 청크 크기)
├── tools/              # 33개 MCP tools
│   ├── rag_tools.py         # RAG 구축/질의 (5 tools)
│   ├── agent_tools.py       # Multi-Agent (6 tools)
│   ├── kg_tools.py          # Knowledge Graph (7 tools)
│   ├── ml_tools.py          # Audio, OCR, Eval (9 tools)
│   └── google_tools.py      # Google Workspace (6 tools)
├── resources/          # 7개 리소스
│   └── session_resources.py # 세션 상태 관리
└── prompts/            # 8개 프롬프트 템플릿
    └── templates.py         # RAG, Agent, KG 프롬프트
```

**핵심 원칙**:
- ✅ **기존 beanllm 코드 재사용** (새로운 코드 작성 X)
- ✅ `from beanllm.facade.core import RAGChain` → 직접 사용

### Playground (통합 UI)

**Backend** (`playground/backend/main.py`):
- FastAPI REST API
- SSE Streaming for tool call progress
- **✅ beanllm Facade/Handler 직접 호출** (MCP Server 통신 X)

**Frontend** (`playground/frontend/`):
- Next.js 15 + React 19
- Unified Chat UI (모든 기능 통합)
- Tool Call Progress 실시간 표시
- SSE Client for streaming

**핵심 원칙**:
- ✅ **Playground backend는 beanllm 직접 사용**
- ✅ MCP Server는 별도 프로세스 (Claude Desktop 전용)
- ✅ Streaming: beanllm 실행 → 진행 상황 SSE → Frontend 표시

---

## 참고 문서

### 필수 문서

- `ARCHITECTURE.md` - 아키텍처 상세 설명
- `DEPENDENCY_RULES.md` - 의존성 규칙
- `.claude/rules/clean-architecture.md` - Clean Architecture 규칙
- `.claude/rules/code-quality.md` - 코드 품질 규칙
- `.claude/MODEL_SELECTION_GUIDE.md` - Claude 모델 선택 가이드

### 개발 가이드

- `.claude/skills/` - 재사용 가능한 개발 패턴
- `.claude/commands/` - 스킬 상세 설명
- `.claude/agents/` - 전문 에이전트 설명

### API 문서

- `docs/api/` - API 레퍼런스
- `playground/README.md` - Playground 사용법

---

## 빠른 시작

### 설치

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -e ".[all,dev]"

# 3. Ollama 설치 (로컬 모델)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b
ollama pull mxbai-embed-large:335m
```

### 사용

```python
from beanllm import Client

# 1. Chat
client = Client(model="qwen2.5:0.5b")
response = await client.chat([
    {"role": "user", "content": "Hello!"}
])
print(response.content)

# 2. RAG
from beanllm import RAGChain

rag = RAGChain.from_documents("./docs")
result = await rag.query("What is beanllm?")
print(result.answer)

# 3. Multi-Agent
from beanllm import MultiAgentChain

agents = MultiAgentChain.create_debate_team()
result = await agents.run("AI의 미래는?")
print(result.final_answer)
```

### Playground 실행

```bash
# Backend
cd playground/backend
uvicorn main:app --reload

# Frontend
cd playground/frontend
npm install
npm run dev
```

### MCP Server 실행 (Claude Desktop)

```bash
# 1. MCP Server 실행
python mcp_server/run.py

# 2. Claude Desktop 설정 (~/.config/claude/claude_desktop_config.json)
{
  "mcpServers": {
    "beanllm": {
      "command": "python",
      "args": ["/path/to/llmkit/mcp_server/run.py"]
    }
  }
}

# 3. Claude Desktop 재시작 후 사용
```

---

## 라이선스

MIT License - 상업적 사용 가능

---

**마지막 업데이트**: 2026-01-21
**버전**: 0.2.2
**문서 버전**: 1.0.0
