# 🚀 beanllm

**Production-ready LLM toolkit with Clean Architecture and unified interface for multiple providers**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/stars/leebeanbin/beanllm?style=social)](https://github.com/leebeanbin/beanllm)

**beanllm** is a comprehensive, production-ready toolkit for building LLM applications with a unified interface across OpenAI, Anthropic, Google, and Ollama. Built with **Clean Architecture** and **SOLID principles** for maintainability and scalability.

---

## ✨ Key Features

### 🎯 **Core Features**
- 🔄 **Unified Interface** - Single API for OpenAI, Anthropic, Google, Ollama
- 🎛️ **Intelligent Adaptation** - Automatic parameter conversion between providers
- 📊 **Model Registry** - Auto-detect available models from API keys
- 🔍 **CLI Tools** - Inspect models and capabilities from command line
- 💰 **Cost Tracking** - Accurate token counting and cost estimation
- 🏗️ **Clean Architecture** - Layered architecture with clear separation of concerns

### 🏗️ **RAG & Document Processing**
- 📄 **Document Loaders** - PDF, CSV, TXT with automatic format detection
- ✂️ **Smart Text Splitters** - Semantic chunking with tiktoken
- 🔍 **Vector Search** - Chroma, FAISS, Pinecone, Qdrant, Weaviate
- 🎯 **RAG Pipeline** - Complete question-answering system in one line
- 🐛 **RAG Debugging** - Comprehensive debugging toolkit

### 🤖 **Advanced LLM Features**
- 🛠️ **Tools & Agents** - Function calling with ReAct pattern
- 🧠 **Memory Systems** - Buffer, window, token-based, summary memory
- ⛓️ **Chains** - Sequential, parallel, and custom chain composition
- 📊 **Output Parsers** - Pydantic, JSON, datetime, enum parsing
- 🔁 **Streaming** - Real-time response streaming with stats

### 📈 **Graph & Multi-Agent**
- 🕸️ **Graph Workflows** - LangGraph-style DAG execution
- 🤝 **Multi-Agent** - Sequential, parallel, hierarchical, debate patterns
- 🔄 **State Management** - Automatic state threading and checkpoints
- 📞 **Communication** - Inter-agent message passing

### 🎨 **Multimodal AI**
- 🖼️ **Vision RAG** - Image-based question answering with CLIP
- 🎙️ **Audio Processing** - Whisper STT, multi-provider TTS
- 🔊 **Audio RAG** - Search and QA across audio files
- 🌐 **Web Search** - Google, Bing, DuckDuckGo integration
- 🧮 **ML Integration** - TensorFlow, PyTorch, Scikit-learn

### 🏭 **Production Features**
- 💵 **Token & Cost** - tiktoken-based accurate counting, cost optimization
- 📝 **Prompt Templates** - Few-shot, chat, chain-of-thought templates
- 📊 **Evaluation** - BLEU, ROUGE, LLM-as-Judge, RAG metrics, Context Recall
- 👤 **Human-in-the-Loop** - 피드백 수집 및 하이브리드 평가
- 🔄 **Continuous Evaluation** - 정기 평가 및 추적
- 📉 **Drift Detection** - 모델 드리프트 감지
- 📈 **Evaluation Dashboard** - 평가 결과 시각화
- 📋 **Rubric-Driven Grading** - 구조화된 루브릭 기반 평가
- ✅ **CheckEval** - 체크리스트 기반 Boolean 평가
- 📊 **Evaluation Analytics** - 트렌드 및 상관관계 분석
- 🎯 **Fine-tuning** - OpenAI fine-tuning API integration
- 🛡️ **Error Handling** - Retry, circuit breaker, rate limiting
- 📈 **Tracing** - Distributed tracing with OpenTelemetry export

---

## 🏗️ Architecture

beanllm은 **Clean Architecture**와 **SOLID 원칙**을 따르는 계층형 아키텍처를 사용합니다.

### 레이어 구조

```
┌─────────────────────────────────────────────────────────┐
│                    Facade Layer                          │
│  (사용자 친화적 API) - Client, RAGChain, Agent 등       │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                    Handler Layer                          │
│  (Controller 역할) - 입력 검증, 에러 처리                  │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                    Service Layer                          │
│  (비즈니스 로직) - 인터페이스 + 구현체                     │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                    Domain Layer                           │
│  (핵심 비즈니스) - 엔티티, 인터페이스, 규칙              │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                Infrastructure Layer                       │
│  (외부 시스템) - Provider, Vector Store 구현              │
└───────────────────────────────────────────────────────────┘
```

### 디렉토리 구조

```
src/beanllm/
├── facade/          # 외부 인터페이스 (Facade 패턴)
├── handler/         # 요청 처리 (Controller 역할)
├── service/         # 비즈니스 로직 (Service 인터페이스 + 구현체)
├── domain/          # 도메인 모델 및 비즈니스 규칙
├── infrastructure/ # 외부 시스템 인터페이스
├── dto/             # 데이터 전송 객체
├── decorators/      # 공통 데코레이터
└── utils/           # 유틸리티 함수
```

### SOLID 원칙 적용

- **SRP**: 각 레이어가 단일 책임만 담당
- **OCP**: 인터페이스 기반 확장 가능
- **LSP**: 인터페이스 구현체는 언제든 교체 가능
- **ISP**: 작은, 특화된 인터페이스
- **DIP**: 인터페이스에 의존, 구현체에 의존하지 않음

자세한 아키텍처 설명은 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

---

## 📦 Installation

### Poetry 사용 (권장)

```bash
# 프로젝트 클론
git clone https://github.com/yourusername/beanllm.git
cd beanllm

# 의존성 설치
poetry install --extras all  # 모든 Provider 포함
# 또는
poetry install --extras openai  # OpenAI만

# 가상 환경 활성화
poetry shell
```

### pip 사용

```bash
# 기본 설치 (의존성 없음)
pip install beanllm

# 특정 Provider 추가
pip install beanllm[openai]
pip install beanllm[anthropic]
pip install beanllm[gemini]
pip install beanllm[ollama]

# 모든 Provider
pip install beanllm[all]

# 개발 도구 포함
pip install beanllm[dev,all]
```

> **참고**: Provider는 선택적 의존성입니다. 필요한 Provider만 설치하면 됩니다.

---

## 🚀 Quick Start

### Environment Setup

`.env` 파일을 프로젝트 루트에 생성하세요:

```bash
# .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OLLAMA_HOST=http://localhost:11434
EOF
```

### Basic Usage

```python
import asyncio
from beanllm import Client

async def main():
    # Unified interface - works with any provider
    client = Client(model="gpt-4o")
    response = await client.chat(
        messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}]
    )
    print(response.content)
    
    # Switch providers seamlessly
    client = Client(model="claude-3-5-sonnet-20241022")
    response = await client.chat(
        messages=[{"role": "user", "content": "Same question, different provider"}]
    )
    
    # Streaming
    async for chunk in client.stream_chat(
        messages=[{"role": "user", "content": "Tell me a story"}]
    ):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### RAG in One Line

```python
import asyncio
from beanllm import RAGChain

async def main():
    # Create RAG system from documents
    rag = RAGChain.from_documents("docs/")
    
    # Ask questions
    answer = await rag.query("What is this document about?")
    print(answer)
    
    # With sources
    result = await rag.query("Explain the main concept", include_sources=True)
    print(result.answer)
    for source in result.sources:
        print(f"Source: {source.metadata.get('source', 'unknown')}")
    
    # Streaming query
    async for chunk in rag.stream_query("질문"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Tools & Agents

```python
import asyncio
from beanllm import Agent, Tool

async def main():
    # Define tools
    @Tool.from_function
    def calculator(expression: str) -> str:
        """Evaluate a math expression"""
        return str(eval(expression))

    # Create agent
    agent = Agent(
        model="gpt-4o-mini",
        tools=[calculator],
        max_iterations=10
    )
    
    # Run agent
    result = await agent.run("What is 25 * 17?")
    print(result.answer)
    print(f"Steps: {result.total_steps}")

asyncio.run(main())
```

### Graph Workflows

```python
import asyncio
from beanllm import StateGraph, Client

async def main():
    client = Client(model="gpt-4o-mini")
    
    # Create graph
    graph = StateGraph()
    
    async def analyze(state):
        response = await client.chat(
            messages=[{"role": "user", "content": f"Analyze: {state['input']}"}]
        )
        state["analysis"] = response.content
        return state
    
    def decide(state):
        score = float(state["analysis"].split("Score:")[1]) if "Score:" in state["analysis"] else 0.5
        return "good" if score > 0.8 else "bad"
    
    # Build graph
    graph.add_node("analyze", analyze)
    graph.add_conditional_edges("analyze", decide, {
        "good": "END",
        "bad": "improve"
    })
    
    # Run
    result = await graph.invoke({"input": "Draft text"})
    print(result)

asyncio.run(main())
```

---

## 📖 Examples

더 많은 사용 예제는 [examples/](examples/) 디렉토리를 참고하세요:

- `basic_usage.py` - 기본 사용법
- `rag_demo.py` - RAG 파이프라인 예제
- `rag_chain_demo.py` - RAG Chain 예제
- `state_graph_demo.py` - Graph Workflow 예제
- `embeddings_demo.py` - 임베딩 예제
- `vector_stores_demo.py` - Vector Store 예제

---

## 📚 Core Modules

### 1. Client & Adapters

Unified interface with automatic parameter adaptation:

```python
from beanllm import Client

# Works across all providers
client = Client(model="gpt-4o")

# Parameters automatically adapted
response = await client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1000,  # → max_completion_tokens for GPT-5
                       # → max_output_tokens for Gemini
                       # → num_predict for Ollama
)
```

### 2. Document Processing

```python
from beanllm import DocumentLoader, RecursiveCharacterTextSplitter

# Load documents
docs = DocumentLoader.load("docs/")  # PDF, CSV, TXT

# Smart splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs)
```

### 3. Embeddings & Vector Stores

```python
from beanllm import OpenAIEmbedding, ChromaVectorStore

# Create embeddings
embedding = OpenAIEmbedding(model="text-embedding-3-small")

# Vector store
store = ChromaVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

# Search
results = store.similarity_search("query", k=5)

# MMR search (diversity)
diverse_results = store.mmr_search("query", k=5, lambda_mult=0.5)
```

### 4. Multi-Agent Systems

```python
import asyncio
from beanllm import MultiAgentCoordinator, Agent

async def main():
    # Create agents
    researcher = Agent(model="gpt-4o-mini", tools=[], max_iterations=10)
    writer = Agent(model="gpt-4o-mini", tools=[], max_iterations=10)
    
    # Coordinate
    coordinator = MultiAgentCoordinator(
        agents={"researcher": researcher, "writer": writer}
    )
    
    result = await coordinator.execute_sequential(
        task="Write an article about quantum computing",
        agent_order=["researcher", "writer"]
    )
    print(result["final_result"])

asyncio.run(main())
```

---

## 🔧 CLI Usage

```bash
# List available models
beanllm list

# Show model details
beanllm show gpt-4o

# Check providers
beanllm providers

# Quick summary
beanllm summary

# Export model info
beanllm export > models.json
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/beanllm --cov-report=html

# Specific module
pytest tests/test_facade/ -v
```

**현재 테스트 커버리지**: 61% (624 tests, 593 passed)

---

## 🛠️ Development

### Makefile 사용 (권장)

```bash
# 개발 도구 설치
make install-dev

# 빠른 자동 수정
make quick-fix

# 타입 체크
make type-check

# 린트 체크
make lint

# 전체 검사 및 수정
make all
```

### 수동 실행

```bash
# Install in editable mode
pip install -e ".[dev,all]"

# Format code
ruff format src/beanllm

# Lint
ruff check src/beanllm

# Type check
mypy src/beanllm
```

---

## 🗺️ Roadmap

### ✅ 완료된 주요 기능
- ✅ Clean Architecture & SOLID principles
- ✅ Unified multi-provider interface (OpenAI, Anthropic, Google, Ollama)
- ✅ RAG pipeline & Document Processing
- ✅ Tools & Agents (ReAct pattern)
- ✅ Graph workflows (LangGraph-style)
- ✅ Multi-agent systems
- ✅ Vision & Audio processing
- ✅ Production features (evaluation, monitoring, cost tracking)
- ✅ 프롬프트 버전 관리 & A/B 테스트
- ✅ 스트리밍 응답 버퍼링
- ✅ 평가 시스템 확장 (Human-in-the-Loop, Continuous Evaluation, Drift Detection)
- ✅ 내부 성능 최적화 (병렬 처리, 배치 검색, 히스토리 압축)

### 📋 계획 중
- ⬜ 벤치마크 시스템

---

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - 빠른 시작 가이드
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 아키텍처 상세 설명
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - PyPI 배포 가이드
- **[docs/theory/](docs/theory/)** - 이론 문서 및 학습 자료
- **[docs/tutorials/](docs/tutorials/)** - 튜토리얼 코드
- **[examples/](examples/)** - 사용 예제 코드

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by:
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM application framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Graph workflow patterns
- **[Anthropic Claude](https://www.anthropic.com/)** - Clear code philosophy

Special thanks to:
- OpenAI for GPT models and APIs
- Anthropic for Claude API
- Google for Gemini API
- Ollama team for local LLM support

---

## 📧 Contact

- **GitHub**: https://github.com/leebeanbin/beanllm
- **Issues**: https://github.com/leebeanbin/beanllm/issues
- **Discussions**: https://github.com/leebeanbin/beanllm/discussions

---

**Built with ❤️ for the LLM community**

Transform your LLM applications from prototype to production with beanllm.
