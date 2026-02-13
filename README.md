<h1 align="center">beanllm</h1>

<p align="center">
  <em>Production-ready LLM toolkit with Clean Architecture and unified interface for multiple providers</em>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/beanllm"><img src="https://badge.fury.io/py/beanllm.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml"><img src="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
</p>

---

**beanllm** is a comprehensive, production-ready toolkit for building LLM applications with a unified interface across OpenAI, Anthropic, Google, DeepSeek, Perplexity, and Ollama. Built with **Clean Architecture** and **SOLID principles**.

## Features Overview

| Module | Highlights |
|--------|------------|
| **LLM Providers** | 7 providers (OpenAI, Claude, Gemini, DeepSeek, Perplexity, Ollama) with smart parameter adaptation |
| **RAG Pipeline** | Document loaders, vector stores, hybrid search, rerankers, HyDE, MultiQuery |
| **Embeddings** | 11 providers, Matryoshka, Code embeddings, CLIP/SigLIP |
| **Retrieval** | ColBERT, ColPali, 5 rerankers, semantic chunking |
| **Evaluation** | RAGAS, DeepEval, TruLens, Human-in-the-loop |
| **Vision** | SAM3, YOLOv12, Florence-2, Qwen3-VL |
| **Audio** | 8 STT engines (Whisper, SenseVoice, Granite) |
| **OCR** | 11 engines (PaddleOCR, Qwen2-VL, DeepSeek) |
| **Optimizer** | Parameter search, benchmarking, A/B testing |
| **Multi-Agent** | Sequential, parallel, hierarchical, debate patterns |
| **Orchestrator** | 10 node types, DAG workflow graph, visual builder |
| **Knowledge Graph** | Multi NER engines, relation extraction, GraphRAG, Neo4j |
| **MCP Server** | Model Context Protocol server for tool integration |

### Key Capabilities

- **Unified Interface** - Single API for 7 LLM providers
- **Smart Parameter Adaptation** - Auto-convert between providers
- **Advanced PDF Processing** - 3-layer architecture (Fast/Accurate/ML)
- **8 Vector Stores** - Chroma, FAISS, Pinecone, Qdrant, Weaviate, Milvus, LanceDB, pgvector
- **Graph Workflows** - LangGraph-style DAG execution
- **Production Ready** - Retry, circuit breaker, rate limiting, tracing
- **Interactive TUI** - OpenCode-style terminal UI with autocomplete

---

## Quick Start

### Installation

```bash
# Basic
pip install beanllm

# With specific providers
pip install beanllm[openai,anthropic,gemini]

# Full installation (all providers + CLI + MCP)
pip install beanllm[all]

# Development
pip install -e ".[dev,all]"
```

### Environment Setup

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OLLAMA_HOST=http://localhost:11434
```

### Basic Chat

```python
import asyncio
from beanllm import Client

async def main():
    client = Client(model="gpt-4o")
    response = await client.chat(
        messages=[{"role": "user", "content": "Explain quantum computing"}]
    )
    print(response.content)

    # Streaming
    async for chunk in client.stream_chat(
        messages=[{"role": "user", "content": "Tell me a story"}]
    ):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### RAG in One Line

```python
from beanllm import RAGChain

async def main():
    rag = RAGChain.from_documents("docs/")
    result = await rag.query("What is this about?", include_sources=True)
    print(result.answer)

asyncio.run(main())
```

### Tools & Agents

```python
from beanllm import Agent, Tool

@Tool.from_function
def calculator(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))

agent = Agent(model="gpt-4o-mini", tools=[calculator])
result = await agent.run("What is 25 * 17?")
```

### Graph Workflows

```python
from beanllm import StateGraph

graph = StateGraph()
graph.add_node("analyze", analyze_fn)
graph.add_node("improve", improve_fn)
graph.add_conditional_edges("analyze", decide, {"good": "END", "bad": "improve"})
graph.set_entry_point("analyze")

result = await graph.invoke({"input": "Draft proposal"})
```

---

## Installation Extras

beanllm uses optional extras to keep the base installation lightweight.

| Extra | Description | Install |
|-------|-------------|---------|
| `openai` | OpenAI provider | `pip install beanllm[openai]` |
| `anthropic` | Anthropic Claude provider | `pip install beanllm[anthropic]` |
| `gemini` | Google Gemini provider | `pip install beanllm[gemini]` |
| `ollama` | Ollama local models | `pip install beanllm[ollama]` |
| `audio` | Whisper STT | `pip install beanllm[audio]` |
| `ml` | ML-based PDF (marker-pdf, torch) | `pip install beanllm[ml]` |
| `cli` | CLI (typer) | `pip install beanllm[cli]` |
| `mcp` | MCP server (fastmcp) | `pip install beanllm[mcp]` |
| `all` | All providers + CLI + MCP | `pip install beanllm[all]` |
| `vector` | ChromaDB vector store | `pip install beanllm[vector]` |
| `semantic` | Semantic chunking (sentence-transformers) | `pip install beanllm[semantic]` |
| `colbert` | ColBERT multi-vector search | `pip install beanllm[colbert]` |
| `colpali` | ColPali vision document search | `pip install beanllm[colpali]` |
| `ragpro` | Enterprise RAG (semantic + colbert + db) | `pip install beanllm[ragpro]` |
| `distributed` | Redis + Kafka | `pip install beanllm[distributed]` |
| `monitoring` | Streamlit dashboard + Plotly | `pip install beanllm[monitoring]` |
| `advanced` | UMAP, HDBSCAN, NetworkX, Bayesian opt | `pip install beanllm[advanced]` |
| `neo4j` | Neo4j graph database | `pip install beanllm[neo4j]` |
| `db` | PostgreSQL + MongoDB drivers | `pip install beanllm[db]` |
| `web` | FastAPI playground backend | `pip install beanllm[web]` |
| `dev` | Development tools (pytest, ruff, mypy, bandit) | `pip install beanllm[dev]` |

---

## Docker

The project includes Docker Compose with profile-based service management.

```bash
# Infrastructure only (MongoDB, Redis, Kafka, Ollama)
docker compose up -d

# Full stack (+ FastAPI backend + Next.js frontend)
docker compose --profile app up -d

# Full stack + admin UIs (Kafka UI, Mongo Express, Redis Commander)
docker compose --profile app --profile ui up -d

# With Neo4j knowledge graph
docker compose --profile neo4j up -d

# With monitoring dashboard
docker compose --profile monitoring up -d

# Stop and remove volumes
docker compose down -v
```

### Services

| Service | Port | Profile |
|---------|------|---------|
| MongoDB | 27017 | default |
| Redis | 6379 | default |
| Kafka | 9092 | default |
| Ollama | 11434 | default |
| Backend (FastAPI) | 8000 | `app` |
| Frontend (Next.js) | 3000 | `app` |
| Neo4j | 7474 / 7687 | `neo4j` |
| Kafka UI | 8080 | `ui` |
| Mongo Express | 8081 | `ui` |
| Redis Commander | 8082 | `ui` |

---

## CLI

```bash
# Interactive TUI (OpenCode-style, no arguments)
beanllm

# Model management
beanllm list              # List available models
beanllm show gpt-4o       # Show model details
beanllm providers          # Check provider status
beanllm summary            # Quick summary statistics
beanllm export             # Export models as JSON

# Advanced
beanllm scan               # Scan APIs for new models
beanllm analyze gpt-4o     # Model analysis with pattern inference

# Admin (Google Workspace)
beanllm admin stats        # Google service statistics
beanllm admin analyze      # Usage analysis with Gemini
beanllm admin optimize     # Cost optimization suggestions
beanllm admin security     # Security event audit
beanllm admin dashboard    # Launch Streamlit dashboard
```

---

## Playground

Full-stack Chat UI built with **FastAPI** (backend) and **Next.js 15 + React 19** (frontend).

### Backend (`playground/backend/`)

- **17 API routers**: chat, agent, multi-agent, RAG, chain, knowledge graph, vision, audio, evaluation, fine-tuning, optimizer, OCR, web search, monitoring, config, models, history
- **Agentic Chat**: automatic intent classification with tool routing
- **Session-based RAG**: per-session document upload and retrieval
- **Redis caching** and **MongoDB** persistence
- **WebSocket** real-time communication with heartbeat
- **SSE streaming** with proper `[DONE]` termination
- **Connection pooling**: httpx, MongoDB, Redis

### Frontend (`playground/frontend/`)

- **Next.js 15** with React 19 and Tailwind CSS
- **Pages**: Chat, Monitoring Dashboard, Settings
- **Features**: streaming responses, session management, API key modal, Google OAuth, model selector

### Setup

See the detailed guides in `playground/backend/`:

- `LOCAL_SETUP.md` - Local development setup
- `START_GUIDE.md` - Getting started guide
- `TROUBLESHOOTING.md` - Common issues and solutions

---

## Model Support

### LLM Providers

- **OpenAI**: GPT-5, GPT-4o, GPT-4.1, GPT-4o-mini
- **Anthropic**: Claude Opus 4, Claude Sonnet 4.5
- **Google**: Gemini 2.5 Pro/Flash
- **DeepSeek**: DeepSeek-V3 (671B MoE)
- **Perplexity**: Sonar (real-time web search)
- **Ollama**: Any local model

### Vision Models

- SAM 3 (zero-shot segmentation)
- YOLOv12 (object detection)
- Qwen3-VL, Florence-2

### Audio (8 STT Engines)

- SenseVoice-Small (15x faster, emotion recognition)
- Granite Speech 8B (WER 5.85%)
- Whisper V3 Turbo, Distil-Whisper, Parakeet, Canary, Moonshine

### Embeddings

- Qwen3-Embedding-8B (multilingual)
- OpenAI text-embedding-3
- Code embeddings, CLIP/SigLIP

---

## Architecture

Built with **Clean Architecture** - dependencies point inward, each layer only knows about the layer directly below it.

```
                       ┌──────────────────────────────┐
                       │        Facade Layer          │
                       │  Client, RAGChain, Agent     │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │       Handler Layer          │
                       │  Validation, decorators      │
                       └──────────────┬───────────────┘
                                      │ interfaces only
                       ┌──────────────▼───────────────┐
                       │       Service Layer          │
                       │  Business logic (I + Impl)   │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │       Domain Layer           │
                       │  Core entities and rules     │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │    Infrastructure Layer      │
                       │  Providers, vector stores    │
                       └──────────────────────────────┘
```

### Project Structure

```
src/beanllm/
├── facade/           # Public API (Client, RAG, Agent, Chain, etc.)
├── handler/          # Request handling (core, advanced, ml)
├── service/          # Business logic interfaces + impl/
├── domain/           # Core models (40+ modules)
├── dto/              # Data transfer objects
├── infrastructure/   # External integrations (60+ files)
├── providers/        # LLM provider implementations
├── decorators/       # Error handling, validation, logging
├── ui/               # Interactive TUI
└── utils/            # CLI, config, streaming, tracer

src/beantui/          # Standalone reusable TUI engine
mcp_server/           # Model Context Protocol server
playground/           # Full-stack Chat UI (FastAPI + Next.js)
```

---

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/leebeanbin/beanllm.git
cd beanllm
pip install -e ".[dev,all]"

# Setup pre-commit hooks (auto quality checks on commit)
make pre-commit
```

### Code Quality

```bash
make quality       # Full: ruff format + lint + mypy + bandit + pytest
make quick-fix     # Auto-fix: ruff lint + format + import sort
make type-check    # MyPy type checking
make lint          # Ruff linting only
make test          # Run pytest
make test-cov      # pytest with HTML coverage report
make clean         # Remove caches and build artifacts
```

### Branch Workflow

```bash
# 1. Create a branch from main
make new-feat NAME=rag-hyde         # feat/rag-hyde
make new-fix NAME=chat-rate-limit   # fix/chat-rate-limit
make new-refactor NAME=service      # refactor/service

# 2. Develop and commit (reference issue numbers)
git commit -m "feat(rag): Add HyDE query expansion

Closes #42"

# 3. Quality check + push + create PR
make pr

# 4. Keep in sync with main
make sync

# 5. After PR is merged, clean up
make done
```

### Pre-commit Hooks

Automatically run on every `git commit`:

| Tool | Purpose |
|------|---------|
| Ruff | Code formatting, linting, import sorting |
| Bandit | Security scanning |

---

## Contributing

1. **Create an Issue** using one of the templates (Feature, Bug, Refactor)
2. **Create a branch**: `make new-feat NAME=your-feature`
3. **Develop** with commits referencing the issue (`Closes #issue_number`)
4. **Run quality checks**: `make quality`
5. **Submit a PR**: `make pr` (auto-fills the PR template)
6. **After merge**: delete the branch on GitHub, then `make done` locally

### Templates

- **Issue templates**: Feature Request, Bug Report, Refactoring
- **PR template**: Summary, Related Issues (`Closes #N`), Changes, Test Plan

---

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src/beanllm --cov-report=html

# Full quality pipeline
make quality
```

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Links

- **GitHub**: https://github.com/leebeanbin/beanllm
- **PyPI**: https://pypi.org/project/beanllm/
- **Issues**: https://github.com/leebeanbin/beanllm/issues
- **Examples**: [examples/](examples/) (20+ working examples)

---

**Built with care for the LLM community**
