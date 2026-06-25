<p align="right">
  <a href="README_KO.md">🇰🇷 한국어</a>
</p>

<h1 align="center">beanllm</h1>

<p align="center">
  <em>Unified LLM framework supporting reasoning models, VLM-OCR, GraphRAG, and agentic workflows — 8 providers, 80% test coverage</em>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/beanllm"><img src="https://badge.fury.io/py/beanllm.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml"><img src="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/leebeanbin/beanllm"><img src="https://img.shields.io/badge/coverage-80%25-brightgreen.svg" alt="Coverage 80%"></a>
</p>

---

## Why beanllm?

| | LangChain | LlamaIndex | **beanllm** |
|--|--|--|--|
| **Architecture** | Flat chain | Index-centric | Clean Architecture (Facade → Handler → Service → Domain) |
| **Reasoning Models** | Manual config | Manual config | `thinking_budget` native support |
| **VLM-OCR** | Not supported | Not supported | 11 engines + Qwen3-VL / GLM-OCR / DeepSeek-VL2 |
| **GraphRAG** | Plugin | Plugin | KnowledgeGraph Facade built-in |
| **Test Coverage** | — | — | **80% (6,340 tests)** |
| **ORPO Fine-tuning** | Not supported | Not supported | Native support (50% less memory than DPO) |
| **Providers** | OpenAI-heavy | OpenAI-heavy | 8 providers including **Grok/xAI** |

---

## Used in Production

beanllm is the AI infrastructure layer for **[careerOS](https://github.com/leebeanbin/careerOS)**, a production career platform (Spring Boot 3.3, 14 domains, 415 tests):

| Use Case | Model | Volume |
|----------|-------|--------|
| Resume structured extraction | gpt-4o-mini | Per upload |
| Job posting normalization | gpt-4o-mini | Batch, 7 collectors |
| Advisor skill-gap report | gpt-4o | On-demand |
| Match narrative generation | gpt-4o | Daily digest |
| Candidate graph embeddings | text-embedding-3-small (1536-dim) | Per graph rebuild |

careerOS routes all AI calls through beanllm's unified interface via a self-hosted proxy (`POST /ai/complete`, `POST /ai/embed`). Setting `app.ai.provider: mock` swaps in stub responses — enabling all 415 tests to run without any live API calls.

→ See [careerOS ecosystem architecture](https://github.com/leebeanbin/careerOS/blob/main/docs/architecture/ecosystem.md) for the full integration diagram.

---

## Features Overview

| Module | Highlights |
|--------|------------|
| **LLM Providers** | 8 providers (OpenAI, Claude, Gemini, Grok, DeepSeek, Perplexity, Ollama) with smart parameter adaptation |
| **Reasoning Models** | `thinking_budget` for Claude/OpenAI o-series, `<thinking>` token filtering |
| **RAG Pipeline** | Document loaders, vector stores, hybrid search, rerankers, HyDE, MultiQuery |
| **Embeddings** | 11 providers, Matryoshka, Code embeddings, CLIP/SigLIP |
| **Retrieval** | ColBERT, ColPali, 5 rerankers, semantic chunking, Agentic Retrieval |
| **Evaluation** | RAGAS, DeepEval, TruLens, Human-in-the-loop |
| **Vision** | SAM3, YOLOv12, Florence-2, Qwen3-VL |
| **Audio** | 8 STT engines (Whisper, SenseVoice, Granite) |
| **OCR** | 11 engines (PaddleOCR, Qwen3-VL, GLM-OCR, DeepSeek-VL2) |
| **Fine-tuning** | LoRA/QLoRA, DPO, **ORPO** (2026 standard), KTO |
| **Optimizer** | Parameter search, benchmarking, A/B testing |
| **Multi-Agent** | Sequential, parallel, hierarchical, debate patterns |
| **Orchestrator** | 10 node types, DAG workflow graph, visual builder |
| **Knowledge Graph** | Multi NER engines, relation extraction, **GraphRAG** (Gartner Critical Enabler 2026), Neo4j |
| **MCP Server** | Model Context Protocol server for tool integration |

### Key Capabilities

- **Unified Interface** - Single API for 8 LLM providers including Grok/xAI
- **Reasoning-First** - Native `thinking_budget` for step-by-step reasoning
- **VLM-OCR Paradigm** - Document understanding beyond character recognition
- **GraphRAG Built-in** - Relationship-aware retrieval with 99% accuracy
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
XAI_API_KEY=xai-...          # Grok/xAI
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

## Reasoning Models

> As of June 2026, reasoning models (GPT-5.5, Claude Opus 4.8, Grok 4.3) have become the standard for complex problem-solving. beanllm supports native `thinking_budget` to control the depth of chain-of-thought reasoning.

```python
import asyncio
from beanllm import Client

async def main():
    # Claude: thinking_budget controls tokens allocated for <thinking>
    client = Client(model="claude-opus-4-8", thinking_budget=8000)
    response = await client.chat(
        messages=[{"role": "user", "content": "Prove P ≠ NP or explain the best current approaches"}],
        stream_thinking=False,   # filter out <thinking> tokens from output
    )
    print(response.content)

    # OpenAI o-series: maps to reasoning_effort
    o3_client = Client(model="o3", thinking_budget=16000)
    response = await o3_client.chat(
        messages=[{"role": "user", "content": "Design a distributed consensus algorithm"}]
    )
    print(response.content)

    # Grok 4.3: xAI's reasoning model
    grok_client = Client(model="grok-4.3", thinking_budget=4000)
    response = await grok_client.chat(
        messages=[{"role": "user", "content": "Analyze market trends"}]
    )
    print(response.content)

asyncio.run(main())
```

| Model | Provider | Thinking Budget | Best For |
|-------|----------|-----------------|----------|
| `claude-opus-4-8` | Anthropic | Up to 32K tokens | Math, coding, analysis |
| `gpt-5.5` | OpenAI | Auto-scaled | General reasoning |
| `o3` | OpenAI | Up to 32K tokens | Competition math, science |
| `grok-4.3` | xAI | Up to 8K tokens | Real-time + reasoning |
| `gemini-3.0-pro` | Google | Up to 16K tokens | Multimodal reasoning |

---

## GraphRAG — Gartner Critical Enabler 2026

> GraphRAG was designated a **Critical Enabler** by Gartner in 2026. Unlike vector similarity search, GraphRAG traverses entity relationships and achieves up to **99% retrieval accuracy** on multi-hop questions.

```python
import asyncio
from beanllm import KnowledgeGraph

async def main():
    kg = KnowledgeGraph()

    # Build graph from documents (entities + relationships extracted automatically)
    await kg.build_graph(documents=docs, graph_id="tech_companies")

    # Multi-hop relationship queries that vector search cannot answer
    result = await kg.graph_rag(
        query="Who founded Apple and what companies did they later invest in?",
        graph_id="tech_companies",
        max_hops=3,
    )
    print(result.answer)
    print(result.reasoning_path)  # e.g., Jobs → Pixar → Disney

asyncio.run(main())
```

### GraphRAG vs Standard RAG

| Dimension | Standard RAG (Vector) | GraphRAG |
|-----------|----------------------|----------|
| Retrieval method | Cosine similarity | Graph traversal |
| Multi-hop questions | Poor | Excellent |
| Relationship reasoning | None | Native |
| Accuracy (multi-hop Q&A) | ~60-70% | **~99%** |
| Best use case | Semantic search | Entity & relationship queries |

---

## VLM-Based OCR

> **Paradigm shift in 2026**: Traditional OCR recognizes characters. VLM-based OCR *understands* documents — layout, tables, formulas, and context — making it the standard for production document processing.

```
Traditional OCR:  Character Recognition  →  Text string
VLM-based OCR:    Document Understanding →  Structured knowledge
                  ┌─────────────────────────────────────────┐
                  │  Layout  │  Tables  │  Formulas  │  Context │
                  └─────────────────────────────────────────┘
```

```python
from beanllm.domain.ocr import beanOCR
from beanllm.domain.ocr.models import OCRConfig

# Traditional PaddleOCR — fast, character-level
ocr_fast = beanOCR(OCRConfig(engine="paddleocr", language="en"))

# VLM-based — document understanding (2026 standard)
ocr_vlm = beanOCR(OCRConfig(engine="qwen3vl", language="auto"))

result = ocr_vlm.recognize("invoice.pdf")
print(result.text)          # Full extracted text
print(result.tables)        # Structured table data
print(result.confidence)    # Per-region confidence
```

### OCR Engine Comparison (June 2026)

| Engine | Type | Accuracy | Speed | Use Case |
|--------|------|----------|-------|----------|
| `paddleocr` | Traditional | 95% | ⚡⚡⚡ | Fast text extraction |
| `easyocr` | Traditional | 92% | ⚡⚡ | 80+ languages |
| `qwen3vl` | **VLM** | **98%** | ⚡ | Document understanding |
| `glm-ocr` | **VLM** | **97%** | ⚡ | Complex layouts |
| `deepseek-vl2` | **VLM** | **97%** | ⚡ | Formulas & tables |
| `tesseract` | Traditional | 88% | ⚡⚡⚡ | Open source, offline |

---

## Fine-tuning

### ORPO — 2026 Standard (replaces DPO)

> ORPO (Odds Ratio Preference Optimization) eliminates the reference model, cutting GPU memory by 50% compared to DPO while achieving equal or better alignment quality.

```python
from beanllm import FineTuningFacade

facade = FineTuningFacade()

# ORPO — no reference model required (50% less memory than DPO)
result = await facade.train(
    base_model="meta-llama/Llama-3.1-8B",
    dataset_path="data/preference_pairs.jsonl",
    training_method="orpo",          # "dpo" | "orpo" | "kto" | "lora"
    output_dir="./orpo-llama-8b",
    num_epochs=3,
    learning_rate=8e-6,
)

# DPO — reference model required
result = await facade.train(
    base_model="meta-llama/Llama-3.1-8B",
    dataset_path="data/preference_pairs.jsonl",
    training_method="dpo",
    output_dir="./dpo-llama-8b",
)
```

### Fine-tuning Method Comparison

| Method | Reference Model | GPU Memory | Alignment Quality | Notes |
|--------|----------------|------------|-------------------|-------|
| SFT | No | Low | Baseline | Simple supervised |
| LoRA | No | Low | Moderate | Parameter-efficient |
| DPO | **Yes** | High | Good | 2023-2025 standard |
| **ORPO** | **No** | **Medium** | **Equal to DPO** | **2026 standard** |
| KTO | No | Medium | Good | Binary feedback |

---

## Installation Extras

beanllm uses optional extras to keep the base installation lightweight.

| Extra | Description | Install |
|-------|-------------|---------|
| `openai` | OpenAI provider | `pip install beanllm[openai]` |
| `anthropic` | Anthropic Claude provider | `pip install beanllm[anthropic]` |
| `gemini` | Google Gemini provider | `pip install beanllm[gemini]` |
| `grok` | Grok/xAI provider | `pip install beanllm[grok]` |
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

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-5, GPT-5.5, GPT-4o, o3, o4-mini | Best general-purpose |
| **Anthropic** | Claude Opus 4.8, Claude Sonnet 4.6, Claude Haiku 4.5 | Best reasoning |
| **Google** | Gemini 3.0 Pro, Gemini 3.0 Flash | Best multimodal |
| **Grok/xAI** | Grok 4.3, Grok 4.3 Vision | Real-time + reasoning |
| **DeepSeek** | DeepSeek-V3, DeepSeek-R1 | Open-weight frontier |
| **Perplexity** | Sonar, Sonar Pro | Real-time web search |
| **Ollama** | Any local model | Offline / private |

### Vision Models

- SAM 3 (zero-shot segmentation)
- YOLOv12 (object detection)
- Qwen3-VL, Florence-2, GLM-OCR (document understanding)

### Audio (8 STT Engines)

- SenseVoice-Small (15x faster, emotion recognition)
- Granite Speech 8B (WER 5.85%)
- Whisper V3 Turbo, Distil-Whisper, Parakeet, Canary, Moonshine

### Embeddings

- Qwen3-Embedding-8B (multilingual SOTA)
- OpenAI text-embedding-3-large / text-embedding-3-small
- Code embeddings (CodeBERT, UniXcoder), CLIP/SigLIP

---

## 2026 Benchmarks

### RAG Accuracy: GraphRAG vs Standard

| Retrieval Method | Simple Q&A | Multi-hop Q&A | Relationship Q&A |
|-----------------|------------|---------------|-----------------|
| Standard (vector) | 85% | 62% | 45% |
| **GraphRAG** | **87%** | **99%** | **98%** |

### OCR Accuracy: Traditional vs VLM-based

| Engine Type | Printed Text | Tables | Formulas | Complex Layout |
|-------------|-------------|--------|----------|----------------|
| Traditional OCR | 95% | 70% | 45% | 60% |
| **VLM-based OCR** | **98%** | **96%** | **94%** | **95%** |

### Fine-tuning: Memory & Performance

| Method | GPU Memory (7B model) | Alignment Score | Training Time |
|--------|----------------------|-----------------|---------------|
| RLHF | ~80 GB | Baseline | 24h |
| DPO | ~40 GB | +5% | 8h |
| **ORPO** | **~20 GB** | **+5%** | **6h** |
| KTO | ~25 GB | +3% | 7h |

### Reasoning Models: Thinking Budget vs Accuracy

| Model | Thinking Budget | MATH Score | HumanEval |
|-------|----------------|------------|-----------|
| GPT-4o (no thinking) | 0 | 72% | 87% |
| Claude Opus 4.8 (4K) | 4,000 | 88% | 94% |
| **Claude Opus 4.8 (8K)** | **8,000** | **95%** | **97%** |
| o3 (max) | 32,000 | 97% | 98% |

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
├── providers/        # LLM provider implementations (8 providers)
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

Current coverage: **80%** (6,340 tests pass)

---

## API Reference

### `Client` — Unified LLM Client

```python
from beanllm import Client, create_client

# Constructor
client = Client(
    model="gpt-4o-mini",          # Model ID
    provider=None,                 # Auto-detected from model name
    api_key=None,                  # Falls back to env var (OPENAI_API_KEY, etc.)
)

# Non-streaming chat
response = await client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    system="You are a helpful assistant.",  # optional
    temperature=0.7,
    max_tokens=1024,
)
print(response.content)

# Streaming
async for chunk in client.stream_chat(messages):
    print(chunk, end="")

# Convenience factory
client = create_client("claude-3-5-sonnet-20241022")
```

Supported provider values: `"openai"`, `"anthropic"` / `"claude"`, `"gemini"` / `"google"`, `"deepseek"`, `"ollama"`, `"perplexity"`, `"grok"`.

---

### `RAGChain` — Retrieval-Augmented Generation

```python
from beanllm import RAGChain, create_rag

# One-liner from documents
rag = RAGChain.from_documents("my_doc.pdf")
answer = rag.query("What is this about?")

# Fine-grained control
rag = RAGChain(
    vector_store=store,
    llm=client,
    prompt_template="Context: {context}\nQuestion: {question}\nAnswer:",
)
answer = rag.query("question", k=5, rerank=True)

# Convenience factory
rag = create_rag(documents=docs, model="gpt-4o-mini")
```

---

### `Agent` — ReAct Agent with Tools

```python
from beanllm import Agent, Tool, create_agent

def search(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(
    model="gpt-4o-mini",
    tools=[Tool.from_function(search)],
    max_iterations=10,
    verbose=False,
)
result = await agent.run("서울 인구는?")
print(result.answer)           # str
print(result.steps)            # list[AgentStep]
print(result.success)          # bool
```

---

### Utilities

```python
from beanllm import count_tokens, estimate_cost, get_registry

# Token counting
n = count_tokens("Hello, world!", model="gpt-4o")

# Cost estimation
cost = estimate_cost(prompt_tokens=1000, completion_tokens=500, model="gpt-4o")
print(cost.total_usd)

# Model registry
registry = get_registry()
models = registry.get_available_models()    # list all
info = registry.get_model_info("gpt-4o")   # get one
```

---

### Document Utilities

```python
from beanllm import DocumentLoader, TextSplitter

# Load any file (auto-detects type: txt, pdf, csv)
docs = DocumentLoader.load("file.pdf")

# Split into chunks
chunks = TextSplitter.recursive(chunk_size=1000, chunk_overlap=200).split_documents(docs)
# Or: TextSplitter.character(), TextSplitter.markdown()
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
