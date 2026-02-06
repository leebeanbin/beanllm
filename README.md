<h1 align="center">🚀 beanllm</h1>

<p align="center">
  <em>Production-ready LLM toolkit with Clean Architecture and unified interface for multiple providers</em>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/beanllm"><img src="https://badge.fury.io/py/beanllm.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml"><img src="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
</p>

**beanllm** is a comprehensive, production-ready toolkit for building LLM applications with a unified interface across OpenAI, Anthropic, Google, DeepSeek, Perplexity, and Ollama. Built with **Clean Architecture** and **SOLID principles**.

---

## 📚 Documentation

- 💡 **[Examples](examples/)** - 20+ working examples
- 📦 **[PyPI Package](https://pypi.org/project/beanllm/)** - Installation and releases
- 🌐 **[Playground](playground/)** - Full-stack Chat UI
  - **Backend**: FastAPI + Agentic Chat (자동 Intent 분류), 세션별 RAG, Redis 캐싱
  - **Frontend**: Next.js 15 + React 19, Settings, Monitoring Dashboard

---

## ✨ Features Overview

### Core Modules (100% Complete)

| Module | Status | Highlights |
|--------|--------|------------|
| **LLM Providers** | ✅ 100% | 7 providers (OpenAI, Claude, Gemini, DeepSeek, Perplexity, Ollama) |
| **RAG Pipeline** | ✅ 100% | Document loaders, vector stores, hybrid search, rerankers |
| **Embeddings** | ✅ 100% | 11 providers, Matryoshka, Code embeddings |
| **Retrieval** | ✅ 100% | HyDE, MultiQuery, ColBERT, ColPali, 5 rerankers |
| **Evaluation** | ✅ 99% | RAGAS, DeepEval, TruLens, Human-in-the-loop |
| **Vision** | ✅ 100% | SAM3, YOLOv12, Florence-2, Qwen3-VL |
| **Audio** | ✅ 100% | 8 STT engines (Whisper, SenseVoice, Granite) |
| **OCR** | ✅ 100% | 11 engines (PaddleOCR, Qwen2-VL, DeepSeek) |
| **Optimizer** | ✅ 100% | Parameter search, benchmarking, A/B testing |
| **Multi-Agent** | ✅ 100% | Sequential, parallel, hierarchical, debate |
| **Orchestrator** | ✅ 100% | 10 node types, workflow graph, visual builder |
| **Knowledge Graph** | ✅ 100% | Multi NER engines, relation extraction, GraphRAG |

### Key Capabilities

- 🔄 **Unified Interface** - Single API for 7 LLM providers
- 🎛️ **Smart Parameter Adaptation** - Auto-convert between providers
- 📑 **Advanced PDF Processing** - 3-layer architecture (Fast/Accurate/ML)
- 🗄️ **8 Vector Stores** - Chroma, FAISS, Pinecone, Qdrant, Weaviate, Milvus, LanceDB, pgvector
- 🕸️ **Graph Workflows** - LangGraph-style DAG execution
- 🛡️ **Production Ready** - Retry, circuit breaker, rate limiting, tracing

---

## 📦 Installation

```bash
# Basic installation
pip install beanllm

# With specific providers
pip install beanllm[openai,anthropic,gemini]

# Full installation
pip install beanllm[all]

# Development
pip install beanllm[dev,all]
```

### Using Poetry

```bash
git clone https://github.com/leebeanbin/beanllm.git
cd beanllm
poetry install --extras all
```

---

## 🚀 Quick Start

### Environment Setup

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OLLAMA_HOST=http://localhost:11434
```

### 💬 Basic Chat

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

### 📚 RAG in One Line

```python
from beanllm import RAGChain

async def main():
    rag = RAGChain.from_documents("docs/")
    result = await rag.query("What is this about?", include_sources=True)
    print(result.answer)

asyncio.run(main())
```

### 🛠️ Tools & Agents

```python
from beanllm import Agent, Tool

@Tool.from_function
def calculator(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))

agent = Agent(model="gpt-4o-mini", tools=[calculator])
result = await agent.run("What is 25 * 17?")
```

### 🕸️ Graph Workflows

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

## 🎯 Model Support

### LLM Providers
- **OpenAI**: GPT-5, GPT-4o, GPT-4.1, GPT-4o-mini
- **Anthropic**: Claude Opus 4, Claude Sonnet 4.5
- **Google**: Gemini 2.5 Pro/Flash
- **DeepSeek**: DeepSeek-V3 (671B MoE)
- **Perplexity**: Sonar (real-time web search)
- **Ollama**: Local LLM support

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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Facade Layer                       │
│  User-friendly API (Client, RAGChain, Agent)        │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                 Handler Layer                       │
│  Input validation, error handling                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                 Service Layer                       │
│  Business logic (interfaces + implementations)      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                 Domain Layer                        │
│  Core business (entities, rules)                    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│            Infrastructure Layer                     │
│  External systems (providers, vector stores)        │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 CLI

```bash
beanllm list          # List available models
beanllm show gpt-4o   # Show model details
beanllm providers     # Check providers
beanllm summary       # Quick summary
```

---

## 🧪 Testing

```bash
pytest                                    # Run all tests
pytest --cov=src/beanllm --cov-report=html  # With coverage
make quality                              # Full quality check (Ruff, mypy, Bandit, pytest)
```

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev,all]"

# Setup pre-commit hooks
make pre-commit

# Code quality
make quick-fix     # Auto-fix with Black, Ruff
make type-check    # MyPy type checking
make lint          # Ruff linting
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework inspiration
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph workflow patterns
- OpenAI, Anthropic, Google, DeepSeek teams

---

## 📧 Contact

- **GitHub**: https://github.com/leebeanbin/beanllm
- **Issues**: https://github.com/leebeanbin/beanllm/issues

---

**Built with ❤️ for the LLM community**
