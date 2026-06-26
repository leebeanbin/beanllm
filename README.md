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

careerOS routes all AI calls through beanllm's unified interface. Setting `app.ai.provider: mock` swaps in stub responses — enabling all 415 tests to run without any live API calls.

---

## Quick Start

```bash
pip install beanllm[openai]        # or [anthropic], [gemini], [all]
```

```python
import asyncio
from beanllm import Client

async def main():
    client = Client(model="gpt-4o")
    response = await client.chat(
        messages=[{"role": "user", "content": "Explain quantum computing"}]
    )
    print(response.content)

asyncio.run(main())
```

```python
from beanllm import RAGChain

rag = RAGChain.from_documents("docs/")
result = await rag.query("What is this about?", include_sources=True)
```

### Providers

| Provider | Install | Env var |
|----------|---------|---------|
| OpenAI | `beanllm[openai]` | `OPENAI_API_KEY` |
| Claude (Anthropic) | `beanllm[anthropic]` | `ANTHROPIC_API_KEY` |
| Gemini (Google) | `beanllm[gemini]` | `GEMINI_API_KEY` |
| Grok / xAI | `beanllm[all]` | `XAI_API_KEY` |
| DeepSeek | `beanllm[all]` | `DEEPSEEK_API_KEY` |
| Perplexity | `beanllm[all]` | `PERPLEXITY_API_KEY` |
| Ollama (local) | `beanllm` (built-in) | `OLLAMA_HOST` (optional) |
| HuggingFace | `beanllm[all]` | `HUGGINGFACE_API_KEY` |

Provider auto-selection, CircuitBreaker, and rate limiting: [wiki/providers.md](wiki/providers.md)

---

## Capabilities

| Module | Highlights |
|--------|------------|
| **LLM Providers** | 8 providers, unified interface, `ModelParameterStrategy` auto-adapts per-model params |
| **Reasoning Models** | Native `thinking_budget` for Claude/OpenAI o-series; `<thinking>` token filtering |
| **RAG Pipeline** | Document loaders, 8 vector stores, hybrid search, HyDE, MultiQuery, rerankers |
| **GraphRAG** | KnowledgeGraph Facade — NER, relation extraction, Neo4j, relationship-aware retrieval |
| **VLM-OCR** | 11 engines (PaddleOCR, Qwen3-VL, GLM-OCR, DeepSeek-VL2); 3-layer PDF processing |
| **Fine-tuning** | LoRA/QLoRA, DPO, ORPO (50% less memory vs DPO), KTO |
| **Multi-Agent** | Sequential, parallel, hierarchical, debate patterns; DAG graph workflows |
| **MCP Server** | Model Context Protocol server for tool integration |

---

## Documentation

| | |
|--|--|
| [Architecture](wiki/architecture.md) | Clean Architecture layers, request flow diagram |
| [Providers](wiki/providers.md) | Setup, CircuitBreaker state machine, rate limiting |
| [Facade API](wiki/facade.md) | Client, RAGChain, Agent usage guide |
| [API Reference](docs/api/) | Full endpoint specs (client, agent, models, RAG) |
| [Decision Log](docs/decision-log/) | 3 architectural decisions |
| [Playbooks](docs/playbooks/) | CircuitBreaker / rate limit incident runbooks |
