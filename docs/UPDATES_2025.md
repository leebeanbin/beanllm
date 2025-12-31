# beanLLM Updates (2024-2025)

## Overview

This document summarizes the latest features and integrations added to beanLLM in 2024-2025.

---

## Vision AI

### Models Added
- **SAM 3** - Latest Segment Anything Model for zero-shot segmentation
- **YOLOv12** - State-of-the-art object detection and segmentation
- **Qwen3-VL** - Vision-language model with VQA, OCR, captioning capabilities
  - 128K context window
  - Multi-image chat support

### Usage
```python
from beanllm.domain.vision import create_vision_task_model

# SAM 3
sam = create_vision_task_model("sam2")
masks = sam.predict(image="photo.jpg", points=[[500, 375]], labels=[1])

# YOLOv12
yolo = create_vision_task_model("yolo", version="12")
detections = yolo.predict(image="photo.jpg", conf=0.5)

# Qwen3-VL
qwen = create_vision_task_model("qwen3vl", model_size="8B")
caption = qwen.caption(image="photo.jpg")
answer = qwen.vqa(image="photo.jpg", question="What is this?")
text = qwen.ocr(image="document.jpg")
```

---

## Embeddings

### Models Added
- **Qwen3-Embedding-8B** - Top multilingual embedding model
- **Code Embeddings** - Specialized embeddings for code search
- **Matryoshka Embeddings** - Dimension reduction support (83% storage savings)

### Usage
```python
from beanllm.domain.embeddings import Qwen3Embedding, CodeEmbedding
from beanllm.domain.embeddings import MatryoshkaEmbedding, truncate_embedding

# Qwen3-Embedding-8B
qwen3 = Qwen3Embedding(model_size="8B")
vectors = qwen3.embed_sync(["text1", "text2"])

# Code embeddings
code_emb = CodeEmbedding(model="jinaai/jina-embeddings-v3")
code_vectors = code_emb.embed_sync(["def foo():", "class Bar:"])

# Matryoshka (dimension reduction)
base_emb = OpenAIEmbedding(model="text-embedding-3-large")
mat_emb = MatryoshkaEmbedding(base_embedding=base_emb, output_dimension=512)
reduced_vectors = mat_emb.embed_sync(["text"])  # 512 dimensions instead of 1536
```

---

## RAG & Retrieval

### Features Added
- **HyDE** - Hypothetical Document Embeddings for query expansion
- **TruLens** - RAG performance evaluation and monitoring
- **Milvus** - High-performance vector database
- **LanceDB** - Modern vector database with SQL support
- **pgvector** - PostgreSQL extension for vector search

### Usage
```python
from beanllm.domain.retrieval import HyDE
from beanllm.domain.vector_stores import MilvusVectorStore, LanceDBVectorStore
from beanllm.domain.evaluation import TruLensEvaluator

# HyDE query expansion
hyde = HyDE(llm=client, embedding=embedding)
expanded_query = hyde.expand_query("What is quantum computing?")

# Milvus vector store
milvus = MilvusVectorStore(
    collection_name="docs",
    embedding=embedding,
    connection_args={"host": "localhost", "port": "19530"}
)

# TruLens evaluation
evaluator = TruLensEvaluator(app_name="my_rag")
results = evaluator.evaluate(query="question", response="answer", context="docs")
```

---

## Document Loaders

### Loaders Added
- **Docling** - Advanced Office file processing (PDF, DOCX, XLSX, PPTX, HTML)
  - 97.9% accuracy
  - Table and image extraction
  - OCR integration
- **JupyterLoader** - Jupyter Notebook (.ipynb) support
  - Code cell extraction
  - Markdown cell extraction
  - Output inclusion options
- **HTMLLoader** - Multi-tier fallback HTML parsing
  - Trafilatura (primary)
  - Readability (fallback 1)
  - BeautifulSoup (fallback 2)

### Usage
```python
from beanllm.domain.loaders import DoclingLoader, JupyterLoader, HTMLLoader

# Docling (Office files)
loader = DoclingLoader(
    "document.docx",
    extract_tables=True,
    extract_images=False,
    ocr_enabled=False
)
docs = loader.load()

# Jupyter Notebook
loader = JupyterLoader(
    "notebook.ipynb",
    include_outputs=True,
    filter_cell_types=["code"]
)
docs = loader.load()

# HTML
loader = HTMLLoader(
    "https://example.com",
    fallback_chain=["trafilatura", "readability", "beautifulsoup"]
)
docs = loader.load()
```

---

## Audio/STT

### Engines Added
- **SenseVoice-Small** - 15x faster than Whisper-Large
  - Multilingual (Chinese, Cantonese, English, Japanese, Korean)
  - Emotion recognition (SER)
  - Audio event detection (AED)
  - 70ms processing time for 10-second audio
- **Granite Speech 8B** - IBM enterprise-grade STT
  - Open ASR Leaderboard #2 (WER 5.85%)
  - 5 languages (English, French, German, Spanish, Portuguese)
  - Translation support
  - Apache 2.0 license

### Total: 8 STT Engines
1. SenseVoice-Small (Alibaba)
2. Granite Speech 8B (IBM)
3. Whisper V3 Turbo (OpenAI)
4. Distil-Whisper
5. Parakeet TDT (NVIDIA)
6. Canary (NVIDIA)
7. Moonshine (Useful Sensors)

### Usage
```python
from beanllm.domain.audio import beanSTT

# SenseVoice (fastest + emotion)
stt = beanSTT(engine="sensevoice", language="ko")
result = stt.transcribe("korean_audio.mp3")
print(result.text)
print(result.metadata["emotion"])  # Emotion recognition

# Granite Speech (enterprise-grade)
stt = beanSTT(engine="granite", language="en")
result = stt.transcribe("audio.mp3")
print(f"WER: {result.metadata['wer']}")  # 5.85%
```

---

## LLM Providers

### Providers Added
- **DeepSeek-V3** - Open-source 671B MoE model
  - 37B active parameters
  - OpenAI-compatible API
  - Cost-efficient
  - Models: deepseek-chat, deepseek-reasoner
- **Perplexity Sonar** - Real-time web search + LLM
  - Llama 3.3 70B based
  - 1200 tokens/second
  - Search Arena #1 (beats GPT-4o Search, Gemini 2.0 Flash)
  - Detailed citations
  - Models: sonar, sonar-pro, sonar-reasoning-pro

### Total: 7 LLM Providers
1. OpenAI (GPT-5, GPT-4o, GPT-4.1)
2. Anthropic (Claude Opus 4, Sonnet 4.5, Haiku 3.5)
3. Google (Gemini 2.5 Pro, Flash)
4. DeepSeek (DeepSeek-V3)
5. Perplexity (Sonar)
6. Ollama (Local LLMs)

### Usage
```python
from beanllm._source_providers import DeepSeekProvider, PerplexityProvider

# DeepSeek
provider = DeepSeekProvider()
response = await provider.chat(
    messages=[{"role": "user", "content": "Explain MoE"}],
    model="deepseek-chat"
)

# Perplexity (real-time search)
provider = PerplexityProvider()
response = await provider.chat(
    messages=[{"role": "user", "content": "What's happening today?"}],
    model="sonar"
)
print(response.usage["citations"])  # Web sources
```

### Environment Variables
```bash
DEEPSEEK_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...
```

---

## Advanced Features

### 1. Structured Outputs
100% schema accuracy with OpenAI strict mode.

**Supported Models:**
- OpenAI: gpt-4o-2024-08-06, gpt-4o-mini
- Anthropic: Claude Sonnet 4.5, Opus 4.1

**Benefits:**
- Zero JSON parsing failures (was 14-20%)
- Server-side schema validation
- Type safety

**Example:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

response = await client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Extract: John, 30, john@example.com"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"}
                },
                "required": ["name", "age", "email"]
            }
        }
    }
)
```

### 2. Prompt Caching
85% latency reduction, 10x cost savings (Anthropic).

**Supported Providers:**
- Anthropic: 200K tokens, 5-minute TTL (default)
- OpenAI: Auto-caching, 24-hour retention (GPT-5.1, GPT-4.1)

**Benefits:**
- Cached tokens cost 10% of regular input tokens
- Ideal for long system prompts and documents
- Automatic cache management

**Example:**
```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": "Long system prompt..." * 1000,
        "cache_control": {"type": "ephemeral"}  # Cache for 5 minutes
    }],
    messages=[{"role": "user", "content": "Question"}],
    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)

# Check cache usage
print(response.usage.cache_creation_input_tokens)  # First time
print(response.usage.cache_read_input_tokens)      # Subsequent calls
```

### 3. Parallel Tool Calling
Concurrent function execution for better performance.

**Supported Providers:**
- OpenAI: Default enabled
- Anthropic: Default disabled (safety-first)

**Benefits:**
- Faster execution for independent tools
- Configurable per-request

**Example:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "..."}},
    {"type": "function", "function": {"name": "get_time", "description": "..."}}
]

# Parallel execution (default)
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather in Seoul and time in Tokyo?"}],
    tools=tools,
    parallel_tool_calls=True  # Execute both simultaneously
)

# Sequential execution
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    parallel_tool_calls=False  # One at a time
)
```

---

## Summary

### New Capabilities
- **Vision**: 3 latest models (SAM 3, YOLOv12, Qwen3-VL)
- **Embeddings**: 3 advanced models (Qwen3, Code, Matryoshka)
- **RAG**: 5 new integrations (HyDE, TruLens, Milvus, LanceDB, pgvector)
- **Loaders**: 3 new loaders (Docling, Jupyter, HTML)
- **Audio**: 2 new STT engines (SenseVoice, Granite) - total 8 engines
- **Providers**: 2 new LLM providers (DeepSeek, Perplexity) - total 7 providers
- **Advanced**: 3 new features (Structured Outputs, Prompt Caching, Parallel Tool Calling)

### Performance Improvements
- **15x faster STT** (SenseVoice vs Whisper-Large)
- **85% latency reduction** (Prompt Caching)
- **83% storage savings** (Matryoshka Embeddings)
- **100% schema accuracy** (Structured Outputs)
- **10x cost reduction** (Prompt Caching)

### Documentation
- [README.md](../README.md) - Main documentation
- [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) - Detailed guide for advanced features
- [API Reference](API_REFERENCE.md) - Complete API documentation

---

**All features are production-ready and fully integrated into beanLLM.**
