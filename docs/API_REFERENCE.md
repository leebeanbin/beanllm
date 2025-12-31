# 📚 beanllm API Reference

Complete API reference for all beanllm components.

## Table of Contents

### Core Components
- [Client](#client) - Basic LLM client
- [LLM Providers](#llm-providers) - 7 LLM providers (OpenAI, Anthropic, Google, DeepSeek, Perplexity, Ollama)
- [RAGChain](#ragchain) - RAG (Retrieval-Augmented Generation) system
- [Agent](#agent) - AI agent with tools
- [Chain](#chain) - Chain execution

### Document Processing
- [beanPDFLoader](#beanpdfloader) - Advanced PDF processing with 3-layer architecture
- [Document Loaders](#document-loaders) - Docling, Jupyter, HTML, Text, CSV loaders
- [Text Splitters](#text-splitters) - Semantic text chunking

### Embeddings & Retrieval
- [Embeddings](#embeddings) - Qwen3-Embedding-8B, Code, Matryoshka embeddings
- [Vector Stores](#vector-stores) - Milvus, LanceDB, pgvector, Chroma, FAISS
- [Retrieval](#retrieval) - HyDE, Hybrid Search, Reranking

### Advanced Features
- [MultiAgentCoordinator](#multiagentcoordinator) - Multi-agent collaboration
- [Graph](#graph) - Graph-based workflows
- [StateGraph](#stategraph) - State-based graph execution
- [Audio](#audio) - 8 STT engines (SenseVoice, Granite, Whisper, etc.)
- [Vision](#vision) - Qwen3-VL, YOLOv12, SAM 3, Florence-2

### Specialized Features
- [VisionRAG](#visionrag) - Vision + RAG with image understanding
- [WebSearch](#websearch) - Web search integration
- [Evaluator](#evaluator) - LLM evaluation (TruLens, RAGAS)
- [FineTuningManager](#finetuningmanager) - Model fine-tuning
- [Advanced LLM Features](#advanced-llm-features) - Structured Outputs, Prompt Caching, Parallel Tool Calling

---

## Installation

```bash
# Basic installation
pip install beanllm

# With all providers
pip install beanllm[all]

# Specific providers
pip install beanllm[openai,anthropic]
```

---

## Quick Start

```python
import asyncio
from beanllm import Client

async def main():
    # Initialize client
    client = Client(model="gpt-4")

    # Simple chat
    response = await client.chat(
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print(response.content)

asyncio.run(main())
```

---

# API Documentation

## Core Components

### Client

기본 LLM 클라이언트. 가장 간단한 채팅 인터페이스를 제공합니다.

#### `__init__(model, provider=None, api_key=None, **kwargs)`

**파라미터:**
- `model` (str): 모델 이름 (예: "gpt-4", "claude-3-opus", "gemini-pro")
- `provider` (str, optional): Provider 이름. 생략 시 모델명에서 자동 감지
- `api_key` (str, optional): API 키. 생략 시 환경변수에서 로드
- `**kwargs`: Provider별 추가 설정

**예제:**
```python
from beanllm import Client

# OpenAI
client = Client(model="gpt-4")

# Anthropic (provider 자동 감지)
client = Client(model="claude-3-opus-20240229")

# 명시적 provider 지정
client = Client(model="gpt-4", provider="openai")
```

#### `chat(messages, system=None, temperature=None, max_tokens=None, **kwargs)` (async)

채팅 완료를 수행합니다.

**파라미터:**
- `messages` (List[Dict[str, str]]): 메시지 리스트 `[{"role": "user", "content": "..."}]`
- `system` (str, optional): 시스템 프롬프트
- `temperature` (float, optional): 샘플링 온도 (0.0-2.0)
- `max_tokens` (int, optional): 최대 생성 토큰 수
- `**kwargs`: 추가 파라미터

**반환:** `ChatResponse`

**예제:**
```python
response = await client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1000
)
print(response.content)
```

#### `stream_chat(messages, **kwargs)` (async)

스트리밍 방식으로 채팅 완료를 수행합니다.

**파라미터:** `chat()`와 동일

**반환:** `AsyncIterator[str]`

**예제:**
```python
async for chunk in client.stream_chat(messages=[{"role": "user", "content": "Tell me a story"}]):
    print(chunk, end="", flush=True)
```

---

### LLM Providers

beanllm supports 7 LLM providers with automatic fallback and unified interface.

#### Supported Providers

| Provider | Models | Features |
|----------|---------|----------|
| **OpenAI** | GPT-4, GPT-4o, GPT-4o-mini | Structured Outputs, Vision, Tool Calling |
| **Anthropic** | Claude Opus 4, Sonnet 4.5, Haiku 3.5 | Prompt Caching, Vision, Tool Calling |
| **Google** | Gemini 2.5 Pro, Flash | Large context (2M tokens), Vision |
| **DeepSeek** | DeepSeek-V3 (671B MoE) | Cost-efficient, OpenAI-compatible |
| **Perplexity** | Sonar, Sonar-Pro | Real-time web search, Citations |
| **Ollama** | Llama 3.3, Qwen2.5, etc. | Local deployment, Privacy |
| **X.AI** | Grok 2 | Coming soon |

#### Usage Examples

```python
from beanllm import Client

# OpenAI
client = Client(model="gpt-4o")

# Anthropic
client = Client(model="claude-sonnet-4-20250514")

# Google Gemini
client = Client(model="gemini-2.5-pro")

# DeepSeek (cost-efficient)
client = Client(model="deepseek-chat")

# Perplexity (real-time web search)
client = Client(model="sonar-pro")

# Ollama (local)
client = Client(model="llama3.3:70b", provider="ollama")
```

#### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="sk-..."
export PERPLEXITY_API_KEY="pplx-..."
export OLLAMA_HOST="http://localhost:11434"  # Optional
```

---

### RAGChain

RAG (Retrieval-Augmented Generation) 시스템. 문서 기반 질의응답을 제공합니다.

#### `from_documents(source, chunk_size=500, chunk_overlap=50, embedding_model="text-embedding-3-small", llm_model="gpt-4o-mini", **kwargs)`

팩토리 메서드로 RAG 시스템을 생성합니다.

**파라미터:**
- `source` (str | List): 문서 경로 또는 문서 리스트
- `chunk_size` (int): 청크 크기
- `chunk_overlap` (int): 청크 겹침
- `embedding_model` (str): 임베딩 모델 이름
- `llm_model` (str): LLM 모델 이름
- `**kwargs`: 추가 설정

**예제:**
```python
from beanllm import RAGChain

rag = RAGChain.from_documents(
    source="documents.txt",
    chunk_size=500,
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4"
)
```

#### `add_documents(documents)` (async)

문서를 벡터 저장소에 추가합니다.

**파라미터:**
- `documents` (List[str] | List[Document]): 문서 리스트

**예제:**
```python
documents = [
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
]
await rag.add_documents(documents)
```

#### `query(question, top_k=3, **kwargs)` (async)

질문에 대한 답변을 생성합니다.

**파라미터:**
- `question` (str): 질문
- `top_k` (int): 검색할 문서 수
- `**kwargs`: 추가 파라미터

**반환:** `RAGResponse`

**예제:**
```python
response = await rag.query(
    question="What is Python?",
    top_k=3
)
print(response.answer)
print(response.sources)  # 사용된 문서들
```

---

### Agent

도구를 사용할 수 있는 AI 에이전트.

#### `__init__(model, tools=None, max_iterations=10, **kwargs)`

**파라미터:**
- `model` (str): LLM 모델 이름
- `tools` (List[Tool], optional): 사용할 도구 리스트
- `max_iterations` (int): 최대 반복 횟수
- `**kwargs`: 추가 설정

**예제:**
```python
from beanllm import Agent
from beanllm import search_web, calculator

agent = Agent(
    model="gpt-4",
    tools=[search_web, calculator]
)
```

#### `run(task, max_iterations=10, **kwargs)` (async)

에이전트를 실행하여 작업을 수행합니다.

**파라미터:**
- `task` (str): 수행할 작업
- `max_iterations` (int): 최대 반복 횟수
- `**kwargs`: 추가 파라미터

**반환:** `AgentResponse`

**예제:**
```python
response = await agent.run(
    task="Calculate 123 * 456 and search for the result online",
    max_iterations=5
)
print(response.final_answer)
print(response.steps)  # 실행 단계
```

---

### Chain

여러 단계를 순차적으로 실행하는 체인.

#### `__init__(client, memory=None, verbose=False)`

**파라미터:**
- `client` (Client): LLM 클라이언트
- `memory` (Memory, optional): 메모리 객체
- `verbose` (bool): 디버그 출력 여부

#### `run(user_input, **kwargs)` (async)

체인을 실행합니다.

**파라미터:**
- `user_input` (str): 사용자 입력
- `**kwargs`: 추가 파라미터

**반환:** `ChainResult`

**예제:**
```python
from beanllm import Chain, Client

client = Client(model="gpt-4")
chain = Chain(client=client)

response = await chain.run("Translate 'hello' to French")
print(response.output)
```

---

## Document Processing

### beanPDFLoader

고급 PDF 처리를 위한 3-Layer 아키텍처 로더.

**3-Layer 아키텍처:**
- **Fast Layer** (PyMuPDF): 빠른 처리 (~130 pages/sec), 이미지 추출
- **Accurate Layer** (pdfplumber): 정확한 테이블 추출 (~10 pages/sec)
- **ML Layer** (marker-pdf): 구조 보존 Markdown 변환 (98% 정확도)

#### `__init__(file_path, strategy="auto", extract_tables=True, extract_images=False, to_markdown=False, **kwargs)`

**파라미터:**
- `file_path` (str | Path): PDF 파일 경로
- `strategy` (str): 파싱 전략
  - `"auto"`: 자동 선택 (기본값)
  - `"fast"`: PyMuPDF (빠른 처리)
  - `"accurate"`: pdfplumber (정확한 테이블)
  - `"ml"`: marker-pdf (ML 기반, optional)
- `extract_tables` (bool): 테이블 추출 여부 (기본: True)
- `extract_images` (bool): 이미지 추출 여부 (기본: False)
- `to_markdown` (bool): Markdown 변환 여부 (기본: False)
- `enable_ocr` (bool): OCR 활성화 (향후 구현)
- `layout_analysis` (bool): 레이아웃 분석 (향후 구현)
- `max_pages` (int, optional): 최대 처리 페이지 수
- `page_range` (tuple[int, int], optional): 처리할 페이지 범위

**예제:**
```python
from beanllm.domain.loaders.pdf import beanPDFLoader

# 기본 사용 (자동 전략)
loader = beanPDFLoader("document.pdf")
docs = loader.load()

# 테이블 추출
loader = beanPDFLoader("report.pdf", extract_tables=True)
docs = loader.load()
tables = loader._result["tables"]

# Markdown 변환
loader = beanPDFLoader("article.pdf", to_markdown=True)
docs = loader.load()
markdown = loader._result["markdown"]

# ML Layer 사용 (marker-pdf 필요)
loader = beanPDFLoader("complex.pdf", strategy="ml", to_markdown=True)
docs = loader.load()
```

#### `load()` → `List[Document]`

PDF를 로딩하여 Document 리스트를 반환합니다.

**반환값:**
- `List[Document]`: 페이지별 Document 리스트

**예제:**
```python
loader = beanPDFLoader("document.pdf")
docs = loader.load()

for doc in docs:
    print(f"Page {doc.metadata['page']}: {doc.content[:100]}...")
```

#### 고급 기능

**1. 테이블 추출 및 변환**

```python
from beanllm.domain.loaders.pdf import beanPDFLoader
from beanllm.domain.loaders.pdf.extractors import TableExtractor

# 테이블 추출
loader = beanPDFLoader("report.pdf", extract_tables=True)
docs = loader.load()

# 테이블 조회
extractor = TableExtractor(docs)
all_tables = extractor.get_all_tables()
high_quality = extractor.get_high_quality_tables(min_confidence=0.8)

# Markdown 변환
markdown_tables = extractor.export_to_markdown()
```

**2. Markdown 변환 및 Layout Analysis**

```python
from beanllm.domain.loaders.pdf import beanPDFLoader
from beanllm.domain.loaders.pdf.utils import LayoutAnalyzer

# Markdown 변환
loader = beanPDFLoader("article.pdf", to_markdown=True)
docs = loader.load()
markdown = loader._result["markdown"]

# Layout 분석
analyzer = LayoutAnalyzer()
for doc in docs:
    page_data = {"text": doc.content, "width": doc.metadata["width"],
                 "height": doc.metadata["height"], "metadata": doc.metadata}
    layout = analyzer.analyze_layout(page_data)
    print(f"Columns: {layout['columns']}, Multi-column: {layout['is_multi_column']}")
```

**3. MarkerEngine (ML Layer)**

```python
# ML Layer 사용 (marker-pdf 설치 필요: pip install beanllm[ml])
from beanllm.domain.loaders.pdf.engines import MarkerEngine

engine = MarkerEngine(
    use_gpu=False,      # GPU 사용 여부
    enable_cache=True,  # 결과 캐싱
    cache_size=10,      # 캐시 크기
)

# 단일 PDF 처리
result = engine.extract("document.pdf", {
    "to_markdown": True,
    "extract_tables": True,
    "extract_images": True,
})

# Batch 처리
results = engine.extract_batch(
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    {"to_markdown": True}
)

# 캐시 통계
stats = engine.get_cache_stats()
print(f"Cache: {stats['cache_size']}/{stats['cache_limit']}")
```

**4. 성능 벤치마크**

```
Engine       Time(s)    Pages/s    Memory(MB)
------------------------------------------------
PyMuPDF      0.03       129.61     0.20
pdfplumber   0.42       9.59       41.41
marker-pdf   ~10s/100pg (GPU), 98% accuracy
```

---

### Document Loaders

다양한 문서 형식 지원 (Office, Jupyter, HTML, Text, CSV).

#### DoclingLoader - Office Files (97.9% accuracy)

```python
from beanllm.domain.loaders import DoclingLoader

# PDF, DOCX, XLSX, PPTX, HTML 지원
loader = DoclingLoader(
    "document.docx",
    extract_tables=True,
    extract_images=False,
    ocr_enabled=False
)
docs = loader.load()

# 테이블 데이터 접근
tables = loader.get_tables()
```

#### JupyterLoader - Jupyter Notebooks

```python
from beanllm.domain.loaders import JupyterLoader

loader = JupyterLoader(
    "notebook.ipynb",
    include_outputs=True,
    filter_cell_types=["code", "markdown"]  # Optional
)
docs = loader.load()
```

#### HTMLLoader - Multi-tier Fallback

```python
from beanllm.domain.loaders import HTMLLoader

# 3-tier fallback: Trafilatura → Readability → BeautifulSoup
loader = HTMLLoader(
    "https://example.com",
    fallback_chain=["trafilatura", "readability", "beautifulsoup"]
)
docs = loader.load()
```

#### Text & CSV Loaders

```python
from beanllm.domain.loaders import TextLoader, CSVLoader

# Text 파일
text_loader = TextLoader("document.txt")
docs = text_loader.load()

# CSV 파일
csv_loader = CSVLoader("data.csv")
docs = csv_loader.load()
```

---

### Text Splitters

의미 단위로 텍스트 분할.

```python
from beanllm.domain.splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "]
)

chunks = splitter.split_documents(docs)
```

---

## Embeddings & Retrieval

### Embeddings

최신 임베딩 모델 지원 (Qwen3-Embedding-8B, Code, Matryoshka).

#### Qwen3-Embedding-8B - Top Multilingual Model

```python
from beanllm.domain.embeddings import Qwen3Embedding

# Qwen3-Embedding (SOTA multilingual)
qwen3 = Qwen3Embedding(model_size="8B")  # or "4B", "2B"
vectors = qwen3.embed_sync(["한글 텍스트", "English text", "日本語"])
```

#### Code Embeddings - Specialized for Code Search

```python
from beanllm.domain.embeddings import CodeEmbedding

# Code-specialized embeddings
code_emb = CodeEmbedding(model="jinaai/jina-embeddings-v3")
code_vectors = code_emb.embed_sync([
    "def hello_world():",
    "class MyClass:",
    "import numpy as np"
])
```

#### Matryoshka Embeddings - 83% Storage Savings

```python
from beanllm.domain.embeddings import MatryoshkaEmbedding, OpenAIEmbedding, truncate_embedding

# Dimension reduction (1536 → 512)
base_emb = OpenAIEmbedding(model="text-embedding-3-large")
mat_emb = MatryoshkaEmbedding(base_embedding=base_emb, output_dimension=512)
reduced_vectors = mat_emb.embed_sync(["text"])  # 512 dims instead of 1536

# Or truncate existing embeddings
full_vector = base_emb.embed_sync(["text"])[0]  # 1536 dims
reduced = truncate_embedding(full_vector, target_dim=512)  # 512 dims
```

---

### Vector Stores

고성능 벡터 데이터베이스 지원 (Milvus, LanceDB, pgvector, Chroma, FAISS).

#### Milvus - High Performance

```python
from beanllm.domain.vector_stores import MilvusVectorStore

milvus = MilvusVectorStore(
    collection_name="docs",
    embedding=embedding,
    connection_args={"host": "localhost", "port": "19530"}
)
milvus.add_documents(docs)
results = milvus.similarity_search("query", k=5)
```

#### LanceDB - Modern Vector DB

```python
from beanllm.domain.vector_stores import LanceDBVectorStore

lancedb = LanceDBVectorStore(
    table_name="docs",
    embedding=embedding,
    uri="./lancedb_data"
)
lancedb.add_documents(docs)
results = lancedb.similarity_search("query", k=5)
```

#### pgvector - PostgreSQL Extension

```python
from beanllm.domain.vector_stores import PGVectorStore

pgvector = PGVectorStore(
    collection_name="docs",
    embedding=embedding,
    connection_string="postgresql://user:pass@localhost/dbname"
)
pgvector.add_documents(docs)
results = pgvector.similarity_search("query", k=5)
```

---

### Retrieval

고급 검색 기법 (HyDE, Hybrid Search, Reranking).

#### HyDE - Hypothetical Document Embeddings

```python
from beanllm.domain.retrieval import HyDE

# Query expansion using LLM
hyde = HyDE(llm=client, embedding=embedding)
expanded_query = hyde.expand_query("What is quantum computing?")
# Returns: hypothetical answer + original query
```

#### Hybrid Search - Combine Vector + Keyword

```python
from beanllm.domain.retrieval import HybridSearch

hybrid = HybridSearch(
    vector_store=vector_store,
    keyword_search=bm25_search,
    alpha=0.5  # 0.5 = equal weight
)
results = hybrid.search("query", k=10)
```

#### Reranking - Cross-Encoder

```python
from beanllm.domain.retrieval import Reranker

reranker = Reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(
    query="query",
    documents=initial_results,
    top_k=5
)
```

---

## Advanced Features

### MultiAgentCoordinator

여러 에이전트가 협업하는 시스템.

#### `__init__(agents, communication_bus=None)`

**파라미터:**
- `agents` (Dict[str, Agent]): 에이전트 딕셔너리 (id: agent)
- `communication_bus` (CommunicationBus, optional): 통신 버스

#### `execute_sequential(task, agent_order, **kwargs)` (async)

순차적으로 에이전트를 실행합니다.

**파라미터:**
- `task` (str): 작업
- `agent_order` (List[str]): 에이전트 실행 순서

#### `execute_debate(task, agent_ids=None, rounds=3, **kwargs)` (async)

토론 방식으로 에이전트를 실행합니다.

**예제:**
```python
from beanllm import MultiAgentCoordinator, Agent

researcher = Agent(model="gpt-4")
writer = Agent(model="gpt-4")

coordinator = MultiAgentCoordinator(
    agents={"researcher": researcher, "writer": writer}
)

result = await coordinator.execute_sequential(
    task="Research AI trends and write a summary",
    agent_order=["researcher", "writer"]
)
```

---

### Graph

그래프 기반 워크플로우.

#### `__init__(enable_cache=True)`

**파라미터:**
- `enable_cache` (bool): 캐싱 활성화 여부

#### `add_node(node)`

그래프에 노드를 추가합니다.

#### `add_edge(from_node, to_node)`

노드 간 엣지를 추가합니다.

#### `run(initial_state, verbose=False)` (async)

그래프를 실행합니다.

---

### StateGraph

상태 기반 그래프 실행 시스템.

#### `__init__(state_schema=None, config=None)`

**파라미터:**
- `state_schema` (Dict, optional): 상태 스키마 정의
- `config` (GraphConfig, optional): 그래프 설정

#### `add_node(name, func)`

상태 그래프에 노드를 추가합니다.

#### `set_entry_point(node_name)`

진입점을 설정합니다.

#### `add_conditional_edge(from_node, condition_func, edge_mapping=None)`

조건부 엣지를 추가합니다.

#### `invoke(initial_state, execution_id=None)` (async)

상태 그래프를 실행합니다.

**예제:**
```python
from beanllm import StateGraph

graph = StateGraph(state_schema={"count": 0, "message": ""})

def increment(state):
    state["count"] += 1
    return state

graph.add_node("increment", increment)
graph.set_entry_point("increment")

result = await graph.invoke({"count": 0, "message": "start"})
```

---

### Audio

8개 STT 엔진 지원 (SenseVoice, Granite, Whisper, etc.)

#### SenseVoice - 15x Faster + Emotion Recognition

```python
from beanllm.domain.audio import beanSTT

# 15x faster than Whisper-Large
stt = beanSTT(engine="sensevoice", language="ko")
result = stt.transcribe("korean_audio.mp3")
print(result.text)
print(result.metadata["emotion"])  # Emotion recognition (SER)
print(result.metadata["events"])   # Audio event detection (AED)
```

#### Granite Speech 8B - Enterprise-grade (WER 5.85%)

```python
from beanllm.domain.audio import beanSTT

# Open ASR Leaderboard #2
stt = beanSTT(engine="granite", language="en")
result = stt.transcribe("english_audio.mp3")
print(f"Transcription: {result.text}")
print(f"WER: {result.metadata.get('wer', 'N/A')}")  # 5.85%
```

#### All 8 STT Engines

```python
from beanllm.domain.audio import beanSTT

# 1. SenseVoice-Small (Alibaba) - 15x faster, emotion
# 2. Granite Speech 8B (IBM) - WER 5.85%, enterprise
# 3. Whisper V3 Turbo (OpenAI) - Balanced
# 4. Distil-Whisper - Efficient
# 5. Parakeet TDT (NVIDIA) - High accuracy
# 6. Canary (NVIDIA) - Multilingual
# 7. Moonshine (Useful Sensors) - Edge devices

engines = ["sensevoice", "granite", "whisper-v3-turbo", "distil-whisper",
           "parakeet", "canary", "moonshine"]
```

#### TextToSpeech - Text-to-Speech

```python
from beanllm import TextToSpeech

tts = TextToSpeech(provider="openai", voice="alloy")
audio_bytes = tts.synthesize("Hello, world!")
```

#### AudioRAG - 오디오 검색 및 QA

```python
from beanllm import AudioRAG

audio_rag = AudioRAG()
audio_rag.add_audio("interview.mp3", audio_id="interview_1")
results = audio_rag.search("What did they say about AI?", top_k=3)
```

---

### Vision

최신 Vision AI 모델 지원 (Qwen3-VL, YOLOv12, SAM 3, Florence-2).

#### Qwen3-VL - Vision-Language Model (128K context)

```python
from beanllm.domain.vision import create_vision_task_model

# Qwen3-VL (VQA, OCR, Captioning, Multi-image Chat)
qwen = create_vision_task_model("qwen3vl", model_size="8B")

# Image Captioning
caption = qwen.caption(image="photo.jpg")

# Visual Question Answering
answer = qwen.vqa(image="photo.jpg", question="What is in this image?")

# OCR Text Extraction
text = qwen.ocr(image="document.jpg")

# Multi-image Chat (128K context)
response = qwen.chat(
    images=["img1.jpg", "img2.jpg", "img3.jpg"],
    prompt="Compare these images and describe the differences"
)
```

#### YOLOv12 - Object Detection & Segmentation

```python
from beanllm.domain.vision import create_vision_task_model

# YOLOv12 (latest)
yolo = create_vision_task_model("yolo", version="12")
detections = yolo.predict(image="photo.jpg", conf=0.5)

for det in detections:
    print(f"Object: {det['class']}, Confidence: {det['confidence']:.2f}")
```

#### SAM 3 - Segment Anything Model

```python
from beanllm.domain.vision import create_vision_task_model

# SAM 3 (Zero-shot segmentation)
sam = create_vision_task_model("sam2")
masks = sam.predict(
    image="photo.jpg",
    points=[[500, 375]],  # Click point
    labels=[1]  # 1=foreground, 0=background
)
```

#### Florence-2 - Unified Vision Tasks

```python
from beanllm.domain.vision import create_vision_task_model

# Florence-2 (Captioning, Detection, OCR, Grounding)
florence = create_vision_task_model("florence2", model_size="large")

# Dense Captioning
captions = florence.caption(image="photo.jpg", task="dense")

# Object Detection
detections = florence.detect(image="photo.jpg")

# OCR
text = florence.ocr(image="document.jpg")
```

---

## Specialized Features

### VisionRAG

이미지 + 텍스트 기반 RAG.

#### `from_images(source, generate_captions=True, llm_model="gpt-4o", **kwargs)`

이미지로부터 VisionRAG를 생성합니다.

**파라미터:**
- `source` (str | List): 이미지 경로 또는 리스트
- `generate_captions` (bool): 자동 캡션 생성 여부
- `llm_model` (str): LLM 모델

**예제:**
```python
from beanllm import VisionRAG

vision_rag = VisionRAG.from_images(
    source="images/",
    generate_captions=True,
    llm_model="gpt-4o"
)

# 이미지 검색 및 질의
response = vision_rag.query(
    question="What objects are in the images?",
    k=3,
    include_images=True
)
```

---

### WebSearch

웹 검색 통합.

#### `search(query, engine=None, **kwargs)`

웹 검색을 수행합니다.

**파라미터:**
- `query` (str): 검색 쿼리
- `engine` (str, optional): 검색 엔진 ("google", "bing", "duckduckgo")

**예제:**
```python
from beanllm import WebSearch

search = WebSearch(default_engine="duckduckgo")
results = search.search("latest AI news")

for result in results:
    print(result.title, result.url)
```

---

### Evaluator

RAG 평가 및 모니터링 (TruLens, RAGAS).

#### TruLens - RAG Performance Evaluation

```python
from beanllm.domain.evaluation import TruLensEvaluator

# TruLens로 RAG 성능 평가
evaluator = TruLensEvaluator(app_name="my_rag")
results = evaluator.evaluate(
    query="What is quantum computing?",
    response="Quantum computing uses quantum mechanics...",
    context=["Document 1 text", "Document 2 text"]
)

# Metrics: Groundedness, Context Relevance, Answer Relevance
print(results.scores)
# {'groundedness': 0.95, 'context_relevance': 0.88, 'answer_relevance': 0.92}
```

#### RAGAS - RAG Assessment

```python
from beanllm.domain.evaluation import RAGASEvaluator

# RAGAS metrics
evaluator = RAGASEvaluator(metrics=["faithfulness", "answer_relevancy"])
result = evaluator.evaluate(
    question="What is Python?",
    answer="Python is a programming language",
    contexts=["Python is a high-level language..."],
    ground_truth="Python is a programming language"  # Optional
)
print(result.scores)
```

#### Traditional Metrics

```python
from beanllm import Evaluator

evaluator = Evaluator(metrics=["bleu", "rouge", "f1"])
result = evaluator.evaluate(
    prediction="The cat sat on the mat",
    reference="A cat was sitting on the mat"
)
print(result.scores)
```

---

### FineTuningManager

모델 파인튜닝.

#### `prepare_and_upload(examples, output_path, validate=True)`

훈련 데이터를 준비하고 업로드합니다.

#### `start_training(model, training_file, validation_file=None, **kwargs)`

파인튜닝 작업을 시작합니다.

**예제:**
```python
from beanllm import FineTuningManager

manager = FineTuningManager(provider="openai")

# 데이터 준비
file_id = manager.prepare_and_upload(
    examples=[...],
    output_path="training.jsonl"
)

# 훈련 시작
job = manager.start_training(
    model="gpt-3.5-turbo",
    training_file=file_id
)

# 진행 상황 확인
progress = manager.get_training_progress(job.id)
```

---

### Advanced LLM Features

고급 LLM 기능 (Structured Outputs, Prompt Caching, Parallel Tool Calling).

자세한 내용은 [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)를 참조하세요.

#### Structured Outputs - 100% Schema Accuracy

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

response = await client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Extract: John Doe, 30, john@example.com"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "strict": True,  # 100% accuracy guarantee
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

#### Prompt Caching - 85% Latency Reduction, 10x Cost Savings

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
print(response.usage.cache_read_input_tokens)  # Cached tokens
```

#### Parallel Tool Calling - Concurrent Execution

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "..."}},
    {"type": "function", "function": {"name": "get_time", "description": "..."}}
]

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather in Seoul and time in Tokyo?"}],
    tools=tools,
    parallel_tool_calls=True  # Execute both simultaneously
)
```

---

## Common Types

### Response Objects

All facade methods return specific response objects:

- `ChatResponse` - Chat completion response
- `RAGResponse` - RAG query response
- `AgentResponse` - Agent execution response
- `AudioResponse` - Audio processing response
- `EvaluationResponse` - Evaluation results
- etc.

### Common Parameters

Most facades support these common parameters:

- `model` (str): Model name (e.g., "gpt-4", "claude-3-opus")
- `temperature` (float): Sampling temperature (0.0 - 2.0)
- `max_tokens` (int): Maximum tokens to generate
- `stream` (bool): Enable streaming responses

---

## Error Handling

```python
import asyncio
from beanllm import Client
from beanllm.utils.exceptions import LLMKitError

async def main():
    try:
        client = Client(model="gpt-4")
        response = await client.chat(
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.content)
    except LLMKitError as e:
        print(f"Error: {e}")

asyncio.run(main())
```

---

## Environment Variables

beanllm uses environment variables for API keys:

```bash
# LLM Providers (7 providers)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="sk-..."
export PERPLEXITY_API_KEY="pplx-..."
export OLLAMA_HOST="http://localhost:11434"  # Optional

# Or use .env file in project root
```

---

## Additional Resources

- [GitHub Repository](https://github.com/leebeanbin/beanllm)
- [PyPI Package](https://pypi.org/project/beanllm/)
- [Examples](../examples/)
- [Architecture Guide](../ARCHITECTURE.md)

---

**Last Updated:** 2025-12-31
**Version:** 0.2.0 (2024-2025 Update)
