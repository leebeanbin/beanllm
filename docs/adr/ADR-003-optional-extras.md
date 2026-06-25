# ADR-003: 선택적 의존성(Optional Extras) 시스템

* **Status:** Accepted
* **Date:** 2024-09
* **Author:** leebeanbin

## Context & Problem Statement

beanllm은 8개 LLM 프로바이더 외에도 OCR (PaddleOCR, Qwen3-VL), 음성 인식 (Whisper 8종), 파인튜닝 (LoRA/ORPO), 벡터스토어 (ChromaDB, Qdrant, FAISS 등 8종), ColBERT, GraphRAG (Neo4j) 등을 지원한다.

모든 기능을 기본 설치에 포함하면:
- `torch` 하나만 2GB+
- PaddleOCR + 모델 가중치 수 GB
- Kafka, Redis, Neo4j 드라이버 등 불필요한 의존성

단순히 GPT-4o로 채팅하려는 사용자가 torch를 설치해야 하는 상황이 발생한다.

## Decision Drivers

* `pip install beanllm` 기본 설치가 수 초 내에 완료되어야 함 (~5MB)
* 사용자는 필요한 기능만 설치 (`pip install beanllm[openai]`)
* CI/CD에서 기능별 독립 테스트 가능 (`pip install beanllm[openai,dev]`)
* 미설치 패키지 접근 시 명확한 에러 메시지 ("beanllm[ml] required for ML-based PDF")

## Considered Options

1. **단일 heavy 패키지** — 모든 의존성 포함, 설치 즉시 모든 기능 사용
2. **기능별 별도 패키지** — `beanllm-openai`, `beanllm-rag` 등 분리 배포
3. **Optional Extras (PEP 508)** ← 선택

## Decision Outcome

Chosen Option: **Option 3**.

`pyproject.toml`의 `[project.optional-dependencies]`로 extras를 선언하고, 런타임에 `try/except`로 선택적 import한다.

**핵심 의존성 (기본 설치):**
```toml
dependencies = [
    "httpx",          # HTTP 클라이언트
    "python-dotenv",  # .env 로드
    "pydantic",       # Response DTO 검증
    "tiktoken",       # 토큰 카운팅
    "PyMuPDF",        # Fast PDF 파싱
    "pdfplumber",     # 테이블 추출
    "numpy",          # 수치 연산
    "rich",           # 터미널 UI
]
```

**선택적 extras:**
```
beanllm[openai]      → openai SDK
beanllm[anthropic]   → anthropic SDK
beanllm[gemini]      → google-generativeai
beanllm[grok]        → xai-sdk
beanllm[ollama]      → ollama Python client
beanllm[audio]       → openai-whisper
beanllm[ml]          → torch, marker-pdf (ML-based OCR)
beanllm[vector]      → chromadb
beanllm[semantic]    → sentence-transformers
beanllm[colbert]     → colbert-ai
beanllm[ragpro]      → semantic + colbert + DB drivers (enterprise RAG)
beanllm[neo4j]       → neo4j driver (GraphRAG)
beanllm[distributed] → redis, kafka-python
beanllm[monitoring]  → streamlit, plotly
beanllm[cli]         → typer (CLI 툴)
beanllm[mcp]         → fastmcp (MCP 서버)
beanllm[dev]         → pytest, ruff, mypy, bandit
beanllm[all]         → 전체 (모든 프로바이더 + CLI + MCP)
```

**런타임 선택적 import 패턴:**
```python
# provider_factory.py
try:
    from .claude_provider import ClaudeProvider
except Exception as e:
    logger.warning(f"Failed to import ClaudeProvider: {e}")
    ClaudeProvider = None   # None이면 ProviderFactory에서 건너뜀

# ML 기능
try:
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def load_ml_model():
    if not ML_AVAILABLE:
        raise ImportError("pip install beanllm[ml] required for ML-based PDF processing")
```

### Consequences

* **Positive:**
  - `pip install beanllm` — 핵심 LLM 채팅 기능 즉시 사용 (수 초, ~5MB)
  - 서버 환경에서 불필요한 GPU 라이브러리(torch) 설치 없이 운영 가능
  - CI/CD에서 기능 조합별 독립 테스트 (`beanllm[openai,dev]`, `beanllm[ml,dev]`)
  - 미설치 기능 접근 시 명확한 에러 → 사용자 혼란 최소화
* **Negative/Trade-offs:**
  - Extras 조합이 20개 → 어떤 것을 설치해야 하는지 문서화 필요
  - `try/except` import 패턴이 모든 optional 모듈에 반복 → 보일러플레이트
  - 의존성 충돌: `beanllm[all]` 설치 시 버전 충돌 가능성 (패키지 수가 많을수록 위험)

---

## Options Comparison Matrix

| Criteria | 단일 heavy 패키지 | 별도 패키지 분리 | Optional Extras |
|---|---|---|---|
| **설치 크기 (최소)** | ❌ 수 GB | ✅ 작음 | ✅ ~5MB |
| **기능 통합성** | ✅ 모두 즉시 | ❌ 패키지 간 호환성 | ✅ 하나의 패키지 |
| **PyPI 관리** | ✅ 1개 | ❌ 20+ 패키지 | ✅ 1개 |
| **선택적 설치** | ❌ 불가 | ✅ | ✅ |
| **import 에러 처리** | ✅ 없음 | ✅ | ⚠️ try/except 필요 |
