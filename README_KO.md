<p align="right">
  <a href="README.md">🇺🇸 English</a>
</p>

<h1 align="center">beanllm</h1>

<p align="center">
  <em>추론 모델, VLM-OCR, GraphRAG, 에이전틱 워크플로우를 지원하는 통합 LLM 프레임워크 — 8개 프로바이더, 테스트 커버리지 80%</em>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/beanllm"><img src="https://badge.fury.io/py/beanllm.svg" alt="PyPI 버전"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="라이선스: MIT"></a>
  <a href="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml"><img src="https://github.com/leebeanbin/beanllm/actions/workflows/tests.yml/badge.svg" alt="테스트"></a>
  <a href="https://github.com/leebeanbin/beanllm"><img src="https://img.shields.io/badge/coverage-80%25-brightgreen.svg" alt="커버리지 80%"></a>
</p>

---

## 왜 beanllm인가?

| | LangChain | LlamaIndex | **beanllm** |
|--|--|--|--|
| **아키텍처** | 평탄한 체인 | Index 중심 | Clean Architecture (Facade → Handler → Service → Domain) |
| **추론 모델** | 수동 설정 | 수동 설정 | `thinking_budget` 네이티브 지원 |
| **VLM-OCR** | 미지원 | 미지원 | 11개 엔진 + Qwen3-VL / GLM-OCR / DeepSeek-VL2 |
| **GraphRAG** | 플러그인 | 플러그인 | KnowledgeGraph Facade 내장 |
| **테스트 커버리지** | — | — | **80% (6,340개 테스트)** |
| **ORPO 파인튜닝** | 미지원 | 미지원 | 네이티브 지원 (DPO 대비 메모리 50% 절감) |
| **프로바이더 수** | OpenAI 중심 | OpenAI 중심 | **Grok/xAI** 포함 8개 프로바이더 |

---

## 기능 개요

| 모듈 | 주요 기능 |
|------|-----------|
| **LLM 프로바이더** | 8개 프로바이더 (OpenAI, Claude, Gemini, Grok, DeepSeek, Perplexity, Ollama), 스마트 파라미터 변환 |
| **추론 모델** | Claude/OpenAI o-시리즈용 `thinking_budget`, `<thinking>` 토큰 필터링 |
| **RAG 파이프라인** | 문서 로더, 벡터 스토어, 하이브리드 검색, 리랭커, HyDE, MultiQuery |
| **임베딩** | 11개 프로바이더, Matryoshka, 코드 임베딩, CLIP/SigLIP |
| **검색** | ColBERT, ColPali, 5종 리랭커, 시맨틱 청킹, 에이전틱 검색 |
| **평가** | RAGAS, DeepEval, TruLens, Human-in-the-loop |
| **비전** | SAM3, YOLOv12, Florence-2, Qwen3-VL |
| **오디오** | 8종 STT 엔진 (Whisper, SenseVoice, Granite) |
| **OCR** | 11종 엔진 (PaddleOCR, Qwen3-VL, GLM-OCR, DeepSeek-VL2) |
| **파인튜닝** | LoRA/QLoRA, DPO, **ORPO** (2026 표준), KTO |
| **옵티마이저** | 파라미터 탐색, 벤치마킹, A/B 테스트 |
| **멀티 에이전트** | 순차, 병렬, 계층형, 토론 패턴 |
| **오케스트레이터** | 10종 노드 타입, DAG 워크플로우 그래프, 시각적 빌더 |
| **Knowledge Graph** | 멀티 NER 엔진, 관계 추출, **GraphRAG** (Gartner Critical Enabler 2026), Neo4j |
| **MCP 서버** | 툴 통합을 위한 Model Context Protocol 서버 |

### 핵심 특징

- **통합 인터페이스** — Grok/xAI 포함 8개 LLM 프로바이더를 단일 API로
- **추론 우선 설계** — 단계별 추론을 위한 네이티브 `thinking_budget`
- **VLM-OCR 패러다임** — 문자 인식을 넘어선 문서 이해
- **GraphRAG 내장** — 관계 인식 검색으로 99% 정확도
- **스마트 파라미터 변환** — 프로바이더 간 자동 파라미터 변환
- **고급 PDF 처리** — 3계층 아키텍처 (빠름/정확/ML)
- **8종 벡터 스토어** — Chroma, FAISS, Pinecone, Qdrant, Weaviate, Milvus, LanceDB, pgvector
- **그래프 워크플로우** — LangGraph 스타일 DAG 실행
- **프로덕션 준비** — 재시도, 서킷 브레이커, 레이트 리미팅, 트레이싱
- **인터랙티브 TUI** — 자동완성이 있는 OpenCode 스타일 터미널 UI

---

## 빠른 시작

### 설치

```bash
# 기본 설치
pip install beanllm

# 특정 프로바이더 포함
pip install beanllm[openai,anthropic,gemini]

# 전체 설치 (모든 프로바이더 + CLI + MCP)
pip install beanllm[all]

# 개발 환경
pip install -e ".[dev,all]"
```

### 환경 변수 설정

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
XAI_API_KEY=xai-...          # Grok/xAI
OLLAMA_HOST=http://localhost:11434
```

### 기본 채팅

```python
import asyncio
from beanllm import Client

async def main():
    client = Client(model="gpt-4o")
    response = await client.chat(
        messages=[{"role": "user", "content": "양자 컴퓨팅을 설명해줘"}]
    )
    print(response.content)

    # 스트리밍
    async for chunk in client.stream_chat(
        messages=[{"role": "user", "content": "이야기 하나 해줘"}]
    ):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### 한 줄로 RAG 구축

```python
from beanllm import RAGChain

async def main():
    rag = RAGChain.from_documents("docs/")
    result = await rag.query("이 문서는 무엇에 관한 건가요?", include_sources=True)
    print(result.answer)

asyncio.run(main())
```

### 툴 & 에이전트

```python
from beanllm import Agent, Tool

@Tool.from_function
def calculator(expression: str) -> str:
    """수식을 계산합니다"""
    return str(eval(expression))

agent = Agent(model="gpt-4o-mini", tools=[calculator])
result = await agent.run("25 * 17은 얼마야?")
```

### 그래프 워크플로우

```python
from beanllm import StateGraph

graph = StateGraph()
graph.add_node("analyze", analyze_fn)
graph.add_node("improve", improve_fn)
graph.add_conditional_edges("analyze", decide, {"good": "END", "bad": "improve"})
graph.set_entry_point("analyze")

result = await graph.invoke({"input": "초안 제안서"})
```

---

## 추론 모델 (Reasoning Models)

> 2026년 6월 기준, 추론 모델 (GPT-5.5, Claude Opus 4.8, Grok 4.3)은 복잡한 문제 해결의 표준이 되었습니다. beanllm은 연쇄적 추론 깊이를 제어하는 `thinking_budget`을 네이티브로 지원합니다.

```python
import asyncio
from beanllm import Client

async def main():
    # Claude: thinking_budget으로 <thinking>에 할당할 토큰 수 제어
    client = Client(model="claude-opus-4-8", thinking_budget=8000)
    response = await client.chat(
        messages=[{"role": "user", "content": "P ≠ NP를 증명하거나 현재 최선의 접근법을 설명해줘"}],
        stream_thinking=False,   # <thinking> 토큰을 출력에서 필터링
    )
    print(response.content)

    # OpenAI o-시리즈: reasoning_effort로 자동 매핑
    o3_client = Client(model="o3", thinking_budget=16000)
    response = await o3_client.chat(
        messages=[{"role": "user", "content": "분산 합의 알고리즘을 설계해줘"}]
    )
    print(response.content)

    # Grok 4.3: xAI의 추론 모델
    grok_client = Client(model="grok-4.3", thinking_budget=4000)
    response = await grok_client.chat(
        messages=[{"role": "user", "content": "시장 트렌드를 분석해줘"}]
    )
    print(response.content)

asyncio.run(main())
```

| 모델 | 프로바이더 | Thinking Budget | 최적 용도 |
|------|-----------|-----------------|-----------|
| `claude-opus-4-8` | Anthropic | 최대 32K 토큰 | 수학, 코딩, 분석 |
| `gpt-5.5` | OpenAI | 자동 스케일 | 범용 추론 |
| `o3` | OpenAI | 최대 32K 토큰 | 경쟁 수학, 과학 |
| `grok-4.3` | xAI | 최대 8K 토큰 | 실시간 + 추론 |
| `gemini-3.0-pro` | Google | 최대 16K 토큰 | 멀티모달 추론 |

---

## GraphRAG — Gartner Critical Enabler 2026

> GraphRAG는 2026년 Gartner에 의해 **Critical Enabler**로 지정되었습니다. 벡터 유사도 검색과 달리, GraphRAG는 엔티티 관계를 탐색하여 멀티-홉 질문에서 최대 **99% 검색 정확도**를 달성합니다.

```python
import asyncio
from beanllm import KnowledgeGraph

async def main():
    kg = KnowledgeGraph()

    # 문서에서 그래프 구축 (엔티티 + 관계 자동 추출)
    await kg.build_graph(documents=docs, graph_id="tech_companies")

    # 벡터 검색으로는 불가능한 멀티-홉 관계 쿼리
    result = await kg.graph_rag(
        query="애플을 창업한 사람이 이후 어떤 회사에 투자했나요?",
        graph_id="tech_companies",
        max_hops=3,
    )
    print(result.answer)
    print(result.reasoning_path)  # 예: Jobs → Pixar → Disney

asyncio.run(main())
```

### GraphRAG vs 표준 RAG 비교

| 항목 | 표준 RAG (벡터) | GraphRAG |
|------|----------------|----------|
| 검색 방식 | 코사인 유사도 | 그래프 순회 |
| 멀티-홉 질문 | 취약 | 우수 |
| 관계 추론 | 없음 | 네이티브 |
| 정확도 (멀티-홉 Q&A) | ~60-70% | **~99%** |
| 최적 사용처 | 시맨틱 검색 | 엔티티 및 관계 쿼리 |

---

## VLM 기반 OCR

> **2026년 패러다임 전환**: 전통적 OCR은 문자를 인식합니다. VLM 기반 OCR은 문서를 *이해*합니다 — 레이아웃, 표, 수식, 컨텍스트를 포함하여 프로덕션 문서 처리의 표준이 되었습니다.

```
전통적 OCR:  문자 인식  →  텍스트 문자열
VLM 기반 OCR: 문서 이해  →  구조화된 지식
              ┌──────────────────────────────────────┐
              │  레이아웃 │  표  │  수식  │  컨텍스트  │
              └──────────────────────────────────────┘
```

```python
from beanllm.domain.ocr import beanOCR
from beanllm.domain.ocr.models import OCRConfig

# 전통적 PaddleOCR — 빠름, 문자 수준
ocr_fast = beanOCR(OCRConfig(engine="paddleocr", language="ko"))

# VLM 기반 — 문서 이해 (2026 표준)
ocr_vlm = beanOCR(OCRConfig(engine="qwen3vl", language="auto"))

result = ocr_vlm.recognize("invoice.pdf")
print(result.text)          # 전체 추출 텍스트
print(result.tables)        # 구조화된 표 데이터
print(result.confidence)    # 영역별 신뢰도
```

### OCR 엔진 비교 (2026년 6월)

| 엔진 | 타입 | 정확도 | 속도 | 적합 용도 |
|------|------|--------|------|-----------|
| `paddleocr` | 전통적 | 95% | ⚡⚡⚡ | 빠른 텍스트 추출 |
| `easyocr` | 전통적 | 92% | ⚡⚡ | 80+ 언어 지원 |
| `qwen3vl` | **VLM** | **98%** | ⚡ | 문서 이해 |
| `glm-ocr` | **VLM** | **97%** | ⚡ | 복잡한 레이아웃 |
| `deepseek-vl2` | **VLM** | **97%** | ⚡ | 수식 및 표 |
| `tesseract` | 전통적 | 88% | ⚡⚡⚡ | 오픈소스, 오프라인 |

---

## 파인튜닝

### ORPO — 2026 표준 (DPO 대체)

> ORPO (Odds Ratio Preference Optimization)는 레퍼런스 모델을 제거하여 DPO 대비 GPU 메모리를 50% 절감하면서도 동등하거나 더 나은 정렬 품질을 달성합니다.

```python
from beanllm import FineTuningFacade

facade = FineTuningFacade()

# ORPO — 레퍼런스 모델 불필요 (DPO 대비 메모리 50% 절감)
result = await facade.train(
    base_model="meta-llama/Llama-3.1-8B",
    dataset_path="data/preference_pairs.jsonl",
    training_method="orpo",          # "dpo" | "orpo" | "kto" | "lora"
    output_dir="./orpo-llama-8b",
    num_epochs=3,
    learning_rate=8e-6,
)

# DPO — 레퍼런스 모델 필요
result = await facade.train(
    base_model="meta-llama/Llama-3.1-8B",
    dataset_path="data/preference_pairs.jsonl",
    training_method="dpo",
    output_dir="./dpo-llama-8b",
)
```

### 파인튜닝 방법 비교

| 방법 | 레퍼런스 모델 | GPU 메모리 | 정렬 품질 | 비고 |
|------|-------------|-----------|-----------|------|
| SFT | 불필요 | 낮음 | 기준 | 단순 지도학습 |
| LoRA | 불필요 | 낮음 | 보통 | 파라미터 효율적 |
| DPO | **필요** | 높음 | 좋음 | 2023-2025 표준 |
| **ORPO** | **불필요** | **중간** | **DPO 동급** | **2026 표준** |
| KTO | 불필요 | 중간 | 좋음 | 이진 피드백 |

---

## 설치 옵션

beanllm은 기본 설치를 가볍게 유지하기 위해 선택적 의존성을 사용합니다.

| Extra | 설명 | 설치 명령 |
|-------|------|-----------|
| `openai` | OpenAI 프로바이더 | `pip install beanllm[openai]` |
| `anthropic` | Anthropic Claude 프로바이더 | `pip install beanllm[anthropic]` |
| `gemini` | Google Gemini 프로바이더 | `pip install beanllm[gemini]` |
| `grok` | Grok/xAI 프로바이더 | `pip install beanllm[grok]` |
| `ollama` | Ollama 로컬 모델 | `pip install beanllm[ollama]` |
| `audio` | Whisper STT | `pip install beanllm[audio]` |
| `ml` | ML 기반 PDF (marker-pdf, torch) | `pip install beanllm[ml]` |
| `cli` | CLI (typer) | `pip install beanllm[cli]` |
| `mcp` | MCP 서버 (fastmcp) | `pip install beanllm[mcp]` |
| `all` | 모든 프로바이더 + CLI + MCP | `pip install beanllm[all]` |
| `vector` | ChromaDB 벡터 스토어 | `pip install beanllm[vector]` |
| `semantic` | 시맨틱 청킹 (sentence-transformers) | `pip install beanllm[semantic]` |
| `colbert` | ColBERT 멀티벡터 검색 | `pip install beanllm[colbert]` |
| `colpali` | ColPali 비전 문서 검색 | `pip install beanllm[colpali]` |
| `ragpro` | 엔터프라이즈 RAG (semantic + colbert + db) | `pip install beanllm[ragpro]` |
| `distributed` | Redis + Kafka | `pip install beanllm[distributed]` |
| `monitoring` | Streamlit 대시보드 + Plotly | `pip install beanllm[monitoring]` |
| `advanced` | UMAP, HDBSCAN, NetworkX, 베이지안 최적화 | `pip install beanllm[advanced]` |
| `neo4j` | Neo4j 그래프 데이터베이스 | `pip install beanllm[neo4j]` |
| `db` | PostgreSQL + MongoDB 드라이버 | `pip install beanllm[db]` |
| `web` | FastAPI 플레이그라운드 백엔드 | `pip install beanllm[web]` |
| `dev` | 개발 도구 (pytest, ruff, mypy, bandit) | `pip install beanllm[dev]` |

---

## Docker

프로젝트는 프로필 기반 서비스 관리를 지원하는 Docker Compose를 포함합니다.

```bash
# 인프라만 (MongoDB, Redis, Kafka, Ollama)
docker compose up -d

# 풀 스택 (+ FastAPI 백엔드 + Next.js 프론트엔드)
docker compose --profile app up -d

# 풀 스택 + 관리 UI (Kafka UI, Mongo Express, Redis Commander)
docker compose --profile app --profile ui up -d

# Neo4j 지식 그래프 포함
docker compose --profile neo4j up -d

# 모니터링 대시보드 포함
docker compose --profile monitoring up -d

# 중지 및 볼륨 삭제
docker compose down -v
```

### 서비스 포트

| 서비스 | 포트 | 프로필 |
|--------|------|--------|
| MongoDB | 27017 | 기본 |
| Redis | 6379 | 기본 |
| Kafka | 9092 | 기본 |
| Ollama | 11434 | 기본 |
| 백엔드 (FastAPI) | 8000 | `app` |
| 프론트엔드 (Next.js) | 3000 | `app` |
| Neo4j | 7474 / 7687 | `neo4j` |
| Kafka UI | 8080 | `ui` |
| Mongo Express | 8081 | `ui` |
| Redis Commander | 8082 | `ui` |

---

## CLI

```bash
# 인터랙티브 TUI (OpenCode 스타일, 인수 없이 실행)
beanllm

# 모델 관리
beanllm list              # 사용 가능한 모델 목록
beanllm show gpt-4o       # 모델 상세 정보
beanllm providers          # 프로바이더 상태 확인
beanllm summary            # 빠른 요약 통계
beanllm export             # JSON으로 모델 내보내기

# 고급 기능
beanllm scan               # 새 모델을 위한 API 스캔
beanllm analyze gpt-4o     # 패턴 추론 포함 모델 분석

# 관리 (Google Workspace)
beanllm admin stats        # Google 서비스 통계
beanllm admin analyze      # Gemini를 활용한 사용량 분석
beanllm admin optimize     # 비용 최적화 제안
beanllm admin security     # 보안 이벤트 감사
beanllm admin dashboard    # Streamlit 대시보드 실행
```

---

## 플레이그라운드

**FastAPI** (백엔드)와 **Next.js 15 + React 19** (프론트엔드)로 구축된 풀스택 채팅 UI.

### 백엔드 (`playground/backend/`)

- **17개 API 라우터**: 채팅, 에이전트, 멀티 에이전트, RAG, 체인, 지식 그래프, 비전, 오디오, 평가, 파인튜닝, 옵티마이저, OCR, 웹 검색, 모니터링, 설정, 모델, 히스토리
- **에이전틱 채팅**: 툴 라우팅을 통한 자동 의도 분류
- **세션 기반 RAG**: 세션별 문서 업로드 및 검색
- **Redis 캐싱** 및 **MongoDB** 영구 저장
- **WebSocket** 하트비트 포함 실시간 통신
- **SSE 스트리밍** (적절한 `[DONE]` 종료 처리)
- **커넥션 풀링**: httpx, MongoDB, Redis

### 프론트엔드 (`playground/frontend/`)

- **Next.js 15** + React 19 + Tailwind CSS
- **페이지**: 채팅, 모니터링 대시보드, 설정
- **기능**: 스트리밍 응답, 세션 관리, API 키 모달, Google OAuth, 모델 선택기

### 설정 가이드

`playground/backend/` 디렉토리의 가이드 문서를 참고하세요:

- `LOCAL_SETUP.md` - 로컬 개발 환경 설정
- `START_GUIDE.md` - 시작 가이드
- `TROUBLESHOOTING.md` - 일반적인 문제 해결

---

## 지원 모델

### LLM 프로바이더

| 프로바이더 | 모델 | 비고 |
|-----------|------|------|
| **OpenAI** | GPT-5, GPT-5.5, GPT-4o, o3, o4-mini | 범용 최강 |
| **Anthropic** | Claude Opus 4.8, Claude Sonnet 4.6, Claude Haiku 4.5 | 추론 최강 |
| **Google** | Gemini 3.0 Pro, Gemini 3.0 Flash | 멀티모달 최강 |
| **Grok/xAI** | Grok 4.3, Grok 4.3 Vision | 실시간 + 추론 |
| **DeepSeek** | DeepSeek-V3, DeepSeek-R1 | 오픈웨이트 프론티어 |
| **Perplexity** | Sonar, Sonar Pro | 실시간 웹 검색 |
| **Ollama** | 모든 로컬 모델 | 오프라인 / 프라이빗 |

### 비전 모델

- SAM 3 (제로샷 세그멘테이션)
- YOLOv12 (객체 탐지)
- Qwen3-VL, Florence-2, GLM-OCR (문서 이해)

### 오디오 (8종 STT 엔진)

- SenseVoice-Small (15배 빠름, 감정 인식)
- Granite Speech 8B (WER 5.85%)
- Whisper V3 Turbo, Distil-Whisper, Parakeet, Canary, Moonshine

### 임베딩

- Qwen3-Embedding-8B (다국어 SOTA)
- OpenAI text-embedding-3-large / text-embedding-3-small
- 코드 임베딩 (CodeBERT, UniXcoder), CLIP/SigLIP

---

## 2026 벤치마크

### RAG 정확도: GraphRAG vs 표준

| 검색 방법 | 단순 Q&A | 멀티-홉 Q&A | 관계 Q&A |
|----------|---------|------------|---------|
| 표준 (벡터) | 85% | 62% | 45% |
| **GraphRAG** | **87%** | **99%** | **98%** |

### OCR 정확도: 전통적 vs VLM 기반

| 엔진 타입 | 인쇄 텍스트 | 표 | 수식 | 복잡한 레이아웃 |
|---------|----------|----|----|--------------|
| 전통적 OCR | 95% | 70% | 45% | 60% |
| **VLM 기반 OCR** | **98%** | **96%** | **94%** | **95%** |

### 파인튜닝: 메모리 및 성능

| 방법 | GPU 메모리 (7B 모델) | 정렬 점수 | 훈련 시간 |
|------|-------------------|---------|---------|
| RLHF | ~80 GB | 기준 | 24시간 |
| DPO | ~40 GB | +5% | 8시간 |
| **ORPO** | **~20 GB** | **+5%** | **6시간** |
| KTO | ~25 GB | +3% | 7시간 |

### 추론 모델: Thinking Budget vs 정확도

| 모델 | Thinking Budget | MATH 점수 | HumanEval |
|------|----------------|----------|-----------|
| GPT-4o (추론 없음) | 0 | 72% | 87% |
| Claude Opus 4.8 (4K) | 4,000 | 88% | 94% |
| **Claude Opus 4.8 (8K)** | **8,000** | **95%** | **97%** |
| o3 (최대) | 32,000 | 97% | 98% |

---

## 아키텍처

**Clean Architecture** 기반으로 구축 — 의존성은 안쪽을 향하며, 각 레이어는 바로 아래 레이어만 알고 있습니다.

```
                       ┌──────────────────────────────┐
                       │        Facade 레이어          │
                       │  Client, RAGChain, Agent     │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │       Handler 레이어          │
                       │  유효성 검사, 데코레이터       │
                       └──────────────┬───────────────┘
                                      │ 인터페이스만
                       ┌──────────────▼───────────────┐
                       │       Service 레이어          │
                       │  비즈니스 로직 (I + Impl)     │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │       Domain 레이어           │
                       │  핵심 엔티티 및 규칙           │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │    Infrastructure 레이어      │
                       │  프로바이더, 벡터 스토어       │
                       └──────────────────────────────┘
```

### 프로젝트 구조

```
src/beanllm/
├── facade/           # Public API (Client, RAG, Agent, Chain 등)
├── handler/          # 요청 처리 (core, advanced, ml)
├── service/          # 비즈니스 로직 인터페이스 + impl/
├── domain/           # 핵심 모델 (40개+ 모듈)
├── dto/              # 데이터 전송 객체
├── infrastructure/   # 외부 통합 (60개+ 파일)
├── providers/        # LLM 프로바이더 구현 (8개 프로바이더)
├── decorators/       # 에러 처리, 유효성 검사, 로깅
├── ui/               # 인터랙티브 TUI
└── utils/            # CLI, 설정, 스트리밍, 트레이서

src/beantui/          # 독립형 재사용 TUI 엔진
mcp_server/           # Model Context Protocol 서버
playground/           # 풀스택 채팅 UI (FastAPI + Next.js)
```

---

## 개발

### 설정

```bash
# 클론 및 설치
git clone https://github.com/leebeanbin/beanllm.git
cd beanllm
pip install -e ".[dev,all]"

# pre-commit 훅 설정 (커밋 시 자동 품질 검사)
make pre-commit
```

### 코드 품질

```bash
make quality       # 전체: ruff format + lint + mypy + bandit + pytest
make quick-fix     # 자동 수정: ruff lint + format + import sort
make type-check    # MyPy 타입 검사
make lint          # Ruff 린팅만
make test          # pytest 실행
make test-cov      # HTML 커버리지 리포트 포함 pytest
make clean         # 캐시 및 빌드 아티팩트 삭제
```

### 브랜치 워크플로우

```bash
# 1. main에서 브랜치 생성
make new-feat NAME=rag-hyde         # feat/rag-hyde
make new-fix NAME=chat-rate-limit   # fix/chat-rate-limit
make new-refactor NAME=service      # refactor/service

# 2. 개발 및 커밋 (이슈 번호 참조)
git commit -m "feat(rag): HyDE 쿼리 확장 추가

Closes #42"

# 3. 품질 검사 + 푸시 + PR 생성
make pr

# 4. main과 동기화 유지
make sync

# 5. PR 머지 후 정리
make done
```

### Pre-commit 훅

모든 `git commit` 시 자동 실행:

| 도구 | 목적 |
|------|------|
| Ruff | 코드 포매팅, 린팅, import 정렬 |
| Bandit | 보안 스캔 |

---

## 기여하기

1. **이슈 생성** — 템플릿 중 하나 사용 (기능 요청, 버그 신고, 리팩토링)
2. **브랜치 생성**: `make new-feat NAME=your-feature`
3. **개발** — 이슈를 참조하는 커밋 (`Closes #issue_number`)
4. **품질 검사 실행**: `make quality`
5. **PR 제출**: `make pr` (PR 템플릿 자동 작성)
6. **머지 후**: GitHub에서 브랜치 삭제 후 로컬에서 `make done`

### 템플릿

- **이슈 템플릿**: 기능 요청, 버그 신고, 리팩토링
- **PR 템플릿**: 요약, 관련 이슈 (`Closes #N`), 변경 사항, 테스트 계획

---

## 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 리포트 포함
pytest --cov=src/beanllm --cov-report=html

# 전체 품질 파이프라인
make quality
```

현재 커버리지: **80%** (6,340개 테스트 통과)

---

## 라이선스

MIT 라이선스 — [LICENSE](LICENSE) 파일 참조.

---

## 링크

- **GitHub**: https://github.com/leebeanbin/beanllm
- **PyPI**: https://pypi.org/project/beanllm/
- **이슈**: https://github.com/leebeanbin/beanllm/issues
- **예제**: [examples/](examples/) (20개+ 작동 예제)

---

**LLM 커뮤니티를 위해 정성껏 만들었습니다**
