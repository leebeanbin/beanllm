# beanLLM 개선 로드맵 2025

> **작성일**: 2025-12-31
> **조사 범위**: Text Embeddings, Audio/STT, Vision, RAG/Retrieval, LLM Providers, Document Loaders
> **목적**: beanLLM 패키지의 2024-2025 최신 기술 조사 및 개선 방향 제시

---

## 📋 Executive Summary

6개 도메인에 대한 종합 조사 결과, beanLLM은 **기본기는 탄탄하나 일부 최신 기술 업데이트가 필요**한 상태입니다.

### 핵심 발견사항

| 도메인 | 현재 상태 | 업데이트 필요성 | 우선순위 |
|--------|----------|---------------|---------|
| **Text Embeddings** | 🟡 일부 최신화 필요 | Voyage v3, Jina v3, Qwen3 추가 | **높음** |
| **Audio/STT** | 🟢 최신 모델 포함 | Canary Qwen 2.5B, SenseVoice 추가 권장 | 중간 |
| **Vision** | 🟡 업데이트 권장 | SAM 3, YOLOv12, VLM 추가 | **높음** |
| **RAG/Retrieval** | 🔴 대폭 개선 필요 | Hybrid Search, Reranking, 평가 도구 | **매우 높음** |
| **LLM Providers** | 🟢 충분한 커버리지 | 신규 프로바이더 선택적 추가 | 낮음 |
| **Document Loaders** | 🔴 주요 형식 누락 | Office 파일, HTML, Jupyter 필수 | **매우 높음** |

### 영향도 높은 개선 항목 Top 10

1. **Hybrid Search 구현** (RAG 품질 대폭 향상) - 🔥 **가장 시급**
2. **Reranker 추가** (검색 정확도 48% 개선) - 🔥 **가장 시급**
3. **Office 파일 로더 추가** (Docling) - 🔥 **가장 시급**
4. **Voyage AI v3, Jina v3 업데이트** (임베딩 성능 향상)
5. **SAM 3, YOLOv12 업데이트** (최신 비전 모델)
6. **RAGAS/TruLens 평가 도구 통합** (RAG 품질 측정)
7. **Qwen3-Embedding, EVA-CLIP 추가** (다국어 지원)
8. **VLM 추가** (Qwen3-VL, InternVL3.5 등)
9. **HTML/Jupyter 로더 추가** (문서 타입 확장)
10. **Canary Qwen 2.5B, SenseVoice 추가** (STT 성능 향상)

---

## 📊 도메인별 상세 분석

### 1. Text Embeddings

#### 현재 상태
- ✅ **최신 모델 포함**: NV-Embed-v2 (72.31 MTEB), OpenAI text-embedding-3
- ✅ **주요 프로바이더**: OpenAI, Gemini, Cohere, Voyage, Jina, Mistral, Ollama, HuggingFace, NVIDIA
- ⚠️ **업데이트 필요**: Voyage v2 → v3, Jina v2 → v3

#### 중요 발견사항

**NV-Embed-v2는 더 이상 압도적 1위가 아님**
- 현재 MTEB 점수: 72.31 (여전히 최상위권)
- 경쟁자 등장: Qwen3-Embedding-8B (70.58), bge-en-icl (71.24), Voyage-3-large (#1 in specific tasks)

**신규 기술 트렌드**
1. **Matryoshka Embeddings**: 단일 모델로 가변 차원 (32-4096) 지원, 비용 절감
2. **Binary/int8 Quantization**: 32배 압축, 96%+ 성능 유지
3. **Hybrid Search**: Dense + Sparse + ColBERT 조합이 최적
4. **In-Context Learning**: bge-en-icl 방식으로 태스크 적응

**업데이트 권장사항** (우선순위 순)

| 순위 | 항목 | 이유 | 난이도 |
|-----|------|------|-------|
| 1 | Voyage AI v3 시리즈 추가 | 특정 벤치마크 1위, 4개 변형 (large, base, 3.5, code-3, multimodal-3) | 낮음 |
| 2 | Jina AI v3 업데이트 | 89개 언어, LoRA 어댑터, Matryoshka 지원 | 낮음 |
| 3 | Qwen3-Embedding-8B 추가 | 119개 언어, 70.58 MTEB, Matryoshka 지원 | 중간 |
| 4 | Matryoshka 지원 구현 | `dimensions=` 파라미터로 가변 차원 활성화 | 중간 |
| 5 | Code 임베딩 추가 | Mistral Codestral Embed, SFR-Embedding-Code-7B, voyage-code-3 | 중간 |
| 6 | 한국어 모델 추가 | KURE, KoE5, bge-m3-korean, KoSimCSE-roberta | 낮음 |
| 7 | Binary/int8 Quantization | 스토리지 비용 32배 절감 | 높음 |

**Quick Win**
```python
# Voyage v3 추가 (기존 Voyage v2 패턴 재사용)
class VoyageV3Embedding(VoyageEmbedding):
    def __init__(self, model: str = "voyage-3-large", **kwargs):
        super().__init__(model=model, **kwargs)
```

---

### 2. Audio/STT

#### 현재 상태
- ✅ **최신 모델 다수 포함**: Whisper V3 Turbo, Distil-Whisper v3, Canary-1B, Canary-Flash, Moonshine
- ✅ **6개 엔진**: Whisper, AssemblyAI, Deepgram, Google, Azure, Amazon
- ⚠️ **업데이트 권장**: Parakeet TDT V3

#### 중요 발견사항

**Whisper V4는 존재하지 않음** - V3가 최신 공식 버전

**새로운 SOTA 모델** (Open ASR Leaderboard 2024-2025)
1. **Canary Qwen 2.5B** - #1 순위 (5.63% WER, RTFx 418)
2. **IBM Granite Speech 8B** - #2 순위 (5.85% WER, Apache 2.0)
3. **SenseVoice-Small** - Whisper-Large 대비 15배 빠름, 한국어 지원

**현재 모델 평가**
- ✅ Whisper V3 Turbo: 최신
- ✅ Distil-Whisper: 최신 (v3)
- ⚠️ Parakeet TDT: V3로 업그레이드 필요
- ✅ Canary-1B, Canary-Flash, Moonshine: 유지

**업데이트 권장사항** (우선순위 순)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 1 | Canary Qwen 2.5B 추가 | Open ASR #1, 5.63% WER | 중간 | 높음 |
| 2 | SenseVoice-Small 추가 | 15배 빠름, 한국어 지원 | 중간 | 높음 |
| 3 | Granite Speech 8B 추가 | Open ASR #2, Apache 2.0 | 중간 | 중간 |
| 4 | Parakeet TDT V3 업그레이드 | 최신 버전 동기화 | 낮음 | 낮음 |

**상용 API 고려사항**
- Deepgram Nova-3, AssemblyAI Universal-Streaming, Google Chirp 3는 이미 지원 가능
- 추가 필요 없음

---

### 3. Vision

#### 현재 상태
- ✅ **이미지 임베딩 (4개)**: CLIP, SigLIP 2, MobileCLIP2, NV-Embed-v2
- ✅ **태스크 모델 (3개)**: YOLOv11, SAM 2, Florence-2
- ⚠️ **VLM 없음**: 멀티모달 언어 모델 부재

#### 중요 발견사항

**이미지 임베딩 SOTA**
- ✅ **SigLIP 2 (2025년 2월)**: 이미 포함, 다국어 지원
- 🆕 **EVA-CLIP-18B (2024)**: 82.0 zero-shot top-1 (ImageNet)
- 🆕 **DINOv2 (2023, 활발 사용)**: 자기지도학습 백본
- ✅ **MobileCLIP2 (2025년 8월)**: 이미 포함, 모바일 최적

**객체 검출 & 세분화 SOTA**
- 🆕 **YOLOv12 (NeurIPS 2025)**: Attention-centric, 40.6% mAP
- 🆕 **RF-DETR (2025)**: 실시간 최초 60+ mAP (60.5 mAP @ 25 FPS)
- 🆕 **SAM 3 (2025년 11월)**: 텍스트 프롬프트, 개념 세분화

**VLM (Vision-Language Models) SOTA**
- 🆕 **Qwen3-VL (2025)**: 128k 컨텍스트, 29개 언어, 2B-32B
- 🆕 **InternVL3.5 (2025년 8월)**: 오픈소스 MLLM SOTA (241B-A28B)
- 🆕 **PaliGemma 2 (2024년 12월)**: Google, OCR/분자 인식 SOTA
- 🆕 **LLaMA 3.2 Vision (2024년 9월)**: Meta, 11B/90B
- 🆕 **Pixtral Large (2024년 11월)**: 124B, LMSys 오픈소스 1위
- 🆕 **Aria (2024년 10월)**: Multimodal native MoE, 비디오 강점

**업데이트 권장사항** (우선순위 순)

#### 필수 업데이트 (Phase 1)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 1 | SAM 3 업그레이드 | 텍스트 프롬프트, 개념 세분화 (SAM 2 대비 2배 성능) | 중간 | 높음 |
| 2 | YOLOv12 업그레이드 | Attention-centric, NeurIPS 2025 | 낮음 | 중간 |
| 3 | Qwen3-VL 추가 | 다국어 VLM, 비디오 지원, 128k 컨텍스트 | 높음 | 높음 |
| 4 | EVA-CLIP 추가 | ImageNet 82.0 zero-shot (1/6 파라미터로 SOTA) | 중간 | 중간 |

#### 고급 추가 (Phase 2)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 5 | DINOv2 추가 | 자기지도학습 백본, 의료/비전 태스크 강점 | 중간 | 중간 |
| 6 | RF-DETR 추가 | 실시간 SOTA (60.5 mAP) | 중간 | 중간 |
| 7 | InternVL3.5 추가 | 오픈소스 MLLM 최강 (perception & reasoning) | 높음 | 높음 |
| 8 | PaliGemma 2 추가 | Google VLM, OCR/분자 인식 특화 | 중간 | 중간 |
| 9 | Depth Anything V2 추가 | Monocular depth estimation SOTA | 중간 | 낮음 |

#### 선택적 추가 (Phase 3)

- LLaMA 3.2 Vision, Pixtral Large, Aria, Phi-3 Vision, DeepSeek-VL2
- Image Quality Assessment: HiRQA, UniQA, LAR-IQA

**Quick Win**
```python
# YOLOv12 업그레이드 (기존 YOLOWrapper 패턴 재사용)
class YOLOWrapper(BaseVisionTaskModel):
    def __init__(self, version: str = "12", model_size: str = "m", task: str = "detect"):
        # version="11" → "12"로 변경만으로 업그레이드
```

---

### 4. RAG/Retrieval 🔥 **가장 시급한 개선 영역**

#### 현재 상태
- ✅ **벡터 DB (5개)**: Chroma, FAISS, Pinecone, Qdrant, Weaviate
- ❌ **Hybrid Search 없음**: BM25 + Dense 결합 부재
- ❌ **Reranker 없음**: 검색 품질 48% 개선 기회 놓침
- ❌ **평가 도구 없음**: RAGAS, TruLens, LangSmith 부재

#### 중요 발견사항

**RAG 핵심 기술 (2024-2025)**

1. **Hybrid Search (필수)** 🔥
   - BM25 + Dense Vectors + SPLADE Sparse Vectors
   - IBM 연구: 3-way retrieval이 최적
   - 검색 품질 대폭 향상 (단일 방법 대비)

2. **Reranking (필수)** 🔥
   - Databricks 연구: 검색 품질 **최대 48% 개선**
   - BGE Reranker v2 (BAAI) - 다국어, 2024년 3월
   - Cohere Rerank 4 (2024년 12월) - 32K context, self-learning

3. **RAG 개선 기법**
   - **HyDE**: 가상 문서 생성으로 의미적 갭 해결
   - **RAPTOR**: 계층적 요약 트리 (QuALITY 20% 향상)
   - **Self-RAG**: 적응형 검색
   - **GraphRAG**: 지식 그래프 통합 (Microsoft, 2024)

4. **Context Engineering**
   - **Position Engineering**: 중요 정보를 프롬프트 상단/하단 배치 (무료, 대폭 성능 향상)
   - "Lost in the Middle" 문제 해결
   - Long Context vs RAG: RAG 우선 접근 권장

5. **Evaluation & Monitoring (필수)** 🔥
   - **RAGAS**: Reference-free RAG 평가 (업계 표준)
   - **TruLens**: RAG Triad (Context Relevance, Groundedness, Answer Relevance)
   - **LangSmith**: End-to-end 플랫폼 (LangChain 통합)

6. **Multi-modal RAG**
   - 텍스트 + 이미지 + 테이블 + 차트 통합
   - ACL 2025 Findings: 최초 포괄적 서베이 논문
   - LanceDB: Multi-modal native 지원

7. **신규 벡터 DB**
   - **Milvus**: 엔터프라이즈 대규모 (100K+ QPS, 수십억 벡터)
   - **LanceDB**: 임베디드, 멀티모달 네이티브 (엣지 AI 최적)
   - **pgvector**: PostgreSQL 확장 (50M 벡터 @ 471 QPS)

**업데이트 권장사항** (우선순위 순)

#### 🔥 Phase 1: 필수 기초 (1-2개월)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 1 | **Hybrid Search 구현** | BM25 + Dense, 검색 품질 대폭 향상 | 중간 | **매우 높음** |
| 2 | **Reranker 추가 (BGE v2-m3)** | 검색 정확도 48% 개선 | 낮음 | **매우 높음** |
| 3 | **RAGAS 평가 통합** | RAG 품질 측정 (Faithfulness, Relevancy) | 낮음 | **높음** |

**구현 예시**
```python
# Hybrid Search
from beanllm.domain.retrieval import HybridRetriever

retriever = HybridRetriever(
    dense_retriever=chroma_retriever,  # 기존 벡터 검색
    sparse_retriever=BM25Retriever(),  # 새로 추가
    fusion_method="rrf"  # Reciprocal Rank Fusion
)

# Reranker
from beanllm.domain.retrieval import Reranker

reranker = Reranker(model="BAAI/bge-reranker-v2-m3")
results = reranker.rerank(query, candidates, top_k=5)

# RAGAS Evaluation
from beanllm.evaluation import RAGASEvaluator

evaluator = RAGASEvaluator()
metrics = evaluator.evaluate(
    questions=[...],
    answers=[...],
    contexts=[...]
)
# → Faithfulness, Answer Relevancy, Context Precision/Recall
```

#### ⚡ Phase 2: 최적화 (3-4개월)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 4 | HyDE 쿼리 확장 | 의미적 갭 해결 | 중간 | 높음 |
| 5 | Position Engineering | 무료로 성능 향상 | 낮음 | 높음 |
| 6 | TruLens 통합 | 시각화 디버깅 | 중간 | 중간 |
| 7 | Milvus 지원 추가 | 엔터프라이즈 대규모 | 중간 | 중간 |
| 8 | LanceDB 지원 추가 | 멀티모달, 엣지 AI | 중간 | 중간 |
| 9 | pgvector 지원 추가 | PostgreSQL 통합 | 낮음 | 중간 |

#### 🚀 Phase 3: 고급 기능 (6개월+)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 10 | RAPTOR 계층적 인덱싱 | QuALITY 20% 향상 | 높음 | 높음 |
| 11 | Self-RAG 구현 | 적응형 검색 | 높음 | 중간 |
| 12 | GraphRAG (Microsoft) | 지식 그래프 통합 | 높음 | 높음 |
| 13 | Multi-modal RAG | 이미지+텍스트+테이블 | 높음 | 높음 |
| 14 | LangSmith 통합 | End-to-end 모니터링 | 중간 | 중간 |

**프레임워크 선택**
- **LlamaIndex**: RAG 품질 우선 (검색 속도 40% 빠름, 정확도 92% vs 85%)
- **LangChain**: 에이전트 & 오케스트레이션
- **권장**: 하이브리드 (LlamaIndex로 검색, LangChain으로 워크플로우)

**성능 목표**
- Recall@10: >85%
- Faithfulness: >0.9 (환각 최소화)
- Answer Relevancy: >0.85
- 쿼리 레이턴시: <2초 (end-to-end)

---

### 5. LLM Providers & Agents

#### 현재 상태
- ✅ **충분한 커버리지**: OpenAI, Anthropic, Google, Cohere, Mistral 등 주요 프로바이더 지원
- ✅ **Agent 프레임워크**: LangChain, LlamaIndex 통합 가능
- ⚠️ **선택적 추가 고려**: 신규 프로바이더

#### 중요 발견사항

**신규 LLM 프로바이더 (2024-2025)**
- xAI Grok 4, Mistral Pixtral Large, DeepSeek-V3, Perplexity Sonar, Cohere Command A
- 오픈소스: Llama 4, Qwen3, Phi-4, Gemma 3

**Agent 프레임워크 트렌드**
- **CrewAI**: Fortune 500 기업 60% 채택, 강력 추천
- **LangGraph**: 프로덕션급, 상태 관리 강점
- **Microsoft Agent Framework**: 엔터프라이즈급

**핵심 기능**
- **Parallel Tool Calling**: 동시 다중 도구 호출
- **Structured Outputs**: OpenAI strict mode로 100% 스키마 정확도
- **Prompt Caching**: 10배 비용 절감

**업데이트 권장사항** (우선순위 낮음)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 1 | Parallel Tool Calling 구현 | 효율성 향상 | 중간 | 중간 |
| 2 | Structured Outputs 지원 | 100% 정확도 | 낮음 | 중간 |
| 3 | Prompt Caching 지원 | 비용 10배 절감 | 중간 | 높음 |
| 4 | DeepSeek-V3 추가 (선택) | 오픈소스 SOTA | 낮음 | 낮음 |
| 5 | xAI Grok 추가 (선택) | 실시간 데이터 접근 | 낮음 | 낮음 |

---

### 6. Document Loaders 🔥 **주요 형식 누락**

#### 현재 상태
- ✅ **지원 형식**: Text, PDF (5개 엔진), CSV, Directory, Image
- ❌ **누락 형식**: Microsoft Office, HTML, Jupyter, JSON/XML, Email

#### 중요 발견사항

**필수 누락 형식**
1. **Microsoft Office** (DOCX, XLSX, PPTX) - 가장 치명적 누락
2. **HTML** (웹 콘텐츠) - 웹 스크래핑 필수
3. **Jupyter Notebook** (.ipynb) - 데이터 과학/개발자
4. **JSON/XML** - 구조화 데이터
5. **Email** (.eml, .msg) - 비즈니스 문서

**최적 솔루션**
- **IBM Docling (2024)**: Microsoft Office 통합 솔루션 (DOCX + XLSX + PPTX)
  - 97.9% 정확도
  - Layout 분석, 표 추출, OCR
  - MIT 라이선스

**업데이트 권장사항** (우선순위 순)

#### 🔥 Phase 1: 필수 (1-2개월)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 1 | **Docling (Office 통합)** | DOCX/XLSX/PPTX, 97.9% 정확도 | 중간 | **매우 높음** |
| 2 | **HTMLLoader** | Trafilatura → Readability → BeautifulSoup fallback | 낮음 | **높음** |
| 3 | **JupyterLoader** | nbformat 사용 | 낮음 | **높음** |

**구현 예시**
```python
# Docling (Office 통합)
from beanllm.domain.loaders import DoclingLoader

loader = DoclingLoader()
docs = loader.load("report.docx")  # DOCX, XLSX, PPTX 모두 지원

# HTML Multi-tier Fallback
from beanllm.domain.loaders import HTMLLoader

loader = HTMLLoader(
    fallback_chain=["trafilatura", "readability", "beautifulsoup"]
)
docs = loader.load("https://example.com/article")

# Jupyter Notebook
from beanllm.domain.loaders import JupyterLoader

loader = JupyterLoader(include_outputs=True)
docs = loader.load("analysis.ipynb")
```

#### ⚡ Phase 2: 확장 (3-4개월)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 4 | JSON/XML Loaders | 구조화 데이터 | 낮음 | 중간 |
| 5 | EmailLoader | .eml, .msg 지원 | 중간 | 중간 |
| 6 | Jina AI Reader | 웹 스크래핑 (무료 API) | 낮음 | 중간 |

#### 🚀 Phase 3: 클라우드 & 고급 (6개월+)

| 순위 | 항목 | 이유 | 난이도 | 영향도 |
|-----|------|------|-------|-------|
| 7 | Notion Loader | 클라우드 문서 | 중간 | 낮음 |
| 8 | Google Drive Loader | 클라우드 스토리지 | 중간 | 낮음 |
| 9 | Database Loaders | SQL, MongoDB 등 | 높음 | 중간 |

---

## 🎯 종합 우선순위 로드맵

### 🔥 Critical (즉시 시작, 1-2개월)

**RAG 기초 구축** - 가장 시급
1. Hybrid Search 구현 (BM25 + Dense)
2. Reranker 추가 (BGE reranker-v2-m3)
3. RAGAS 평가 통합

**Document Loaders 보강** - 매우 중요
4. Docling 추가 (Office 파일)
5. HTMLLoader 추가 (Multi-tier fallback)
6. JupyterLoader 추가

**Vision 업데이트**
7. SAM 3 업그레이드
8. YOLOv12 업그레이드

**Embeddings 업데이트**
9. Voyage AI v3 추가
10. Jina AI v3 업데이트

**예상 효과**
- RAG 품질: **40-50% 향상** (Hybrid Search + Reranking)
- 문서 지원: **Microsoft Office 커버리지 100%**
- 비전 성능: **SAM 2배, YOLO 2% mAP 향상**

### ⚡ High Priority (3-4개월)

**RAG 최적화**
11. HyDE 쿼리 확장
12. Position Engineering
13. TruLens 통합
14. Milvus, LanceDB, pgvector 추가

**Embeddings 확장**
15. Qwen3-Embedding-8B 추가
16. Matryoshka 지원 구현
17. Code 임베딩 추가

**Vision 확장**
18. Qwen3-VL 추가 (VLM)
19. EVA-CLIP 추가
20. DINOv2 추가

**Audio/STT 강화**
21. Canary Qwen 2.5B 추가
22. SenseVoice-Small 추가

**Document Loaders 확장**
23. JSON/XML Loaders
24. EmailLoader
25. Jina AI Reader

**예상 효과**
- RAG 정확도: **추가 20% 향상**
- 다국어 지원: **119개 언어** (Qwen3)
- STT 성능: **15배 빠른 속도** (SenseVoice)

### 🚀 Medium Priority (6-12개월)

**고급 RAG**
26. RAPTOR 계층적 인덱싱
27. Self-RAG 구현
28. GraphRAG (Microsoft)
29. Multi-modal RAG

**Vision 고급 기능**
30. InternVL3.5 추가 (대형 VLM)
31. PaliGemma 2 추가 (Google VLM)
32. Depth Anything V2 추가
33. RF-DETR 추가

**Embeddings 고급 기능**
34. Binary/int8 Quantization
35. 한국어 모델 추가 (KURE, KoE5)

**Audio/STT 추가**
36. Granite Speech 8B 추가

**Document Loaders 클라우드**
37. Notion, Google Drive Loaders
38. Database Loaders

**LLM/Agent 개선**
39. Parallel Tool Calling
40. Structured Outputs
41. Prompt Caching

**예상 효과**
- RAG 고급 쿼리: **40-50% 성능 향상** (RAPTOR, GraphRAG)
- 멀티모달: **이미지+텍스트+비디오 통합**
- 비용: **10배 절감** (Prompt Caching)

---

## 📈 Quick Wins vs Long-term Investments

### ⚡ Quick Wins (낮은 난이도, 높은 영향도)

| 항목 | 난이도 | 영향도 | 예상 시간 | ROI |
|------|-------|-------|----------|-----|
| **Reranker 추가 (BGE v2-m3)** | 낮음 | 매우 높음 | 1주 | ⭐⭐⭐⭐⭐ |
| **RAGAS 통합** | 낮음 | 높음 | 1주 | ⭐⭐⭐⭐⭐ |
| **HTMLLoader 추가** | 낮음 | 높음 | 3일 | ⭐⭐⭐⭐⭐ |
| **JupyterLoader 추가** | 낮음 | 높음 | 2일 | ⭐⭐⭐⭐⭐ |
| **Voyage v3 업데이트** | 낮음 | 높음 | 1일 | ⭐⭐⭐⭐⭐ |
| **Jina v3 업데이트** | 낮음 | 높음 | 1일 | ⭐⭐⭐⭐⭐ |
| **YOLOv12 업그레이드** | 낮음 | 중간 | 2일 | ⭐⭐⭐⭐ |
| **Position Engineering** | 낮음 | 높음 | 1일 | ⭐⭐⭐⭐⭐ |

**추천 순서** (1-2주 내 완료 가능)
1. Reranker 추가 (1주) → **검색 48% 개선**
2. RAGAS 통합 (1주) → **RAG 품질 측정**
3. Voyage/Jina v3 업데이트 (1일) → **임베딩 성능 향상**
4. HTMLLoader (3일) → **웹 콘텐츠 지원**
5. Position Engineering (1일) → **무료 성능 향상**

### 🏗️ Long-term Investments (높은 난이도, 높은 영향도)

| 항목 | 난이도 | 영향도 | 예상 시간 | 전략적 가치 |
|------|-------|-------|----------|-----------|
| **Hybrid Search** | 중간 | 매우 높음 | 2-3주 | ⭐⭐⭐⭐⭐ |
| **Docling (Office)** | 중간 | 매우 높음 | 2주 | ⭐⭐⭐⭐⭐ |
| **Qwen3-VL (VLM)** | 높음 | 높음 | 3-4주 | ⭐⭐⭐⭐⭐ |
| **RAPTOR** | 높음 | 높음 | 4주 | ⭐⭐⭐⭐ |
| **GraphRAG** | 높음 | 높음 | 6주 | ⭐⭐⭐⭐ |
| **Multi-modal RAG** | 높음 | 높음 | 8주 | ⭐⭐⭐⭐⭐ |
| **Binary Quantization** | 높음 | 높음 | 3주 | ⭐⭐⭐⭐ |

---

## 🛠️ 구현 체크리스트

### Phase 1: 기초 (Month 1-2)

#### RAG 기초
- [ ] BM25 검색 구현
- [ ] Hybrid Search 통합 (Dense + BM25)
- [ ] BGE Reranker v2-m3 추가
- [ ] RAGAS 평가 통합
- [ ] Position Engineering 구현

#### Document Loaders
- [ ] Docling 통합 (DOCX, XLSX, PPTX)
- [ ] HTMLLoader (Multi-tier fallback)
- [ ] JupyterLoader (nbformat)

#### Embeddings
- [ ] Voyage AI v3 추가
- [ ] Jina AI v3 업데이트

#### Vision
- [ ] SAM 3 업그레이드
- [ ] YOLOv12 업그레이드

**마일스톤**: RAG 품질 40% 향상, Office 파일 지원

### Phase 2: 최적화 (Month 3-4)

#### RAG 최적화
- [ ] HyDE 쿼리 확장
- [ ] TruLens 통합
- [ ] Milvus 지원
- [ ] LanceDB 지원
- [ ] pgvector 지원

#### Embeddings 확장
- [ ] Qwen3-Embedding-8B
- [ ] Matryoshka 지원 (`dimensions=` 파라미터)
- [ ] Code 임베딩 (Codestral, SFR-Code-7B, voyage-code-3)

#### Vision 확장
- [ ] Qwen3-VL (VLM)
- [ ] EVA-CLIP
- [ ] DINOv2

#### Audio/STT
- [ ] Canary Qwen 2.5B
- [ ] SenseVoice-Small
- [ ] Parakeet TDT V3 업그레이드

#### Document Loaders 확장
- [ ] JSON/XML Loaders
- [ ] EmailLoader
- [ ] Jina AI Reader

**마일스톤**: 다국어 119개 언어, 멀티모달 VLM, STT 15배 빠름

### Phase 3: 고급 기능 (Month 6-12)

#### 고급 RAG
- [ ] RAPTOR 계층적 인덱싱
- [ ] Self-RAG 구현
- [ ] GraphRAG (Microsoft)
- [ ] Multi-modal RAG (이미지, 테이블, 비디오)
- [ ] LangSmith 통합

#### Vision 고급
- [ ] InternVL3.5 (대형 VLM)
- [ ] PaliGemma 2 (Google VLM)
- [ ] Depth Anything V2
- [ ] RF-DETR

#### Embeddings 고급
- [ ] Binary/int8 Quantization (32배 압축)
- [ ] 한국어 모델 (KURE, KoE5, bge-m3-korean)

#### Document Loaders 클라우드
- [ ] Notion Loader
- [ ] Google Drive Loader
- [ ] Database Loaders (SQL, MongoDB)

#### LLM/Agent
- [ ] Parallel Tool Calling
- [ ] Structured Outputs (strict mode)
- [ ] Prompt Caching

**마일스톤**: 엔터프라이즈급 RAG, 비용 10배 절감, 멀티모달 통합

---

## 💰 예상 비용 & 리소스

### 개발 리소스 추정

| Phase | 인력 | 기간 | 총 공수 |
|-------|------|------|---------|
| Phase 1 (기초) | 2명 | 2개월 | 4인월 |
| Phase 2 (최적화) | 2-3명 | 2개월 | 5인월 |
| Phase 3 (고급) | 3-4명 | 6개월 | 20인월 |
| **총계** | - | **10개월** | **29인월** |

### 외부 의존성 비용

| 항목 | 라이선스 | 비용 | 비고 |
|------|---------|------|------|
| Docling | MIT | 무료 | ✅ |
| BGE Reranker v2 | MIT | 무료 | ✅ |
| RAGAS | Apache 2.0 | 무료 | ✅ |
| Cohere Rerank 4 | 상용 | $0.50/1K requests | 선택적 |
| LangSmith | 상용 | $39/월~ | 선택적 |
| Voyage v3 API | 상용 | $0.12/1M tokens | 기존 |
| Jina v3 API | 상용 | $0.02/1M tokens | 기존 |

**대부분 오픈소스/무료** → 인프라 비용 최소화

### 인프라 비용

| 항목 | 스펙 | 월 비용 | 비고 |
|------|------|---------|------|
| GPU 서버 (개발/테스트) | A100 40GB | $500-1000 | 선택적 (로컬 모델용) |
| 벡터 DB (Managed) | Pinecone/Milvus | $0-500 | 스케일에 따라 |
| **총계** | - | **$0-1500/월** | 최소 설정 가능 |

---

## 📚 참고 문서

### 상세 기술 조사 문서

1. **Text Embeddings**: `docs/TEXT_EMBEDDING_SURVEY_2024_2025.md` (작성 완료)
2. **Audio/STT**: `docs/AUDIO_STT_SURVEY_2024_2025.md` (작성 완료)
3. **Vision**: `docs/VISION_TECHNOLOGY_SURVEY_2024_2025.md` (작성 완료)
4. **RAG/Retrieval**: `docs/RAG_TECHNOLOGY_SURVEY_2024_2025.md` (작성 완료)
5. **LLM/Agents**: `docs/LLM_AGENT_SURVEY_2024_2025.md` (작성 완료)
6. **Document Loaders**: `docs/DOCUMENT_LOADERS_SURVEY_2024_2025.md` (작성 완료)

### 외부 리소스

#### RAG
- [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) - 고급 RAG 기법
- [microsoft/graphrag](https://github.com/microsoft/graphrag) - GraphRAG 공식
- [RAGAS Official](https://www.ragas.io/) - RAG 평가

#### Embeddings
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - 임베딩 벤치마크
- [Voyage AI v3](https://docs.voyageai.com/) - 최신 API 문서
- [Jina AI v3](https://jina.ai/embeddings/) - 최신 임베딩

#### Vision
- [Papers with Code - Vision](https://paperswithcode.com/area/computer-vision) - 최신 논문
- [GitHub - DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [GitHub - QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

#### Audio/STT
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) - STT 벤치마크
- [GitHub - nvidia/Canary](https://github.com/NVIDIA/NeMo) - Canary 모델

#### Document Loaders
- [IBM Docling](https://github.com/DS4SD/docling) - Office 파일 파서
- [Jina AI Reader](https://jina.ai/reader/) - 웹 스크래핑

---

## 🎯 결론 및 권장 시작 순서

### 즉시 시작 (Week 1-2) - Quick Wins

```
1일차: Voyage v3, Jina v3 업데이트 (1일)
2-3일차: HTMLLoader, JupyterLoader 추가 (2일)
4-8일차: Reranker 추가 (BGE v2-m3) (1주)
9-13일차: RAGAS 통합 (1주)
14일차: Position Engineering (1일)
```

**예상 효과**: 검색 48% 개선, RAG 품질 측정 가능

### 1개월 목표 - 기초 완성

```
Week 3-4: Hybrid Search 구현 (BM25 + Dense) (2주)
Week 5-6: Docling 추가 (Office 파일) (2주)
Week 7-8: SAM 3, YOLOv12 업그레이드 (2주)
```

**예상 효과**: RAG 품질 60% 향상, Office 파일 100% 커버

### 3개월 목표 - 최적화 완료

```
Month 2: RAG 최적화 (HyDE, TruLens, 벡터 DB 확장)
Month 3: Embeddings 확장 (Qwen3, Matryoshka, Code)
         Vision 확장 (Qwen3-VL, EVA-CLIP, DINOv2)
         Audio/STT 강화 (Canary Qwen 2.5B, SenseVoice)
```

**예상 효과**: 다국어 119개 언어, VLM 지원, STT 15배 빠름

### 12개월 목표 - 엔터프라이즈급

```
Month 6-12: 고급 RAG (RAPTOR, GraphRAG, Multi-modal)
            고급 Vision (InternVL3.5, PaliGemma 2)
            고급 Embeddings (Binary Quantization, 한국어)
            클라우드 연동 (Notion, Google Drive, Database)
            프로덕션 최적화 (Prompt Caching, 모니터링)
```

**예상 효과**: 엔터프라이즈급 RAG, 비용 10배 절감, 멀티모달 통합

---

**최종 권장사항**: **RAG 기초 구축 (Hybrid Search + Reranking + RAGAS)**과 **Office 파일 지원 (Docling)**을 최우선으로 시작하고, 점진적으로 확장하는 것이 가장 효율적인 접근입니다.

**문서 버전**: 1.0
**최종 업데이트**: 2025-12-31
**작성자**: beanLLM Development Team
