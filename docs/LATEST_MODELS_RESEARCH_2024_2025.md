# 최신 모델 리서치 (2024-2025)

beanLLM의 각 도메인에 적용 가능한 최신 모델과 프레임워크 조사 결과입니다.

---

## 1. OCR (광학 문자 인식) ✅ 완료

### 현재 상태
- **기존 엔진 (7개)**: PaddleOCR, EasyOCR, TrOCR, Nougat, Surya, Tesseract, Cloud API
- **신규 추가 (3개)**: Qwen2.5-VL, MiniCPM-o 2.6, DeepSeek-OCR

### 최신 모델 (2024-2025)
| 모델 | 파라미터 | 특징 | 성능 | 상태 |
|------|----------|------|------|------|
| MiniCPM-o 2.6 | 8B | OCRBench 1위, GPT-4o 능가 | 96% | ✅ 구현됨 |
| Qwen2.5-VL | 2B/7B/72B | 오픈소스 최고 성능 | 95% | ✅ 구현됨 |
| DeepSeek-OCR | 3B | 토큰 압축, 메모리 효율 | 94% | ✅ 구현됨 |
| GOT-OCR 2.0 | - | 고정밀 OCR | - | ⏳ 향후 고려 |

### Sources
- [Northflank - Best STT Models 2025](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2025-benchmarks)
- [OCRBench Rankings](https://huggingface.co/spaces/mteb/leaderboard)

---

## 2. 텍스트 임베딩 (Text Embeddings)

### 현재 상태
- **구현된 Provider**: OpenAI, Gemini, Voyage, Jina, Mistral, Cohere (모두 API 기반)
- **로컬 모델**: 없음

### 최신 모델 (2024-2025)
| 모델 | 파라미터 | MTEB 점수 | 특징 | 권장도 |
|------|----------|-----------|------|--------|
| NVIDIA NV-Embed | - | 69.32 | MTEB 1위 (2024) | ⭐⭐⭐ |
| SFR-Embedding-Mistral | 7B | - | E5-mistral 기반, 고성능 | ⭐⭐⭐ |
| Alibaba-NLP GTE | 1.5B | - | 컴팩트, 1024-d, Matryoshka | ⭐⭐ |
| Google Gemma Embedding | 300M | - | 100+ 언어, 리소스 제한 환경 | ⭐⭐ |

### 권장 사항
1. **로컬 모델 지원 추가**
   - `NVIDIAEmbedding` 클래스 추가
   - `HuggingFaceEmbedding` 범용 클래스 추가 (SFR, Alibaba, 등)
   - Sentence Transformers 통합

2. **Matryoshka 임베딩 지원**
   - 가변 차원 임베딩 (128d, 256d, 512d, 1024d)

### Sources
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [NVIDIA NV-Embed Blog](https://developer.nvidia.com/blog/nvidia-text-embedding-model-tops-mteb-leaderboard/)
- [Modal - Top MTEB Models](https://modal.com/blog/mteb-leaderboard-article)

---

## 3. 비전 임베딩 (Vision Embeddings)

### 현재 상태
- **구현된 모델**: CLIP (OpenAI)
- **멀티모달**: 기본 MultimodalEmbedding

### 최신 모델 (2024-2025)
| 모델 | 특징 | 성능 | 권장도 |
|------|------|------|--------|
| SigLIP 2 (Google) | 다국어, self-distillation | CLIP 능가 | ⭐⭐⭐ |
| MobileCLIP2 (Apple) | 모바일 최적화, 2x 경량 | SigLIP-SO400M 동급 | ⭐⭐⭐ |
| Voyage-Multimodal-3 | 텍스트+이미지+스크린샷 | 범용성 높음 | ⭐⭐ |
| EVA-CLIP | 고해상도, 정밀 검색 | 우수 | ⭐⭐ |
| AIMv2 | Autoregressive, 멀티모달 | 최신 아키텍처 | ⭐ |

### 권장 사항
1. **SigLIP 2 지원 추가**
   - `SigLIPEmbedding` 클래스 생성
   - 다국어 zero-shot 분류 지원

2. **MobileCLIP2 지원 추가**
   - 모바일/엣지 디바이스용
   - `MobileCLIPEmbedding` 클래스

### Sources
- [SigLIP 2 Blog](https://huggingface.co/blog/siglip2)
- [Top Embedding Models 2025](https://artsmart.ai/blog/top-embedding-models-in-2025/)
- [Voyage Multimodal 3](https://blog.voyageai.com/2024/11/12/voyage-multimodal-3/)

---

## 4. 음성 인식 (Speech Recognition / Audio)

### 현재 상태
- **구현 상태**: Type definitions만 존재 (실제 구현 없음)
- **WhisperModel enum**: 정의만 있음

### 최신 모델 (2024-2025)
| 모델 | 파라미터 | RTFx | WER | 특징 | 권장도 |
|------|----------|------|-----|------|--------|
| Whisper Large V3 Turbo | 809M | - | 7.4% | 6x 빠름, 99+ 언어 | ⭐⭐⭐ |
| Distil-Whisper | 756M | - | ~8% | 6x 빠름, 압축 | ⭐⭐⭐ |
| NVIDIA Parakeet TDT | 1.1B | >2000 | - | 실시간 최적화 | ⭐⭐⭐ |
| Canary-1B | 1B | - | 6.67% | 다국어, 번역 | ⭐⭐ |
| Canary-1B-Flash | 1B | >1000 | - | 초고속 추론 | ⭐⭐ |
| Moonshine | <100M | - | - | 온디바이스, 초경량 | ⭐ |

### 권장 사항
1. **beanSTT 클래스 구현** (OCR과 유사한 구조)
   ```python
   from beanllm.domain.audio import beanSTT

   stt = beanSTT(engine="whisper-v3-turbo", language="ko")
   result = stt.transcribe("audio.mp3")
   ```

2. **지원 엔진**
   - `whisper-v3-turbo`: Whisper Large V3 Turbo
   - `distil-whisper`: Distil-Whisper
   - `parakeet`: NVIDIA Parakeet TDT
   - `canary`: Canary-1B
   - `moonshine`: Moonshine (온디바이스)

### Sources
- [Northflank - Best Open-Source STT 2025](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2025-benchmarks)
- [Modal - Open Source STT](https://modal.com/blog/open-source-stt)
- [AssemblyAI - Top 8 STT Options](https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications)

---

## 5. LLM 평가 (Evaluation)

### 현재 상태
- **구현된 메트릭**: ExactMatch, F1, BLEU, ROUGE, Semantic Similarity, LLMJudge
- **프레임워크**: 자체 구현 Evaluator

### 최신 프레임워크 (2024-2025)
| 프레임워크 | 다운로드 | 특징 | 권장도 |
|------------|----------|------|--------|
| DeepEval | 500K/월 | 14+ 메트릭, RAG/fine-tuning | ⭐⭐⭐ |
| LM Evaluation Harness | - | EleutherAI, CI/CD 파이프라인 | ⭐⭐⭐ |
| Confident AI | - | 최고 메트릭, 프로덕션 | ⭐⭐ |
| Ragas | - | RAG 전문, Faithfulness | ⭐⭐ |
| OpenAI Evals | - | 커뮤니티 기반 | ⭐ |

### 주요 벤치마크
- **기본**: GLUE, SuperGLUE, HellaSwag, MMLU
- **고급**: MMLU-Pro (>90% 넘어선 난이도)
- **특화**: MT-Bench (다중턴), GPQA-Diamond (대학원 수준), ARC-AGI (추론), GAIA (AGI)

### 권장 사항
1. **DeepEval 통합**
   - `DeepEvalMetric` 클래스 추가
   - RAG 평가 메트릭 활용

2. **LM Evaluation Harness 통합**
   - 표준 벤치마크 실행
   - `LMEvalBenchmark` 클래스

3. **벤치마크 실행 유틸리티**
   ```python
   from beanllm.domain.evaluation import run_benchmark

   result = run_benchmark(model, benchmark="mmlu-pro")
   ```

### Sources
- [Top 5 LLM Evaluation Frameworks](https://dev.to/guybuildingai/-top-5-open-source-llm-evaluation-frameworks-in-2024-98m)
- [5 LLM Evaluation Tools 2025](https://humanloop.com/blog/best-llm-evaluation-tools)
- [LLM Benchmarks 2025](https://llm-stats.com/benchmarks)

---

## 6. 파인튜닝 (Fine-tuning)

### 현재 상태
- **구현된 Provider**: OpenAI API 기반만
- **로컬 파인튜닝**: 없음

### 최신 프레임워크 (2024-2025)
| 프레임워크 | 특징 | 강점 | 권장도 |
|------------|------|------|--------|
| Axolotl | 커뮤니티 기반 | 초보자 친화적, multi-GPU | ⭐⭐⭐ |
| Unsloth | 속도 최적화 | single-GPU 최고 속도 | ⭐⭐⭐ |
| Torchtune | PyTorch 네이티브 | PyTorch 통합, 멀티노드 | ⭐⭐⭐ |
| LlamaFactory | 범용성 | 100+ 모델, config 기반 | ⭐⭐⭐ |
| Hugging Face PEFT | 표준 | LoRA/QLoRA 표준 | ⭐⭐ |

### PEFT 기법
- **LoRA**: 1-5% 파라미터만 학습 (Adapter)
- **QLoRA**: 4-bit 양자화 + LoRA (70B를 단일 GPU에서)
- **Spectrum (2024)**: SNR 분석, 상위 30% 레이어만 학습

### 권장 스택 (2025)
```
QLoRA / Spectrum
+ FlashAttention-2
+ Liger Kernels
+ Gradient Checkpointing
```

### 권장 사항
1. **PEFT Provider 추가**
   ```python
   from beanllm.domain.finetuning import PEFTProvider

   provider = PEFTProvider(
       framework="axolotl",
       method="qlora",
       model="meta-llama/Llama-3-8B"
   )
   job = provider.create_job(config)
   ```

2. **지원 프레임워크**
   - Axolotl (초보자, multi-GPU)
   - Unsloth (single-GPU 최적화)
   - LlamaFactory (범용)

### Sources
- [LLM Fine-Tuning Tools 2025](https://labelyourdata.com/articles/llm-fine-tuning/top-llm-tools-for-fine-tuning)
- [Fine-Tune LLMs 2025 Guide](https://www.philschmid.de/fine-tune-llms-in-2025)
- [LoRA vs QLoRA Comparison](https://www.index.dev/blog/top-ai-fine-tuning-tools-lora-vs-qlora-vs-full)

---

## 7. 문서 파싱 (Document Parsing / PDF Loaders)

### 현재 상태
- **구현**: beanPDFLoader (기본)
- **기능**: 테이블, 이미지 추출

### 최신 모델/툴킷 (2024-2025)
| 도구 | 제공자 | 특징 | 권장도 |
|------|--------|------|--------|
| PDF-Extract-Kit | OpenDataLab | DocLayout-YOLO, StructTable-InternVL2 | ⭐⭐⭐ |
| Docling | IBM | DocLayNet, TableFormer, 고정밀 | ⭐⭐⭐ |
| MinerU | - | PDF-Extract-Kit 기반, OCR+Table | ⭐⭐ |
| DocLayout-YOLO | - | GL-CRM, 빠른 레이아웃 검출 | ⭐⭐ |
| LlamaParse | LlamaIndex | 초고속 (~6s), API 기반 | ⭐⭐ |

### VLM 기반 파싱
- GPT-4V, Qwen, InternVL: 멀티모달 end-to-end
- Nougat, Fox, GOT: 문서 전문 VLM

### 권장 사항
1. **PDF-Extract-Kit 통합**
   - DocLayout-YOLO로 레이아웃 검출
   - StructTable-InternVL2로 테이블 인식

2. **Docling 통합**
   - 고정밀 파싱
   - `DoclingLoader` 클래스

3. **beanPDFLoader 고도화**
   ```python
   from beanllm.domain.loaders import beanPDFLoader

   loader = beanPDFLoader(
       "document.pdf",
       engine="docling",  # or "pdf-extract-kit"
       extract_tables=True,
       extract_images=True,
       layout_model="doclayout-yolo"
   )
   docs = loader.load()
   ```

### Sources
- [PDF-Extract-Kit GitHub](https://github.com/opendatalab/PDF-Extract-Kit)
- [PDF Parsing Benchmark 2025](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [Document Parsing Survey 2024](https://arxiv.org/html/2410.21169v4)

---

## 8. 비전 모델 (Object Detection / Segmentation)

### 현재 상태
- **구현**: CLIP 임베딩만
- **고급 비전 기능**: 없음

### 최신 모델 (2024-2025)
| 모델 | 제공자 | 특징 | 권장도 |
|------|--------|------|--------|
| SAM 3 (2025) | Meta | 텍스트 프롬프트, 3D 재구성 | ⭐⭐⭐ |
| Florence-2 | Microsoft | 멀티태스크 VLM, zero-shot | ⭐⭐⭐ |
| YOLOv12 | - | 속도+정확도, real-time | ⭐⭐⭐ |
| Grounding DINO | - | Open-set 검출, 텍스트 기반 | ⭐⭐ |
| RF-DETR | - | 고정밀 검출 | ⭐⭐ |

### 권장 사항
1. **비전 도메인 확장**
   - Object Detection: `beanDetector` 클래스
   - Segmentation: `beanSegmenter` 클래스 (SAM 3 기반)
   - VLM: `beanVision` 범용 클래스 (Florence-2)

2. **사용 예시**
   ```python
   from beanllm.domain.vision import beanDetector, beanSegmenter

   # Object Detection
   detector = beanDetector(model="yolov12")
   results = detector.detect("image.jpg")

   # Segmentation (텍스트 프롬프트)
   segmenter = beanSegmenter(model="sam3")
   masks = segmenter.segment("image.jpg", prompt="person wearing red shirt")
   ```

### Sources
- [SAM 3 Announcement](https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/)
- [Florence-2 Overview](https://www.ultralytics.com/blog/florence-2-microsofts-latest-vision-language-model)
- [Object Detection SOTA 2025](https://hiringnet.com/object-detection-state-of-the-art-models-in-2025/)

---

## 우선순위 권장 사항

### 🔥 즉시 구현 권장 (High Priority)
1. **음성 인식 (Audio/STT)** - 현재 구현 없음, 수요 높음
   - Whisper V3 Turbo, Distil-Whisper, Parakeet 지원

2. **비전 임베딩 업데이트** - SigLIP 2, MobileCLIP2 추가
   - CLIP 대비 성능 향상

3. **PDF 파싱 고도화** - PDF-Extract-Kit, Docling 통합
   - 테이블/레이아웃 검출 정확도 향상

### ⭐ 중요 (Medium Priority)
4. **텍스트 임베딩 로컬 모델** - NVIDIA NV-Embed, SFR 지원
   - API 의존성 감소, 비용 절감

5. **평가 프레임워크 통합** - DeepEval, LM Eval Harness
   - RAG 평가, 표준 벤치마크

### 💡 향후 고려 (Low Priority)
6. **파인튜닝 로컬 지원** - Axolotl, Unsloth
   - 로컬 파인튜닝 수요 있을 시

7. **비전 모델 확장** - SAM 3, Florence-2
   - Object Detection/Segmentation 필요 시

---

## 구현 가이드

### 1단계: 음성 인식 (beanSTT)
```python
# src/beanllm/domain/audio/bean_stt.py
class beanSTT:
    def __init__(self, engine="whisper-v3-turbo", language="auto"):
        self.engine = engine
        self.language = language

    def transcribe(self, audio_path):
        # Whisper/Parakeet/Canary 엔진 선택
        # 오디오 파일 로드
        # 전사 실행
        return TranscriptionResult(...)
```

### 2단계: 비전 임베딩 (SigLIP 2)
```python
# src/beanllm/domain/vision/embeddings/siglip.py
class SigLIPEmbedding(BaseEmbedding):
    def __init__(self, model_name="google/siglip2-so400m-patch14-384"):
        # HuggingFace 모델 로드
        # Processor 초기화

    def embed(self, images, texts=None):
        # 이미지-텍스트 임베딩
        return embeddings
```

### 3단계: PDF 파싱 (PDF-Extract-Kit)
```python
# src/beanllm/domain/loaders/pdf/engines/pdf_extract_kit.py
class PDFExtractKitEngine:
    def __init__(self):
        # DocLayout-YOLO 로드
        # StructTable-InternVL2 로드

    def parse(self, pdf_path):
        # 레이아웃 검출
        # 테이블 추출
        # 구조화된 Document 반환
```

---

## 참고 문헌

### 종합 리소스
- [Awesome LLM Evaluation](https://alopatenko.github.io/LLMEvaluation/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

### 모델 허브
- [Hugging Face](https://huggingface.co/)
- [Model Scope](https://modelscope.cn/)
- [Papers with Code](https://paperswithcode.com/)

---

**생성일**: 2025-12-30
**작성자**: Claude Code
**목적**: beanLLM 도메인별 최신 모델 리서치 및 업데이트 가이드
