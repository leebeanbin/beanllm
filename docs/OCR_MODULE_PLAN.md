# beanOCR 모듈 구현 계획

**작성일**: 2025-12-30
**상태**: 계획 단계
**예상 기간**: 2주

---

## 🎯 목표

스캔된 문서, 이미지 기반 PDF를 고품질 텍스트로 변환하는 OCR 모듈 구현

**핵심 가치**:
- 90-96% 정확도 (PaddleOCR 기준)
- 다국어 지원 (한글, 중국어, 일본어 최적화)
- 7개 엔진 선택 가능 (용도별 최적화)
- LLM 후처리로 98%+ 정확도
- Hybrid 전략으로 95% 비용 절감

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           beanOCR (Facade)              │
│  - 사용자 친화적 API                     │
│  - 자동 엔진 선택                        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      OCR Engine Manager                 │
│  - 7개 엔진 관리                        │
│  - Fallback 처리                        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Preprocessing Pipeline               │
│  - 이미지 전처리                        │
│  - 노이즈 제거, 대비 조정               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     OCR Engines (7개)                   │
│  - PaddleOCR (메인)                     │
│  - EasyOCR (대체)                       │
│  - TrOCR (손글씨)                       │
│  - Nougat (학술)                        │
│  - Surya (복잡한 레이아웃)              │
│  - Tesseract 5.x (Fallback)             │
│  - Cloud API (대체)                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Postprocessing Pipeline              │
│  - LLM 오류 수정                        │
│  - 맞춤법 검사                          │
│  - 품질 검증                            │
└─────────────────────────────────────────┘
```

---

## 📦 Phase 1: 핵심 구조 (Week 1)

### TODO-OCR-101: 기본 인터페이스 및 모델

**예상 시간**: 4시간

```python
# src/beanllm/domain/ocr/__init__.py
from .bean_ocr import beanOCR
from .models import OCRResult, OCRConfig

__all__ = ["beanOCR", "OCRResult", "OCRConfig"]
```

```python
# src/beanllm/domain/ocr/models.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BoundingBox:
    """텍스트 영역 좌표"""
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0

@dataclass
class OCRTextLine:
    """OCR로 인식된 텍스트 라인"""
    text: str
    bbox: BoundingBox
    confidence: float
    language: str = "en"

@dataclass
class OCRResult:
    """OCR 결과"""
    text: str  # 전체 텍스트
    lines: List[OCRTextLine]  # 라인별 정보
    language: str
    confidence: float  # 평균 신뢰도
    engine: str  # 사용된 엔진
    processing_time: float
    metadata: dict = field(default_factory=dict)

@dataclass
class OCRConfig:
    """OCR 설정"""
    engine: str = "paddleocr"  # paddleocr, easyocr, trrocr, nougat, surya, tesseract
    language: str = "auto"  # auto, ko, zh, ja, en
    use_gpu: bool = True
    enable_preprocessing: bool = True
    enable_llm_postprocessing: bool = False
    llm_model: Optional[str] = None
    confidence_threshold: float = 0.5
    # 전처리 옵션
    denoise: bool = True
    contrast_adjustment: bool = True
    rotation_correction: bool = True
    # 후처리 옵션
    spell_check: bool = False
    grammar_check: bool = False
```

---

### TODO-OCR-102: beanOCR 메인 클래스

**예상 시간**: 6시간

```python
# src/beanllm/domain/ocr/bean_ocr.py
class beanOCR:
    """
    통합 OCR 인터페이스

    Example:
        ```python
        from beanllm.domain.ocr import beanOCR

        # 기본 사용
        ocr = beanOCR(engine="paddleocr", language="ko")
        result = ocr.recognize("scanned_image.jpg")
        print(result.text)

        # LLM 후처리 활성화
        ocr = beanOCR(
            engine="paddleocr",
            enable_llm_postprocessing=True,
            llm_model="gpt-4o-mini"
        )
        result = ocr.recognize("noisy_image.jpg")

        # PDF 페이지 OCR
        result = ocr.recognize_pdf_page(pdf_path, page_num=0)
        ```
    """

    def __init__(self, config: Optional[OCRConfig] = None, **kwargs):
        self.config = config or OCRConfig(**kwargs)
        self._engine = None
        self._preprocessor = None
        self._postprocessor = None
        self._init_components()

    def _init_components(self):
        """컴포넌트 초기화"""
        # 엔진 초기화
        self._engine = self._create_engine(self.config.engine)

        # 전처리기
        if self.config.enable_preprocessing:
            self._preprocessor = ImagePreprocessor()

        # 후처리기
        if self.config.enable_llm_postprocessing:
            self._postprocessor = LLMPostprocessor(
                model=self.config.llm_model
            )

    def recognize(self, image_or_path, **kwargs) -> OCRResult:
        """
        이미지 OCR 인식

        Args:
            image_or_path: 이미지 경로 또는 numpy array
            **kwargs: 추가 옵션

        Returns:
            OCRResult
        """
        start_time = time.time()

        # 1. 이미지 로드
        image = self._load_image(image_or_path)

        # 2. 전처리
        if self._preprocessor:
            image = self._preprocessor.process(image, self.config)

        # 3. OCR 실행
        raw_result = self._engine.recognize(image, self.config)

        # 4. 후처리
        if self._postprocessor:
            raw_result = self._postprocessor.process(raw_result, self.config)

        # 5. OCRResult 생성
        result = OCRResult(
            text=raw_result["text"],
            lines=raw_result["lines"],
            language=raw_result.get("language", self.config.language),
            confidence=raw_result["confidence"],
            engine=self.config.engine,
            processing_time=time.time() - start_time,
            metadata=raw_result.get("metadata", {}),
        )

        return result

    def recognize_pdf_page(self, pdf_path, page_num: int) -> OCRResult:
        """PDF 페이지 OCR"""
        # PyMuPDF로 페이지 → 이미지 변환
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)  # 고해상도
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        doc.close()

        return self.recognize(image)

    def batch_recognize(self, images: List, **kwargs) -> List[OCRResult]:
        """배치 OCR"""
        results = []
        for img in images:
            result = self.recognize(img, **kwargs)
            results.append(result)
        return results
```

---

## 🚀 Phase 2: OCR 엔진 구현 (Week 1-2)

### TODO-OCR-201: PaddleOCR 엔진 (메인)

**우선순위**: P0
**예상 시간**: 8시간

```python
# src/beanllm/domain/ocr/engines/paddleocr_engine.py
class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR 엔진 (메인)

    Features:
    - 90-96% 정확도
    - 빠른 처리 속도
    - 다국어 지원 (80+ languages)
    - GPU 가속
    """

    def __init__(self):
        super().__init__(name="PaddleOCR")
        self._check_dependencies()
        self._init_ocr()

    def _check_dependencies(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR is required. "
                "Install it with: pip install paddleocr"
            )

    def _init_ocr(self):
        from paddleocr import PaddleOCR
        # 언어별 모델 초기화 (lazy loading)
        self._models = {}

    def recognize(self, image, config: OCRConfig) -> dict:
        """PaddleOCR 실행"""
        from paddleocr import PaddleOCR

        # 언어별 모델 선택
        lang = config.language if config.language != "auto" else "ch"
        if lang not in self._models:
            self._models[lang] = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=config.use_gpu,
                show_log=False,
            )

        # OCR 실행
        result = self._models[lang].ocr(image, cls=True)

        # 결과 변환
        return self._convert_result(result, config)

    def _convert_result(self, raw_result, config) -> dict:
        """PaddleOCR 결과 → 표준 형식"""
        lines = []
        text_parts = []

        for line_data in raw_result[0]:
            bbox_coords, (text, confidence) = line_data

            # BoundingBox 생성
            bbox = BoundingBox(
                x0=bbox_coords[0][0],
                y0=bbox_coords[0][1],
                x1=bbox_coords[2][0],
                y1=bbox_coords[2][1],
                confidence=confidence,
            )

            # OCRTextLine 생성
            if confidence >= config.confidence_threshold:
                line = OCRTextLine(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    language=config.language,
                )
                lines.append(line)
                text_parts.append(text)

        full_text = "\n".join(text_parts)
        avg_confidence = sum(l.confidence for l in lines) / len(lines) if lines else 0.0

        return {
            "text": full_text,
            "lines": lines,
            "confidence": avg_confidence,
            "language": config.language,
        }
```

**다국어 최적화**:
```python
# 언어별 모델 설정
LANGUAGE_MODELS = {
    "ko": "korean",  # 한글
    "zh": "ch",      # 중국어
    "ja": "japan",   # 일본어
    "en": "en",      # 영어
}

# CJK 언어 전처리 최적화
def optimize_for_cjk(image, language):
    if language in ["ko", "zh", "ja"]:
        # 해상도 증가 (CJK는 세밀함)
        image = increase_resolution(image, factor=1.5)
        # 대비 강화
        image = enhance_contrast(image, method="CLAHE")
    return image
```

---

### TODO-OCR-202: 대체 엔진 구현

**우선순위**: P1
**예상 시간**: 각 2-4시간

1. **EasyOCR** (대체 엔진)
   - PaddleOCR와 유사한 성능
   - Fallback 용도

2. **TrOCR** (손글씨 전문)
   - Transformer 기반
   - 손글씨 90%+ 정확도

3. **Nougat** (학술 논문)
   - 수식, 표 특화
   - LaTeX 변환

4. **Surya** (복잡한 레이아웃)
   - 2024년 최신 모델
   - 다단, 복잡한 구조

5. **Tesseract 5.x** (Fallback)
   - 오픈소스
   - 안정성

---

## 🔧 Phase 3: 전처리 & 후처리 (Week 2)

### TODO-OCR-301: 이미지 전처리 파이프라인

**예상 시간**: 6시간

```python
# src/beanllm/domain/ocr/preprocessing.py
class ImagePreprocessor:
    """
    OCR 전처리 파이프라인

    Features:
    - 노이즈 제거
    - 대비 조정
    - 회전 보정
    - 이진화
    - 해상도 최적화
    """

    def process(self, image, config: OCRConfig):
        """전처리 실행"""
        if config.denoise:
            image = self.denoise(image)

        if config.contrast_adjustment:
            image = self.adjust_contrast(image)

        if config.rotation_correction:
            image = self.correct_rotation(image)

        image = self.binarize(image)
        image = self.optimize_resolution(image)

        return image

    def denoise(self, image):
        """노이즈 제거 (Non-local Means Denoising)"""
        import cv2
        return cv2.fastNlMeansDenoisingColored(image)

    def adjust_contrast(self, image):
        """대비 조정 (CLAHE)"""
        import cv2
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def correct_rotation(self, image):
        """회전 보정 (Hough Transform)"""
        # Skew 각도 감지 및 보정
        pass

    def binarize(self, image):
        """이진화 (Otsu's method)"""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
```

---

### TODO-OCR-302: LLM 후처리

**예상 시간**: 8시간

```python
# src/beanllm/domain/ocr/postprocessing.py
class LLMPostprocessor:
    """
    LLM 기반 OCR 후처리

    Features:
    - 오타 수정
    - 문맥 기반 보정
    - 맞춤법 검사
    - 98%+ 정확도
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        from ...facade.client import Client
        self.llm = Client(model=model)

    async def process(self, ocr_result: dict, config: OCRConfig) -> dict:
        """LLM 후처리"""
        original_text = ocr_result["text"]

        # LLM에 오류 수정 요청
        prompt = f"""
다음 OCR 결과에서 오타를 수정해주세요.
원본 의미를 유지하면서 맞춤법과 문법을 교정하세요.

원본 OCR 결과:
{original_text}

수정된 텍스트만 출력하세요:
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 낮은 온도로 일관성 유지
        )

        corrected_text = response.content.strip()

        # 신뢰도 향상
        ocr_result["text"] = corrected_text
        ocr_result["confidence"] = min(ocr_result["confidence"] + 0.1, 1.0)
        ocr_result["metadata"]["llm_corrected"] = True

        return ocr_result
```

---

## 💰 Phase 4: Hybrid 전략 (비용 절감)

### TODO-OCR-401: Hybrid OCR 전략

**예상 시간**: 4시간

```python
class HybridOCRStrategy:
    """
    Local + Cloud Hybrid 전략

    Features:
    - 로컬 OCR 우선 (무료)
    - 신뢰도 낮으면 Cloud API (유료)
    - 95% 비용 절감
    """

    def __init__(self, local_engine="paddleocr", cloud_api="google_vision"):
        self.local_ocr = beanOCR(engine=local_engine)
        self.cloud_ocr = CloudOCRClient(api=cloud_api)

    async def recognize(self, image, min_confidence=0.85):
        # 1. 로컬 OCR 시도
        local_result = self.local_ocr.recognize(image)

        # 2. 신뢰도 체크
        if local_result.confidence >= min_confidence:
            return local_result  # 로컬 결과 사용 (무료)

        # 3. 신뢰도 낮으면 Cloud API
        cloud_result = await self.cloud_ocr.recognize(image)
        return cloud_result  # Cloud 결과 사용 (유료, 하지만 5%만)
```

---

## 📊 성능 목표

| 항목 | 목표 |
|------|------|
| 정확도 (일반 문서) | 90-96% |
| 정확도 (LLM 후처리) | 98%+ |
| 처리 속도 (GPU) | ~1초/페이지 |
| 다국어 지원 | 80+ languages |
| 한글 정확도 | 95%+ |
| 비용 절감 (Hybrid) | 95% |

---

## 🧪 테스트 계획

1. **단위 테스트** (80개 예상)
   - 각 엔진별 기본 기능
   - 전처리 파이프라인
   - 후처리 LLM

2. **통합 테스트**
   - 다국어 문서
   - 손글씨 문서
   - 학술 논문

3. **성능 테스트**
   - 정확도 벤치마크
   - 처리 속도
   - GPU vs CPU

---

## 📦 의존성

```toml
# pyproject.toml
[project.optional-dependencies]
ocr = [
    "paddleocr>=2.7.0",
    "easyocr>=1.7.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
]

ocr-full = [
    "paddleocr>=2.7.0",
    "easyocr>=1.7.0",
    "transformers>=4.35.0",  # TrOCR, Nougat
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "pytesseract>=0.3.10",  # Tesseract
    "surya-ocr>=0.4.0",  # Surya
]
```

---

## 🗓️ 구현 일정

| Week | Task | Hours |
|------|------|-------|
| Week 1 | Phase 1-2 (핵심 + PaddleOCR) | 20h |
| Week 2 | Phase 2-3 (대체 엔진 + 전후처리) | 24h |
| Week 3 | Phase 4 + 테스트 | 16h |

**Total**: ~60 hours (2-3주)
