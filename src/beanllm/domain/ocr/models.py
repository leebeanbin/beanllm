"""
OCR 데이터 모델

OCR 결과와 설정을 위한 데이터 클래스 정의.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

__all__ = [
    "BoundingBox",
    "OCRTextLine",
    "OCRResult",
    "DenoiseConfig",
    "ContrastConfig",
    "BinarizeConfig",
    "DeskewConfig",
    "SharpenConfig",
    "ResizeConfig",
    "OCRConfig",
]


@dataclass
class BoundingBox:
    """
    텍스트 영역의 경계 상자 (Bounding Box)

    좌표계: 이미지 좌상단이 (0, 0)

    Attributes:
        x0: 좌상단 X 좌표
        y0: 좌상단 Y 좌표
        x1: 우하단 X 좌표
        y1: 우하단 Y 좌표
        confidence: 영역 감지 신뢰도 (0.0-1.0)

    Example:
        ```python
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50, confidence=0.95)
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0
        ```
    """

    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        """경계 상자 너비"""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """경계 상자 높이"""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """경계 상자 면적"""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """경계 상자 중심점 (x, y)"""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x0={self.x0:.1f}, y0={self.y0:.1f}, "
            f"x1={self.x1:.1f}, y1={self.y1:.1f}, conf={self.confidence:.2f})"
        )


@dataclass
class OCRTextLine:
    """
    OCR로 인식된 텍스트 라인

    한 줄의 텍스트와 위치, 신뢰도 정보를 포함합니다.

    Attributes:
        text: 인식된 텍스트 내용
        bbox: 텍스트 영역의 경계 상자
        confidence: 텍스트 인식 신뢰도 (0.0-1.0)
        language: 텍스트 언어 (예: "ko", "en", "zh", "ja")

    Example:
        ```python
        line = OCRTextLine(
            text="안녕하세요",
            bbox=BoundingBox(10, 20, 100, 50, 0.95),
            confidence=0.92,
            language="ko"
        )
        print(f"Text: {line.text}, Confidence: {line.confidence:.2%}")
        ```
    """

    text: str
    bbox: BoundingBox
    confidence: float
    language: str = "en"

    def __repr__(self) -> str:
        return f"OCRTextLine(text='{self.text[:20]}...', conf={self.confidence:.2f})"


@dataclass
class OCRResult:
    """
    OCR 인식 결과

    전체 OCR 결과와 메타데이터를 포함합니다.

    Attributes:
        text: 전체 텍스트 (라인별 텍스트를 합친 결과)
        lines: 라인별 OCR 결과 리스트
        language: 인식된 언어
        confidence: 평균 신뢰도 (0.0-1.0)
        engine: 사용된 OCR 엔진 이름
        processing_time: 처리 시간 (초)
        metadata: 추가 메타데이터 딕셔너리

    Example:
        ```python
        result = OCRResult(
            text="안녕하세요\\n반갑습니다",
            lines=[line1, line2],
            language="ko",
            confidence=0.92,
            engine="PaddleOCR",
            processing_time=1.23,
            metadata={"llm_corrected": True}
        )
        print(f"Text: {result.text}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Engine: {result.engine}")
        ```
    """

    text: str
    lines: List[OCRTextLine]
    language: str
    confidence: float
    engine: str
    processing_time: float
    metadata: Dict = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        """인식된 라인 수"""
        return len(self.lines)

    @property
    def average_line_confidence(self) -> float:
        """라인별 평균 신뢰도"""
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    @property
    def low_confidence_lines(self, threshold: float = 0.7) -> List[OCRTextLine]:
        """신뢰도가 낮은 라인 목록 (기본값: 0.7 미만)"""
        return [line for line in self.lines if line.confidence < threshold]

    def __repr__(self) -> str:
        return (
            f"OCRResult(engine={self.engine}, lang={self.language}, "
            f"lines={len(self.lines)}, conf={self.confidence:.2f})"
        )


@dataclass
class DenoiseConfig:
    """
    노이즈 제거 세부 설정

    Attributes:
        enabled: 노이즈 제거 활성화
        gaussian_kernel: Gaussian blur 커널 크기 (기본: (3, 3))
        median_kernel: Median filter 커널 크기 (기본: 3)
        strength: 노이즈 제거 강도 (light/medium/strong)
    """
    enabled: bool = True
    gaussian_kernel: tuple[int, int] = (3, 3)
    median_kernel: int = 3
    strength: Literal["light", "medium", "strong"] = "medium"


@dataclass
class ContrastConfig:
    """
    대비 조정 세부 설정 (CLAHE)

    Attributes:
        enabled: 대비 조정 활성화
        clip_limit: CLAHE clip limit (기본: 2.0, 높을수록 강함)
        tile_grid_size: CLAHE tile grid 크기 (기본: (8, 8))
    """
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


@dataclass
class BinarizeConfig:
    """
    이진화 세부 설정

    Attributes:
        enabled: 이진화 활성화
        method: 이진화 방법 (otsu/adaptive/manual)
        threshold: 수동 임계값 (method='manual'일 때, 0-255)
        block_size: Adaptive 블록 크기 (method='adaptive'일 때, 홀수)
        c: Adaptive 상수 (method='adaptive'일 때)
    """
    enabled: bool = True
    method: Literal["otsu", "adaptive", "manual"] = "otsu"
    threshold: int = 127
    block_size: int = 11
    c: int = 2


@dataclass
class DeskewConfig:
    """
    기울기 보정 세부 설정

    Attributes:
        enabled: 기울기 보정 활성화
        angle_threshold: 보정 각도 임계값 (degrees, 이 값 이상만 보정)
    """
    enabled: bool = True
    angle_threshold: float = 0.5


@dataclass
class SharpenConfig:
    """
    선명화 세부 설정

    Attributes:
        enabled: 선명화 활성화
        strength: 선명화 강도 (0.0-1.0, 기본: 0.5)
    """
    enabled: bool = True
    strength: float = 0.5


@dataclass
class ResizeConfig:
    """
    크기 조정 세부 설정

    Attributes:
        enabled: 크기 조정 활성화
        max_size: 최대 크기 (픽셀, None이면 조정 안 함)
        interpolation: 보간 방법 (area/linear/cubic)
    """
    enabled: bool = True
    max_size: Optional[int] = None
    interpolation: Literal["area", "linear", "cubic"] = "area"


@dataclass
class OCRConfig:
    """
    OCR 설정

    OCR 엔진 선택, 언어 설정, 전처리/후처리 옵션을 포함합니다.

    Attributes:
        engine: OCR 엔진 선택
            - "paddleocr": PaddleOCR (메인, 90-96% 정확도)
            - "easyocr": EasyOCR (대체)
            - "trocr": TrOCR (손글씨 전문)
            - "nougat": Nougat (학술 논문, 수식)
            - "surya": Surya (복잡한 레이아웃)
            - "tesseract": Tesseract 5.x (Fallback)
            - "cloud": Cloud API (Google Vision, AWS Textract 등)
            - "qwen2vl-2b/7b/72b": Qwen2.5-VL (오픈소스 최고 성능, 2024-2025)
            - "minicpm": MiniCPM-o 2.6 (OCRBench 1위, 2024-2025)
            - "deepseek-ocr": DeepSeek-OCR (토큰 압축, 효율적, 2024-2025)

        language: 언어 설정
            - "auto": 자동 감지
            - "ko": 한국어
            - "en": 영어
            - "zh": 중국어
            - "ja": 일본어
            - 기타 80+ languages

        use_gpu: GPU 사용 여부 (기본: True)
        confidence_threshold: 최소 신뢰도 임계값 (기본: 0.5)

        전처리 옵션:
        - enable_preprocessing: 전처리 활성화 (기본: True)
        - denoise: 노이즈 제거
        - contrast_adjustment: 대비 조정 (CLAHE)
        - rotation_correction: 회전 보정

        후처리 옵션:
        - enable_llm_postprocessing: LLM 후처리 활성화 (기본: False)
        - llm_model: LLM 모델 (예: "gpt-4o-mini")
        - spell_check: 맞춤법 검사
        - grammar_check: 문법 검사

    Example:
        ```python
        # 기본 설정
        config = OCRConfig(engine="paddleocr", language="ko")

        # 고급 설정
        config = OCRConfig(
            engine="paddleocr",
            language="ko",
            use_gpu=True,
            enable_preprocessing=True,
            denoise=True,
            contrast_adjustment=True,
            enable_llm_postprocessing=True,
            llm_model="gpt-4o-mini"
        )

        # 손글씨 전용
        config = OCRConfig(engine="trocr", language="en", use_gpu=True)
        ```
    """

    # 엔진 설정
    engine: str = "paddleocr"
    language: str = "auto"
    use_gpu: bool = True
    confidence_threshold: float = 0.5

    # 전처리 옵션 (레거시 호환성)
    enable_preprocessing: bool = True
    denoise: bool = True
    contrast_adjustment: bool = True
    binarize: bool = True
    deskew: bool = True
    sharpen: bool = False

    # 전처리 세부 설정 (고급)
    denoise_config: Optional[DenoiseConfig] = None
    contrast_config: Optional[ContrastConfig] = None
    binarize_config: Optional[BinarizeConfig] = None
    deskew_config: Optional[DeskewConfig] = None
    sharpen_config: Optional[SharpenConfig] = None
    resize_config: Optional[ResizeConfig] = None

    # 후처리 옵션
    enable_llm_postprocessing: bool = False
    llm_model: Optional[str] = None
    spell_check: bool = False
    grammar_check: bool = False

    # 고급 옵션
    batch_size: int = 1
    max_image_size: Optional[int] = None  # 최대 이미지 크기 (픽셀)
    output_format: str = "text"  # text, json, markdown

    def __post_init__(self):
        """설정 유효성 검증 및 기본값 초기화"""
        # 엔진 유효성 검사
        valid_engines = {
            "paddleocr",
            "easyocr",
            "trocr",
            "nougat",
            "surya",
            "tesseract",
            "cloud",
            "cloud-google",
            "cloud-aws",
            "qwen2vl",
            "qwen2vl-2b",
            "qwen2vl-7b",
            "qwen2vl-72b",
            "minicpm",
            "deepseek-ocr",
        }
        if self.engine not in valid_engines:
            raise ValueError(
                f"Invalid engine: {self.engine}. "
                f"Must be one of {valid_engines}"
            )

        # 언어 유효성 검사 (일부만 체크)
        if self.language not in ["auto", "ko", "en", "zh", "ja"]:
            # 경고만 출력 (80+ languages 지원하므로)
            import warnings

            warnings.warn(
                f"Language '{self.language}' may not be supported by all engines. "
                f"Common languages: auto, ko, en, zh, ja"
            )

        # 신뢰도 임계값 범위 검사
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # LLM 후처리 설정 검증
        if self.enable_llm_postprocessing and not self.llm_model:
            raise ValueError(
                "llm_model must be specified when enable_llm_postprocessing is True"
            )

        # 세부 설정 초기화 (레거시 bool 필드 기반)
        if self.denoise_config is None:
            self.denoise_config = DenoiseConfig(enabled=self.denoise)
        if self.contrast_config is None:
            self.contrast_config = ContrastConfig(enabled=self.contrast_adjustment)
        if self.binarize_config is None:
            self.binarize_config = BinarizeConfig(enabled=self.binarize)
        if self.deskew_config is None:
            self.deskew_config = DeskewConfig(enabled=self.deskew)
        if self.sharpen_config is None:
            self.sharpen_config = SharpenConfig(enabled=self.sharpen)
        if self.resize_config is None:
            self.resize_config = ResizeConfig(
                enabled=(self.max_image_size is not None),
                max_size=self.max_image_size
            )

    def __repr__(self) -> str:
        return (
            f"OCRConfig(engine={self.engine}, lang={self.language}, "
            f"gpu={self.use_gpu}, preprocess={self.enable_preprocessing}, "
            f"llm_postprocess={self.enable_llm_postprocessing})"
        )
