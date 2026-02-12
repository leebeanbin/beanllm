"""
OCR Router

Optical Character Recognition endpoints.
Uses Python best practices: context managers, type hints.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from utils.file_upload import save_upload_to_temp, temp_directory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ocr", tags=["OCR"])


# ============================================================================
# Response Models
# ============================================================================


class BoundingBox(BaseModel):
    """Bounding box for text region"""

    x: int
    y: int
    width: int
    height: int


class OCRLine(BaseModel):
    """Single line of OCR result"""

    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None


class OCRResponse(BaseModel):
    """Response from OCR recognition"""

    text: str
    confidence: float
    processing_time: float
    engine: str
    language: str
    num_lines: int
    lines: List[OCRLine] = Field(default_factory=list)


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_bbox(bbox: Any) -> Optional[Dict[str, int]]:
    """Extract bounding box using duck typing"""
    if bbox is None:
        return None

    return {
        "x": getattr(bbox, "x", 0),
        "y": getattr(bbox, "y", 0),
        "width": getattr(bbox, "width", 0),
        "height": getattr(bbox, "height", 0),
    }


def _extract_lines(lines: List[Any]) -> List[Dict[str, Any]]:
    """Extract OCR lines using duck typing"""
    if not lines:
        return []

    return [
        {
            "text": getattr(line, "text", str(line)),
            "confidence": getattr(line, "confidence", 0.0),
            "bbox": _extract_bbox(getattr(line, "bbox", None)),
        }
        for line in lines
    ]


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/recognize", response_model=OCRResponse)
async def ocr_recognize(
    file: UploadFile = File(..., description="Image file to recognize"),
    engine: str = "paddleocr",
    language: str = "auto",
    use_gpu: bool = True,
    confidence_threshold: float = 0.5,
    enable_preprocessing: bool = True,
    denoise: bool = True,
    contrast_adjustment: bool = True,
    binarize: bool = True,
    deskew: bool = True,
    sharpen: bool = False,
    enable_llm_postprocessing: bool = False,
    llm_model: Optional[str] = None,
    spell_check: bool = False,
    grammar_check: bool = False,
    max_image_size: Optional[int] = None,
    output_format: str = "text",
) -> OCRResponse:
    """
    Perform OCR on an uploaded image.

    Supported engines:
    - paddleocr: Fast, accurate (default)
    - easyocr: Multi-language support
    - tesseract: Classic engine
    - trocr: Transformer-based
    - qwen2vl-2b: Vision-language model
    """
    try:
        from beanllm.domain.ocr import OCRConfig, beanOCR

        # Allowed image extensions for OCR
        ocr_extensions = frozenset([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"])

        async with temp_directory() as temp_dir:
            # Save with sanitized filename + streaming + size check
            tmp_path = await save_upload_to_temp(
                file,
                temp_dir,
                allowed_extensions=ocr_extensions,
                fallback_ext=".jpg",
            )

            # Create OCR config
            ocr_config = OCRConfig(
                engine=engine,
                language=language,
                use_gpu=use_gpu,
                confidence_threshold=confidence_threshold,
                enable_preprocessing=enable_preprocessing,
                denoise=denoise,
                contrast_adjustment=contrast_adjustment,
                binarize=binarize,
                deskew=deskew,
                sharpen=sharpen,
                enable_llm_postprocessing=enable_llm_postprocessing,
                llm_model=llm_model,
                spell_check=spell_check,
                grammar_check=grammar_check,
                max_image_size=max_image_size,
                output_format=output_format,
            )

            # Run OCR
            ocr = beanOCR(config=ocr_config)
            result = ocr.recognize(str(tmp_path))

            # Extract lines using duck typing
            lines_data = _extract_lines(getattr(result, "lines", []))
            lines = [
                OCRLine(
                    text=line["text"],
                    confidence=line["confidence"],
                    bbox=BoundingBox(**line["bbox"]) if line["bbox"] else None,
                )
                for line in lines_data
            ]

            return OCRResponse(
                text=getattr(result, "text", ""),
                confidence=getattr(result, "confidence", 0.0),
                processing_time=getattr(result, "processing_time", 0.0),
                engine=getattr(result, "engine", engine),
                language=getattr(result, "language", language),
                num_lines=len(lines),
                lines=lines,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR error: {e}", exc_info=True)
        raise HTTPException(500, f"OCR processing failed: {str(e)}")


@router.get("/engines")
async def ocr_list_engines() -> Dict[str, Any]:
    """List available OCR engines with their capabilities"""
    return {
        "engines": [
            {
                "id": "paddleocr",
                "name": "PaddleOCR",
                "description": "Fast and accurate, supports multiple languages",
                "gpu_support": True,
            },
            {
                "id": "easyocr",
                "name": "EasyOCR",
                "description": "Easy to use, good multi-language support",
                "gpu_support": True,
            },
            {
                "id": "tesseract",
                "name": "Tesseract",
                "description": "Classic OCR engine, widely used",
                "gpu_support": False,
            },
            {
                "id": "trocr",
                "name": "TrOCR",
                "description": "Transformer-based, high accuracy",
                "gpu_support": True,
            },
            {
                "id": "qwen2vl-2b",
                "name": "Qwen2-VL",
                "description": "Vision-language model, contextual understanding",
                "gpu_support": True,
            },
        ]
    }
