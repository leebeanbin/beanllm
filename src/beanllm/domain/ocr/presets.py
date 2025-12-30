"""
OCR 설정 프리셋 관리

문서 타입별 최적 OCR 설정을 제공하고 커스텀 프리셋을 관리.

Features:
- 문서 타입별 최적 프리셋 (영수증, 명함, 스캔 문서 등)
- 커스텀 프리셋 저장/로드 (JSON)
- 프리셋 목록 조회
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    BinarizeConfig,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    OCRConfig,
    ResizeConfig,
    SharpenConfig,
)

logger = logging.getLogger(__name__)


class ConfigPresets:
    """
    OCR 설정 프리셋 관리자

    문서 타입별 최적 설정과 커스텀 프리셋을 관리합니다.

    Features:
    - 내장 프리셋 (receipt, business_card, scanned_document, handwriting, academic_paper)
    - 커스텀 프리셋 저장/로드
    - 프리셋 목록 조회

    Example:
        ```python
        from beanllm.domain.ocr import ConfigPresets, beanOCR

        presets = ConfigPresets()

        # 영수증 OCR 최적 설정
        config = presets.get('receipt')
        ocr = beanOCR(config=config)
        result = ocr.recognize("receipt.jpg")

        # 커스텀 프리셋 저장
        custom_config = OCRConfig(
            denoise=True,
            contrast_adjustment=True,
            binarize=True
        )
        presets.save('my_preset', custom_config)

        # 커스텀 프리셋 로드
        loaded_config = presets.load('my_preset')

        # 프리셋 목록
        print(presets.list())  # ['receipt', 'business_card', ...]
        ```
    """

    def __init__(self, presets_dir: Optional[Path] = None):
        """
        프리셋 관리자 초기화

        Args:
            presets_dir: 커스텀 프리셋 저장 디렉토리 (기본: ~/.beanllm/ocr_presets)
        """
        if presets_dir is None:
            presets_dir = Path.home() / ".beanllm" / "ocr_presets"
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        self._builtin_presets = self._init_builtin_presets()

    def _init_builtin_presets(self) -> Dict[str, OCRConfig]:
        """
        내장 프리셋 초기화

        Returns:
            Dict[str, OCRConfig]: 프리셋 이름 → 설정
        """
        return {
            # 영수증: 노이즈 많고 저품질, 강한 전처리 필요
            "receipt": OCRConfig(
                engine="paddleocr",
                language="auto",
                denoise=True,
                denoise_config=DenoiseConfig(enabled=True, strength="strong"),
                contrast_adjustment=True,
                contrast_config=ContrastConfig(enabled=True, clip_limit=3.0),
                binarize=True,
                binarize_config=BinarizeConfig(enabled=True, method="adaptive"),
                deskew=True,
                sharpen=True,
                sharpen_config=SharpenConfig(enabled=True, strength=0.7),
            ),
            # 명함: 작은 텍스트, 고해상도 필요
            "business_card": OCRConfig(
                engine="paddleocr",
                language="auto",
                denoise=True,
                denoise_config=DenoiseConfig(enabled=True, strength="light"),
                contrast_adjustment=True,
                contrast_config=ContrastConfig(enabled=True, clip_limit=2.0),
                binarize=False,  # 명함은 일반적으로 깨끗해서 이진화 불필요
                deskew=True,
                sharpen=True,
                sharpen_config=SharpenConfig(enabled=True, strength=0.5),
            ),
            # 스캔 문서: 깨끗한 이미지, 최소 전처리
            "scanned_document": OCRConfig(
                engine="paddleocr",
                language="auto",
                denoise=False,
                contrast_adjustment=True,
                contrast_config=ContrastConfig(enabled=True, clip_limit=1.5),
                binarize=False,
                deskew=True,
                deskew_config=DeskewConfig(enabled=True, angle_threshold=0.3),
                sharpen=False,
            ),
            # 손글씨: TrOCR 엔진 + 최소 전처리
            "handwriting": OCRConfig(
                engine="trocr",
                language="en",
                denoise=True,
                denoise_config=DenoiseConfig(enabled=True, strength="light"),
                contrast_adjustment=True,
                binarize=False,  # 손글씨는 이진화하면 오히려 정확도 떨어짐
                deskew=False,  # 손글씨는 기울기 다양
                sharpen=False,
            ),
            # 학술 논문: Nougat 엔진 + LaTeX 출력
            "academic_paper": OCRConfig(
                engine="nougat",
                language="en",
                denoise=False,
                contrast_adjustment=False,
                binarize=False,
                deskew=True,
                sharpen=False,
            ),
            # 저해상도 이미지: 강한 전처리 + 선명화
            "low_quality": OCRConfig(
                engine="paddleocr",
                language="auto",
                denoise=True,
                denoise_config=DenoiseConfig(enabled=True, strength="strong"),
                contrast_adjustment=True,
                contrast_config=ContrastConfig(enabled=True, clip_limit=3.5),
                binarize=True,
                binarize_config=BinarizeConfig(enabled=True, method="adaptive"),
                deskew=True,
                sharpen=True,
                sharpen_config=SharpenConfig(enabled=True, strength=1.0),
            ),
            # 복잡한 레이아웃: Surya 엔진
            "complex_layout": OCRConfig(
                engine="surya",
                language="auto",
                denoise=True,
                denoise_config=DenoiseConfig(enabled=True, strength="medium"),
                contrast_adjustment=True,
                binarize=False,
                deskew=True,
                sharpen=False,
            ),
        }

    def get(self, preset_name: str) -> OCRConfig:
        """
        프리셋 가져오기 (내장 또는 커스텀)

        Args:
            preset_name: 프리셋 이름

        Returns:
            OCRConfig: OCR 설정

        Raises:
            ValueError: 프리셋을 찾을 수 없음

        Example:
            ```python
            presets = ConfigPresets()
            config = presets.get('receipt')
            ```
        """
        # 내장 프리셋 확인
        if preset_name in self._builtin_presets:
            return self._builtin_presets[preset_name]

        # 커스텀 프리셋 확인
        preset_path = self.presets_dir / f"{preset_name}.json"
        if preset_path.exists():
            return self.load(preset_name)

        raise ValueError(
            f"Preset '{preset_name}' not found. "
            f"Available presets: {self.list()}"
        )

    def save(self, preset_name: str, config: OCRConfig) -> None:
        """
        커스텀 프리셋 저장

        Args:
            preset_name: 프리셋 이름
            config: OCR 설정

        Example:
            ```python
            custom_config = OCRConfig(
                denoise=True,
                contrast_adjustment=True
            )
            presets.save('my_preset', custom_config)
            ```
        """
        preset_path = self.presets_dir / f"{preset_name}.json"

        # OCRConfig를 JSON 직렬화 가능한 dict로 변환
        config_dict = self._config_to_dict(config)

        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Preset '{preset_name}' saved to {preset_path}")

    def load(self, preset_name: str) -> OCRConfig:
        """
        커스텀 프리셋 로드

        Args:
            preset_name: 프리셋 이름

        Returns:
            OCRConfig: OCR 설정

        Raises:
            FileNotFoundError: 프리셋 파일을 찾을 수 없음

        Example:
            ```python
            config = presets.load('my_preset')
            ```
        """
        preset_path = self.presets_dir / f"{preset_name}.json"

        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset '{preset_name}' not found at {preset_path}"
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return self._dict_to_config(config_dict)

    def list(self) -> List[str]:
        """
        사용 가능한 프리셋 목록 (내장 + 커스텀)

        Returns:
            List[str]: 프리셋 이름 리스트

        Example:
            ```python
            presets = ConfigPresets()
            print(presets.list())
            # ['receipt', 'business_card', 'scanned_document', ...]
            ```
        """
        # 내장 프리셋
        builtin = list(self._builtin_presets.keys())

        # 커스텀 프리셋
        custom = [
            p.stem
            for p in self.presets_dir.glob("*.json")
        ]

        return sorted(set(builtin + custom))

    def delete(self, preset_name: str) -> None:
        """
        커스텀 프리셋 삭제

        Args:
            preset_name: 프리셋 이름

        Raises:
            ValueError: 내장 프리셋은 삭제 불가
            FileNotFoundError: 프리셋 파일을 찾을 수 없음

        Example:
            ```python
            presets.delete('my_preset')
            ```
        """
        if preset_name in self._builtin_presets:
            raise ValueError(
                f"Cannot delete builtin preset '{preset_name}'"
            )

        preset_path = self.presets_dir / f"{preset_name}.json"

        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset '{preset_name}' not found at {preset_path}"
            )

        preset_path.unlink()
        logger.info(f"Preset '{preset_name}' deleted")

    def _config_to_dict(self, config: OCRConfig) -> dict:
        """OCRConfig를 dict로 변환 (JSON 직렬화용)"""
        # dataclass를 dict로 변환
        import dataclasses

        return dataclasses.asdict(config)

    def _dict_to_config(self, config_dict: dict) -> OCRConfig:
        """dict를 OCRConfig로 변환"""
        # 세부 설정 복원
        if config_dict.get("denoise_config"):
            config_dict["denoise_config"] = DenoiseConfig(**config_dict["denoise_config"])
        if config_dict.get("contrast_config"):
            config_dict["contrast_config"] = ContrastConfig(**config_dict["contrast_config"])
        if config_dict.get("binarize_config"):
            config_dict["binarize_config"] = BinarizeConfig(**config_dict["binarize_config"])
        if config_dict.get("deskew_config"):
            config_dict["deskew_config"] = DeskewConfig(**config_dict["deskew_config"])
        if config_dict.get("sharpen_config"):
            config_dict["sharpen_config"] = SharpenConfig(**config_dict["sharpen_config"])
        if config_dict.get("resize_config"):
            config_dict["resize_config"] = ResizeConfig(**config_dict["resize_config"])

        return OCRConfig(**config_dict)

    def __repr__(self) -> str:
        return f"ConfigPresets(presets_dir={self.presets_dir}, available={len(self.list())})"
