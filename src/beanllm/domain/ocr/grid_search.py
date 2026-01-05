"""
OCR Grid Search Tuner

파라미터 조합을 자동으로 테스트하여 최적 설정을 찾습니다.

Features:
- 파라미터 그리드 정의
- 모든 조합 자동 테스트
- 최적 설정 추천
- 진행률 표시
"""

import itertools
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from PIL import Image

from .bean_ocr import beanOCR
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


class GridSearchTuner:
    """
    OCR 파라미터 Grid Search 튜너

    파라미터 그리드를 정의하면 모든 조합을 자동으로 테스트하여
    최적 설정을 찾아줍니다.

    Features:
    - 파라미터 그리드 자동 탐색
    - 진행률 표시
    - 최적 설정 추천
    - 결과 비교 테이블

    Example:
        ```python
        from beanllm.domain.ocr import GridSearchTuner, beanOCR

        tuner = GridSearchTuner(beanOCR())

        # 파라미터 그리드 정의
        best_config, results = tuner.search(
            image="document.jpg",
            param_grid={
                'denoise_strength': ['light', 'medium', 'strong'],
                'clip_limit': [1.5, 2.0, 2.5, 3.0],
                'binarize': [True, False],
            },
            metric='confidence'
        )

        print(f"Best config: {best_config}")
        # → 자동으로 3×4×2 = 24가지 조합 테스트
        ```
    """

    def __init__(self, ocr: Optional[beanOCR] = None, verbose: bool = True):
        """
        Grid Search 튜너 초기화

        Args:
            ocr: beanOCR 인스턴스 (없으면 기본 생성)
            verbose: 진행률 표시 여부
        """
        self.ocr = ocr or beanOCR()
        self.verbose = verbose

    def search(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        param_grid: Dict[str, List[Any]],
        metric: Literal["confidence", "speed", "text_length"] = "confidence",
        n_top: int = 5,
    ) -> tuple[OCRConfig, List[Dict]]:
        """
        Grid Search 실행

        Args:
            image: 입력 이미지
            param_grid: 파라미터 그리드
                {
                    'denoise_strength': ['light', 'medium', 'strong'],
                    'clip_limit': [1.5, 2.0, 2.5],
                    'binarize': [True, False],
                    'binarize_method': ['otsu', 'adaptive'],
                    ...
                }
            metric: 평가 기준 (confidence/speed/text_length)
            n_top: 상위 N개 결과 출력

        Returns:
            (best_config, all_results):
                - best_config: 최적 설정
                - all_results: 모든 실험 결과 리스트

        Example:
            ```python
            best_config, results = tuner.search(
                "document.jpg",
                param_grid={
                    'denoise_strength': ['medium', 'strong'],
                    'clip_limit': [2.0, 3.0],
                },
                metric='confidence'
            )
            ```
        """
        # 파라미터 조합 생성
        param_combinations = self._generate_combinations(param_grid)
        total_combinations = len(param_combinations)

        if self.verbose:
            print("=" * 80)
            print(f"🔍 OCR Grid Search: {total_combinations} combinations")
            print("=" * 80)
            print(f"Parameters: {list(param_grid.keys())}")
            print(f"Metric: {metric}")
            print("=" * 80)

        # 각 조합 테스트
        results = []
        for idx, params in enumerate(param_combinations, 1):
            if self.verbose:
                print(f"\n[{idx}/{total_combinations}] Testing: {self._format_params(params)}")

            # OCRConfig 생성
            config = self._params_to_config(params)

            # OCR 실행
            start_time = time.time()
            original_config = self.ocr.config
            self.ocr.config = config

            try:
                result = self.ocr.recognize(image)
            except Exception as e:
                logger.error(f"Error with params {params}: {e}")
                continue
            finally:
                self.ocr.config = original_config

            processing_time = time.time() - start_time

            # 결과 저장
            result_dict = {
                "params": params,
                "config": config,
                "result": result,
                "confidence": result.confidence,
                "processing_time": processing_time,
                "text_length": len(result.text),
                "line_count": len(result.lines),
            }
            results.append(result_dict)

            if self.verbose:
                print(
                    f"  → Confidence: {result.confidence:.2%}, "
                    f"Time: {processing_time:.2f}s, "
                    f"Length: {len(result.text)}"
                )

        # 결과 정렬
        if metric == "confidence":
            results.sort(key=lambda x: x["confidence"], reverse=True)
        elif metric == "speed":
            results.sort(key=lambda x: x["processing_time"])
        elif metric == "text_length":
            results.sort(key=lambda x: x["text_length"], reverse=True)

        # Top N 결과 출력
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🏆 Top {n_top} Results (by {metric})")
            print("=" * 80)

            for idx, r in enumerate(results[:n_top], 1):
                print(f"\n#{idx}: {self._format_params(r['params'])}")
                print(f"   Confidence: {r['confidence']:.2%}")
                print(f"   Time: {r['processing_time']:.2f}s")
                print(f"   Text Length: {r['text_length']}")

            print("\n" + "=" * 80)

        # 최적 설정 반환
        best_config = results[0]["config"] if results else self.ocr.config

        if self.verbose:
            print("\n✅ Best configuration found!")
            print(f"   {self._format_params(results[0]['params'])}")
            print(f"   Confidence: {results[0]['confidence']:.2%}")

        return best_config, results

    def _generate_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        파라미터 그리드에서 모든 조합 생성

        Args:
            param_grid: 파라미터 그리드

        Returns:
            List[Dict]: 파라미터 조합 리스트
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        # 모든 조합 생성 (Cartesian product)
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _params_to_config(self, params: Dict[str, Any]) -> OCRConfig:
        """
        파라미터 dict를 OCRConfig로 변환

        Args:
            params: 파라미터 dict

        Returns:
            OCRConfig: OCR 설정
        """
        # 기본 설정
        config_kwargs = {
            "engine": params.get("engine", "paddleocr"),
            "language": params.get("language", "auto"),
        }

        # Denoise
        denoise_enabled = params.get("denoise", True)
        denoise_strength = params.get("denoise_strength", "medium")
        config_kwargs["denoise"] = denoise_enabled
        config_kwargs["denoise_config"] = DenoiseConfig(
            enabled=denoise_enabled,
            strength=denoise_strength,
        )

        # Contrast
        contrast_enabled = params.get("contrast", True)
        clip_limit = params.get("clip_limit", 2.0)
        config_kwargs["contrast_adjustment"] = contrast_enabled
        config_kwargs["contrast_config"] = ContrastConfig(
            enabled=contrast_enabled,
            clip_limit=clip_limit,
        )

        # Binarize
        binarize_enabled = params.get("binarize", False)
        binarize_method = params.get("binarize_method", "otsu")
        threshold = params.get("threshold", 127)
        config_kwargs["binarize"] = binarize_enabled
        config_kwargs["binarize_config"] = BinarizeConfig(
            enabled=binarize_enabled,
            method=binarize_method,
            threshold=threshold,
        )

        # Deskew
        deskew_enabled = params.get("deskew", True)
        angle_threshold = params.get("angle_threshold", 0.5)
        config_kwargs["deskew"] = deskew_enabled
        config_kwargs["deskew_config"] = DeskewConfig(
            enabled=deskew_enabled,
            angle_threshold=angle_threshold,
        )

        # Sharpen
        sharpen_enabled = params.get("sharpen", False)
        sharpen_strength = params.get("sharpen_strength", 0.5)
        config_kwargs["sharpen"] = sharpen_enabled
        config_kwargs["sharpen_config"] = SharpenConfig(
            enabled=sharpen_enabled,
            strength=sharpen_strength,
        )

        # Resize
        max_size = params.get("max_size", None)
        config_kwargs["resize_config"] = ResizeConfig(
            enabled=(max_size is not None),
            max_size=max_size,
        )

        return OCRConfig(**config_kwargs)

    def _format_params(self, params: Dict[str, Any]) -> str:
        """파라미터를 읽기 쉽게 포맷"""
        items = [f"{k}={v}" for k, v in params.items()]
        return ", ".join(items)

    def compare_results(
        self,
        results: List[Dict],
        top_n: int = 10,
    ) -> None:
        """
        Grid Search 결과 비교 테이블 출력

        Args:
            results: search() 결과
            top_n: 상위 N개만 출력

        Example:
            ```python
            best_config, results = tuner.search(...)
            tuner.compare_results(results, top_n=5)
            ```
        """
        if not results:
            print("No results to compare")
            return

        print("\n" + "=" * 100)
        print(" " * 40 + "GRID SEARCH RESULTS")
        print("=" * 100)

        header = f"{'Rank':<6} | {'Confidence':>11} | {'Time':>8} | {'Length':>8} | {'Parameters':<50}"
        print(header)
        print("-" * 100)

        for idx, r in enumerate(results[:top_n], 1):
            conf = r["confidence"]
            time_val = r["processing_time"]
            length = r["text_length"]
            params_str = self._format_params(r["params"])[:48]

            # 1등 표시
            rank_str = f"#{idx}"
            if idx == 1:
                rank_str += " 🏆"

            row = f"{rank_str:<6} | {conf:>10.2%} | {time_val:>7.2f}s | {length:>8} | {params_str}"
            print(row)

        print("=" * 100)

    def export_best_config(
        self,
        best_config: OCRConfig,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        최적 설정을 프리셋으로 저장

        Args:
            best_config: 최적 설정
            save_path: 저장 경로 (없으면 ~/.beanllm/ocr_presets/grid_search_best.json)

        Example:
            ```python
            best_config, _ = tuner.search(...)
            tuner.export_best_config(best_config, "best_receipt_config")
            ```
        """
        from .presets import ConfigPresets

        presets = ConfigPresets()

        if save_path:
            preset_name = Path(save_path).stem
        else:
            preset_name = "grid_search_best"

        presets.save(preset_name, best_config)
        print(f"✅ Best config saved as preset: '{preset_name}'")

    def __repr__(self) -> str:
        return f"GridSearchTuner(engine={self.ocr.config.engine})"
