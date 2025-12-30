"""
OCR A/B 테스트 도구

여러 OCR 설정으로 동시에 실행하고 결과를 비교하여 최적 설정을 찾습니다.

Features:
- 여러 설정으로 동시 실행
- 결과 비교 테이블
- 최적 설정 추천 (신뢰도, 속도, 텍스트 길이 기준)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from PIL import Image

from .bean_ocr import beanOCR
from .models import OCRConfig, OCRResult

logger = logging.getLogger(__name__)


class OCRExperiment:
    """
    OCR A/B 테스트 도구

    여러 OCR 설정으로 실행하고 결과를 비교하여 최적 설정을 찾습니다.

    Features:
    - 여러 설정으로 동시 실행
    - 결과 비교 테이블 출력
    - 최적 설정 추천

    Example:
        ```python
        from beanllm.domain.ocr import OCRExperiment, OCRConfig, beanOCR

        exp = OCRExperiment(beanOCR())

        # 여러 설정으로 실험
        results = exp.run_experiments(
            "document.jpg",
            configs=[
                OCRConfig(denoise=True, binarize=False),
                OCRConfig(denoise=False, binarize=True),
                OCRConfig(denoise=True, binarize=True),
            ]
        )

        # 결과 비교
        exp.compare_results(results)

        # 최적 설정 추천
        best_config = exp.get_best_config(results, metric='confidence')
        print(f"Best config: {best_config}")
        ```
    """

    def __init__(self, ocr: Optional[beanOCR] = None):
        """
        실험 도구 초기화

        Args:
            ocr: beanOCR 인스턴스 (없으면 기본 생성)
        """
        self.ocr = ocr or beanOCR()

    def run_experiments(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        configs: List[OCRConfig],
        labels: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        여러 설정으로 OCR 실험 실행

        Args:
            image: 입력 이미지
            configs: OCR 설정 리스트
            labels: 설정 라벨 (없으면 "Config 1", "Config 2", ...)

        Returns:
            List[Dict]: 실험 결과 리스트
                [
                    {
                        "label": "Config 1",
                        "config": OCRConfig(...),
                        "result": OCRResult(...),
                        "processing_time": 1.23,
                        "text_length": 1234,
                        "line_count": 56,
                        "avg_confidence": 0.92,
                    },
                    ...
                ]

        Example:
            ```python
            results = exp.run_experiments(
                "document.jpg",
                configs=[config1, config2, config3]
            )
            ```
        """
        if labels is None:
            labels = [f"Config {i+1}" for i in range(len(configs))]

        if len(labels) != len(configs):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of configs ({len(configs)})"
            )

        results = []

        for label, config in zip(labels, configs):
            logger.info(f"Running experiment: {label}")
            start_time = time.time()

            # OCR 실행 (config 임시 교체)
            original_config = self.ocr.config
            self.ocr.config = config

            try:
                result = self.ocr.recognize(image)
            finally:
                self.ocr.config = original_config

            processing_time = time.time() - start_time

            # 결과 정리
            results.append({
                "label": label,
                "config": config,
                "result": result,
                "processing_time": processing_time,
                "text_length": len(result.text),
                "line_count": len(result.lines),
                "avg_confidence": result.confidence,
            })

        return results

    def compare_results(
        self,
        results: List[Dict],
        show_text: bool = False,
        max_text_preview: int = 50,
    ) -> None:
        """
        실험 결과 비교 테이블 출력

        Args:
            results: run_experiments() 결과
            show_text: 인식된 텍스트 미리보기 표시 여부
            max_text_preview: 텍스트 미리보기 최대 길이

        Example:
            ```python
            exp.compare_results(results)
            # Output:
            # ┌──────────┬────────┬────────┬────────┬────────┐
            # │  Label   │ Length │ Lines  │ Conf   │ Time   │
            # ├──────────┼────────┼────────┼────────┼────────┤
            # │ Config 1 │  1234  │   56   │ 0.92   │ 1.23s  │
            # │ Config 2 │  1189  │   54   │ 0.88   │ 0.98s  │
            # │ Config 3 │  1256  │   58   │ 0.95 ⭐│ 1.45s  │
            # └──────────┴────────┴────────┴────────┴────────┘
            ```
        """
        if not results:
            print("No results to compare")
            return

        # 최고 성능 찾기
        best_confidence_idx = max(range(len(results)), key=lambda i: results[i]["avg_confidence"])
        best_speed_idx = min(range(len(results)), key=lambda i: results[i]["processing_time"])

        # 테이블 헤더
        print("\n" + "=" * 90)
        print(" " * 30 + "OCR EXPERIMENT RESULTS")
        print("=" * 90)

        header = f"{'Label':<20} | {'Length':>8} | {'Lines':>6} | {'Confidence':>11} | {'Time':>8}"
        print(header)
        print("-" * 90)

        # 결과 행
        for idx, r in enumerate(results):
            label = r["label"]
            length = r["text_length"]
            lines = r["line_count"]
            conf = r["avg_confidence"]
            time_val = r["processing_time"]

            # 최고 성능 표시
            conf_str = f"{conf:.2%}"
            if idx == best_confidence_idx:
                conf_str += " ⭐"

            time_str = f"{time_val:.2f}s"
            if idx == best_speed_idx:
                time_str += " ⚡"

            row = f"{label:<20} | {length:>8} | {lines:>6} | {conf_str:>11} | {time_str:>8}"
            print(row)

        print("=" * 90)

        # 텍스트 미리보기
        if show_text:
            print("\n" + "-" * 90)
            print("TEXT PREVIEW:")
            print("-" * 90)
            for r in results:
                text_preview = r["result"].text[:max_text_preview]
                if len(r["result"].text) > max_text_preview:
                    text_preview += "..."
                print(f"\n[{r['label']}]")
                print(text_preview)
            print("-" * 90)

    def get_best_config(
        self,
        results: List[Dict],
        metric: Literal["confidence", "speed", "text_length"] = "confidence",
    ) -> OCRConfig:
        """
        최적 설정 추천

        Args:
            results: run_experiments() 결과
            metric: 평가 기준
                - "confidence": 신뢰도 기준 (높을수록 좋음)
                - "speed": 처리 속도 기준 (빠를수록 좋음)
                - "text_length": 텍스트 길이 기준 (길수록 좋음)

        Returns:
            OCRConfig: 최적 설정

        Example:
            ```python
            best_config = exp.get_best_config(results, metric='confidence')
            ```
        """
        if not results:
            raise ValueError("No results to analyze")

        if metric == "confidence":
            best_idx = max(range(len(results)), key=lambda i: results[i]["avg_confidence"])
        elif metric == "speed":
            best_idx = min(range(len(results)), key=lambda i: results[i]["processing_time"])
        elif metric == "text_length":
            best_idx = max(range(len(results)), key=lambda i: results[i]["text_length"])
        else:
            raise ValueError(
                f"Invalid metric: {metric}. "
                f"Must be one of: confidence, speed, text_length"
            )

        best_result = results[best_idx]
        logger.info(
            f"Best config by {metric}: {best_result['label']} "
            f"(confidence={best_result['avg_confidence']:.2%}, "
            f"time={best_result['processing_time']:.2f}s)"
        )

        return best_result["config"]

    def get_detailed_comparison(self, results: List[Dict]) -> Dict:
        """
        상세 비교 통계

        Args:
            results: run_experiments() 결과

        Returns:
            Dict: 상세 통계
                {
                    "best_confidence": {...},
                    "best_speed": {...},
                    "best_text_length": {...},
                    "avg_confidence": 0.90,
                    "avg_speed": 1.2,
                }

        Example:
            ```python
            stats = exp.get_detailed_comparison(results)
            print(f"Avg confidence: {stats['avg_confidence']:.2%}")
            ```
        """
        if not results:
            return {}

        # 최고 성능
        best_conf_idx = max(range(len(results)), key=lambda i: results[i]["avg_confidence"])
        best_speed_idx = min(range(len(results)), key=lambda i: results[i]["processing_time"])
        best_length_idx = max(range(len(results)), key=lambda i: results[i]["text_length"])

        # 평균
        avg_confidence = sum(r["avg_confidence"] for r in results) / len(results)
        avg_speed = sum(r["processing_time"] for r in results) / len(results)
        avg_length = sum(r["text_length"] for r in results) / len(results)

        return {
            "best_confidence": results[best_conf_idx],
            "best_speed": results[best_speed_idx],
            "best_text_length": results[best_length_idx],
            "avg_confidence": avg_confidence,
            "avg_speed": avg_speed,
            "avg_text_length": avg_length,
            "total_experiments": len(results),
        }

    def __repr__(self) -> str:
        return f"OCRExperiment(engine={self.ocr.config.engine})"
