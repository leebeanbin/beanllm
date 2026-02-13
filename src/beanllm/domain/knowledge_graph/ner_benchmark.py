"""NER benchmark - NERBenchmark class and quick_benchmark function."""

from __future__ import annotations

from typing import Dict, List, Optional

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_factory import NEREngineFactory
from beanllm.domain.knowledge_graph.ner_models import BenchmarkResult, BenchmarkSample
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


# 샘플 테스트 데이터
SAMPLE_TEST_DATA: List[BenchmarkSample] = [
    BenchmarkSample(
        text="Steve Jobs founded Apple Inc. in Cupertino, California in 1976.",
        entities=[
            {"text": "Steve Jobs", "label": "PERSON", "start": 0, "end": 10},
            {"text": "Apple Inc.", "label": "ORG", "start": 19, "end": 29},
            {"text": "Cupertino", "label": "LOC", "start": 33, "end": 42},
            {"text": "California", "label": "LOC", "start": 44, "end": 54},
            {"text": "1976", "label": "DATE", "start": 58, "end": 62},
        ],
    ),
    BenchmarkSample(
        text="Elon Musk is the CEO of Tesla and SpaceX.",
        entities=[
            {"text": "Elon Musk", "label": "PERSON", "start": 0, "end": 9},
            {"text": "Tesla", "label": "ORG", "start": 24, "end": 29},
            {"text": "SpaceX", "label": "ORG", "start": 34, "end": 40},
        ],
    ),
    BenchmarkSample(
        text="Google was founded by Larry Page and Sergey Brin in 1998.",
        entities=[
            {"text": "Google", "label": "ORG", "start": 0, "end": 6},
            {"text": "Larry Page", "label": "PERSON", "start": 22, "end": 32},
            {"text": "Sergey Brin", "label": "PERSON", "start": 37, "end": 48},
            {"text": "1998", "label": "DATE", "start": 52, "end": 56},
        ],
    ),
    BenchmarkSample(
        text="Microsoft released Windows 11 in October 2021.",
        entities=[
            {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
            {"text": "Windows 11", "label": "PRODUCT", "start": 19, "end": 29},
            {"text": "October 2021", "label": "DATE", "start": 33, "end": 45},
        ],
    ),
    BenchmarkSample(
        text="Amazon Web Services is headquartered in Seattle, Washington.",
        entities=[
            {"text": "Amazon Web Services", "label": "ORG", "start": 0, "end": 19},
            {"text": "Seattle", "label": "LOC", "start": 39, "end": 46},
            {"text": "Washington", "label": "LOC", "start": 48, "end": 58},
        ],
    ),
]


class NERBenchmark:
    """
    NER 엔진 벤치마크

    다양한 NER 엔진의 정확도와 속도를 비교합니다.

    Example:
        ```python
        # 테스트 데이터 준비
        test_data = [
            BenchmarkSample(
                text="Steve Jobs founded Apple Inc. in Cupertino.",
                entities=[
                    {"text": "Steve Jobs", "label": "PERSON", "start": 0, "end": 10},
                    {"text": "Apple Inc.", "label": "ORG", "start": 19, "end": 29},
                    {"text": "Cupertino", "label": "LOC", "start": 33, "end": 42},
                ]
            ),
            # ...
        ]

        # 벤치마크 실행
        benchmark = NERBenchmark(engines=[
            NEREngineFactory.create("spacy"),
            NEREngineFactory.create("huggingface"),
            NEREngineFactory.create("gliner"),
        ])

        results = benchmark.run(test_data)
        print(benchmark.get_report())
        ```
    """

    # 레이블 정규화 매핑
    LABEL_MAPPING = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "LOCATION": "LOCATION",
        "DATE": "DATE",
        "TIME": "DATE",
        "MISC": "OTHER",
        "PRODUCT": "PRODUCT",
        "TECHNOLOGY": "TECHNOLOGY",
        "EVENT": "EVENT",
    }

    def __init__(self, engines: List[BaseNEREngine]) -> None:
        """
        벤치마크 초기화

        Args:
            engines: 테스트할 NER 엔진 목록
        """
        self.engines = engines
        self._results: Dict[str, BenchmarkResult] = {}

    def run(
        self,
        test_data: List[BenchmarkSample],
        normalize_labels: bool = True,
    ) -> Dict[str, BenchmarkResult]:
        """
        벤치마크 실행

        Args:
            test_data: 테스트 데이터
            normalize_labels: 레이블 정규화 여부

        Returns:
            엔진별 벤치마크 결과
        """
        self._results = {}

        for engine in self.engines:
            logger.info(f"Benchmarking: {engine.name}")

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_latency: float = 0.0
            detailed = []

            for sample in test_data:
                # 엔티티 추출
                result = engine.extract_with_timing(sample.text)
                total_latency += result.latency_ms

                # 예측 엔티티 정규화
                predicted = set()
                for ent in result.entities:
                    label = self._normalize_label(ent.label) if normalize_labels else ent.label
                    predicted.add((ent.text.lower(), label))

                # 정답 엔티티 정규화
                gold = set()
                for gold_ent in sample.entities:
                    label = (
                        self._normalize_label(gold_ent["label"])
                        if normalize_labels
                        else gold_ent["label"]
                    )
                    gold.add((gold_ent["text"].lower(), label))

                # TP, FP, FN 계산
                tp = len(predicted & gold)
                fp = len(predicted - gold)
                fn = len(gold - predicted)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                detailed.append(
                    {
                        "text": sample.text[:50] + "...",
                        "predicted": list(predicted),
                        "gold": list(gold),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

            # 메트릭 계산
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            self._results[engine.name] = BenchmarkResult(
                engine_name=engine.name,
                precision=precision,
                recall=recall,
                f1_score=f1,
                avg_latency_ms=total_latency / len(test_data) if test_data else 0,
                total_samples=len(test_data),
                detailed_results=detailed,
            )

        return self._results

    def _normalize_label(self, label: str) -> str:
        """레이블 정규화"""
        return self.LABEL_MAPPING.get(label.upper(), label.upper())

    def get_report(self, format: str = "markdown") -> str:
        """
        벤치마크 리포트 생성

        Args:
            format: "markdown" or "text"
        """
        if not self._results:
            return "No benchmark results. Run benchmark first."

        if format == "markdown":
            lines = [
                "# NER Benchmark Results",
                "",
                "| Engine | Precision | Recall | F1 Score | Avg Latency |",
                "|--------|-----------|--------|----------|-------------|",
            ]

            for name, result in sorted(
                self._results.items(),
                key=lambda x: x[1].f1_score,
                reverse=True,
            ):
                lines.append(
                    f"| {name} | {result.precision:.4f} | {result.recall:.4f} | "
                    f"{result.f1_score:.4f} | {result.avg_latency_ms:.1f}ms |"
                )

            # 최고 성능 엔진
            best = max(self._results.values(), key=lambda x: x.f1_score)
            fastest = min(self._results.values(), key=lambda x: x.avg_latency_ms)

            lines.extend(
                [
                    "",
                    "## Summary",
                    f"- **Best F1 Score**: {best.engine_name} ({best.f1_score:.4f})",
                    f"- **Fastest**: {fastest.engine_name} ({fastest.avg_latency_ms:.1f}ms)",
                    f"- **Samples**: {best.total_samples}",
                ]
            )

            return "\n".join(lines)

        # Text format
        lines = ["NER Benchmark Results", "=" * 50]
        for name, result in self._results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Precision: {result.precision:.4f}")
            lines.append(f"  Recall: {result.recall:.4f}")
            lines.append(f"  F1 Score: {result.f1_score:.4f}")
            lines.append(f"  Avg Latency: {result.avg_latency_ms:.1f}ms")

        return "\n".join(lines)

    def get_best_engine(self, metric: str = "f1_score") -> Optional[str]:
        """최고 성능 엔진 반환"""
        if not self._results:
            return None

        return max(
            self._results.items(),
            key=lambda x: getattr(x[1], metric, 0),
        )[0]


def quick_benchmark(engine_types: Optional[List[str]] = None) -> str:
    """
    빠른 벤치마크 실행

    Args:
        engine_types: 테스트할 엔진 타입 목록 (기본: ["spacy"])

    Returns:
        벤치마크 리포트
    """
    engine_types = engine_types or ["spacy"]
    engines = []

    for et in engine_types:
        try:
            engine = NEREngineFactory.create(et)
            engines.append(engine)
        except Exception as e:
            logger.warning(f"Failed to create {et} engine: {e}")

    if not engines:
        return "No engines available for benchmark"

    benchmark = NERBenchmark(engines=engines)
    benchmark.run(SAMPLE_TEST_DATA)
    return benchmark.get_report()
