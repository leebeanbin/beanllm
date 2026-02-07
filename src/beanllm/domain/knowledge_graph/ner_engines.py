"""
NER Engines - Named Entity Recognition 엔진 모음

다양한 NER 모델을 통합하여 사용할 수 있습니다.
정확도 벤치마크 기능도 제공합니다.

지원 엔진:
1. spaCy NER - 빠르고 정확한 rule-based + statistical NER
2. Hugging Face NER - BERT/RoBERTa 기반 transformer NER
3. GLiNER - Zero-shot NER (새로운 엔티티 타입에 유연)
4. Flair NER - Contextual string embeddings 기반
5. LLM NER - GPT/Claude 등 LLM 기반 (고비용, 고정확도)

Example:
    ```python
    from beanllm.domain.knowledge_graph.ner_engines import (
        NEREngineFactory,
        NERBenchmark,
    )

    # 엔진 생성
    spacy_engine = NEREngineFactory.create("spacy")
    hf_engine = NEREngineFactory.create("huggingface", model="dslim/bert-base-NER")
    gliner_engine = NEREngineFactory.create("gliner")

    # 엔티티 추출
    entities = spacy_engine.extract("Steve Jobs founded Apple.")

    # 벤치마크
    benchmark = NERBenchmark(engines=[spacy_engine, hf_engine, gliner_engine])
    results = benchmark.run(test_data)
    print(benchmark.get_report())
    ```
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class NEREntity:
    """NER 추출 결과"""

    text: str
    label: str  # PERSON, ORG, LOC, DATE, etc.
    start: int
    end: int
    confidence: float = 1.0
    source: str = ""  # 어떤 엔진에서 추출했는지


@dataclass
class NERResult:
    """NER 엔진 결과"""

    entities: List[NEREntity]
    engine_name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNEREngine(ABC):
    """NER 엔진 기본 클래스"""

    def __init__(self, name: str):
        self.name = name
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """모델 로드 (lazy loading)"""
        pass

    @abstractmethod
    def extract(self, text: str) -> List[NEREntity]:
        """엔티티 추출"""
        pass

    def extract_with_timing(self, text: str) -> NERResult:
        """타이밍 포함 추출"""
        if not self._is_loaded:
            self.load()

        start = time.time()
        entities = self.extract(text)
        latency = (time.time() - start) * 1000

        return NERResult(
            entities=entities,
            engine_name=self.name,
            latency_ms=latency,
        )


class SpacyNEREngine(BaseNEREngine):
    """
    spaCy NER 엔진

    장점: 빠름, 정확함, 다국어 지원
    단점: 고정된 엔티티 타입

    Models:
    - en_core_web_sm (small, fast)
    - en_core_web_md (medium)
    - en_core_web_lg (large, accurate)
    - en_core_web_trf (transformer, most accurate)
    - ko_core_news_sm (Korean)
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        disable: Optional[List[str]] = None,
    ):
        super().__init__(f"spacy:{model}")
        self.model_name = model
        self.disable = disable or ["parser", "tagger", "lemmatizer"]
        self._nlp = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            import spacy

            self._nlp = spacy.load(self.model_name, disable=self.disable)
            self._is_loaded = True
            logger.info(f"spaCy model loaded: {self.model_name}")
        except OSError:
            logger.warning(f"spaCy model not found: {self.model_name}")
            logger.info(f"Install with: python -m spacy download {self.model_name}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                NEREntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence
                    source=self.name,
                )
            )

        return entities


class HuggingFaceNEREngine(BaseNEREngine):
    """
    Hugging Face NER 엔진

    장점: 다양한 모델, 높은 정확도
    단점: GPU 권장, 무거움

    Recommended Models:
    - dslim/bert-base-NER (English, general)
    - dslim/bert-large-NER (English, accurate)
    - Jean-Baptiste/camembert-ner (French)
    - xlm-roberta-large-finetuned-conll03-english (Multilingual)
    """

    def __init__(
        self,
        model: str = "dslim/bert-base-NER",
        device: Optional[str] = None,
        aggregation_strategy: str = "simple",
    ):
        super().__init__(f"hf:{model.split('/')[-1]}")
        self.model_name = model
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self._pipeline = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy,
            )
            self._is_loaded = True
            logger.info(f"HuggingFace NER model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()

        results = self._pipeline(text)
        entities = []

        for item in results:
            entities.append(
                NEREntity(
                    text=item.get("word", item.get("entity_group", "")),
                    label=item.get("entity_group", item.get("entity", ""))
                    .replace("B-", "")
                    .replace("I-", ""),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                    confidence=item.get("score", 1.0),
                    source=self.name,
                )
            )

        return entities


class GLiNEREngine(BaseNEREngine):
    """
    GLiNER - Zero-shot NER 엔진

    장점: 커스텀 엔티티 타입 지원, 학습 불필요
    단점: 속도가 느릴 수 있음

    Models:
    - urchade/gliner_small (fast)
    - urchade/gliner_medium (balanced)
    - urchade/gliner_large (accurate)
    - urchade/gliner_multi (multilingual)
    """

    def __init__(
        self,
        model: str = "urchade/gliner_small",
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        super().__init__(f"gliner:{model.split('/')[-1]}")
        self.model_name = model
        self.labels = labels or [
            "person",
            "organization",
            "location",
            "date",
            "product",
            "technology",
            "event",
        ]
        self.threshold = threshold
        self._model = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(self.model_name)
            self._is_loaded = True
            logger.info(f"GLiNER model loaded: {self.model_name}")
        except ImportError:
            logger.error("GLiNER not installed. Install with: pip install gliner")
            raise
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()

        results = self._model.predict_entities(text, self.labels, threshold=self.threshold)
        entities = []

        for item in results:
            entities.append(
                NEREntity(
                    text=item["text"],
                    label=item["label"].upper(),
                    start=item["start"],
                    end=item["end"],
                    confidence=item.get("score", 1.0),
                    source=self.name,
                )
            )

        return entities

    def set_labels(self, labels: List[str]) -> None:
        """커스텀 레이블 설정"""
        self.labels = labels


class FlairNEREngine(BaseNEREngine):
    """
    Flair NER 엔진

    장점: 높은 정확도, contextual embeddings
    단점: 무거움

    Models:
    - flair/ner-english (English)
    - flair/ner-english-large (English, accurate)
    - flair/ner-multi (Multilingual)
    """

    def __init__(self, model: str = "flair/ner-english"):
        super().__init__(f"flair:{model.split('/')[-1]}")
        self.model_name = model
        self._tagger = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            from flair.data import Sentence
            from flair.models import SequenceTagger

            self._tagger = SequenceTagger.load(self.model_name)
            self._Sentence = Sentence
            self._is_loaded = True
            logger.info(f"Flair NER model loaded: {self.model_name}")
        except ImportError:
            logger.error("Flair not installed. Install with: pip install flair")
            raise
        except Exception as e:
            logger.error(f"Failed to load Flair model: {e}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()

        sentence = self._Sentence(text)
        self._tagger.predict(sentence)

        entities = []
        for entity in sentence.get_spans("ner"):
            entities.append(
                NEREntity(
                    text=entity.text,
                    label=entity.get_label("ner").value,
                    start=entity.start_position,
                    end=entity.end_position,
                    confidence=entity.get_label("ner").score,
                    source=self.name,
                )
            )

        return entities


class LLMNEREngine(BaseNEREngine):
    """
    LLM 기반 NER 엔진

    장점: 유연한 엔티티 타입, 높은 정확도
    단점: 비용, 속도
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        labels: Optional[List[str]] = None,
    ):
        super().__init__("llm")
        self._llm_function = llm_function
        self.labels = labels or [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "DATE",
            "PRODUCT",
            "TECHNOLOGY",
        ]
        self._is_loaded = True

    def load(self) -> None:
        pass  # No loading needed

    def extract(self, text: str) -> List[NEREntity]:
        import json
        import re

        prompt = f"""Extract named entities from the following text.

Text: {text}

Entity types to extract: {', '.join(self.labels)}

Return as JSON array:
[{{"text": "entity text", "label": "ENTITY_TYPE", "start": 0, "end": 10, "confidence": 0.95}}]

JSON:"""

        response = self._llm_function(prompt)

        # Parse JSON
        entities = []
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for item in data:
                    entities.append(
                        NEREntity(
                            text=item.get("text", ""),
                            label=item.get("label", "OTHER"),
                            start=item.get("start", 0),
                            end=item.get("end", 0),
                            confidence=item.get("confidence", 0.9),
                            source=self.name,
                        )
                    )
            except json.JSONDecodeError:
                pass

        return entities


class NEREngineFactory:
    """NER 엔진 팩토리"""

    _engines = {
        "spacy": SpacyNEREngine,
        "huggingface": HuggingFaceNEREngine,
        "hf": HuggingFaceNEREngine,
        "gliner": GLiNEREngine,
        "flair": FlairNEREngine,
    }

    @classmethod
    def create(cls, engine_type: str, **kwargs) -> BaseNEREngine:
        """
        NER 엔진 생성

        Args:
            engine_type: "spacy", "huggingface", "gliner", "flair"
            **kwargs: 엔진별 설정

        Returns:
            BaseNEREngine 인스턴스
        """
        engine_class = cls._engines.get(engine_type.lower())
        if not engine_class:
            raise ValueError(
                f"Unknown engine type: {engine_type}. Available: {list(cls._engines.keys())}"
            )

        return engine_class(**kwargs)

    @classmethod
    def create_llm(cls, llm_function: Callable[[str], str], **kwargs) -> LLMNEREngine:
        """LLM NER 엔진 생성"""
        return LLMNEREngine(llm_function=llm_function, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """사용 가능한 엔진 목록"""
        return list(cls._engines.keys())


@dataclass
class BenchmarkSample:
    """벤치마크 샘플"""

    text: str
    entities: List[Dict[str, Any]]  # [{"text": "...", "label": "PERSON", "start": 0, "end": 10}]


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""

    engine_name: str
    precision: float
    recall: float
    f1_score: float
    avg_latency_ms: float
    total_samples: int
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)


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

    def __init__(self, engines: List[BaseNEREngine]):
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
            total_latency = 0
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
                for ent in sample.entities:
                    label = (
                        self._normalize_label(ent["label"]) if normalize_labels else ent["label"]
                    )
                    gold.add((ent["text"].lower(), label))

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


# 샘플 테스트 데이터
SAMPLE_TEST_DATA = [
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
