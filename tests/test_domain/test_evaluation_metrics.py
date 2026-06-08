"""
Evaluation Metrics 테스트 - 순수 Python 메트릭 (ExactMatch, F1, BLEU, ROUGE, RAG metrics)
"""

import pytest

from beanllm.domain.evaluation.base_metric import BaseMetric
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.metrics.rag_metrics import (
    ContextPrecisionMetric,
    ContextRecallMetric,
)
from beanllm.domain.evaluation.metrics.similarity import (
    BLEUMetric,
    ExactMatchMetric,
    F1ScoreMetric,
    ROUGEMetric,
)
from beanllm.domain.evaluation.results import EvaluationResult


class ConcreteMetric(BaseMetric):
    """BaseMetric의 concrete subclass for testing"""

    def __init__(self):
        super().__init__("test_metric", MetricType.SIMILARITY)

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        score = 1.0 if prediction == reference else 0.0
        return EvaluationResult(metric_name=self.name, score=score)


class TestBaseMetric:
    def test_create_metric(self) -> None:
        metric = ConcreteMetric()
        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.SIMILARITY

    def test_compute(self) -> None:
        metric = ConcreteMetric()
        result = metric.compute("hello", "hello")
        assert result.score == 1.0

    def test_batch_compute(self) -> None:
        metric = ConcreteMetric()
        preds = ["hello", "world", "foo"]
        refs = ["hello", "world", "bar"]
        batch = metric.batch_compute(preds, refs)
        assert batch.average_score == pytest.approx(2 / 3)
        assert len(batch.results) == 3

    def test_batch_compute_mismatched_lengths(self) -> None:
        metric = ConcreteMetric()
        with pytest.raises(ValueError, match="same length"):
            metric.batch_compute(["a", "b"], ["a"])

    def test_batch_compute_metadata(self) -> None:
        metric = ConcreteMetric()
        batch = metric.batch_compute(["a"], ["a"])
        assert batch.metadata["count"] == 1


class TestExactMatchMetric:
    @pytest.fixture
    def metric(self) -> ExactMatchMetric:
        return ExactMatchMetric()

    def test_exact_match(self, metric: ExactMatchMetric) -> None:
        result = metric.compute("Hello world", "Hello world")
        assert result.score == 1.0

    def test_no_match(self, metric: ExactMatchMetric) -> None:
        result = metric.compute("Hello", "World")
        assert result.score == 0.0

    def test_case_sensitive_default(self, metric: ExactMatchMetric) -> None:
        result = metric.compute("Hello", "hello")
        assert result.score == 0.0

    def test_case_insensitive(self) -> None:
        metric = ExactMatchMetric(case_sensitive=False)
        result = metric.compute("HELLO", "hello")
        assert result.score == 1.0

    def test_whitespace_normalization(self, metric: ExactMatchMetric) -> None:
        result = metric.compute("hello   world", "hello world")
        assert result.score == 1.0

    def test_no_whitespace_normalization(self) -> None:
        metric = ExactMatchMetric(normalize_whitespace=False)
        result = metric.compute("hello   world", "hello world")
        assert result.score == 0.0

    def test_result_metadata(self, metric: ExactMatchMetric) -> None:
        result = metric.compute("pred", "ref")
        assert "prediction" in result.metadata
        assert "reference" in result.metadata


class TestF1ScoreMetric:
    @pytest.fixture
    def metric(self) -> F1ScoreMetric:
        return F1ScoreMetric()

    def test_perfect_match(self, metric: F1ScoreMetric) -> None:
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_partial_match(self, metric: F1ScoreMetric) -> None:
        result = metric.compute("the cat sat", "the cat sat on the mat")
        assert 0.0 < result.score < 1.0

    def test_no_overlap(self, metric: F1ScoreMetric) -> None:
        result = metric.compute("apple banana", "cat dog fish")
        assert result.score == 0.0

    def test_empty_prediction(self, metric: F1ScoreMetric) -> None:
        result = metric.compute("", "cat dog")
        assert result.score == 0.0

    def test_result_metadata_has_precision_recall(self, metric: F1ScoreMetric) -> None:
        result = metric.compute("the cat", "the cat")
        assert "precision" in result.metadata
        assert "recall" in result.metadata
        assert "common_tokens" in result.metadata


class TestBLEUMetric:
    @pytest.fixture
    def metric(self) -> BLEUMetric:
        return BLEUMetric()

    def test_perfect_match(self, metric: BLEUMetric) -> None:
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_partial_match(self, metric: BLEUMetric) -> None:
        result = metric.compute("the cat sat", "the cat sat on the mat")
        assert 0.0 <= result.score <= 1.0

    def test_no_overlap(self, metric: BLEUMetric) -> None:
        result = metric.compute("foo bar baz", "one two three four five")
        assert result.score == 0.0

    def test_short_prediction_brevity_penalty(self, metric: BLEUMetric) -> None:
        result_short = metric.compute("cat", "the cat sat on the mat")
        result_full = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result_short.score <= result_full.score

    def test_result_metadata(self, metric: BLEUMetric) -> None:
        result = metric.compute("hello world", "hello world")
        assert "precisions" in result.metadata
        assert "brevity_penalty" in result.metadata

    def test_custom_ngram_range(self) -> None:
        metric = BLEUMetric(max_n=2)
        result = metric.compute("hello world", "hello world")
        assert result.score > 0

    def test_empty_prediction(self, metric: BLEUMetric) -> None:
        result = metric.compute("", "hello world")
        assert result.score == 0.0

    def test_brevity_penalty_longer_than_ref(self) -> None:
        metric = BLEUMetric(max_n=1)
        result = metric.compute(
            "the cat sat on the mat yes it did",
            "the cat sat on the mat",
        )
        assert result.metadata["brevity_penalty"] == 1.0


class TestROUGEMetric:
    def test_rouge_1_perfect(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-1")
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_rouge_1_partial(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-1")
        result = metric.compute("the cat sat", "the cat sat on the mat")
        assert 0.0 < result.score < 1.0

    def test_rouge_2_perfect(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-2")
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_rouge_2_no_bigrams(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-2")
        result = metric.compute("hello", "world")
        assert result.score == 0.0

    def test_rouge_l_perfect(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-l")
        result = metric.compute("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)

    def test_rouge_l_partial(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-l")
        result = metric.compute("cat sat mat", "the cat sat on the mat")
        assert 0.0 < result.score <= 1.0

    def test_rouge_invalid_type(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-99")
        with pytest.raises(ValueError, match="Unknown ROUGE"):
            metric.compute("hello", "world")

    def test_rouge_result_metadata(self) -> None:
        metric = ROUGEMetric(rouge_type="rouge-1")
        result = metric.compute("cat sat", "cat sat mat")
        assert "precision" in result.metadata
        assert "recall" in result.metadata
        assert "f1" in result.metadata


class TestContextPrecisionMetric:
    @pytest.fixture
    def metric(self) -> ContextPrecisionMetric:
        return ContextPrecisionMetric()

    def test_no_contexts_returns_zero(self, metric: ContextPrecisionMetric) -> None:
        result = metric.compute("answer", "question", contexts=None)
        assert result.score == 0.0
        assert "error" in result.metadata

    def test_empty_contexts_returns_zero(self, metric: ContextPrecisionMetric) -> None:
        result = metric.compute("answer", "question", contexts=[])
        assert result.score == 0.0

    def test_relevant_contexts(self, metric: ContextPrecisionMetric) -> None:
        answer = "machine learning is used to train neural networks"
        contexts = [
            "machine learning algorithms train models using data and neural networks",
            "completely unrelated topic about cooking recipes",
        ]
        result = metric.compute(answer, "what is ml", contexts=contexts)
        assert 0.0 <= result.score <= 1.0

    def test_all_relevant_contexts(self, metric: ContextPrecisionMetric) -> None:
        answer = "Python is a programming language with many libraries"
        contexts = [
            "Python is a popular programming language with many libraries for data science",
            "Python programming language supports multiple paradigms and has many libraries",
        ]
        result = metric.compute(answer, "what is python", contexts=contexts)
        assert result.score > 0.0

    def test_metadata_contains_counts(self, metric: ContextPrecisionMetric) -> None:
        result = metric.compute(
            "test answer",
            "test question",
            contexts=["ctx1", "ctx2"],
        )
        assert "total_contexts" in result.metadata


class TestContextRecallMetric:
    @pytest.fixture
    def metric(self) -> ContextRecallMetric:
        return ContextRecallMetric()

    def test_no_contexts_returns_zero(self, metric: ContextRecallMetric) -> None:
        result = metric.compute(
            "answer",
            "question",
            contexts=None,
            ground_truth_contexts=["gt1"],
        )
        assert result.score == 0.0

    def test_no_ground_truth_returns_zero(self, metric: ContextRecallMetric) -> None:
        result = metric.compute(
            "answer",
            "question",
            contexts=["ctx1"],
            ground_truth_contexts=None,
        )
        assert result.score == 0.0

    def test_perfect_recall(self, metric: ContextRecallMetric) -> None:
        gt = ["machine learning neural networks deep learning models"]
        retrieved = ["machine learning neural networks deep learning models training"]
        result = metric.compute("answer", "question", contexts=retrieved, ground_truth_contexts=gt)
        assert result.score == 1.0

    def test_zero_recall(self, metric: ContextRecallMetric) -> None:
        gt = ["specific topic about quantum computing algorithms"]
        retrieved = ["recipe for chocolate cake baking"]
        result = metric.compute("answer", "question", contexts=retrieved, ground_truth_contexts=gt)
        assert result.score == 0.0

    def test_partial_recall(self, metric: ContextRecallMetric) -> None:
        gt = [
            "machine learning neural networks",
            "completely different unrelated topic xyz",
        ]
        retrieved = ["machine learning neural networks deep learning", "other info"]
        result = metric.compute("answer", "question", contexts=retrieved, ground_truth_contexts=gt)
        assert 0.0 <= result.score <= 1.0

    def test_metadata_contains_counts(self, metric: ContextRecallMetric) -> None:
        result = metric.compute(
            "answer",
            "question",
            contexts=["ctx1", "ctx2"],
            ground_truth_contexts=["gt1"],
        )
        assert "retrieved_count" in result.metadata
        assert "ground_truth_count" in result.metadata

    def test_token_based_compute(self, metric: ContextRecallMetric) -> None:
        # No embedding function, uses token-based recall
        assert metric.embedding_function is None
        result = metric.compute(
            "answer",
            "question",
            contexts=["hello world foo bar"],
            ground_truth_contexts=["hello world test"],
        )
        assert isinstance(result.score, float)
