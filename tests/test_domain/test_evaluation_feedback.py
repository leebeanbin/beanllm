"""
Evaluation Human Feedback 및 Custom Metrics 테스트
"""

import pytest

from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.human_feedback import (
    ComparisonFeedback,
    ComparisonWinner,
    FeedbackType,
    HumanFeedback,
    HumanFeedbackCollector,
)
from beanllm.domain.evaluation.metrics.custom import CustomMetric


class TestHumanFeedback:
    def test_create_rating_feedback(self) -> None:
        fb = HumanFeedback(
            feedback_id="fb1",
            feedback_type=FeedbackType.RATING,
            output="AI response",
            rating=0.8,
        )
        assert fb.feedback_id == "fb1"
        assert fb.rating == 0.8
        assert fb.feedback_type == FeedbackType.RATING

    def test_rating_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Rating must be between"):
            HumanFeedback(
                feedback_id="fb2",
                feedback_type=FeedbackType.RATING,
                output="output",
                rating=1.5,
            )

    def test_rating_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            HumanFeedback(
                feedback_id="fb3",
                feedback_type=FeedbackType.RATING,
                output="output",
                rating=-0.1,
            )

    def test_rating_none_allowed(self) -> None:
        fb = HumanFeedback(
            feedback_id="fb4",
            feedback_type=FeedbackType.COMMENT,
            output="output",
            rating=None,
        )
        assert fb.rating is None

    def test_to_dict(self) -> None:
        fb = HumanFeedback(
            feedback_id="fb5",
            feedback_type=FeedbackType.RATING,
            output="test output",
            rating=0.9,
            comment="Great response",
        )
        d = fb.to_dict()
        assert d["feedback_id"] == "fb5"
        assert d["rating"] == 0.9
        assert d["comment"] == "Great response"
        assert "timestamp" in d

    def test_feedback_type_values(self) -> None:
        assert FeedbackType.RATING.value == "rating"
        assert FeedbackType.COMPARISON.value == "comparison"
        assert FeedbackType.CORRECTION.value == "correction"
        assert FeedbackType.COMMENT.value == "comment"

    def test_feedback_has_timestamp(self) -> None:
        fb = HumanFeedback(
            feedback_id="fb6",
            feedback_type=FeedbackType.COMMENT,
            output="output",
        )
        assert fb.timestamp is not None


class TestComparisonFeedback:
    def test_create_comparison_feedback(self) -> None:
        fb = ComparisonFeedback(
            feedback_id="cmp1",
            output_a="Response A text",
            output_b="Response B text",
            winner=ComparisonWinner.A,
        )
        assert fb.output_a == "Response A text"
        assert fb.output_b == "Response B text"
        assert fb.winner == ComparisonWinner.A
        assert fb.feedback_type == FeedbackType.COMPARISON

    def test_comparison_winner_b(self) -> None:
        fb = ComparisonFeedback(
            feedback_id="cmp2",
            output_a="A",
            output_b="B",
            winner=ComparisonWinner.B,
        )
        assert fb.winner == ComparisonWinner.B

    def test_comparison_winner_tie(self) -> None:
        fb = ComparisonFeedback(
            feedback_id="cmp3",
            output_a="A",
            output_b="B",
            winner=ComparisonWinner.TIE,
        )
        assert fb.winner == ComparisonWinner.TIE

    def test_comparison_to_dict(self) -> None:
        fb = ComparisonFeedback(
            feedback_id="cmp4",
            output_a="Response A",
            output_b="Response B",
            winner=ComparisonWinner.A,
            criteria="quality",
        )
        d = fb.to_dict()
        assert "output_a" in d
        assert "output_b" in d
        assert d["winner"] == "A"
        assert d["criteria"] == "quality"

    def test_comparison_combined_output(self) -> None:
        fb = ComparisonFeedback(
            feedback_id="cmp5",
            output_a="First",
            output_b="Second",
            winner=ComparisonWinner.TIE,
        )
        assert "Output A: First" in fb.output
        assert "Output B: Second" in fb.output

    def test_comparison_winner_values(self) -> None:
        assert ComparisonWinner.A.value == "A"
        assert ComparisonWinner.B.value == "B"
        assert ComparisonWinner.TIE.value == "TIE"


class TestHumanFeedbackCollector:
    @pytest.fixture
    def collector(self) -> HumanFeedbackCollector:
        return HumanFeedbackCollector()

    def test_collect_rating(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_rating("AI response text", rating=0.9)
        assert fb.rating == 0.9
        assert fb.feedback_type == FeedbackType.RATING
        assert len(collector.get_all_feedbacks()) == 1

    def test_collect_rating_with_criteria(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_rating(
            "AI response", rating=0.7, criteria="accuracy", comment="Mostly correct"
        )
        assert fb.criteria == "accuracy"
        assert fb.comment == "Mostly correct"

    def test_collect_comparison(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_comparison(
            output_a="Response A here",
            output_b="Response B here",
            winner=ComparisonWinner.A,
        )
        assert isinstance(fb, ComparisonFeedback)
        assert fb.winner == ComparisonWinner.A

    def test_collect_correction(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_correction(
            output="Original wrong text",
            corrected_output="Corrected right text",
            comment="Fixed the error",
        )
        assert fb.feedback_type == FeedbackType.CORRECTION
        assert "Original" in fb.comment
        assert "Corrected" in fb.comment
        assert "Fixed the error" in fb.comment

    def test_collect_comment(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_comment("AI output", comment="This is great!")
        assert fb.feedback_type == FeedbackType.COMMENT
        assert fb.comment == "This is great!"

    def test_get_feedback_by_id(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_rating("output", rating=0.5)
        found = collector.get_feedback(fb.feedback_id)
        assert found is not None
        assert found.feedback_id == fb.feedback_id

    def test_get_feedback_not_found(self, collector: HumanFeedbackCollector) -> None:
        result = collector.get_feedback("nonexistent_id")
        assert result is None

    def test_get_feedbacks_by_type(self, collector: HumanFeedbackCollector) -> None:
        collector.collect_rating("output1", rating=0.9)
        collector.collect_comment("output2", comment="hello")
        collector.collect_rating("output3", rating=0.5)

        ratings = collector.get_feedbacks_by_type(FeedbackType.RATING)
        assert len(ratings) == 2
        comments = collector.get_feedbacks_by_type(FeedbackType.COMMENT)
        assert len(comments) == 1

    def test_clear(self, collector: HumanFeedbackCollector) -> None:
        collector.collect_rating("output", rating=0.8)
        collector.clear()
        assert len(collector.get_all_feedbacks()) == 0

    def test_feedback_ids_are_unique(self, collector: HumanFeedbackCollector) -> None:
        fb1 = collector.collect_rating("out1", rating=0.5)
        fb2 = collector.collect_rating("out2", rating=0.7)
        assert fb1.feedback_id != fb2.feedback_id

    def test_multiple_feedback_types_tracked(self, collector: HumanFeedbackCollector) -> None:
        collector.collect_rating("out1", rating=0.9)
        collector.collect_comparison("A", "B", winner=ComparisonWinner.B)
        collector.collect_correction("bad", "good")
        collector.collect_comment("out", comment="ok")

        all_fb = collector.get_all_feedbacks()
        assert len(all_fb) == 4

    def test_collect_correction_without_comment(self, collector: HumanFeedbackCollector) -> None:
        fb = collector.collect_correction("orig", "corrected")
        assert "Original: orig" in fb.comment
        assert "Corrected: corrected" in fb.comment


class TestCustomMetric:
    def test_custom_metric_exact_length(self) -> None:
        def length_match(pred: str, ref: str) -> float:
            return 1.0 if len(pred) == len(ref) else 0.0

        metric = CustomMetric(name="length_match", compute_fn=length_match)
        result = metric.compute("hello", "world")
        assert result.score == 1.0

    def test_custom_metric_score_range(self) -> None:
        def overlap_ratio(pred: str, ref: str) -> float:
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            if not ref_words:
                return 0.0
            return len(pred_words & ref_words) / len(ref_words)

        metric = CustomMetric(name="overlap", compute_fn=overlap_ratio)
        result = metric.compute("the cat sat", "the cat sat on the mat")
        assert 0.0 <= result.score <= 1.0

    def test_custom_metric_metadata(self) -> None:
        metric = CustomMetric(name="my_metric", compute_fn=lambda p, r: 0.5)
        result = metric.compute("pred", "ref")
        assert result.metadata.get("type") == "custom"

    def test_custom_metric_name(self) -> None:
        metric = CustomMetric(name="my_custom_score", compute_fn=lambda p, r: 1.0)
        assert metric.name == "my_custom_score"

    def test_custom_metric_type(self) -> None:
        metric = CustomMetric(
            name="test", compute_fn=lambda p, r: 0.0, metric_type=MetricType.QUALITY
        )
        assert metric.metric_type == MetricType.QUALITY

    def test_custom_metric_batch(self) -> None:
        metric = CustomMetric(name="ones", compute_fn=lambda p, r: 1.0)
        batch = metric.batch_compute(["a", "b", "c"], ["x", "y", "z"])
        assert batch.average_score == 1.0
        assert len(batch.results) == 3
