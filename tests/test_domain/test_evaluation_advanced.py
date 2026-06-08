"""
Advanced tests for beanllm evaluation domain:
  - rubric.py   : RubricCriterion, Rubric, RubricGrader
  - checklist.py: ChecklistItem, Checklist, ChecklistGrader
  - continuous.py: EvaluationTask, EvaluationRun, ContinuousEvaluator
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.evaluation.checklist import Checklist, ChecklistGrader, ChecklistItem
from beanllm.domain.evaluation.continuous import (
    ContinuousEvaluator,
    EvaluationRun,
    EvaluationTask,
)
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.evaluator import Evaluator
from beanllm.domain.evaluation.results import BatchEvaluationResult, EvaluationResult
from beanllm.domain.evaluation.rubric import Rubric, RubricCriterion, RubricGrader

# ---------------------------------------------------------------------------
# Helpers / shared factories
# ---------------------------------------------------------------------------


def make_criterion(
    name: str = "accuracy",
    description: str = "How accurate",
    weight: float = 1.0,
    levels: dict | None = None,
) -> RubricCriterion:
    return RubricCriterion(
        name=name,
        description=description,
        weight=weight,
        levels=levels,
    )


def make_rubric(name: str = "test_rubric", num_criteria: int = 2) -> Rubric:
    criteria = [
        make_criterion(name=f"criterion_{i}", description=f"Desc {i}", weight=1.0)
        for i in range(num_criteria)
    ]
    return Rubric(
        name=name,
        description="A test rubric",
        criteria=criteria,
    )


def make_checklist_item(
    question: str = "Is it correct?",
    description: str | None = None,
    weight: float = 1.0,
    required: bool = False,
) -> ChecklistItem:
    return ChecklistItem(
        question=question,
        description=description,
        weight=weight,
        required=required,
    )


def make_checklist(name: str = "test_checklist", num_items: int = 3) -> Checklist:
    items = [
        make_checklist_item(
            question=f"Question {i}?",
            description=f"Check {i}",
            weight=1.0,
        )
        for i in range(num_items)
    ]
    return Checklist(
        name=name,
        description="A test checklist",
        items=items,
    )


def make_mock_client(
    content: str = "criterion_0: excellent - great\ncriterion_1: good - ok",
) -> MagicMock:
    """Return a fake LLM client whose .chat() returns a mock response."""
    client = MagicMock()
    response = MagicMock()
    response.content = content
    client.chat.return_value = response
    return client


def make_evaluator_with_rubric_grader(rubric: Rubric, use_llm: bool = False) -> Evaluator:
    grader = RubricGrader(rubric=rubric, use_llm=use_llm)
    evaluator = Evaluator(metrics=[grader])
    return evaluator


# ============================================================================
# RubricCriterion tests
# ============================================================================


class TestRubricCriterion:
    def test_creation_with_defaults(self) -> None:
        criterion = make_criterion()
        assert criterion.name == "accuracy"
        assert criterion.description == "How accurate"
        assert criterion.weight == 1.0

    def test_default_levels_populated_on_post_init(self) -> None:
        criterion = make_criterion(levels=None)
        assert criterion.levels is not None
        assert "excellent" in criterion.levels
        assert "good" in criterion.levels
        assert "satisfactory" in criterion.levels
        assert "needs_improvement" in criterion.levels
        assert "poor" in criterion.levels

    def test_default_level_scores(self) -> None:
        criterion = make_criterion(levels=None)
        assert criterion.levels["excellent"] == 1.0
        assert criterion.levels["good"] == 0.8
        assert criterion.levels["satisfactory"] == 0.6
        assert criterion.levels["needs_improvement"] == 0.4
        assert criterion.levels["poor"] == 0.2

    def test_custom_levels_preserved(self) -> None:
        custom_levels = {"high": 1.0, "low": 0.2}
        criterion = make_criterion(levels=custom_levels)
        assert criterion.levels == custom_levels

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            make_criterion(weight=-0.1)

    def test_zero_weight_allowed(self) -> None:
        criterion = make_criterion(weight=0.0)
        assert criterion.weight == 0.0

    def test_weight_greater_than_one_allowed(self) -> None:
        criterion = make_criterion(weight=5.0)
        assert criterion.weight == 5.0


# ============================================================================
# Rubric tests
# ============================================================================


class TestRubric:
    def test_creation_basic(self) -> None:
        rubric = make_rubric()
        assert rubric.name == "test_rubric"
        assert rubric.description == "A test rubric"
        assert len(rubric.criteria) == 2

    def test_empty_criteria_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one criterion"):
            Rubric(name="empty", description="no criteria", criteria=[])

    def test_weights_normalized_to_sum_one(self) -> None:
        criteria = [
            make_criterion("c1", weight=1.0),
            make_criterion("c2", weight=3.0),
        ]
        rubric = Rubric(name="weighted", description="test", criteria=criteria)
        total = sum(c.weight for c in rubric.criteria)
        assert total == pytest.approx(1.0)

    def test_equal_weights_each_half(self) -> None:
        criteria = [make_criterion("c1", weight=1.0), make_criterion("c2", weight=1.0)]
        rubric = Rubric(name="equal", description="equal weights", criteria=criteria)
        assert rubric.criteria[0].weight == pytest.approx(0.5)
        assert rubric.criteria[1].weight == pytest.approx(0.5)

    def test_single_criterion_weight_normalized_to_one(self) -> None:
        criteria = [make_criterion("sole", weight=42.0)]
        rubric = Rubric(name="single", description="one criterion", criteria=criteria)
        assert rubric.criteria[0].weight == pytest.approx(1.0)

    def test_metadata_defaults_to_empty_dict(self) -> None:
        rubric = make_rubric()
        assert rubric.metadata == {}

    def test_metadata_preserved(self) -> None:
        rubric = Rubric(
            name="meta",
            description="meta test",
            criteria=[make_criterion()],
            metadata={"version": "1.0"},
        )
        assert rubric.metadata["version"] == "1.0"


# ============================================================================
# RubricGrader tests
# ============================================================================


class TestRubricGraderInit:
    def test_init_sets_name(self) -> None:
        rubric = make_rubric("my_rubric")
        grader = RubricGrader(rubric=rubric)
        assert grader.name == "rubric_my_rubric"

    def test_init_metric_type_quality(self) -> None:
        grader = RubricGrader(rubric=make_rubric())
        assert grader.metric_type == MetricType.QUALITY

    def test_init_no_client_by_default(self) -> None:
        grader = RubricGrader(rubric=make_rubric())
        assert grader.client is None

    def test_init_use_llm_default_true(self) -> None:
        grader = RubricGrader(rubric=make_rubric())
        assert grader.use_llm is True

    def test_get_client_raises_when_none(self) -> None:
        grader = RubricGrader(rubric=make_rubric(), use_llm=True)
        with pytest.raises(RuntimeError, match="LLM client not available"):
            grader._get_client()

    def test_get_client_returns_injected_client(self) -> None:
        mock = make_mock_client()
        grader = RubricGrader(rubric=make_rubric(), client=mock)
        assert grader._get_client() is mock


class TestRubricGraderCreatePrompt:
    def test_prompt_contains_rubric_name(self) -> None:
        rubric = make_rubric("writing_quality")
        grader = RubricGrader(rubric=rubric)
        prompt = grader._create_rubric_prompt("Some prediction")
        assert "writing_quality" in prompt

    def test_prompt_contains_criterion_names(self) -> None:
        rubric = make_rubric(num_criteria=2)
        grader = RubricGrader(rubric=rubric)
        prompt = grader._create_rubric_prompt("pred")
        assert "criterion_0" in prompt
        assert "criterion_1" in prompt

    def test_prompt_contains_prediction(self) -> None:
        rubric = make_rubric()
        grader = RubricGrader(rubric=rubric)
        prompt = grader._create_rubric_prompt("My unique prediction text")
        assert "My unique prediction text" in prompt

    def test_prompt_contains_reference_when_provided(self) -> None:
        rubric = make_rubric()
        grader = RubricGrader(rubric=rubric)
        prompt = grader._create_rubric_prompt("pred", reference="Reference answer")
        assert "Reference answer" in prompt

    def test_prompt_no_reference_section_when_none(self) -> None:
        rubric = make_rubric()
        grader = RubricGrader(rubric=rubric)
        prompt = grader._create_rubric_prompt("pred", reference=None)
        assert "Reference:" not in prompt

    def test_prompt_contains_format_instructions(self) -> None:
        grader = RubricGrader(rubric=make_rubric())
        prompt = grader._create_rubric_prompt("pred")
        assert "CRITERION_NAME" in prompt


class TestRubricGraderComputeManual:
    def test_compute_manual_all_excellent(self) -> None:
        rubric = make_rubric(num_criteria=2)
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {c.name: "excellent" for c in rubric.criteria}
        result = grader._compute_manual("prediction text", manual)
        assert isinstance(result, EvaluationResult)
        assert result.score == pytest.approx(1.0)

    def test_compute_manual_all_poor(self) -> None:
        rubric = make_rubric(num_criteria=2)
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {c.name: "poor" for c in rubric.criteria}
        result = grader._compute_manual("prediction text", manual)
        assert result.score == pytest.approx(0.2)

    def test_compute_manual_mixed_levels(self) -> None:
        criteria = [
            make_criterion("c1", weight=1.0),
            make_criterion("c2", weight=1.0),
        ]
        rubric = Rubric(name="mixed", description="mix", criteria=criteria)
        grader = RubricGrader(rubric=rubric, use_llm=False)
        # After normalisation each weight = 0.5
        manual = {"c1": "excellent", "c2": "poor"}
        result = grader._compute_manual("pred", manual)
        # (1.0 * 0.5 + 0.2 * 0.5) = 0.6
        assert result.score == pytest.approx(0.6)

    def test_compute_manual_missing_criterion_skipped(self) -> None:
        rubric = make_rubric(num_criteria=2)
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {"criterion_0": "excellent"}  # criterion_1 missing
        result = grader._compute_manual("pred", manual)
        # Only criterion_0 (w=0.5) counted → 1.0*0.5 / 0.5 = 1.0
        assert result.score == pytest.approx(1.0)

    def test_compute_manual_metadata_contains_rubric_name(self) -> None:
        rubric = make_rubric("named_rubric")
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {c.name: "good" for c in rubric.criteria}
        result = grader._compute_manual("pred", manual)
        assert result.metadata["rubric_name"] == "named_rubric"

    def test_compute_manual_metadata_manual_evaluation_true(self) -> None:
        rubric = make_rubric()
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {c.name: "good" for c in rubric.criteria}
        result = grader._compute_manual("pred", manual)
        assert result.metadata["manual_evaluation"] is True

    def test_compute_manual_explanation_generated(self) -> None:
        rubric = make_rubric()
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {c.name: "satisfactory" for c in rubric.criteria}
        result = grader._compute_manual("pred", manual)
        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_compute_delegates_to_compute_manual_when_manual_scores_given(self) -> None:
        rubric = make_rubric(num_criteria=1)
        grader = RubricGrader(rubric=rubric, use_llm=False)
        manual = {"criterion_0": "excellent"}
        result = grader.compute("pred", manual_scores=manual)
        assert result.score == pytest.approx(1.0)

    def test_compute_raises_when_no_manual_and_use_llm_false(self) -> None:
        grader = RubricGrader(rubric=make_rubric(), use_llm=False)
        with pytest.raises(ValueError):
            grader.compute("pred")


class TestRubricGraderParseLLMResponse:
    def test_parse_standard_format(self) -> None:
        criteria = [make_criterion("clarity", weight=1.0)]
        rubric = Rubric(name="r", description="d", criteria=criteria)
        grader = RubricGrader(rubric=rubric)
        llm_output = "clarity: excellent - Very clear and concise"
        parsed = grader._parse_llm_response(llm_output)
        assert "clarity" in parsed
        assert parsed["clarity"]["level"] == "excellent"

    def test_parse_fallback_pattern(self) -> None:
        criteria = [make_criterion("relevance", weight=1.0)]
        rubric = Rubric(name="r", description="d", criteria=criteria)
        grader = RubricGrader(rubric=rubric)
        llm_output = "relevance good"
        parsed = grader._parse_llm_response(llm_output)
        assert "relevance" in parsed
        assert parsed["relevance"]["level"].lower() == "good"

    def test_parse_missing_criterion_not_in_result(self) -> None:
        criteria = [make_criterion("missing_one", weight=1.0)]
        rubric = Rubric(name="r", description="d", criteria=criteria)
        grader = RubricGrader(rubric=rubric)
        llm_output = "totally unrelated text without criterion name"
        parsed = grader._parse_llm_response(llm_output)
        assert parsed == {}


class TestRubricGraderGenerateExplanation:
    def test_explanation_contains_rubric_name(self) -> None:
        rubric = make_rubric("rubric_alpha")
        grader = RubricGrader(rubric=rubric)
        explanation = grader._generate_explanation({}, 0.5)
        assert "rubric_alpha" in explanation

    def test_explanation_contains_final_score(self) -> None:
        rubric = make_rubric()
        grader = RubricGrader(rubric=rubric)
        explanation = grader._generate_explanation({}, 0.75)
        assert "0.750" in explanation

    def test_explanation_not_evaluated_for_missing_criteria(self) -> None:
        rubric = make_rubric(num_criteria=1)
        grader = RubricGrader(rubric=rubric)
        explanation = grader._generate_explanation({}, 0.0)
        assert "not evaluated" in explanation


class TestRubricGraderBatchCompute:
    def test_batch_compute_equal_lengths(self) -> None:
        rubric = make_rubric(num_criteria=1)
        grader = RubricGrader(rubric=rubric, use_llm=False)
        # batch_compute doesn't support manual_scores kwarg from BaseMetric,
        # but it will call compute without manual_scores (which raises when use_llm=False).
        # We must use a grader that can handle no-manual path → attach a client.
        mock_client = make_mock_client("criterion_0: good - fine")
        grader2 = RubricGrader(rubric=rubric, client=mock_client, use_llm=True)
        batch = grader2.batch_compute(["pred1", "pred2"], ["ref1", "ref2"])
        assert isinstance(batch, BatchEvaluationResult)
        assert len(batch.results) == 2

    def test_batch_compute_mismatched_lengths_raises(self) -> None:
        rubric = make_rubric(num_criteria=1)
        mock_client = make_mock_client("criterion_0: good - fine")
        grader = RubricGrader(rubric=rubric, client=mock_client, use_llm=True)
        with pytest.raises(ValueError, match="same length"):
            grader.batch_compute(["a"], ["b", "c"])


# ============================================================================
# ChecklistItem tests
# ============================================================================


class TestChecklistItem:
    def test_creation_defaults(self) -> None:
        item = make_checklist_item()
        assert item.question == "Is it correct?"
        assert item.description is None
        assert item.weight == 1.0
        assert item.required is False

    def test_required_flag_set(self) -> None:
        item = make_checklist_item(required=True)
        assert item.required is True

    def test_description_set(self) -> None:
        item = make_checklist_item(description="Check this carefully")
        assert item.description == "Check this carefully"

    def test_custom_weight(self) -> None:
        item = make_checklist_item(weight=3.0)
        assert item.weight == 3.0


# ============================================================================
# Checklist tests
# ============================================================================


class TestChecklist:
    def test_creation_basic(self) -> None:
        cl = make_checklist()
        assert cl.name == "test_checklist"
        assert len(cl.items) == 3

    def test_empty_items_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one item"):
            Checklist(name="empty", description="no items", items=[])

    def test_weights_normalized(self) -> None:
        items = [
            make_checklist_item("q1", weight=1.0),
            make_checklist_item("q2", weight=3.0),
        ]
        cl = Checklist(name="w", description="weighted", items=items)
        total = sum(i.weight for i in cl.items)
        assert total == pytest.approx(1.0)

    def test_equal_weights_after_normalization(self) -> None:
        items = [make_checklist_item("q1"), make_checklist_item("q2")]
        cl = Checklist(name="eq", description="equal", items=items)
        assert cl.items[0].weight == pytest.approx(0.5)
        assert cl.items[1].weight == pytest.approx(0.5)

    def test_metadata_defaults_empty(self) -> None:
        cl = make_checklist()
        assert cl.metadata == {}


# ============================================================================
# ChecklistGrader tests
# ============================================================================


class TestChecklistGraderInit:
    def test_init_sets_name(self) -> None:
        cl = make_checklist("my_list")
        grader = ChecklistGrader(checklist=cl)
        assert grader.name == "checklist_my_list"

    def test_init_metric_type_quality(self) -> None:
        grader = ChecklistGrader(checklist=make_checklist())
        assert grader.metric_type == MetricType.QUALITY

    def test_get_client_raises_when_none(self) -> None:
        grader = ChecklistGrader(checklist=make_checklist())
        with pytest.raises(RuntimeError, match="LLM client not available"):
            grader._get_client()

    def test_get_client_returns_injected(self) -> None:
        mock = make_mock_client()
        grader = ChecklistGrader(checklist=make_checklist(), client=mock)
        assert grader._get_client() is mock


class TestChecklistGraderCreatePrompt:
    def test_prompt_contains_checklist_name(self) -> None:
        cl = make_checklist("safety_check")
        grader = ChecklistGrader(checklist=cl)
        prompt = grader._create_checklist_prompt("Some response")
        assert "safety_check" in prompt

    def test_prompt_contains_questions(self) -> None:
        cl = make_checklist(num_items=2)
        grader = ChecklistGrader(checklist=cl)
        prompt = grader._create_checklist_prompt("pred")
        assert "Question 0?" in prompt
        assert "Question 1?" in prompt

    def test_prompt_contains_required_marker(self) -> None:
        items = [make_checklist_item(required=True)]
        cl = Checklist(name="req", description="required items", items=items)
        grader = ChecklistGrader(checklist=cl)
        prompt = grader._create_checklist_prompt("pred")
        assert "REQUIRED" in prompt

    def test_prompt_contains_prediction(self) -> None:
        cl = make_checklist()
        grader = ChecklistGrader(checklist=cl)
        prompt = grader._create_checklist_prompt("special prediction")
        assert "special prediction" in prompt

    def test_prompt_contains_reference_when_given(self) -> None:
        cl = make_checklist()
        grader = ChecklistGrader(checklist=cl)
        prompt = grader._create_checklist_prompt("pred", reference="Gold answer")
        assert "Gold answer" in prompt


class TestChecklistGraderComputeManual:
    def test_all_true_gives_score_one(self) -> None:
        cl = make_checklist(num_items=3)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        answers = {0: True, 1: True, 2: True}
        result = grader._compute_manual("pred", answers)
        assert result.score == pytest.approx(1.0)

    def test_all_false_gives_score_zero(self) -> None:
        cl = make_checklist(num_items=3)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        answers = {0: False, 1: False, 2: False}
        result = grader._compute_manual("pred", answers)
        assert result.score == pytest.approx(0.0)

    def test_mixed_answers_correct_score(self) -> None:
        items = [make_checklist_item(f"q{i}") for i in range(2)]
        cl = Checklist(name="c", description="d", items=items)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        answers = {0: True, 1: False}
        result = grader._compute_manual("pred", answers)
        assert result.score == pytest.approx(0.5)

    def test_required_item_failed_caps_score_at_half(self) -> None:
        items = [
            make_checklist_item("required q", required=True),
            make_checklist_item("optional q"),
        ]
        cl = Checklist(name="cap", description="capped", items=items)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        # required=True item answered False → cap at 0.5
        answers = {0: False, 1: True}
        result = grader._compute_manual("pred", answers)
        assert result.score <= 0.5

    def test_required_failed_listed_in_metadata(self) -> None:
        items = [make_checklist_item("must pass", required=True)]
        cl = Checklist(name="req", description="required", items=items)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        result = grader._compute_manual("pred", {0: False})
        assert "must pass" in result.metadata["required_failed"]

    def test_metadata_manual_evaluation_flag(self) -> None:
        cl = make_checklist(num_items=1)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        result = grader._compute_manual("pred", {0: True})
        assert result.metadata["manual_evaluation"] is True

    def test_missing_answer_treated_as_false(self) -> None:
        cl = make_checklist(num_items=2)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        result = grader._compute_manual("pred", {0: True})  # index 1 missing → False
        assert result.score == pytest.approx(0.5)

    def test_compute_delegates_to_manual_when_answers_given(self) -> None:
        cl = make_checklist(num_items=1)
        grader = ChecklistGrader(checklist=cl, use_llm=False)
        result = grader.compute("pred", manual_answers={0: True})
        assert result.score == pytest.approx(1.0)

    def test_compute_raises_no_answers_no_llm(self) -> None:
        grader = ChecklistGrader(checklist=make_checklist(), use_llm=False)
        with pytest.raises(ValueError):
            grader.compute("pred")


class TestChecklistGraderParseLLMResponse:
    def test_parse_yes_answer(self) -> None:
        items = [make_checklist_item("q1")]
        cl = Checklist(name="c", description="d", items=items)
        grader = ChecklistGrader(checklist=cl)
        llm_output = "1. YES - Looks good"
        parsed = grader._parse_llm_response(llm_output)
        assert parsed[0] is True

    def test_parse_no_answer(self) -> None:
        items = [make_checklist_item("q1")]
        cl = Checklist(name="c", description="d", items=items)
        grader = ChecklistGrader(checklist=cl)
        llm_output = "1. NO - Does not meet criteria"
        parsed = grader._parse_llm_response(llm_output)
        assert parsed[0] is False

    def test_parse_fallback_yes(self) -> None:
        items = [make_checklist_item("q1")]
        cl = Checklist(name="c", description="d", items=items)
        grader = ChecklistGrader(checklist=cl)
        llm_output = "1. YES"
        parsed = grader._parse_llm_response(llm_output)
        assert parsed[0] is True

    def test_parse_unmatched_defaults_false(self) -> None:
        items = [make_checklist_item("q1")]
        cl = Checklist(name="c", description="d", items=items)
        grader = ChecklistGrader(checklist=cl)
        llm_output = "completely unrelated text"
        parsed = grader._parse_llm_response(llm_output)
        assert parsed[0] is False

    def test_parse_multiple_items(self) -> None:
        items = [make_checklist_item(f"q{i}") for i in range(3)]
        cl = Checklist(name="c", description="d", items=items)
        grader = ChecklistGrader(checklist=cl)
        llm_output = "1. YES - good\n2. NO - bad\n3. YES - ok"
        parsed = grader._parse_llm_response(llm_output)
        assert parsed[0] is True
        assert parsed[1] is False
        assert parsed[2] is True


class TestChecklistGraderBatchCompute:
    def test_batch_compute_returns_batch_result(self) -> None:
        cl = make_checklist(num_items=1)
        llm_out = "1. YES - ok"
        mock_client = MagicMock()
        response = MagicMock()
        response.content = llm_out
        mock_client.chat.return_value = response
        grader = ChecklistGrader(checklist=cl, client=mock_client, use_llm=True)
        batch = grader.batch_compute(["pred1", "pred2"], ["ref1", "ref2"])
        assert isinstance(batch, BatchEvaluationResult)
        assert len(batch.results) == 2

    def test_batch_compute_mismatched_raises(self) -> None:
        cl = make_checklist(num_items=1)
        mock_client = MagicMock()
        response = MagicMock()
        response.content = "1. YES"
        mock_client.chat.return_value = response
        grader = ChecklistGrader(checklist=cl, client=mock_client, use_llm=True)
        with pytest.raises(ValueError, match="same length"):
            grader.batch_compute(["a"], ["b", "c"])


# ============================================================================
# EvaluationTask / EvaluationRun dataclass tests
# ============================================================================


class TestEvaluationTaskDataclass:
    def test_creation_required_fields(self) -> None:
        evaluator = Evaluator()
        task = EvaluationTask(
            task_id="task_1",
            name="My Task",
            evaluator=evaluator,
            test_cases=[{"prediction": "p", "reference": "r"}],
        )
        assert task.task_id == "task_1"
        assert task.name == "My Task"
        assert task.enabled is True
        assert task.schedule is None
        assert task.metadata == {}

    def test_creation_with_schedule(self) -> None:
        evaluator = Evaluator()
        task = EvaluationTask(
            task_id="t2",
            name="scheduled",
            evaluator=evaluator,
            test_cases=[],
            schedule="0 9 * * *",
        )
        assert task.schedule == "0 9 * * *"

    def test_enabled_can_be_set_false(self) -> None:
        evaluator = Evaluator()
        task = EvaluationTask(
            task_id="t3",
            name="disabled",
            evaluator=evaluator,
            test_cases=[],
            enabled=False,
        )
        assert task.enabled is False


class TestEvaluationRunDataclass:
    def test_creation(self) -> None:
        now = datetime.now()
        run = EvaluationRun(
            run_id="run_0",
            task_id="task_1",
            timestamp=now,
            results=[],
            average_score=0.75,
        )
        assert run.run_id == "run_0"
        assert run.task_id == "task_1"
        assert run.timestamp == now
        assert run.average_score == 0.75
        assert run.metadata == {}

    def test_metadata_field(self) -> None:
        now = datetime.now()
        run = EvaluationRun(
            run_id="run_1",
            task_id="t",
            timestamp=now,
            results=[],
            average_score=0.0,
            metadata={"note": "test"},
        )
        assert run.metadata["note"] == "test"


# ============================================================================
# ContinuousEvaluator tests
# ============================================================================


class TestContinuousEvaluatorInit:
    def test_init_default(self) -> None:
        ce = ContinuousEvaluator()
        assert ce.storage_path is None
        assert ce._tasks == {}
        assert ce._runs == []
        assert ce._scheduler is None
        assert ce._run_counter == 0

    def test_init_with_storage_path(self) -> None:
        ce = ContinuousEvaluator(storage_path="/tmp/eval")
        assert ce.storage_path == "/tmp/eval"


class TestContinuousEvaluatorTaskManagement:
    def test_add_task_returns_task(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        task = ce.add_task("t1", "Task One", ev, [])
        assert isinstance(task, EvaluationTask)
        assert task.task_id == "t1"

    def test_add_task_stored_internally(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "Task One", ev, [])
        assert "t1" in ce._tasks

    def test_get_task_returns_correct_task(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "Task One", ev, [])
        task = ce.get_task("t1")
        assert task is not None
        assert task.task_id == "t1"

    def test_get_task_returns_none_for_unknown(self) -> None:
        ce = ContinuousEvaluator()
        assert ce.get_task("nonexistent") is None

    def test_list_tasks_empty(self) -> None:
        ce = ContinuousEvaluator()
        assert ce.list_tasks() == []

    def test_list_tasks_returns_all(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [])
        ce.add_task("t2", "T2", ev, [])
        tasks = ce.list_tasks()
        assert len(tasks) == 2
        ids = {t.task_id for t in tasks}
        assert ids == {"t1", "t2"}

    def test_remove_task_returns_true(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [])
        result = ce.remove_task("t1")
        assert result is True

    def test_remove_task_actually_removes(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [])
        ce.remove_task("t1")
        assert ce.get_task("t1") is None

    def test_remove_nonexistent_task_returns_false(self) -> None:
        ce = ContinuousEvaluator()
        result = ce.remove_task("does_not_exist")
        assert result is False

    def test_add_task_with_metadata(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        task = ce.add_task("t1", "T1", ev, [], metadata={"key": "val"})
        assert task.metadata["key"] == "val"


class TestContinuousEvaluatorRunsRetrieval:
    def _make_run(
        self, run_id: str, task_id: str, score: float, offset_hours: int = 0
    ) -> EvaluationRun:
        ts = datetime.now() - timedelta(hours=offset_hours)
        return EvaluationRun(
            run_id=run_id,
            task_id=task_id,
            timestamp=ts,
            results=[],
            average_score=score,
        )

    def test_get_runs_empty(self) -> None:
        ce = ContinuousEvaluator()
        assert ce.get_runs() == []

    def test_get_runs_all(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("r1", "t1", 0.8))
        ce._runs.append(self._make_run("r2", "t2", 0.6))
        runs = ce.get_runs()
        assert len(runs) == 2

    def test_get_runs_filtered_by_task_id(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("r1", "t1", 0.9))
        ce._runs.append(self._make_run("r2", "t2", 0.5))
        runs = ce.get_runs(task_id="t1")
        assert all(r.task_id == "t1" for r in runs)
        assert len(runs) == 1

    def test_get_runs_with_limit(self) -> None:
        ce = ContinuousEvaluator()
        for i in range(5):
            ce._runs.append(self._make_run(f"r{i}", "t1", float(i) / 5, offset_hours=i))
        runs = ce.get_runs(limit=3)
        assert len(runs) == 3

    def test_get_runs_sorted_newest_first(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("old", "t1", 0.3, offset_hours=10))
        ce._runs.append(self._make_run("new", "t1", 0.9, offset_hours=0))
        runs = ce.get_runs(task_id="t1")
        assert runs[0].run_id == "new"

    def test_get_latest_run_returns_most_recent(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("r_old", "t1", 0.3, offset_hours=5))
        ce._runs.append(self._make_run("r_new", "t1", 0.9, offset_hours=0))
        latest = ce.get_latest_run("t1")
        assert latest is not None
        assert latest.run_id == "r_new"

    def test_get_latest_run_none_when_no_runs(self) -> None:
        ce = ContinuousEvaluator()
        assert ce.get_latest_run("nonexistent") is None


class TestContinuousEvaluatorScoreTrend:
    def _make_run(self, task_id: str, score: float, offset_hours: int = 0) -> EvaluationRun:
        ts = datetime.now() - timedelta(hours=offset_hours)
        return EvaluationRun(
            run_id=f"run_{offset_hours}",
            task_id=task_id,
            timestamp=ts,
            results=[],
            average_score=score,
        )

    def test_trend_no_data(self) -> None:
        ce = ContinuousEvaluator()
        trend = ce.get_score_trend("task_missing")
        assert trend["trend"] == "no_data"
        assert trend["runs_count"] == 0
        assert trend["average_score"] is None

    def test_trend_single_run_stable(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("t1", 0.8, offset_hours=1))
        trend = ce.get_score_trend("t1")
        assert trend["trend"] == "stable"
        assert trend["runs_count"] == 1

    def test_trend_contains_scores_list(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("t1", 0.6, offset_hours=2))
        ce._runs.append(self._make_run("t1", 0.8, offset_hours=1))
        trend = ce.get_score_trend("t1")
        assert "scores" in trend
        assert len(trend["scores"]) == 2

    def test_trend_contains_timestamps(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("t1", 0.7, offset_hours=1))
        trend = ce.get_score_trend("t1")
        assert "timestamps" in trend

    def test_trend_window_days_filters_old_runs(self) -> None:
        ce = ContinuousEvaluator()
        # Within 7-day window
        ce._runs.append(self._make_run("t1", 0.9, offset_hours=1))
        # Outside 7-day window (9 days ago)
        old_ts = datetime.now() - timedelta(days=9)
        old_run = EvaluationRun(
            run_id="old_run",
            task_id="t1",
            timestamp=old_ts,
            results=[],
            average_score=0.1,
        )
        ce._runs.append(old_run)
        trend = ce.get_score_trend("t1", window_days=7)
        assert trend["runs_count"] == 1

    def test_trend_average_score_calculated(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("t1", 0.6, offset_hours=2))
        ce._runs.append(self._make_run("t1", 1.0, offset_hours=1))
        trend = ce.get_score_trend("t1")
        assert trend["average_score"] == pytest.approx(0.8)

    def test_trend_min_max_present(self) -> None:
        ce = ContinuousEvaluator()
        ce._runs.append(self._make_run("t1", 0.3, offset_hours=3))
        ce._runs.append(self._make_run("t1", 0.9, offset_hours=2))
        ce._runs.append(self._make_run("t1", 0.6, offset_hours=1))
        trend = ce.get_score_trend("t1")
        assert trend["min_score"] == pytest.approx(0.3)
        assert trend["max_score"] == pytest.approx(0.9)


class TestContinuousEvaluatorScheduler:
    def test_start_scheduler_raises_without_apscheduler(self) -> None:
        """When apscheduler is NOT installed the error should propagate."""
        import beanllm.domain.evaluation.continuous as cont_mod

        original = cont_mod.APSCHEDULER_AVAILABLE
        try:
            cont_mod.APSCHEDULER_AVAILABLE = False
            ce = ContinuousEvaluator()
            with pytest.raises(ImportError, match="apscheduler"):
                ce.start_scheduler()
        finally:
            cont_mod.APSCHEDULER_AVAILABLE = original

    def test_stop_scheduler_noop_when_none(self) -> None:
        """stop_scheduler() should not raise when no scheduler was started."""
        ce = ContinuousEvaluator()
        ce.stop_scheduler()  # Should not raise

    def test_add_task_with_schedule_raises_without_apscheduler(self) -> None:
        import beanllm.domain.evaluation.continuous as cont_mod

        original = cont_mod.APSCHEDULER_AVAILABLE
        try:
            cont_mod.APSCHEDULER_AVAILABLE = False
            ce = ContinuousEvaluator()
            ev = Evaluator()
            with pytest.raises((ImportError, ValueError)):
                ce.add_task("t1", "T1", ev, [], schedule="0 9 * * *")
        finally:
            cont_mod.APSCHEDULER_AVAILABLE = original


class TestContinuousEvaluatorRunTask:
    @pytest.mark.asyncio
    async def test_run_task_raises_for_unknown_task(self) -> None:
        ce = ContinuousEvaluator()
        with pytest.raises(ValueError, match="not found"):
            await ce.run_task("nonexistent")

    @pytest.mark.asyncio
    async def test_run_task_raises_for_disabled_task(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [], metadata=None)
        ce._tasks["t1"].enabled = False
        with pytest.raises(ValueError, match="disabled"):
            await ce.run_task("t1")

    @pytest.mark.asyncio
    async def test_run_task_increments_run_counter(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [{"prediction": "p", "reference": "r"}])
        assert ce._run_counter == 0
        await ce.run_task("t1")
        assert ce._run_counter == 1

    @pytest.mark.asyncio
    async def test_run_task_stores_run_in_runs_list(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [])
        await ce.run_task("t1")
        assert len(ce._runs) == 1

    @pytest.mark.asyncio
    async def test_run_task_returns_evaluation_run(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [])
        run = await ce.run_task("t1")
        assert isinstance(run, EvaluationRun)
        assert run.task_id == "t1"

    @pytest.mark.asyncio
    async def test_run_task_run_id_format(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "T1", ev, [])
        run = await ce.run_task("t1")
        assert run.run_id == "run_0"

    @pytest.mark.asyncio
    async def test_run_task_metadata_contains_task_name(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        ce.add_task("t1", "Named Task", ev, [])
        run = await ce.run_task("t1")
        assert run.metadata["task_name"] == "Named Task"

    @pytest.mark.asyncio
    async def test_run_task_with_test_cases_evaluates_each(self) -> None:
        ce = ContinuousEvaluator()
        ev = Evaluator()
        test_cases = [
            {"prediction": "p1", "reference": "r1"},
            {"prediction": "p2", "reference": "r2"},
        ]
        ce.add_task("t1", "T1", ev, test_cases)
        run = await ce.run_task("t1")
        assert run.metadata["test_cases_count"] == 2
