"""
Text similarity metrics: ExactMatch, F1, BLEU, ROUGE.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from beanllm.domain.evaluation.base_metric import BaseMetric
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.results import EvaluationResult


class ExactMatchMetric(BaseMetric):
    """
    Exact Match (정확한 일치)

    예측과 참조가 정확히 일치하는지 평가
    """

    def __init__(
        self,
        case_sensitive: bool = True,
        normalize_whitespace: bool = True,
    ):
        super().__init__("exact_match", MetricType.SIMILARITY)
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        pred = prediction
        ref = reference

        # 정규화
        if self.normalize_whitespace:
            pred = " ".join(pred.split())
            ref = " ".join(ref.split())

        if not self.case_sensitive:
            pred = pred.lower()
            ref = ref.lower()

        # 일치 여부
        score = 1.0 if pred == ref else 0.0

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={"prediction": prediction, "reference": reference},
        )


class F1ScoreMetric(BaseMetric):
    """
    F1 Score (토큰 기반)

    예측과 참조의 토큰 오버랩을 기반으로 F1 계산
    """

    def __init__(self) -> None:
        super().__init__("f1_score", MetricType.SIMILARITY)

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토큰화"""
        return text.lower().split()

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        # 공통 토큰
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                metadata={"precision": 0.0, "recall": 0.0},
            )

        # Precision & Recall
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0

        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return EvaluationResult(
            metric_name=self.name,
            score=f1,
            metadata={
                "precision": precision,
                "recall": recall,
                "common_tokens": num_common,
            },
        )


class BLEUMetric(BaseMetric):
    """
    BLEU Score (Bilingual Evaluation Understudy)

    기계번역 평가에 주로 사용되는 메트릭
    N-gram precision 기반
    """

    def __init__(
        self,
        max_n: int = 4,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__("bleu", MetricType.SIMILARITY)
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """N-gram 추출"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def _modified_precision(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str],
        n: int,
    ) -> float:
        """Modified n-gram precision"""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if not pred_ngrams:
            return 0.0

        # Clipped count
        clipped_count = 0
        for ngram, count in pred_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        # Precision
        total_pred = sum(pred_ngrams.values())
        return clipped_count / total_pred if total_pred > 0 else 0.0

    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """Brevity penalty (짧은 문장 패널티)"""
        if pred_len > ref_len:
            return 1.0
        elif pred_len == 0:
            return 0.0
        else:
            return math.exp(1 - ref_len / pred_len)

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        # N-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            p = self._modified_precision(pred_tokens, ref_tokens, n)
            precisions.append(p)

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            geo_mean = 0.0
        else:
            log_sum = sum(w * math.log(p) for w, p in zip(self.weights, precisions))
            geo_mean = math.exp(log_sum)

        # Brevity penalty
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))

        # BLEU score
        bleu = bp * geo_mean

        return EvaluationResult(
            metric_name=self.name,
            score=bleu,
            metadata={
                "precisions": precisions,
                "brevity_penalty": bp,
                "pred_length": len(pred_tokens),
                "ref_length": len(ref_tokens),
            },
        )


class ROUGEMetric(BaseMetric):
    """
    ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

    요약 평가에 주로 사용되는 메트릭
    """

    def __init__(self, rouge_type: str = "rouge-1") -> None:
        """
        Args:
            rouge_type: "rouge-1", "rouge-2", "rouge-l"
        """
        super().__init__(f"rouge_{rouge_type}", MetricType.SIMILARITY)
        self.rouge_type = rouge_type

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """N-gram 추출"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def _rouge_n(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str],
        n: int,
    ) -> dict:
        """ROUGE-N 계산"""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        # Overlap
        overlap = sum((pred_ngrams & ref_ngrams).values())

        # Precision, Recall, F1
        pred_total = sum(pred_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = overlap / pred_total if pred_total > 0 else 0.0
        recall = overlap / ref_total if ref_total > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Longest Common Subsequence 길이"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _rouge_l(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str],
    ) -> dict:
        """ROUGE-L 계산"""
        lcs = self._lcs_length(pred_tokens, ref_tokens)

        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)

        precision = lcs / pred_len if pred_len > 0 else 0.0
        recall = lcs / ref_len if ref_len > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        if self.rouge_type == "rouge-1":
            scores = self._rouge_n(pred_tokens, ref_tokens, 1)
        elif self.rouge_type == "rouge-2":
            scores = self._rouge_n(pred_tokens, ref_tokens, 2)
        elif self.rouge_type == "rouge-l":
            scores = self._rouge_l(pred_tokens, ref_tokens)
        else:
            raise ValueError(f"Unknown ROUGE type: {self.rouge_type}")

        return EvaluationResult(
            metric_name=self.name,
            score=scores["f1"],
            metadata=scores,
        )
