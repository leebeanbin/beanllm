"""
llmkit.evaluation - LLM Evaluation Metrics
LLM 평가 메트릭 시스템

이 모듈은 LLM 출력을 평가하기 위한 다양한 메트릭을 제공합니다.
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MetricType(Enum):
    """메트릭 타입"""

    SIMILARITY = "similarity"  # 텍스트 유사도
    SEMANTIC = "semantic"  # 의미론적 유사도
    QUALITY = "quality"  # 품질 평가
    RAG = "rag"  # RAG 전용
    CUSTOM = "custom"  # 사용자 정의


@dataclass
class EvaluationResult:
    """평가 결과"""

    metric_name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.metric_name}: {self.score:.4f}"


@dataclass
class BatchEvaluationResult:
    """배치 평가 결과"""

    results: List[EvaluationResult]
    average_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric_name: str) -> Optional[EvaluationResult]:
        """특정 메트릭 결과 가져오기"""
        for result in self.results:
            if result.metric_name == metric_name:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "results": [
                {
                    "metric": r.metric_name,
                    "score": r.score,
                    "metadata": r.metadata,
                    "explanation": r.explanation,
                }
                for r in self.results
            ],
            "average_score": self.average_score,
            "metadata": self.metadata,
        }


class BaseMetric(ABC):
    """메트릭 베이스 클래스"""

    def __init__(self, name: str, metric_type: MetricType):
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """메트릭 계산"""
        pass

    def batch_compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> BatchEvaluationResult:
        """배치 평가"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        results = []
        for pred, ref in zip(predictions, references):
            result = self.compute(pred, ref, **kwargs)
            results.append(result)

        average_score = sum(r.score for r in results) / len(results)

        return BatchEvaluationResult(
            results=results, average_score=average_score, metadata={"count": len(results)}
        )


# ===== Text Similarity Metrics =====


class ExactMatchMetric(BaseMetric):
    """
    Exact Match (정확한 일치)

    예측과 참조가 정확히 일치하는지 평가
    """

    def __init__(self, case_sensitive: bool = True, normalize_whitespace: bool = True):
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

    def __init__(self):
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
                metric_name=self.name, score=0.0, metadata={"precision": 0.0, "recall": 0.0}
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
            metadata={"precision": precision, "recall": recall, "common_tokens": num_common},
        )


class BLEUMetric(BaseMetric):
    """
    BLEU Score (Bilingual Evaluation Understudy)

    기계번역 평가에 주로 사용되는 메트릭
    N-gram precision 기반
    """

    def __init__(self, max_n: int = 4, weights: Optional[List[float]] = None):
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

    def _modified_precision(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
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

    def __init__(self, rouge_type: str = "rouge-1"):
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

    def _rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
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

    def _rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
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

        return EvaluationResult(metric_name=self.name, score=scores["f1"], metadata=scores)


# ===== Semantic Similarity Metrics =====


class SemanticSimilarityMetric(BaseMetric):
    """
    의미론적 유사도 (Embedding 기반)

    두 텍스트의 의미적 유사성을 임베딩 벡터의 코사인 유사도로 측정
    """

    def __init__(self, embedding_model=None):
        super().__init__("semantic_similarity", MetricType.SEMANTIC)
        self.embedding_model = embedding_model

    def _get_embedding_model(self):
        """임베딩 모델 lazy loading"""
        if self.embedding_model is None:
            # llmkit의 기본 임베딩 사용
            try:
                from .embeddings import OpenAIEmbedding

                self.embedding_model = OpenAIEmbedding()
            except Exception:
                raise RuntimeError(
                    "Embedding model not available. " "Please provide an embedding model."
                )
        return self.embedding_model

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        model = self._get_embedding_model()

        # 임베딩 생성
        pred_emb = model.embed(prediction)
        ref_emb = model.embed(reference)

        # 코사인 유사도
        similarity = self._cosine_similarity(pred_emb, ref_emb)

        return EvaluationResult(
            metric_name=self.name,
            score=similarity,
            metadata={"embedding_model": str(type(model).__name__)},
        )


# ===== LLM-as-Judge Metrics =====


class LLMJudgeMetric(BaseMetric):
    """
    LLM-as-a-Judge

    LLM을 사용하여 출력 품질 평가
    """

    def __init__(self, client=None, criterion: str = "quality", use_reference: bool = True):
        super().__init__(f"llm_judge_{criterion}", MetricType.QUALITY)
        self.client = client
        self.criterion = criterion
        self.use_reference = use_reference

    def _get_client(self):
        """클라이언트 lazy loading"""
        if self.client is None:
            try:
                from .client import create_client

                self.client = create_client()
            except Exception:
                raise RuntimeError("LLM client not available. " "Please provide a client.")
        return self.client

    def _create_judge_prompt(
        self, prediction: str, reference: Optional[str], criterion: str
    ) -> str:
        """Judge 프롬프트 생성"""
        if criterion == "quality":
            instruction = (
                "Evaluate the quality of the response. "
                "Consider accuracy, completeness, and clarity."
            )
        elif criterion == "relevance":
            instruction = (
                "Evaluate how relevant the response is to the reference. "
                "Consider whether it addresses the same topic and intent."
            )
        elif criterion == "factuality":
            instruction = (
                "Evaluate the factual accuracy of the response. "
                "Check if the information is correct and verifiable."
            )
        elif criterion == "coherence":
            instruction = (
                "Evaluate the coherence of the response. "
                "Check if it's well-structured and logically consistent."
            )
        elif criterion == "helpfulness":
            instruction = (
                "Evaluate how helpful the response is. "
                "Consider usefulness, actionability, and clarity."
            )
        else:
            instruction = f"Evaluate the {criterion} of the response."

        prompt_parts = [instruction]

        if self.use_reference and reference:
            prompt_parts.append(f"\nReference: {reference}")

        prompt_parts.append(f"\nResponse to evaluate: {prediction}")
        prompt_parts.append(
            "\nProvide a score from 0 to 1 (where 1 is best) and a brief explanation."
            "\nFormat your response as: SCORE: <number> EXPLANATION: <text>"
        )

        return "\n".join(prompt_parts)

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        client = self._get_client()

        # Judge 프롬프트 생성
        prompt = self._create_judge_prompt(
            prediction, reference if self.use_reference else None, self.criterion
        )

        # LLM 평가
        response = client.chat([{"role": "user", "content": prompt}])
        judge_output = response.content

        # 점수 추출
        score_match = re.search(r"SCORE:\s*([\d.]+)", judge_output)
        if score_match:
            score = float(score_match.group(1))
        else:
            # 폴백: 0-10 스케일 찾기
            score_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*(?:10|1)", judge_output)
            if score_match:
                score = float(score_match.group(1))
                if score > 1:
                    score = score / 10
            else:
                score = 0.5  # 기본값

        # 설명 추출
        explanation_match = re.search(r"EXPLANATION:\s*(.+)", judge_output, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else judge_output

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={"criterion": self.criterion},
            explanation=explanation,
        )


# ===== RAG-Specific Metrics =====


class AnswerRelevanceMetric(BaseMetric):
    """
    Answer Relevance (RAG)

    생성된 답변이 질문과 얼마나 관련있는지 평가
    """

    def __init__(self, client=None):
        super().__init__("answer_relevance", MetricType.RAG)
        self.client = client

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변
            reference: 원래 질문
        """
        question = reference
        answer = prediction

        # LLM-as-judge 사용
        judge = LLMJudgeMetric(client=self.client, criterion="relevance", use_reference=True)

        result = judge.compute(answer, question)
        result.metric_name = self.name

        return result


class ContextPrecisionMetric(BaseMetric):
    """
    Context Precision (RAG)

    검색된 컨텍스트가 질문에 대한 답변과 얼마나 관련있는지 평가
    """

    def __init__(self):
        super().__init__("context_precision", MetricType.RAG)

    def compute(
        self, prediction: str, reference: str, contexts: Optional[List[str]] = None, **kwargs
    ) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변
            reference: 원래 질문
            contexts: 검색된 컨텍스트 리스트
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name, score=0.0, metadata={"error": "No contexts provided"}
            )

        # 각 컨텍스트가 답변 생성에 사용되었는지 확인
        # 간단한 휴리스틱: 답변에 컨텍스트의 단어가 포함되어 있는지
        answer_tokens = set(prediction.lower().split())
        relevant_count = 0

        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(answer_tokens & ctx_tokens)
            # 충분한 오버랩이 있으면 관련있다고 판단
            if overlap >= min(3, len(ctx_tokens) * 0.3):
                relevant_count += 1

        precision = relevant_count / len(contexts)

        return EvaluationResult(
            metric_name=self.name,
            score=precision,
            metadata={"total_contexts": len(contexts), "relevant_contexts": relevant_count},
        )


class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness (RAG)

    생성된 답변이 제공된 컨텍스트에 충실한지 평가 (환각 검출)
    """

    def __init__(self, client=None):
        super().__init__("faithfulness", MetricType.RAG)
        self.client = client

    def _get_client(self):
        """클라이언트 lazy loading"""
        if self.client is None:
            try:
                from .client import create_client

                self.client = create_client()
            except Exception:
                raise RuntimeError("LLM client not available")
        return self.client

    def compute(
        self, prediction: str, reference: str, contexts: Optional[List[str]] = None, **kwargs
    ) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변
            reference: (사용안함)
            contexts: 검색된 컨텍스트 리스트
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name, score=0.0, metadata={"error": "No contexts provided"}
            )

        client = self._get_client()

        # Faithfulness 평가 프롬프트
        context_text = "\n\n".join(contexts)
        prompt = (
            f"Given the following context:\n{context_text}\n\n"
            f"Evaluate if the following statement is faithful to the context "
            f"(i.e., all information is supported by the context):\n{prediction}\n\n"
            f"Respond with a score from 0 to 1, where 1 means fully faithful.\n"
            f"Format: SCORE: <number>"
        )

        response = client.chat([{"role": "user", "content": prompt}])
        output = response.content

        # 점수 추출
        score_match = re.search(r"SCORE:\s*([\d.]+)", output)
        score = float(score_match.group(1)) if score_match else 0.5

        return EvaluationResult(
            metric_name=self.name, score=score, metadata={"contexts_count": len(contexts)}
        )


# ===== Custom Metrics =====


class CustomMetric(BaseMetric):
    """
    사용자 정의 메트릭

    커스텀 평가 함수를 사용하여 메트릭 생성
    """

    def __init__(
        self,
        name: str,
        compute_fn: Callable[[str, str], float],
        metric_type: MetricType = MetricType.CUSTOM,
    ):
        super().__init__(name, metric_type)
        self.compute_fn = compute_fn

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        score = self.compute_fn(prediction, reference)

        return EvaluationResult(metric_name=self.name, score=score, metadata={"type": "custom"})


# ===== Evaluator =====


class Evaluator:
    """
    통합 평가기

    여러 메트릭을 한 번에 실행
    """

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics or []

    def add_metric(self, metric: BaseMetric) -> "Evaluator":
        """메트릭 추가"""
        self.metrics.append(metric)
        return self

    def evaluate(self, prediction: str, reference: str, **kwargs) -> BatchEvaluationResult:
        """모든 메트릭으로 평가"""
        results = []

        for metric in self.metrics:
            try:
                result = metric.compute(prediction, reference, **kwargs)
                results.append(result)
            except Exception as e:
                # 에러가 나도 다른 메트릭은 계속 실행
                results.append(
                    EvaluationResult(metric_name=metric.name, score=0.0, metadata={"error": str(e)})
                )

        if not results:
            average_score = 0.0
        else:
            average_score = sum(r.score for r in results) / len(results)

        return BatchEvaluationResult(
            results=results, average_score=average_score, metadata={"metrics_count": len(results)}
        )

    def batch_evaluate(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> List[BatchEvaluationResult]:
        """배치 평가"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        batch_results = []
        for pred, ref in zip(predictions, references):
            result = self.evaluate(pred, ref, **kwargs)
            batch_results.append(result)

        return batch_results


# ===== 유틸리티 함수 =====


def evaluate_text(
    prediction: str, reference: str, metrics: Optional[List[str]] = None, **kwargs
) -> BatchEvaluationResult:
    """
    간편한 텍스트 평가

    Args:
        prediction: 예측 텍스트
        reference: 참조 텍스트
        metrics: 사용할 메트릭 이름 리스트 (기본: ["bleu", "rouge", "f1"])
    """
    if metrics is None:
        metrics = ["bleu", "rouge-1", "f1"]

    evaluator = Evaluator()

    for metric_name in metrics:
        if metric_name == "bleu":
            evaluator.add_metric(BLEUMetric())
        elif metric_name.startswith("rouge"):
            evaluator.add_metric(ROUGEMetric(rouge_type=metric_name))
        elif metric_name == "f1":
            evaluator.add_metric(F1ScoreMetric())
        elif metric_name == "exact_match":
            evaluator.add_metric(ExactMatchMetric())
        elif metric_name == "semantic":
            evaluator.add_metric(SemanticSimilarityMetric())
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    return evaluator.evaluate(prediction, reference, **kwargs)


def evaluate_rag(
    question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None, **kwargs
) -> BatchEvaluationResult:
    """
    RAG 시스템 평가

    Args:
        question: 원래 질문
        answer: 생성된 답변
        contexts: 검색된 컨텍스트
        ground_truth: 정답 (있는 경우)
    """
    evaluator = Evaluator()

    # Answer Relevance
    evaluator.add_metric(AnswerRelevanceMetric())

    # Context Precision
    evaluator.add_metric(ContextPrecisionMetric())

    # Faithfulness
    evaluator.add_metric(FaithfulnessMetric())

    # Ground truth가 있으면 일반 메트릭도 추가
    if ground_truth:
        evaluator.add_metric(F1ScoreMetric())
        evaluator.add_metric(ROUGEMetric("rouge-l"))

    return evaluator.evaluate(
        prediction=answer, reference=ground_truth or question, contexts=contexts, **kwargs
    )


def create_evaluator(metric_names: List[str]) -> Evaluator:
    """간편한 Evaluator 생성"""
    evaluator = Evaluator()

    for name in metric_names:
        if name == "bleu":
            evaluator.add_metric(BLEUMetric())
        elif name.startswith("rouge"):
            evaluator.add_metric(ROUGEMetric(rouge_type=name))
        elif name == "f1":
            evaluator.add_metric(F1ScoreMetric())
        elif name == "exact_match":
            evaluator.add_metric(ExactMatchMetric())
        elif name == "semantic":
            evaluator.add_metric(SemanticSimilarityMetric())
        else:
            raise ValueError(f"Unknown metric: {name}")

    return evaluator
