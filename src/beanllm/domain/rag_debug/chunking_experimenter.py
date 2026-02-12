"""
Chunking Experimenter - 청킹 전략 실험 및 개선 도구

다양한 청킹 전략을 테스트하고, 결과를 비교하여
최적의 청킹 설정을 찾습니다.

주요 기능:
1. 청킹 전략 A/B 테스트
2. 검색 품질 기반 청킹 평가
3. 피드백 기반 청킹 개선
4. 파라미터 그리드 서치

Example:
    ```python
    from beanllm.domain.rag_debug import ChunkingExperimenter

    # 실험 설정
    experimenter = ChunkingExperimenter(
        documents=documents,
        test_queries=["What is RAG?", "How does embedding work?"],
        embedding_function=embed_fn
    )

    # 전략 비교
    results = experimenter.compare_strategies([
        {"type": "recursive", "chunk_size": 500, "chunk_overlap": 50},
        {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
        {"type": "semantic", "threshold": 0.5, "max_chunk_size": 1000},
    ])

    # 최적 전략 찾기
    best = experimenter.find_best_strategy()
    print(f"Best strategy: {best['strategy']}")
    print(f"Score: {best['score']:.4f}")

    # 피드백 기반 개선
    experimenter.add_feedback(query="What is RAG?", chunk_id="chunk_3", rating=0.2)
    improved = experimenter.improve_from_feedback()
    ```
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class ChunkingResult:
    """청킹 실험 결과"""

    strategy_name: str
    strategy_config: Dict[str, Any]
    chunks: List[str]
    chunk_count: int
    avg_chunk_size: float
    retrieval_scores: List[float]  # 각 쿼리별 검색 점수
    avg_retrieval_score: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkFeedback:
    """청크 피드백"""

    query: str
    chunk_id: str
    chunk_content: str
    rating: float  # 0.0 (bad) ~ 1.0 (good)
    feedback_type: str  # "relevance", "completeness", "coherence"
    comment: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ChunkingExperimenter:
    """
    청킹 전략 실험 및 개선 도구

    다양한 청킹 전략을 테스트하고, 검색 품질을 비교하여
    최적의 청킹 설정을 찾습니다.

    Features:
    - 청킹 전략 A/B 테스트
    - 파라미터 그리드 서치
    - 피드백 기반 개선 루프
    - 청킹 품질 메트릭

    Example:
        ```python
        experimenter = ChunkingExperimenter(
            documents=documents,
            test_queries=queries,
            embedding_function=embed_fn
        )

        # 전략 비교
        results = experimenter.compare_strategies([
            {"type": "recursive", "chunk_size": 500},
            {"type": "semantic", "threshold": 0.5},
        ])

        # 최적 전략
        best = experimenter.find_best_strategy()
        ```
    """

    def __init__(
        self,
        documents: List[str],
        test_queries: List[str],
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        ground_truth: Optional[Dict[str, List[int]]] = None,
        **kwargs: Any,
    ):
        """
        ChunkingExperimenter 초기화

        Args:
            documents: 원본 문서 목록
            test_queries: 테스트용 쿼리 목록
            embedding_function: 임베딩 함수 (str -> List[float])
            ground_truth: 각 쿼리의 정답 청크 인덱스
                예: {"query1": [0, 2, 5], "query2": [1, 3]}
            **kwargs: 추가 옵션
        """
        self.documents = documents
        self.test_queries = test_queries
        self.embedding_function = embedding_function
        self.ground_truth = ground_truth or {}
        self.kwargs = kwargs

        # 실험 결과 저장
        self._results: List[ChunkingResult] = []
        self._feedbacks: List[ChunkFeedback] = []
        self._current_chunks: Dict[str, List[str]] = {}  # strategy_name -> chunks

        logger.info(
            f"ChunkingExperimenter initialized: {len(documents)} docs, {len(test_queries)} queries"
        )

    def _get_splitter(self, config: Dict[str, Any]):
        """설정에 따른 Splitter 생성"""
        splitter_type = config.get("type", "recursive")

        if splitter_type == "recursive":
            from beanllm.domain.splitters import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
                separators=config.get("separators"),
            )

        elif splitter_type == "character":
            from beanllm.domain.splitters import CharacterTextSplitter

            return CharacterTextSplitter(
                separator=config.get("separator", "\n\n"),
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
            )

        elif splitter_type == "token":
            from beanllm.domain.splitters import TokenTextSplitter

            return TokenTextSplitter(
                encoding_name=config.get("encoding_name", "cl100k_base"),
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
            )

        elif splitter_type == "semantic":
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            return SemanticTextSplitter(
                model=config.get("model", "all-MiniLM-L6-v2"),
                threshold=config.get("threshold", 0.5),
                min_chunk_size=config.get("min_chunk_size", 100),
                max_chunk_size=config.get("max_chunk_size", 1000),
                use_semchunk=config.get("use_semchunk", False),
            )

        elif splitter_type == "coherence":
            from beanllm.domain.splitters.semantic import CoherenceTextSplitter

            return CoherenceTextSplitter(
                model=config.get("model", "all-MiniLM-L6-v2"),
                num_clusters=config.get("num_clusters", "auto"),
                max_chunk_size=config.get("max_chunk_size", 1000),
            )

        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

    def _chunk_documents(self, config: Dict[str, Any]) -> List[str]:
        """문서를 청킹"""
        splitter = self._get_splitter(config)

        all_chunks = []
        for doc in self.documents:
            chunks = splitter.split_text(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def _compute_similarity(self, query: str, chunks: List[str]) -> List[float]:
        """쿼리와 청크 간 유사도 계산"""
        if self.embedding_function is None:
            # 간단한 단어 겹침 기반 유사도
            query_words = set(query.lower().split())
            similarities = []
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                overlap = len(query_words & chunk_words)
                union = len(query_words | chunk_words)
                sim = overlap / union if union > 0 else 0
                similarities.append(sim)
            return similarities

        # 임베딩 기반 유사도
        import math

        query_emb = self.embedding_function(query)
        similarities = []

        for chunk in chunks:
            chunk_emb = self.embedding_function(chunk)

            # 코사인 유사도
            dot_product = sum(a * b for a, b in zip(query_emb, chunk_emb))
            mag1 = math.sqrt(sum(a * a for a in query_emb))
            mag2 = math.sqrt(sum(b * b for b in chunk_emb))

            if mag1 > 0 and mag2 > 0:
                sim = dot_product / (mag1 * mag2)
            else:
                sim = 0.0

            similarities.append(sim)

        return similarities

    def _evaluate_retrieval(self, query: str, chunks: List[str], top_k: int = 5) -> float:
        """검색 품질 평가"""
        similarities = self._compute_similarity(query, chunks)

        # Top-k 평균 유사도
        sorted_sims = sorted(similarities, reverse=True)[:top_k]
        avg_sim = sum(sorted_sims) / len(sorted_sims) if sorted_sims else 0.0

        # Ground truth가 있으면 Recall@k 계산
        if query in self.ground_truth:
            gt_indices = set(self.ground_truth[query])
            top_k_indices = set(
                sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[
                    :top_k
                ]
            )
            recall = len(gt_indices & top_k_indices) / len(gt_indices) if gt_indices else 0.0
            # Recall과 유사도 조합
            return (avg_sim + recall) / 2

        return avg_sim

    def run_experiment(
        self, config: Dict[str, Any], strategy_name: Optional[str] = None
    ) -> ChunkingResult:
        """
        단일 청킹 전략 실험 실행

        Args:
            config: 청킹 설정
            strategy_name: 전략 이름 (기본: auto-generated)

        Returns:
            ChunkingResult 객체
        """
        import time

        if strategy_name is None:
            strategy_name = f"{config.get('type', 'unknown')}_{config.get('chunk_size', 'auto')}"

        logger.info(f"Running experiment: {strategy_name}")
        start_time = time.time()

        # 1. 청킹
        chunks = self._chunk_documents(config)
        self._current_chunks[strategy_name] = chunks

        # 2. 각 쿼리에 대해 검색 품질 평가
        retrieval_scores = []
        for query in self.test_queries:
            score = self._evaluate_retrieval(query, chunks)
            retrieval_scores.append(score)

        # 3. 결과 집계
        latency_ms = (time.time() - start_time) * 1000
        avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

        result = ChunkingResult(
            strategy_name=strategy_name,
            strategy_config=config,
            chunks=chunks,
            chunk_count=len(chunks),
            avg_chunk_size=avg_chunk_size,
            retrieval_scores=retrieval_scores,
            avg_retrieval_score=sum(retrieval_scores) / len(retrieval_scores)
            if retrieval_scores
            else 0,
            latency_ms=latency_ms,
            metadata={
                "doc_count": len(self.documents),
                "query_count": len(self.test_queries),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        self._results.append(result)

        logger.info(
            f"Experiment complete: {strategy_name} | "
            f"chunks={len(chunks)}, avg_score={result.avg_retrieval_score:.4f}"
        )

        return result

    def compare_strategies(self, configs: List[Dict[str, Any]]) -> List[ChunkingResult]:
        """
        여러 청킹 전략 비교

        Args:
            configs: 청킹 설정 목록

        Returns:
            ChunkingResult 목록 (점수 내림차순)
        """
        results = []
        for i, config in enumerate(configs):
            name = config.get("name", f"strategy_{i}")
            result = self.run_experiment(config, strategy_name=name)
            results.append(result)

        # 점수순 정렬
        results.sort(key=lambda r: r.avg_retrieval_score, reverse=True)

        return results

    def grid_search(
        self,
        splitter_type: str = "recursive",
        chunk_sizes: List[int] = [256, 512, 1000, 2000],
        chunk_overlaps: List[int] = [0, 50, 100, 200],
        **fixed_params: Any,
    ) -> List[ChunkingResult]:
        """
        청킹 파라미터 그리드 서치

        Args:
            splitter_type: 스플리터 타입
            chunk_sizes: 테스트할 청크 크기 목록
            chunk_overlaps: 테스트할 오버랩 크기 목록
            **fixed_params: 고정 파라미터

        Returns:
            모든 조합의 결과 (점수 내림차순)
        """
        configs = []
        for size in chunk_sizes:
            for overlap in chunk_overlaps:
                if overlap < size:  # 오버랩은 청크 크기보다 작아야 함
                    config = {
                        "type": splitter_type,
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "name": f"{splitter_type}_s{size}_o{overlap}",
                        **fixed_params,
                    }
                    configs.append(config)

        logger.info(f"Grid search: {len(configs)} configurations")
        return self.compare_strategies(configs)

    def find_best_strategy(self) -> Optional[Dict[str, Any]]:
        """
        최적 전략 찾기

        Returns:
            최적 전략 정보 (없으면 None)
        """
        if not self._results:
            logger.warning("No experiment results. Run experiments first.")
            return None

        best = max(self._results, key=lambda r: r.avg_retrieval_score)

        return {
            "strategy": best.strategy_name,
            "config": best.strategy_config,
            "score": best.avg_retrieval_score,
            "chunk_count": best.chunk_count,
            "avg_chunk_size": best.avg_chunk_size,
        }

    def add_feedback(
        self,
        query: str,
        chunk_id: str,
        rating: float,
        feedback_type: str = "relevance",
        comment: Optional[str] = None,
    ):
        """
        청크 피드백 추가

        Args:
            query: 쿼리
            chunk_id: 청크 ID (예: "strategy_name:chunk_index")
            rating: 평점 (0.0 ~ 1.0)
            feedback_type: 피드백 유형 ("relevance", "completeness", "coherence")
            comment: 추가 코멘트
        """
        # 청크 내용 찾기
        chunk_content = ""
        if ":" in chunk_id:
            strategy, idx_str = chunk_id.rsplit(":", 1)
            if strategy in self._current_chunks:
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(self._current_chunks[strategy]):
                        chunk_content = self._current_chunks[strategy][idx][:200]
                except ValueError:
                    pass

        feedback = ChunkFeedback(
            query=query,
            chunk_id=chunk_id,
            chunk_content=chunk_content,
            rating=rating,
            feedback_type=feedback_type,
            comment=comment,
        )

        self._feedbacks.append(feedback)
        logger.info(f"Feedback added: {chunk_id} = {rating:.2f}")

    def get_feedback_summary(self) -> Dict[str, Any]:
        """피드백 요약"""
        if not self._feedbacks:
            return {"total": 0, "by_type": {}, "avg_rating": 0.0}

        by_type: Dict[str, List[float]] = {}
        for fb in self._feedbacks:
            if fb.feedback_type not in by_type:
                by_type[fb.feedback_type] = []
            by_type[fb.feedback_type].append(fb.rating)

        type_summary = {
            t: {"count": len(ratings), "avg": sum(ratings) / len(ratings)}
            for t, ratings in by_type.items()
        }

        all_ratings = [fb.rating for fb in self._feedbacks]

        return {
            "total": len(self._feedbacks),
            "by_type": type_summary,
            "avg_rating": sum(all_ratings) / len(all_ratings),
            "low_rated_chunks": [fb.chunk_id for fb in self._feedbacks if fb.rating < 0.3],
        }

    def improve_from_feedback(self, min_rating_threshold: float = 0.3) -> Dict[str, Any]:
        """
        피드백 기반 개선 제안

        Args:
            min_rating_threshold: 낮은 평점 임계값

        Returns:
            개선 제안
        """
        summary = self.get_feedback_summary()

        if summary["total"] == 0:
            return {"suggestions": ["피드백이 없습니다. 청크를 평가해주세요."]}

        suggestions = []

        # 낮은 평점 청크 분석
        low_rated = summary.get("low_rated_chunks", [])
        if low_rated:
            suggestions.append(
                f"{len(low_rated)}개의 청크가 낮은 평점을 받았습니다. "
                "청크 크기를 줄이거나 시맨틱 청킹을 시도해보세요."
            )

        # 피드백 유형별 분석
        by_type = summary.get("by_type", {})
        if "relevance" in by_type and by_type["relevance"]["avg"] < 0.5:
            suggestions.append(
                "관련성 점수가 낮습니다. 청크 오버랩을 늘리거나 시맨틱 청킹을 사용해보세요."
            )

        if "completeness" in by_type and by_type["completeness"]["avg"] < 0.5:
            suggestions.append(
                "완전성 점수가 낮습니다. 청크 크기를 늘려 더 많은 컨텍스트를 포함하세요."
            )

        if "coherence" in by_type and by_type["coherence"]["avg"] < 0.5:
            suggestions.append(
                "일관성 점수가 낮습니다. 시맨틱 청킹이나 문장 경계 기반 분할을 시도해보세요."
            )

        # 자동 재실험 제안
        if summary["avg_rating"] < 0.5:
            suggestions.append(
                "전체 평균 평점이 낮습니다. 다른 청킹 전략을 시도해보세요:\n"
                "  - 시맨틱 청킹: SemanticTextSplitter(threshold=0.5)\n"
                "  - 청크 크기 증가: chunk_size=1500, chunk_overlap=200\n"
                "  - 일관성 기반: CoherenceTextSplitter()"
            )

        return {
            "feedback_summary": summary,
            "suggestions": suggestions,
            "recommended_configs": self._generate_improved_configs(summary),
        }

    def _generate_improved_configs(self, feedback_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """피드백 기반 개선된 설정 생성"""
        configs = []

        avg_rating = feedback_summary.get("avg_rating", 0.5)

        # 현재 최고 점수 전략 기반
        best = self.find_best_strategy()
        if best:
            base_config = best["config"].copy()

            # 평점이 낮으면 청크 크기 조정
            if avg_rating < 0.5:
                # 청크 크기 증가
                configs.append(
                    {
                        **base_config,
                        "chunk_size": int(base_config.get("chunk_size", 1000) * 1.5),
                        "chunk_overlap": int(base_config.get("chunk_overlap", 100) * 1.5),
                        "name": "improved_larger_chunks",
                    }
                )

                # 시맨틱 청킹 추천
                configs.append(
                    {
                        "type": "semantic",
                        "threshold": 0.5,
                        "max_chunk_size": 1200,
                        "name": "improved_semantic",
                    }
                )

        return configs

    def get_comparison_report(self) -> str:
        """실험 결과 비교 리포트 생성"""
        if not self._results:
            return "No experiment results."

        lines = ["# Chunking Strategy Comparison Report", ""]

        # 결과 테이블
        lines.append("## Results (sorted by score)")
        lines.append("")
        lines.append("| Strategy | Score | Chunks | Avg Size | Latency |")
        lines.append("|----------|-------|--------|----------|---------|")

        for r in sorted(self._results, key=lambda x: x.avg_retrieval_score, reverse=True):
            lines.append(
                f"| {r.strategy_name} | {r.avg_retrieval_score:.4f} | "
                f"{r.chunk_count} | {r.avg_chunk_size:.0f} | {r.latency_ms:.1f}ms |"
            )

        # 최고 전략
        best = self.find_best_strategy()
        if best:
            lines.append("")
            lines.append("## Best Strategy")
            lines.append(f"- **Name**: {best['strategy']}")
            lines.append(f"- **Score**: {best['score']:.4f}")
            lines.append(f"- **Config**: {best['config']}")

        # 피드백 요약
        fb_summary = self.get_feedback_summary()
        if fb_summary["total"] > 0:
            lines.append("")
            lines.append("## Feedback Summary")
            lines.append(f"- Total feedbacks: {fb_summary['total']}")
            lines.append(f"- Average rating: {fb_summary['avg_rating']:.2f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ChunkingExperimenter("
            f"docs={len(self.documents)}, "
            f"queries={len(self.test_queries)}, "
            f"experiments={len(self._results)})"
        )
