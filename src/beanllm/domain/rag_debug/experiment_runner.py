"""
Chunking Experimenter - Experiment execution logic.

Extracted from chunking_experimenter.py for single responsibility.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

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
    retrieval_scores: List[float]
    avg_retrieval_score: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_splitter(config: Dict[str, Any]) -> Any:
    """Create splitter from config."""
    splitter_type = config.get("type", "recursive")

    if splitter_type == "recursive":
        from beanllm.domain.splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            separators=config.get("separators"),
        )
    if splitter_type == "character":
        from beanllm.domain.splitters import CharacterTextSplitter

        return CharacterTextSplitter(
            separator=config.get("separator", "\n\n"),
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
        )
    if splitter_type == "token":
        from beanllm.domain.splitters import TokenTextSplitter

        return TokenTextSplitter(
            encoding_name=config.get("encoding_name", "cl100k_base"),
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
        )
    if splitter_type == "semantic":
        from beanllm.domain.splitters.semantic import SemanticTextSplitter

        return SemanticTextSplitter(
            model=config.get("model", "all-MiniLM-L6-v2"),
            threshold=config.get("threshold", 0.5),
            min_chunk_size=config.get("min_chunk_size", 100),
            max_chunk_size=config.get("max_chunk_size", 1000),
            use_semchunk=config.get("use_semchunk", False),
        )
    if splitter_type == "coherence":
        from beanllm.domain.splitters.semantic import CoherenceTextSplitter

        return CoherenceTextSplitter(
            model=config.get("model", "all-MiniLM-L6-v2"),
            num_clusters=config.get("num_clusters", "auto"),
            max_chunk_size=config.get("max_chunk_size", 1000),
        )
    raise ValueError(f"Unknown splitter type: {splitter_type}")


def chunk_documents(documents: List[str], config: Dict[str, Any]) -> List[str]:
    """Chunk documents with given config."""
    splitter = get_splitter(config)
    all_chunks: List[str] = []
    for doc in documents:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
    return all_chunks


def compute_similarity(
    query: str,
    chunks: List[str],
    embedding_function: Optional[Callable[[str], List[float]]],
) -> List[float]:
    """Compute query-chunk similarities."""
    if embedding_function is None:
        query_words = set(query.lower().split())
        similarities = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            union = len(query_words | chunk_words)
            sim = overlap / union if union > 0 else 0
            similarities.append(sim)
        return similarities

    import math

    query_emb = embedding_function(query)
    similarities = []
    for chunk in chunks:
        chunk_emb = embedding_function(chunk)
        dot_product = sum(a * b for a, b in zip(query_emb, chunk_emb))
        mag1 = math.sqrt(sum(a * a for a in query_emb))
        mag2 = math.sqrt(sum(b * b for b in chunk_emb))
        sim = dot_product / (mag1 * mag2) if (mag1 > 0 and mag2 > 0) else 0.0
        similarities.append(sim)
    return similarities


def evaluate_retrieval(
    query: str,
    chunks: List[str],
    embedding_function: Optional[Callable[[str], List[float]]],
    ground_truth: Dict[str, List[int]],
    top_k: int = 5,
) -> float:
    """Evaluate retrieval quality for one query."""
    similarities = compute_similarity(query, chunks, embedding_function)
    sorted_sims = sorted(similarities, reverse=True)[:top_k]
    avg_sim = sum(sorted_sims) / len(sorted_sims) if sorted_sims else 0.0

    if query in ground_truth:
        gt_indices = set(ground_truth[query])
        top_k_indices = set(
            sorted(
                range(len(similarities)),
                key=lambda i: similarities[i],
                reverse=True,
            )[:top_k]
        )
        recall = (
            len(gt_indices & top_k_indices) / len(gt_indices)
            if gt_indices
            else 0.0
        )
        return (avg_sim + recall) / 2
    return avg_sim


def run_experiment(
    documents: List[str],
    test_queries: List[str],
    config: Dict[str, Any],
    embedding_function: Optional[Callable[[str], List[float]]],
    ground_truth: Dict[str, List[int]],
    strategy_name: Optional[str] = None,
) -> ChunkingResult:
    """Run a single chunking strategy experiment."""
    if strategy_name is None:
        strategy_name = f"{config.get('type', 'unknown')}_{config.get('chunk_size', 'auto')}"

    logger.info(f"Running experiment: {strategy_name}")
    start_time = time.time()

    chunks = chunk_documents(documents, config)
    retrieval_scores = []
    for query in test_queries:
        score = evaluate_retrieval(
            query, chunks, embedding_function, ground_truth
        )
        retrieval_scores.append(score)

    latency_ms = (time.time() - start_time) * 1000
    avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

    result = ChunkingResult(
        strategy_name=strategy_name,
        strategy_config=config,
        chunks=chunks,
        chunk_count=len(chunks),
        avg_chunk_size=avg_chunk_size,
        retrieval_scores=retrieval_scores,
        avg_retrieval_score=(
            sum(retrieval_scores) / len(retrieval_scores)
            if retrieval_scores
            else 0
        ),
        latency_ms=latency_ms,
        metadata={
            "doc_count": len(documents),
            "query_count": len(test_queries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    logger.info(
        f"Experiment complete: {strategy_name} | "
        f"chunks={len(chunks)}, avg_score={result.avg_retrieval_score:.4f}"
    )
    return result


def compare_strategies(
    documents: List[str],
    test_queries: List[str],
    configs: List[Dict[str, Any]],
    embedding_function: Optional[Callable[[str], List[float]]],
    ground_truth: Dict[str, List[int]],
    run_experiment_fn: Callable[..., ChunkingResult],
) -> List[ChunkingResult]:
    """Compare multiple chunking strategies."""
    results: List[ChunkingResult] = []
    for i, config in enumerate(configs):
        name = config.get("name", f"strategy_{i}")
        result = run_experiment_fn(config, strategy_name=name)
        results.append(result)
    results.sort(key=lambda r: r.avg_retrieval_score, reverse=True)
    return results


def build_grid_configs(
    splitter_type: str = "recursive",
    chunk_sizes: List[int] = (256, 512, 1000, 2000),
    chunk_overlaps: List[int] = (0, 50, 100, 200),
    **fixed_params: Any,
) -> List[Dict[str, Any]]:
    """Build config list for grid search."""
    configs = []
    for size in chunk_sizes:
        for overlap in chunk_overlaps:
            if overlap < size:
                configs.append(
                    {
                        "type": splitter_type,
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "name": f"{splitter_type}_s{size}_o{overlap}",
                        **fixed_params,
                    }
                )
    return configs
