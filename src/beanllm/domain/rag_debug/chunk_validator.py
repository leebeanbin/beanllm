"""
ChunkValidator - 청크(Document) 검증
SOLID 원칙:
- SRP: 청크 검증만 담당
- OCP: 새로운 검증 규칙 추가 가능
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkValidator:
    """
    청크 검증기

    책임:
    - 청크 크기 검증
    - 중복 청크 탐지
    - 메타데이터 검증
    - 청크 간 overlap 검증
    - 권장사항 생성
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        overlap_threshold: float = 0.9,
    ) -> None:
        """
        Args:
            min_chunk_size: 최소 청크 크기 (characters)
            max_chunk_size: 최대 청크 크기 (characters)
            overlap_threshold: 중복 판단 임계값 (Jaccard similarity)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_threshold = overlap_threshold

    def validate_size(
        self, documents: List[Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        청크 크기 검증

        Args:
            documents: Document 리스트

        Returns:
            Tuple[List[Dict], Dict]:
                - issues: 문제가 있는 청크 목록
                - distribution: 크기 분포 {"0-200": 10, "200-500": 50, ...}
        """
        logger.info(f"Validating chunk sizes for {len(documents)} documents")

        issues = []
        sizes = []

        for i, doc in enumerate(documents):
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            size = len(content)
            sizes.append(size)

            if size < self.min_chunk_size:
                issues.append(
                    {
                        "type": "size_too_small",
                        "chunk_id": i,
                        "size": size,
                        "min_size": self.min_chunk_size,
                        "details": f"Chunk size {size} < {self.min_chunk_size}",
                    }
                )
            elif size > self.max_chunk_size:
                issues.append(
                    {
                        "type": "size_too_large",
                        "chunk_id": i,
                        "size": size,
                        "max_size": self.max_chunk_size,
                        "details": f"Chunk size {size} > {self.max_chunk_size}",
                    }
                )

        # Compute size distribution
        distribution = self._compute_size_distribution(sizes)

        logger.info(
            f"Size validation completed: {len(issues)} issues found, "
            f"distribution: {distribution}"
        )

        return issues, distribution

    def _compute_size_distribution(self, sizes: List[int]) -> Dict[str, int]:
        """
        크기 분포 계산

        Args:
            sizes: 청크 크기 리스트

        Returns:
            Dict: 크기 구간별 개수
        """
        bins = [0, 200, 500, 1000, 2000, float("inf")]
        labels = ["0-200", "200-500", "500-1000", "1000-2000", "2000+"]

        distribution = {label: 0 for label in labels}

        for size in sizes:
            for i in range(len(bins) - 1):
                if bins[i] <= size < bins[i + 1]:
                    distribution[labels[i]] += 1
                    break

        return distribution

    def detect_duplicates(
        self, documents: List[Any], threshold: Optional[float] = None
    ) -> List[Tuple[int, int]]:
        """
        중복 청크 탐지 (Jaccard similarity 기반)

        Args:
            documents: Document 리스트
            threshold: Jaccard similarity 임계값 (None이면 self.overlap_threshold 사용)

        Returns:
            List[Tuple[int, int]]: 중복 청크 인덱스 쌍

        Mathematical Details:
            Jaccard similarity = |A ∩ B| / |A ∪ B|
            - A, B: 두 청크의 단어 집합
            - 0 ~ 1 사이 값 (1에 가까울수록 유사)
        """
        threshold = threshold or self.overlap_threshold
        logger.info(
            f"Detecting duplicates in {len(documents)} documents "
            f"(threshold={threshold})"
        )

        duplicates = []

        # Compute Jaccard similarity for all pairs (O(n²))
        # For large datasets, use LSH (Locality Sensitive Hashing)
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                content_i = (
                    documents[i].page_content
                    if hasattr(documents[i], "page_content")
                    else str(documents[i])
                )
                content_j = (
                    documents[j].page_content
                    if hasattr(documents[j], "page_content")
                    else str(documents[j])
                )

                similarity = self._jaccard_similarity(content_i, content_j)

                if similarity >= threshold:
                    duplicates.append((i, j))

        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Jaccard similarity 계산

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            float: Jaccard similarity (0 ~ 1)
        """
        # Tokenize (simple word-based)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            return 1.0  # Both empty

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        return intersection / union

    def validate_metadata(
        self, documents: List[Any], required_keys: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        메타데이터 검증

        Args:
            documents: Document 리스트
            required_keys: 필수 메타데이터 키 목록

        Returns:
            List[Dict]: 메타데이터 문제 목록
        """
        required_keys = required_keys or []
        logger.info(
            f"Validating metadata for {len(documents)} documents "
            f"(required keys: {required_keys})"
        )

        issues = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            # Check for missing required keys
            for key in required_keys:
                if key not in metadata:
                    issues.append(
                        {
                            "type": "missing_metadata",
                            "chunk_id": i,
                            "missing_key": key,
                            "details": f"Required metadata key '{key}' is missing",
                        }
                    )

            # Check for empty metadata
            if not metadata:
                issues.append(
                    {
                        "type": "empty_metadata",
                        "chunk_id": i,
                        "details": "Metadata is empty",
                    }
                )

        logger.info(f"Metadata validation completed: {len(issues)} issues found")
        return issues

    def check_overlap(
        self, documents: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        청크 간 overlap 통계

        Args:
            documents: Document 리스트

        Returns:
            Dict: Overlap 통계 또는 None (계산 불가능 시)

        Note:
            순차적으로 인접한 청크 간 overlap만 확인
            전체 pair-wise는 O(n²)로 비효율적
        """
        logger.info(f"Checking overlap for {len(documents)} documents")

        if len(documents) < 2:
            logger.warning("Need at least 2 documents to check overlap")
            return None

        overlaps = []

        for i in range(len(documents) - 1):
            content_i = (
                documents[i].page_content
                if hasattr(documents[i], "page_content")
                else str(documents[i])
            )
            content_j = (
                documents[i + 1].page_content
                if hasattr(documents[i + 1], "page_content")
                else str(documents[i + 1])
            )

            # Find longest common substring (LCS)
            lcs_length = self._longest_common_substring_length(content_i, content_j)
            overlap_ratio = lcs_length / min(len(content_i), len(content_j))

            overlaps.append(
                {
                    "chunk_pair": (i, i + 1),
                    "lcs_length": lcs_length,
                    "overlap_ratio": overlap_ratio,
                }
            )

        # Statistics
        overlap_ratios = [o["overlap_ratio"] for o in overlaps]
        stats = {
            "num_pairs": len(overlaps),
            "avg_overlap_ratio": sum(overlap_ratios) / len(overlap_ratios)
            if overlap_ratios
            else 0.0,
            "max_overlap_ratio": max(overlap_ratios) if overlap_ratios else 0.0,
            "min_overlap_ratio": min(overlap_ratios) if overlap_ratios else 0.0,
        }

        logger.info(f"Overlap check completed: {stats}")
        return stats

    def _longest_common_substring_length(self, text1: str, text2: str) -> int:
        """
        최장 공통 부분 문자열 길이 (LCS)

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            int: LCS 길이

        Algorithm:
            Dynamic Programming O(m*n)
        """
        m, n = len(text1), len(text2)
        if m == 0 or n == 0:
            return 0

        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])

        return max_length

    def generate_recommendations(
        self,
        size_issues: List[Dict[str, Any]],
        duplicates: List[Tuple[int, int]],
        metadata_issues: List[Dict[str, Any]],
        overlap_stats: Optional[Dict[str, Any]],
    ) -> List[str]:
        """
        검증 결과 기반 권장사항 생성

        Args:
            size_issues: 크기 문제 목록
            duplicates: 중복 청크 목록
            metadata_issues: 메타데이터 문제 목록
            overlap_stats: Overlap 통계

        Returns:
            List[str]: 권장사항 목록
        """
        recommendations = []

        # Size recommendations
        too_small = sum(1 for issue in size_issues if issue["type"] == "size_too_small")
        too_large = sum(1 for issue in size_issues if issue["type"] == "size_too_large")

        if too_small > 0:
            recommendations.append(
                f"⚠️  {too_small} chunks are too small (< {self.min_chunk_size} chars). "
                "Consider increasing chunk_size or reducing chunk_overlap."
            )

        if too_large > 0:
            recommendations.append(
                f"⚠️  {too_large} chunks are too large (> {self.max_chunk_size} chars). "
                "Consider decreasing chunk_size."
            )

        # Duplicate recommendations
        if len(duplicates) > 0:
            recommendations.append(
                f"⚠️  Found {len(duplicates)} duplicate chunk pairs. "
                "Consider deduplication or adjusting chunk_overlap."
            )

        # Metadata recommendations
        if len(metadata_issues) > 0:
            recommendations.append(
                f"⚠️  {len(metadata_issues)} metadata issues found. "
                "Ensure all chunks have proper metadata (source, page, etc.)."
            )

        # Overlap recommendations
        if overlap_stats and overlap_stats["avg_overlap_ratio"] > 0.5:
            recommendations.append(
                f"⚠️  High overlap detected (avg {overlap_stats['avg_overlap_ratio']:.2%}). "
                "Consider reducing chunk_overlap parameter."
            )
        elif overlap_stats and overlap_stats["avg_overlap_ratio"] < 0.1:
            recommendations.append(
                f"💡 Low overlap detected (avg {overlap_stats['avg_overlap_ratio']:.2%}). "
                "This may cause loss of context. Consider increasing chunk_overlap."
            )

        if not recommendations:
            recommendations.append("✅ No major issues detected. Chunks look good!")

        return recommendations

    def validate_all(
        self,
        documents: List[Any],
        required_metadata_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        전체 검증 파이프라인

        Args:
            documents: Document 리스트
            required_metadata_keys: 필수 메타데이터 키

        Returns:
            Dict: 검증 결과
                - total_chunks: 총 청크 수
                - size_issues: 크기 문제
                - size_distribution: 크기 분포
                - duplicates: 중복 청크
                - metadata_issues: 메타데이터 문제
                - overlap_stats: Overlap 통계
                - recommendations: 권장사항
        """
        logger.info(f"Starting full validation for {len(documents)} documents")

        # 1. Size validation
        size_issues, size_distribution = self.validate_size(documents)

        # 2. Duplicate detection
        duplicates = self.detect_duplicates(documents)

        # 3. Metadata validation
        metadata_issues = self.validate_metadata(documents, required_metadata_keys)

        # 4. Overlap check
        overlap_stats = self.check_overlap(documents)

        # 5. Generate recommendations
        recommendations = self.generate_recommendations(
            size_issues, duplicates, metadata_issues, overlap_stats
        )

        results = {
            "total_chunks": len(documents),
            "valid_chunks": len(documents)
            - len(size_issues)
            - len(duplicates)
            - len(metadata_issues),
            "size_issues": size_issues,
            "size_distribution": size_distribution,
            "duplicate_chunks": duplicates,
            "metadata_issues": metadata_issues,
            "overlap_stats": overlap_stats,
            "recommendations": recommendations,
        }

        logger.info("Full validation completed")
        return results
