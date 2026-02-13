"""
EmbeddingAnalyzer - Embedding 분석 (UMAP, t-SNE, Clustering)
SOLID 원칙:
- SRP: Embedding 차원 축소 및 클러스터링만 담당
- OCP: 새로운 분석 방법 추가 가능
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingAnalyzer:
    """
    Embedding 분석기

    책임:
    - 고차원 embedding을 2D/3D로 축소 (UMAP, t-SNE)
    - 클러스터링 (HDBSCAN, KMeans)
    - 이상치 탐지
    - 클러스터링 품질 측정 (Silhouette score)

    Mathematical Foundation:
    - UMAP: Uniform Manifold Approximation and Projection
      - 위상학적 구조 보존
      - t-SNE보다 빠르고 전역 구조 보존 우수
    - t-SNE: t-Distributed Stochastic Neighbor Embedding
      - 지역적 구조 보존에 강점
    - HDBSCAN: Hierarchical Density-Based Spatial Clustering
      - 밀도 기반, 노이즈 처리, 클러스터 수 자동 결정
    """

    def __init__(self) -> None:
        """EmbeddingAnalyzer 초기화"""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """필수 라이브러리 확인"""
        try:
            import hdbscan  # noqa: F401
            import umap  # noqa: F401
            from sklearn.manifold import TSNE  # noqa: F401
            from sklearn.metrics import silhouette_score  # noqa: F401
        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Install with: pip install beanllm[advanced]")
            raise ImportError(
                "Advanced features require additional dependencies. "
                "Install with: pip install beanllm[advanced]"
            ) from e

    def reduce_dimensions_umap(
        self,
        embeddings: List[List[float]],
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> np.ndarray:
        """
        UMAP을 사용한 차원 축소

        Args:
            embeddings: 고차원 embedding vectors
            n_components: 축소할 차원 (2 or 3)
            n_neighbors: 이웃 수 (작을수록 지역 구조, 클수록 전역 구조)
            min_dist: 포인트 간 최소 거리 (작을수록 밀집)
            metric: 거리 메트릭 ("cosine", "euclidean", etc.)
            random_state: Random seed

        Returns:
            np.ndarray: 축소된 embeddings (n_samples, n_components)

        Mathematical Details:
            UMAP은 Riemannian manifold를 가정하고, fuzzy simplicial set을
            사용하여 고차원 구조를 저차원으로 projection합니다.
            - 위상학적 구조 보존
            - Cross-entropy 최적화
            - t-SNE 대비 빠르고 확장성 우수
        """
        import umap

        logger.info(
            f"Running UMAP: {len(embeddings)} embeddings, {len(embeddings[0])}D → {n_components}D"
        )

        embeddings_array = np.array(embeddings)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        reduced = reducer.fit_transform(embeddings_array)
        logger.info(f"UMAP completed: output shape {reduced.shape}")

        return cast(np.ndarray, reduced)

    def reduce_dimensions_tsne(
        self,
        embeddings: List[List[float]],
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        t-SNE를 사용한 차원 축소

        Args:
            embeddings: 고차원 embedding vectors
            n_components: 축소할 차원 (2 or 3)
            perplexity: 이웃 수 (5-50, 데이터셋 크기에 따라)
            learning_rate: 학습률 (10-1000)
            n_iter: 최적화 반복 횟수
            random_state: Random seed

        Returns:
            np.ndarray: 축소된 embeddings (n_samples, n_components)

        Mathematical Details:
            t-SNE는 고차원에서의 Gaussian 분포와 저차원에서의
            Student's t-distribution 간 KL divergence를 최소화합니다.
            - 지역 구조 보존에 강점
            - 전역 구조는 상대적으로 약함
            - 계산 복잡도: O(n²)
        """
        from sklearn.manifold import TSNE

        logger.info(
            f"Running t-SNE: {len(embeddings)} embeddings, {len(embeddings[0])}D → {n_components}D"
        )

        embeddings_array = np.array(embeddings)

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
        )

        reduced = tsne.fit_transform(embeddings_array)
        logger.info(f"t-SNE completed: output shape {reduced.shape}")

        return cast(np.ndarray, reduced)

    def cluster_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        HDBSCAN을 사용한 클러스터링

        Args:
            embeddings: Embedding vectors (reduced or original)
            min_cluster_size: 최소 클러스터 크기
            min_samples: 핵심 포인트 판단 이웃 수
            metric: 거리 메트릭

        Returns:
            Tuple[np.ndarray, Dict]:
                - labels: 클러스터 레이블 (-1은 noise)
                - stats: 클러스터링 통계

        Mathematical Details:
            HDBSCAN은 밀도 기반 계층적 클러스터링:
            1. Mutual reachability graph 구성
            2. Minimum spanning tree 생성
            3. Hierarchical clustering
            4. Extract optimal flat clustering
            - 노이즈 자동 탐지 (label = -1)
            - 클러스터 수 자동 결정
        """
        import hdbscan

        logger.info(
            f"Running HDBSCAN: {len(embeddings)} points, min_cluster_size={min_cluster_size}"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )

        labels = clusterer.fit_predict(embeddings)

        # Compute statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        cluster_sizes = {}
        for label in set(labels):
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))

        stats = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(labels) if len(labels) > 0 else 0.0,
            "cluster_sizes": cluster_sizes,
        }

        logger.info(
            f"HDBSCAN completed: {n_clusters} clusters, "
            f"{n_noise} noise points ({stats['noise_ratio']:.2%})"
        )

        return labels, stats

    def detect_outliers(self, embeddings: np.ndarray, labels: np.ndarray) -> List[int]:
        """
        이상치 탐지

        Args:
            embeddings: Embedding vectors
            labels: 클러스터 레이블 (HDBSCAN 결과)

        Returns:
            List[int]: 이상치 인덱스 목록

        Note:
            HDBSCAN의 noise points (label=-1)를 outliers로 간주
        """
        outlier_indices = np.where(labels == -1)[0].tolist()

        logger.info(f"Detected {len(outlier_indices)} outliers")

        return cast(List[int], outlier_indices)

    def compute_silhouette_score(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Optional[float]:
        """
        Silhouette score 계산 (클러스터링 품질 측정)

        Args:
            embeddings: Embedding vectors
            labels: 클러스터 레이블

        Returns:
            float: Silhouette score (-1 ~ 1, 높을수록 좋음)
                   None if cannot compute (e.g., only 1 cluster)

        Mathematical Details:
            Silhouette score = (b - a) / max(a, b)
            - a: 같은 클러스터 내 평균 거리 (cohesion)
            - b: 가장 가까운 다른 클러스터까지 평균 거리 (separation)
            - 1에 가까울수록: 잘 분리됨
            - 0에 가까울수록: 경계에 위치
            - -1에 가까울수록: 잘못된 클러스터
        """
        from sklearn.metrics import silhouette_score

        # Remove noise points for silhouette calculation
        mask = labels != -1
        if np.sum(mask) < 2:
            logger.warning("Not enough non-noise points for silhouette score")
            return None

        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]

        # Need at least 2 clusters
        if len(set(filtered_labels)) < 2:
            logger.warning("Need at least 2 clusters for silhouette score")
            return None

        try:
            score = silhouette_score(filtered_embeddings, filtered_labels)
            logger.info(f"Silhouette score: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.error(f"Error computing silhouette score: {e}")
            return None

    def analyze(
        self,
        embeddings: List[List[float]],
        method: str = "umap",
        n_clusters: int = 5,
        n_components: int = 2,
        detect_outliers: bool = True,
    ) -> Dict[str, Any]:
        """
        전체 embedding 분석 파이프라인

        Args:
            embeddings: 고차원 embedding vectors
            method: 차원 축소 방법 ("umap" or "tsne")
            n_clusters: HDBSCAN min_cluster_size
            n_components: 축소 차원 (2 or 3)
            detect_outliers: 이상치 탐지 여부

        Returns:
            Dict: 분석 결과
                - reduced_embeddings: 축소된 embeddings
                - labels: 클러스터 레이블
                - cluster_stats: 클러스터 통계
                - outliers: 이상치 인덱스
                - silhouette_score: 클러스터링 품질
        """
        logger.info(
            f"Starting embedding analysis: method={method}, "
            f"n_clusters={n_clusters}, n_components={n_components}"
        )

        # 1. Dimension reduction
        if method == "umap":
            reduced = self.reduce_dimensions_umap(embeddings, n_components=n_components)
        elif method == "tsne":
            reduced = self.reduce_dimensions_tsne(embeddings, n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'.")

        # 2. Clustering
        labels, cluster_stats = self.cluster_hdbscan(reduced, min_cluster_size=n_clusters)

        # 3. Outlier detection
        outliers = []
        if detect_outliers:
            outliers = self.detect_outliers(reduced, labels)

        # 4. Silhouette score
        silhouette = self.compute_silhouette_score(reduced, labels)

        results = {
            "reduced_embeddings": reduced.tolist(),
            "labels": labels.tolist(),
            "cluster_stats": cluster_stats,
            "outliers": outliers,
            "silhouette_score": silhouette,
            "method": method,
            "n_components": n_components,
        }

        logger.info("Embedding analysis completed")
        return results
