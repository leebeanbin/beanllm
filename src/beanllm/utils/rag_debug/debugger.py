"""
RAG Debug Utils - RAG 파이프라인 디버깅 및 검증 도구
중간 과정을 확인하고 문제를 찾는 데 도움
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # numpy 대체 함수들
    class np:
        @staticmethod
        def array(x):
            return x

        @staticmethod
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        @staticmethod
        def linalg_norm(x):
            return sum(v**2 for v in x) ** 0.5

        class linalg:
            @staticmethod
            def norm(x):
                return sum(v**2 for v in x) ** 0.5


# 순환 참조 방지를 위해 TYPE_CHECKING 사용
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...domain.loaders import Document
else:
    # 런타임에만 import (순환 참조 방지)
    # Document는 함수 내부에서만 import
    Document = Any  # type: ignore


@dataclass
class EmbeddingInfo:
    """임베딩 정보"""

    text: str
    vector: List[float]
    dimension: int
    norm: float  # 벡터 크기
    preview: List[float]  # 앞 10개 값


@dataclass
class SimilarityInfo:
    """유사도 정보"""

    text1: str
    text2: str
    cosine_similarity: float
    euclidean_distance: float
    interpretation: str  # 해석


class RAGDebugger:
    """
    RAG 파이프라인 디버깅 도구

    Example:
        debugger = RAGDebugger()

        # 임베딩 확인
        debugger.inspect_embedding(text, vector)

        # 유사도 확인
        debugger.compare_texts(text1, text2, embedding_function)

        # Vector Store 확인
        debugger.inspect_vector_store(store, sample_queries)
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 상세 출력 여부
        """
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        """Verbose 모드일 때만 출력"""
        if self.verbose:
            print(*args, **kwargs)

    # ==================== 임베딩 검증 ====================

    def inspect_embedding(
        self, text: str, vector: List[float], show_preview: int = 10
    ) -> EmbeddingInfo:
        """
        단일 임베딩 검사

        Args:
            text: 원본 텍스트
            vector: 임베딩 벡터
            show_preview: 미리보기 개수

        Returns:
            EmbeddingInfo
        """
        dimension = len(vector)
        norm = float(np.linalg.norm(vector))
        preview = vector[:show_preview]

        info = EmbeddingInfo(
            text=text, vector=vector, dimension=dimension, norm=norm, preview=preview
        )

        self._print(f"\n{'=' * 60}")
        self._print("📊 Embedding 정보")
        self._print(f"{'=' * 60}")
        self._print(f"텍스트: {text[:100]}...")
        self._print(f"차원: {dimension}")
        self._print(f"벡터 크기 (norm): {norm:.4f}")
        self._print(f"미리보기 ({show_preview}개):")
        self._print(f"  {preview}")
        self._print(f"{'=' * 60}\n")

        return info

    def compare_embeddings(self, embeddings: List[Tuple[str, List[float]]]) -> None:
        """
        여러 임베딩 비교

        Args:
            embeddings: [(text, vector), ...] 리스트

        Example:
            debugger.compare_embeddings([
                ("강아지", vec1),
                ("개", vec2),
                ("자동차", vec3)
            ])
        """
        self._print(f"\n{'=' * 60}")
        self._print("📊 Embeddings 비교")
        self._print(f"{'=' * 60}")

        # 각 임베딩 기본 정보
        for text, vector in embeddings:
            norm = float(np.linalg.norm(vector))
            self._print(f"\n{text}:")
            self._print(f"  차원: {len(vector)}")
            self._print(f"  Norm: {norm:.4f}")
            self._print(f"  앞 5개: {vector[:5]}")

        # 유사도 매트릭스
        self._print(f"\n{'=' * 60}")
        self._print("유사도 매트릭스 (Cosine Similarity):")
        self._print(f"{'=' * 60}")

        texts = [t for t, _ in embeddings]
        vectors = [v for _, v in embeddings]

        # 헤더
        header = f"{'':15}"
        for text in texts:
            header += f"{text[:12]:>15}"
        self._print(header)
        self._print("-" * (15 + 15 * len(texts)))

        # 각 행
        for i, text1 in enumerate(texts):
            row = f"{text1[:12]:15}"
            for j, text2 in enumerate(texts):
                sim = self._cosine_similarity(vectors[i], vectors[j])
                row += f"{sim:>15.3f}"
            self._print(row)

        self._print(f"{'=' * 60}\n")

    # ==================== 유사도 계산 ====================

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """유클리드 거리 계산"""
        if HAS_NUMPY:
            import numpy as real_np

            a_arr = real_np.array(a)
            b_arr = real_np.array(b)
            return float(real_np.linalg.norm(a_arr - b_arr))
        else:
            # numpy 없이 계산
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _interpret_similarity(self, cosine_sim: float) -> str:
        """유사도 해석"""
        if cosine_sim >= 0.9:
            return "매우 유사 (거의 같은 의미)"
        elif cosine_sim >= 0.7:
            return "유사 (관련있는 내용)"
        elif cosine_sim >= 0.5:
            return "어느정도 관련 (약한 연관성)"
        elif cosine_sim >= 0.3:
            return "약간 관련 (거의 무관)"
        else:
            return "무관 (전혀 다른 의미)"

    def compare_texts(self, text1: str, text2: str, embedding_function) -> SimilarityInfo:
        """
        두 텍스트의 유사도 계산

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
            embedding_function: 임베딩 함수

        Returns:
            SimilarityInfo
        """
        # 임베딩 생성
        vectors = embedding_function([text1, text2])
        vec1, vec2 = vectors[0], vectors[1]

        # 유사도 계산
        cosine_sim = self._cosine_similarity(vec1, vec2)
        euclidean_dist = self._euclidean_distance(vec1, vec2)
        interpretation = self._interpret_similarity(cosine_sim)

        info = SimilarityInfo(
            text1=text1,
            text2=text2,
            cosine_similarity=cosine_sim,
            euclidean_distance=euclidean_dist,
            interpretation=interpretation,
        )

        self._print(f"\n{'=' * 60}")
        self._print("📊 텍스트 유사도")
        self._print(f"{'=' * 60}")
        self._print(f"텍스트 1: {text1[:50]}...")
        self._print(f"텍스트 2: {text2[:50]}...")
        self._print(f"\n코사인 유사도: {cosine_sim:.4f}")
        self._print(f"유클리드 거리: {euclidean_dist:.4f}")
        self._print(f"해석: {interpretation}")
        self._print(f"{'=' * 60}\n")

        return info

    # ==================== 청크 검증 ====================

    def inspect_chunks(self, chunks: List[Any], show_samples: int = 3) -> Dict[str, Any]:
        """
        텍스트 청크 검사

        Args:
            chunks: 청크 리스트
            show_samples: 샘플 개수

        Returns:
            청크 통계
        """
        if not chunks:
            self._print("⚠️  청크가 비어있습니다!")
            return {}

        # 통계
        total_chunks = len(chunks)
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)

        stats = {
            "total_chunks": total_chunks,
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length,
            "chunk_lengths": chunk_lengths,
        }

        self._print(f"\n{'=' * 60}")
        self._print("📄 청크 정보")
        self._print(f"{'=' * 60}")
        self._print(f"총 청크 수: {total_chunks}")
        self._print(f"평균 길이: {avg_length:.1f} 문자")
        self._print(f"최소 길이: {min_length} 문자")
        self._print(f"최대 길이: {max_length} 문자")

        # 샘플 출력
        self._print(f"\n샘플 청크 (처음 {show_samples}개):")
        for i, chunk in enumerate(chunks[:show_samples], 1):
            self._print(f"\n[Chunk {i}] ({len(chunk.content)} 문자)")
            self._print(f"  내용: {chunk.content[:100]}...")
            if chunk.metadata:
                self._print(f"  메타: {chunk.metadata}")

        self._print(f"{'=' * 60}\n")

        return stats

    # ==================== Vector Store 검증 ====================

    def inspect_vector_store(self, store, sample_queries: List[str], k: int = 3) -> Dict[str, Any]:
        """
        Vector Store 검사

        Args:
            store: VectorStore 인스턴스
            sample_queries: 테스트 쿼리들
            k: 반환할 결과 수

        Returns:
            검색 결과
        """
        self._print(f"\n{'=' * 60}")
        self._print("🔍 Vector Store 검사")
        self._print(f"{'=' * 60}")

        results = {}

        for query in sample_queries:
            self._print(f'\n쿼리: "{query}"')
            self._print("-" * 60)

            try:
                search_results = store.similarity_search(query, k=k)

                if not search_results:
                    self._print("  ⚠️  결과 없음")
                    results[query] = []
                    continue

                results[query] = search_results

                for i, result in enumerate(search_results, 1):
                    score = result.score
                    content = result.document.content
                    metadata = result.document.metadata

                    self._print(f"\n  [{i}] Score: {score:.4f}")
                    self._print(f"      Content: {content[:100]}...")
                    if metadata:
                        self._print(f"      Metadata: {metadata}")

                    # 점수 해석
                    interpretation = self._interpret_similarity(score)
                    self._print(f"      해석: {interpretation}")

            except Exception as e:
                self._print(f"  ❌ 에러: {e}")
                results[query] = None

        self._print(f"\n{'=' * 60}\n")

        return results

    # ==================== 전체 파이프라인 검증 ====================

    def validate_rag_pipeline(
        self,
        documents: List[Any],
        chunks: List[Any],
        embedding_function,
        store,
        test_queries: List[str],
    ) -> Dict[str, Any]:
        """
        전체 RAG 파이프라인 검증

        Args:
            documents: 원본 문서
            chunks: 분할된 청크
            embedding_function: 임베딩 함수
            store: VectorStore
            test_queries: 테스트 쿼리

        Returns:
            전체 검증 결과
        """
        self._print(f"\n{'#' * 60}")
        self._print("# RAG 파이프라인 전체 검증")
        self._print(f"{'#' * 60}\n")

        report = {}

        # 1. 문서 확인
        self._print("1️⃣  원본 문서 확인")
        report["documents"] = {
            "count": len(documents),
            "total_length": sum(len(doc.content) for doc in documents),
        }
        self._print(
            f"   ✓ {len(documents)}개 문서, 총 {report['documents']['total_length']} 문자\n"
        )

        # 2. 청크 확인
        self._print("2️⃣  청크 확인")
        chunk_stats = self.inspect_chunks(chunks, show_samples=2)
        report["chunks"] = chunk_stats

        # 3. 임베딩 테스트
        self._print("3️⃣  임베딩 테스트")
        test_text = chunks[0].content[:100] if chunks else "Test"
        test_vector = embedding_function([test_text])[0]
        self.inspect_embedding(test_text, test_vector, show_preview=5)
        report["embedding_dim"] = len(test_vector)

        # 4. Vector Store 테스트
        self._print("4️⃣  Vector Store 테스트")
        search_results = self.inspect_vector_store(store, test_queries, k=3)
        report["search_results"] = search_results

        # 5. 종합 평가
        self._print(f"\n{'=' * 60}")
        self._print("📊 종합 평가")
        self._print(f"{'=' * 60}")

        issues = []

        # 청크 크기 확인
        if chunk_stats.get("avg_length", 0) < 50:
            issues.append("⚠️  청크가 너무 작음 (평균 < 50)")
        elif chunk_stats.get("avg_length", 0) > 2000:
            issues.append("⚠️  청크가 너무 큼 (평균 > 2000)")

        # 검색 결과 확인
        empty_results = sum(1 for r in search_results.values() if not r)
        if empty_results > 0:
            issues.append(f"⚠️  {empty_results}개 쿼리에서 결과 없음")

        # 낮은 점수 확인
        low_scores = []
        for query, results in search_results.items():
            if results and results[0].score < 0.5:
                low_scores.append((query, results[0].score))

        if low_scores:
            issues.append(f"⚠️  {len(low_scores)}개 쿼리에서 낮은 점수 (< 0.5)")

        if not issues:
            self._print("✅ 문제 없음 - 파이프라인이 정상적으로 작동합니다!")
        else:
            self._print("발견된 문제:")
            for issue in issues:
                self._print(f"  {issue}")

        self._print(f"{'=' * 60}\n")

        report["issues"] = issues

        return report


# ==================== 편의 함수 ====================


def inspect_embedding(text: str, embedding_function, show_preview: int = 10) -> EmbeddingInfo:
    """
    임베딩 검사 (간단한 버전)

    Example:
        from beanllm import Embedding, inspect_embedding

        embed_func = Embedding.openai().embed_sync
        info = inspect_embedding("Hello world", embed_func)
    """
    vector = embedding_function([text])[0]
    debugger = RAGDebugger(verbose=True)
    return debugger.inspect_embedding(text, vector, show_preview)


def compare_texts(text1: str, text2: str, embedding_function) -> SimilarityInfo:
    """
    두 텍스트 유사도 비교 (간단한 버전)

    Example:
        from beanllm import Embedding, compare_texts

        embed_func = Embedding.openai().embed_sync
        info = compare_texts("강아지", "개", embed_func)
    """
    debugger = RAGDebugger(verbose=True)
    return debugger.compare_texts(text1, text2, embedding_function)


def validate_pipeline(
    documents: List[Any],
    chunks: List[Any],
    embedding_function,
    store,
    test_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    전체 RAG 파이프라인 검증 (간단한 버전)

    Example:
        from beanllm import validate_pipeline

        report = validate_pipeline(
            documents=docs,
            chunks=chunks,
            embedding_function=embed_func,
            store=store,
            test_queries=["What is AI?", "How does ML work?"]
        )
    """
    if test_queries is None:
        # 기본 쿼리
        test_queries = [chunks[0].content[:50]] if chunks else ["test"]

    debugger = RAGDebugger(verbose=True)
    return debugger.validate_rag_pipeline(
        documents, chunks, embedding_function, store, test_queries
    )


# ==================== 시각화 유틸리티 ====================


def visualize_embeddings_2d(texts: List[str], embedding_function, save_path: Optional[str] = None):
    """
    임베딩을 2D로 시각화 (기존 함수 - 하위 호환성 유지)

    Args:
        texts: 텍스트 리스트
        embedding_function: 임베딩 함수
        save_path: 저장 경로 (선택)

    Example:
        from beanllm import Embedding, visualize_embeddings_2d

        texts = ["강아지", "개", "고양이", "자동차", "비행기"]
        embed_func = Embedding.openai().embed_sync
        visualize_embeddings_2d(texts, embed_func)
    """
    # 새로운 함수로 위임
    visualize_embeddings(
        texts,
        embedding_function,
        method="tsne",
        dimensions=2,
        save_path=save_path,
        interactive=False,
    )


def visualize_embeddings(
    texts: List[str],
    embedding_function,
    method: str = "tsne",  # "tsne" 또는 "pca"
    dimensions: int = 2,  # 2 또는 3
    save_path: Optional[str] = None,
    interactive: bool = False,  # plotly 사용
):
    """
    임베딩을 2D/3D로 시각화 (확장된 함수)

    Args:
        texts: 텍스트 리스트
        embedding_function: 임베딩 함수
        method: 차원 축소 방법 ("tsne" 또는 "pca")
        dimensions: 차원 수 (2 또는 3)
        save_path: 저장 경로
        interactive: 인터랙티브 플롯 (plotly)

    Example:
        from beanllm import Embedding, visualize_embeddings

        texts = ["AI", "ML", "DL", "강아지", "고양이"]
        embed_func = Embedding.openai().embed_sync

        # 2D 시각화
        visualize_embeddings(texts, embed_func, method="tsne", dimensions=2)

        # 3D 인터랙티브 시각화
        visualize_embeddings(texts, embed_func, method="pca", dimensions=3, interactive=True)
    """
    # 임베딩 생성
    vectors = embedding_function(texts)
    vectors_array = np.array(vectors)

    # 차원 축소
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            reducer = TSNE(
                n_components=dimensions,
                random_state=42,
                perplexity=min(30, len(texts) - 1),
            )
        except ImportError:
            print("⚠️  scikit-learn 필요:")
            print("   pip install scikit-learn")
            return
    elif method == "pca":
        try:
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=dimensions, random_state=42)
        except ImportError:
            print("⚠️  scikit-learn 필요:")
            print("   pip install scikit-learn")
            return
    else:
        raise ValueError(f"Unknown method: {method}")

    vectors_reduced = reducer.fit_transform(vectors_array)

    # 시각화
    if interactive:
        # Plotly 사용 (3D 지원, 인터랙티브)
        try:
            import plotly.graph_objects as go

            if dimensions == 3:
                fig = go.Figure(
                    data=go.Scatter3d(
                        x=vectors_reduced[:, 0],
                        y=vectors_reduced[:, 1],
                        z=vectors_reduced[:, 2],
                        mode="markers+text",
                        text=texts,
                        marker=dict(size=8, color=vectors_reduced[:, 0]),
                    )
                )
            else:
                fig = go.Figure(
                    data=go.Scatter(
                        x=vectors_reduced[:, 0],
                        y=vectors_reduced[:, 1],
                        mode="markers+text",
                        text=texts,
                        marker=dict(size=12),
                    )
                )

            fig.update_layout(title=f"Embeddings 시각화 ({method.upper()}, {dimensions}D)")
            fig.show()

            if save_path:
                fig.write_html(save_path.replace(".png", ".html"))

        except ImportError:
            # plotly 없으면 matplotlib 사용
            interactive = False

    if not interactive:
        # Matplotlib 사용
        try:
            import matplotlib.pyplot as plt

            if dimensions == 3:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    vectors_reduced[:, 0],
                    vectors_reduced[:, 1],
                    vectors_reduced[:, 2],
                    s=200,
                    alpha=0.6,
                )
                for i, text in enumerate(texts):
                    ax.text(
                        vectors_reduced[i, 0],
                        vectors_reduced[i, 1],
                        vectors_reduced[i, 2],
                        text,
                        fontsize=10,
                    )
            else:
                plt.figure(figsize=(12, 8))
                plt.scatter(
                    vectors_reduced[:, 0],
                    vectors_reduced[:, 1],
                    s=200,
                    alpha=0.6,
                )
                for i, text in enumerate(texts):
                    plt.annotate(
                        text,
                        (vectors_reduced[i, 0], vectors_reduced[i, 1]),
                        fontsize=12,
                    )

            plt.title(f"Embeddings 시각화 ({method.upper()}, {dimensions}D)", fontsize=16)
            if dimensions == 2:
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ 저장: {save_path}")

            plt.show()

        except ImportError:
            print("⚠️  matplotlib 필요:")
            print("   pip install matplotlib")
            return


def similarity_heatmap(
    texts: List[str],
    embedding_function,
    save_path: Optional[str] = None,
    cluster: bool = True,  # 클러스터링 적용
    method: str = "ward",  # 클러스터링 방법
):
    """
    유사도 히트맵 생성 (확장된 함수)

    Args:
        texts: 텍스트 리스트
        embedding_function: 임베딩 함수
        save_path: 저장 경로 (선택)
        cluster: 클러스터링 적용 여부
        method: 클러스터링 방법 ("ward", "complete", "average")

    Example:
        from beanllm import Embedding, similarity_heatmap

        texts = ["AI", "ML", "DL", "NLP", "CV"]
        embed_func = Embedding.openai().embed_sync
        similarity_heatmap(texts, embed_func, cluster=True)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("⚠️  matplotlib, seaborn, scikit-learn 필요:")
        print("   pip install matplotlib seaborn scikit-learn")
        return

    # 임베딩 생성
    vectors = embedding_function(texts)
    vectors_array = np.array(vectors)

    # 유사도 행렬 계산
    similarity_matrix = cosine_similarity(vectors_array)

    # 클러스터링 적용
    if cluster:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage

            # 계층적 클러스터링
            linkage_matrix = linkage(vectors_array, method=method)

            # 클러스터 순서로 재정렬
            order = leaves_list(linkage_matrix)
            similarity_matrix = similarity_matrix[order][:, order]
            texts_ordered = [texts[i] for i in order]
        except ImportError:
            print("⚠️  scipy 필요 (클러스터링용):")
            print("   pip install scipy")
            cluster = False
            texts_ordered = texts
    else:
        texts_ordered = texts

    # 히트맵 생성
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        xticklabels=texts_ordered,
        yticklabels=texts_ordered,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        square=True,
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    plt.title("유사도 히트맵", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 저장: {save_path}")

    plt.show()
