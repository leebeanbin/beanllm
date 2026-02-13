"""RAG Debug - 시각화 유틸리티 (임베딩 2D/3D, 유사도 히트맵)."""

from __future__ import annotations

from typing import List, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


def visualize_embeddings_2d(
    texts: List[str], embedding_function, save_path: Optional[str] = None
) -> None:
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
) -> None:
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
    if not HAS_NUMPY or np is None:
        print("⚠️  numpy 필요:")
        print("   pip install numpy")
        return

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
) -> None:
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
    if not HAS_NUMPY or np is None:
        print("⚠️  numpy 필요:")
        print("   pip install numpy")
        return

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
