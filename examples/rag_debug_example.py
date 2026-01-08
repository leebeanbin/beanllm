"""
RAG Debug Example - Phase 2 완성 데모
RAG 디버깅 전체 워크플로우를 보여주는 예제

이 예제는 다음을 시연합니다:
1. RAGDebug 세션 시작
2. Embedding 분석 (UMAP + 클러스터링)
3. 청크 검증 (크기, 중복, 메타데이터)
4. 파라미터 튜닝 (top_k, score_threshold)
5. 리포트 내보내기 (JSON, Markdown, HTML)
"""

import asyncio
from pathlib import Path

# Facade (Public API)
from beanllm.facade.rag_debug_facade import RAGDebug

# CLI Commands (Optional, for Rich UI)
from beanllm.ui.repl.rag_commands import RAGDebugCommands

# Visualizers (Optional, for Rich UI)
from beanllm.ui.visualizers.embedding_viz import EmbeddingVisualizer
from beanllm.ui.visualizers.metrics_viz import MetricsVisualizer


async def example_basic_api():
    """
    Example 1: Basic API Usage (Facade Pattern)

    가장 간단한 사용법 - Facade를 통한 직접 호출
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic API Usage (Facade)")
    print("=" * 60 + "\n")

    # Mock VectorStore (실제로는 Chroma, FAISS 등 사용)
    class MockVectorStore:
        def __init__(self):
            self._documents = [
                {"page_content": f"Document {i}" * 50, "metadata": {"source": f"doc{i}.txt"}}
                for i in range(100)
            ]

        def similarity_search(self, query, k=4):
            return self._documents[:k]

    vector_store = MockVectorStore()

    # Create RAGDebug instance
    debug = RAGDebug(
        vector_store=vector_store,
        session_name="production_debug",
    )

    # Start session
    print("Starting debug session...")
    session = await debug.start()
    print(f"✅ Session started: {session.session_id}")
    print(f"   Documents: {session.num_documents}")
    print(f"   Embeddings: {session.num_embeddings}")

    # Analyze embeddings (requires beanllm[advanced])
    try:
        print("\nAnalyzing embeddings (UMAP + clustering)...")
        analysis = await debug.analyze_embeddings(
            method="umap",
            n_clusters=5,
            detect_outliers=True,
        )
        print(f"✅ Analysis completed:")
        print(f"   Clusters: {analysis.num_clusters}")
        print(f"   Outliers: {len(analysis.outliers)}")
        print(f"   Silhouette Score: {analysis.silhouette_score:.4f}")
    except ImportError:
        print("⚠️  Skipping embedding analysis (install beanllm[advanced])")

    # Validate chunks
    print("\nValidating chunks...")
    validation = await debug.validate_chunks(
        size_threshold=2000,
    )
    print(f"✅ Validation completed:")
    print(f"   Total chunks: {validation.total_chunks}")
    print(f"   Valid chunks: {validation.valid_chunks}")
    print(f"   Issues: {len(validation.issues)}")
    print(f"   Duplicates: {len(validation.duplicate_chunks)}")

    # Tune parameters
    print("\nTuning parameters...")
    tuning = await debug.tune_parameters(
        parameters={"top_k": 10, "score_threshold": 0.7},
        test_queries=["What is RAG?", "How does retrieval work?"],
    )
    print(f"✅ Tuning completed:")
    print(f"   Avg Score: {tuning.avg_score:.4f}")
    if tuning.comparison_with_baseline:
        improvement = tuning.comparison_with_baseline.get("improvement_pct", 0.0)
        print(f"   vs Baseline: {improvement:+.2f}%")

    # Export report
    print("\nExporting debug report...")
    output_dir = Path("./debug_reports")
    output_dir.mkdir(exist_ok=True)
    results = await debug.export_report(
        output_dir=str(output_dir),
        formats=["json", "markdown"],
    )
    print(f"✅ Report exported:")
    for fmt, path in results.items():
        print(f"   {fmt.upper()}: {path}")

    print("\n✅ Basic API example completed!")


async def example_one_stop():
    """
    Example 2: One-Stop Full Analysis

    한 번에 모든 분석 실행 - run_full_analysis()
    """
    print("\n" + "=" * 60)
    print("Example 2: One-Stop Full Analysis")
    print("=" * 60 + "\n")

    class MockVectorStore:
        def __init__(self):
            self._documents = [
                {"page_content": f"Sample text {i}" * 30, "metadata": {"id": i}}
                for i in range(50)
            ]

        def similarity_search(self, query, k=4):
            return self._documents[:k]

    vector_store = MockVectorStore()
    debug = RAGDebug(vector_store=vector_store)

    # Run everything at once
    print("Running full analysis...")
    results = await debug.run_full_analysis(
        analyze_embeddings=False,  # Skip if advanced deps not installed
        validate_chunks=True,
        tune_parameters=False,  # Skip if no test queries
    )

    print(f"\n✅ Full analysis completed!")
    print(f"   Session ID: {results['session'].session_id}")

    if "chunk_validation" in results:
        validation = results["chunk_validation"]
        print(f"   Chunk Validation: {validation.valid_chunks}/{validation.total_chunks} valid")

    if "embedding_analysis" in results:
        analysis = results["embedding_analysis"]
        print(f"   Embedding Analysis: {analysis.num_clusters} clusters")


async def example_rich_cli():
    """
    Example 3: Rich CLI Interface

    Rich UI를 사용한 인터랙티브 디버깅
    """
    print("\n" + "=" * 60)
    print("Example 3: Rich CLI Interface")
    print("=" * 60 + "\n")

    class MockVectorStore:
        def __init__(self):
            self._documents = [
                {"page_content": f"Content {i}" * 40, "metadata": {"idx": i}}
                for i in range(80)
            ]

        def similarity_search(self, query, k=4):
            return self._documents[:k]

    vector_store = MockVectorStore()

    # Create CLI commands interface
    commands = RAGDebugCommands(vector_store=vector_store)

    # Start session with Rich UI
    await commands.cmd_start(session_name="rich_demo")

    # Validate chunks with Rich UI
    await commands.cmd_validate(size_threshold=1500)

    # Note: Embedding analysis requires advanced deps
    # await commands.cmd_analyze(method="umap", n_clusters=5)

    # Tune parameters with Rich UI
    await commands.cmd_tune(
        parameters={"top_k": 8},
        test_queries=["sample query"],
    )

    # Export report with Rich UI
    output_dir = Path("./cli_reports")
    output_dir.mkdir(exist_ok=True)
    await commands.cmd_export(
        output_dir=str(output_dir),
        formats=["json"],
    )

    print("\n✅ Rich CLI example completed!")


async def example_visualizers():
    """
    Example 4: Standalone Visualizers

    시각화 도구만 독립적으로 사용
    """
    print("\n" + "=" * 60)
    print("Example 4: Standalone Visualizers")
    print("=" * 60 + "\n")

    # Embedding Visualizer
    print("Embedding Visualization Demo:")
    print("-" * 60)

    embedding_viz = EmbeddingVisualizer()

    # Mock data: 2D coordinates
    import random
    random.seed(42)

    reduced_embeddings = [
        [random.uniform(-5, 5), random.uniform(-5, 5)]
        for _ in range(50)
    ]

    labels = [i % 3 for i in range(50)]  # 3 clusters
    outliers = [5, 15, 35]  # Some outliers

    embedding_viz.plot_scatter(
        reduced_embeddings=reduced_embeddings,
        labels=labels,
        outliers=outliers,
        width=60,
        height=20,
        title="Demo: Embedding Clusters",
    )

    # Cluster summary
    embedding_viz.show_cluster_summary(
        cluster_sizes={0: 18, 1: 16, 2: 13, -1: 3},
        silhouette_score=0.68,
        method="UMAP",
    )

    # Metrics Visualizer
    print("\n\nMetrics Visualization Demo:")
    print("-" * 60)

    metrics_viz = MetricsVisualizer()

    # Search performance dashboard
    metrics_viz.show_search_dashboard(
        metrics={
            "avg_score": 0.75,
            "avg_latency_ms": 145,
            "total_queries": 50,
            "top_k": 4,
            "score_threshold": 0.5,
        }
    )

    # Parameter comparison
    metrics_viz.compare_parameters(
        baseline={"top_k": 4, "avg_score": 0.72, "avg_latency_ms": 120},
        new={"top_k": 10, "avg_score": 0.78, "avg_latency_ms": 180},
    )

    # Chunk statistics
    metrics_viz.show_chunk_statistics(
        stats={
            "total_chunks": 200,
            "avg_size": 1200,
            "min_size": 100,
            "max_size": 1950,
            "duplicates": 5,
            "avg_overlap_ratio": 0.15,
        }
    )

    print("\n✅ Visualizers example completed!")


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("RAG Debug - Complete Integration Example")
    print("Phase 2 Implementation Complete")
    print("=" * 60)

    # Example 1: Basic API
    await example_basic_api()

    # Example 2: One-stop analysis
    await example_one_stop()

    # Example 3: Rich CLI
    await example_rich_cli()

    # Example 4: Visualizers
    await example_visualizers()

    print("\n" + "=" * 60)
    print("🎉 All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
