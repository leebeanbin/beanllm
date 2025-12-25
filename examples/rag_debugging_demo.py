"""
RAG Debugging Demo - RAG 파이프라인 디버깅하기
중간 과정을 확인하고 문제를 찾는 방법
"""
import asyncio
from pathlib import Path
from beanllm import (
    DocumentLoader,
    TextSplitter,
    Embedding,
    VectorStore,
    from_documents,
    # 디버깅 도구들
    RAGDebugger,
    inspect_embedding,
    compare_texts,
    validate_pipeline,
    visualize_embeddings_2d,
    similarity_heatmap
)


def demo_inspect_embedding():
    """임베딩 검사"""
    print("\n" + "="*60)
    print("1️⃣  임베딩 검사")
    print("="*60)

    try:
        # 임베딩 함수 준비
        embed_func = Embedding.openai().embed_sync

        # 텍스트 임베딩
        text = "Machine learning is a subset of artificial intelligence"

        # ✅ 방법 1: 간단한 함수
        info = inspect_embedding(text, embed_func, show_preview=10)

        print(f"\n결과:")
        print(f"  차원: {info.dimension}")
        print(f"  벡터 크기: {info.norm:.4f}")
        print(f"  미리보기: {info.preview[:5]}")

    except Exception as e:
        print(f"⚠️  OpenAI API 키 필요: {e}")
        print("   더미 임베딩으로 대체...")

        # 더미 임베딩
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]
        text = "Test text"
        info = inspect_embedding(text, embed_func, show_preview=5)


def demo_compare_texts():
    """텍스트 유사도 비교"""
    print("\n" + "="*60)
    print("2️⃣  텍스트 유사도 비교")
    print("="*60)

    try:
        embed_func = Embedding.openai().embed_sync

        # 유사한 텍스트
        print("\n[테스트 1] 유사한 텍스트:")
        info1 = compare_texts("강아지", "개", embed_func)
        print(f"  → 유사도: {info1.cosine_similarity:.3f} ({info1.interpretation})")

        # 다른 텍스트
        print("\n[테스트 2] 다른 텍스트:")
        info2 = compare_texts("강아지", "자동차", embed_func)
        print(f"  → 유사도: {info2.cosine_similarity:.3f} ({info2.interpretation})")

        # 관련된 텍스트
        print("\n[테스트 3] 관련된 텍스트:")
        info3 = compare_texts(
            "Machine learning is amazing",
            "Deep learning is powerful",
            embed_func
        )
        print(f"  → 유사도: {info3.cosine_similarity:.3f} ({info3.interpretation})")

    except Exception as e:
        print(f"⚠️  {e}")


def demo_inspect_chunks():
    """청크 검사"""
    print("\n" + "="*60)
    print("3️⃣  청크 검사")
    print("="*60)

    # 테스트 문서 생성
    test_file = Path("debug_test.txt")
    test_file.write_text("""
Artificial Intelligence is transforming technology.
Machine learning algorithms learn patterns from data.
Deep learning uses neural networks with multiple layers.
Natural language processing enables computers to understand text.
Computer vision allows machines to interpret images.
    """.strip(), encoding="utf-8")

    try:
        # 문서 로딩 및 분할
        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs, chunk_size=100)

        # 청크 검사
        debugger = RAGDebugger()
        stats = debugger.inspect_chunks(chunks, show_samples=3)

        print(f"\n통계:")
        print(f"  총 청크: {stats['total_chunks']}")
        print(f"  평균 길이: {stats['avg_length']:.1f}")

        # 청크 크기 분포 확인
        print(f"\n청크 크기 분포:")
        for i, length in enumerate(stats['chunk_lengths'][:5], 1):
            print(f"  Chunk {i}: {length} 문자")

    finally:
        if test_file.exists():
            test_file.unlink()


def demo_inspect_vector_store():
    """Vector Store 검사"""
    print("\n" + "="*60)
    print("4️⃣  Vector Store 검사")
    print("="*60)

    # 더미 임베딩 (API 키 없이도 테스트 가능)
    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    try:
        # Vector Store 생성 및 문서 추가
        from beanllm import Document

        docs = [
            Document(content="Python is a programming language"),
            Document(content="JavaScript is used for web development"),
            Document(content="Machine learning learns from data"),
            Document(content="Deep learning uses neural networks")
        ]

        store = from_documents(docs, embed_func, provider="chroma")

        # Vector Store 검사
        debugger = RAGDebugger()
        results = debugger.inspect_vector_store(
            store,
            sample_queries=[
                "programming",
                "artificial intelligence",
                "web"
            ],
            k=2
        )

        print(f"\n검색 결과 요약:")
        for query, query_results in results.items():
            if query_results:
                print(f"  '{query}': {len(query_results)}개 결과")
            else:
                print(f"  '{query}': 결과 없음")

    except Exception as e:
        print(f"⚠️  {e}")


def demo_validate_pipeline():
    """전체 파이프라인 검증"""
    print("\n" + "="*60)
    print("5️⃣  전체 RAG 파이프라인 검증")
    print("="*60)

    # 테스트 파일
    test_file = Path("pipeline_test.txt")
    test_file.write_text("""
Artificial Intelligence encompasses various technologies.
Machine learning is a subset of AI that learns from data.
Deep learning uses neural networks with multiple layers.
Natural language processing deals with text understanding.
Computer vision enables image and video analysis.
Reinforcement learning learns through trial and error.
    """.strip(), encoding="utf-8")

    try:
        # 1. 문서 로딩
        print("\n📄 문서 로딩...")
        docs = DocumentLoader.load(test_file)
        print(f"   ✓ {len(docs)}개 문서 로드")

        # 2. 텍스트 분할
        print("\n✂️  텍스트 분할...")
        chunks = TextSplitter.split(docs, chunk_size=150)
        print(f"   ✓ {len(chunks)}개 청크 생성")

        # 3. 임베딩 및 Vector Store
        print("\n🔢 임베딩 및 Vector Store...")
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        store = from_documents(chunks, embed_func, provider="chroma")
        print(f"   ✓ Vector Store 생성 완료")

        # 4. 전체 검증
        print("\n🔍 전체 파이프라인 검증...")
        report = validate_pipeline(
            documents=docs,
            chunks=chunks,
            embedding_function=embed_func,
            store=store,
            test_queries=[
                "What is machine learning?",
                "Tell me about neural networks",
                "How does AI work?"
            ]
        )

        # 결과 출력
        print("\n📊 검증 보고서:")
        print(f"  문서 수: {report['documents']['count']}")
        print(f"  청크 수: {report['chunks']['total_chunks']}")
        print(f"  임베딩 차원: {report['embedding_dim']}")
        print(f"  발견된 문제: {len(report['issues'])}개")

        if report['issues']:
            print("\n문제 목록:")
            for issue in report['issues']:
                print(f"  {issue}")

    except Exception as e:
        print(f"⚠️  {e}")
        import traceback
        traceback.print_exc()

    finally:
        if test_file.exists():
            test_file.unlink()


def demo_compare_multiple():
    """여러 텍스트 비교"""
    print("\n" + "="*60)
    print("6️⃣  여러 텍스트 비교 (유사도 매트릭스)")
    print("="*60)

    try:
        embed_func = Embedding.openai().embed_sync

        # 여러 텍스트
        texts = [
            "강아지",
            "개",
            "고양이",
            "자동차",
            "비행기"
        ]

        # 임베딩
        print("\n임베딩 생성 중...")
        vectors = embed_func(texts)

        # 비교
        debugger = RAGDebugger()
        embeddings = list(zip(texts, vectors))
        debugger.compare_embeddings(embeddings)

    except Exception as e:
        print(f"⚠️  {e}")


def demo_visualization():
    """시각화 (선택적)"""
    print("\n" + "="*60)
    print("7️⃣  임베딩 시각화 (선택적)")
    print("="*60)

    try:
        # scikit-learn, matplotlib 확인
        import sklearn
        import matplotlib

        print("\n시각화 라이브러리 설치됨!")
        print("실제 시각화를 보려면 주석을 해제하세요.\n")

        # 주석을 해제하면 시각화 실행:
        # embed_func = Embedding.openai().embed_sync
        # texts = ["AI", "ML", "DL", "NLP", "CV", "강아지", "고양이", "자동차"]
        #
        # # 2D 시각화
        # visualize_embeddings_2d(texts, embed_func, save_path="embeddings_2d.png")
        #
        # # 유사도 히트맵
        # similarity_heatmap(texts, embed_func, save_path="similarity_heatmap.png")

    except ImportError:
        print("\n⚠️  시각화를 위해 추가 패키지 필요:")
        print("   pip install scikit-learn matplotlib seaborn")


def demo_advanced_debugging():
    """고급 디버깅"""
    print("\n" + "="*60)
    print("8️⃣  고급 디버깅 - RAGDebugger 클래스 직접 사용")
    print("="*60)

    # RAGDebugger 인스턴스 생성
    debugger = RAGDebugger(verbose=True)

    try:
        embed_func = Embedding.openai().embed_sync

        # 1. 임베딩 검사
        print("\n[1] 임베딩 상세 검사:")
        text = "The quick brown fox jumps over the lazy dog"
        vector = embed_func([text])[0]
        info = debugger.inspect_embedding(text, vector, show_preview=15)

        # 2. 유사도 분석
        print("\n[2] 유사도 상세 분석:")
        info = debugger.compare_texts(
            "artificial intelligence",
            "machine learning",
            embed_func
        )

        print(f"\n분석 결과:")
        print(f"  코사인 유사도: {info.cosine_similarity:.4f}")
        print(f"  유클리드 거리: {info.euclidean_distance:.4f}")
        print(f"  해석: {info.interpretation}")

    except Exception as e:
        print(f"⚠️  {e}")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🔍 RAG 디버깅 도구 데모")
    print("="*60)
    print("\nRAG 파이프라인 개발 시 유용한 디버깅 도구들을 소개합니다.")
    print("각 단계에서 무슨 일이 일어나는지 확인할 수 있습니다!\n")

    demo_inspect_embedding()
    demo_compare_texts()
    demo_inspect_chunks()
    demo_inspect_vector_store()
    demo_validate_pipeline()
    demo_compare_multiple()
    demo_visualization()
    demo_advanced_debugging()

    print("\n" + "="*60)
    print("🎉 RAG 디버깅 데모 완료!")
    print("="*60)
    print("\n✨ 주요 기능:")
    print("  1. inspect_embedding() - 임베딩 검사")
    print("  2. compare_texts() - 텍스트 유사도 비교")
    print("  3. validate_pipeline() - 전체 파이프라인 검증")
    print("  4. visualize_embeddings_2d() - 2D 시각화")
    print("  5. similarity_heatmap() - 유사도 히트맵")
    print("\n💡 이제 RAG 파이프라인 개발이 훨씬 쉬워집니다!")


if __name__ == "__main__":
    main()
