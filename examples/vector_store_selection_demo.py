"""
Vector Store 선택 방법 - 3가지 방법
Embedding과 같은 패턴으로 통일
"""
from beanllm import (
    VectorStore,
    Document,
    Embedding,
    create_vector_store,
    from_documents
)


def demo_auto_selection():
    """방법 1: 자동 선택 (가장 간단!)"""
    print("\n" + "="*60)
    print("1️⃣  자동 선택 (추천!)")
    print("="*60)

    # 더미 임베딩 (API 키 없이도 테스트)
    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    try:
        # provider 생략 → 자동으로 가장 좋은 provider 선택
        store = VectorStore(embedding_function=embed_func)

        print("\n✓ VectorStore 생성 (provider 자동 선택)")
        print(f"  선택된 provider: {store.__class__.__name__}")

        # 문서 추가
        docs = [
            Document(content="AI is amazing"),
            Document(content="ML is powerful"),
            Document(content="DL is deep")
        ]

        store.add_documents(docs)
        print(f"  ✓ {len(docs)}개 문서 추가")

        # 검색
        results = store.similarity_search("artificial intelligence", k=2)
        print(f"  ✓ 검색: {len(results)}개 결과")

    except Exception as e:
        print(f"  ⚠️  {e}")

    print("\n💡 provider를 생략하면 자동으로 chroma 선택!")


def demo_explicit_selection():
    """방법 2: 명시적 선택"""
    print("\n" + "="*60)
    print("2️⃣  명시적 선택")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    # Chroma 명시
    print("\n[1] Chroma 명시:")
    try:
        store = VectorStore(
            provider="chroma",
            embedding_function=embed_func,
            collection_name="explicit_demo"
        )
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # FAISS 명시
    print("\n[2] FAISS 명시:")
    try:
        store = VectorStore(
            provider="faiss",
            dimension=384,
            embedding_function=embed_func
        )
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    print("\n💡 특정 provider를 사용하고 싶을 때 명시!")


def demo_factory_methods():
    """방법 3: 팩토리 메서드 (기존 방식)"""
    print("\n" + "="*60)
    print("3️⃣  팩토리 메서드")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    # Chroma
    print("\n[1] VectorStore.chroma():")
    try:
        store = VectorStore.chroma(
            embedding_function=embed_func,
            collection_name="factory_demo"
        )
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # FAISS
    print("\n[2] VectorStore.faiss():")
    try:
        store = VectorStore.faiss(
            dimension=384,
            embedding_function=embed_func
        )
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # Pinecone
    print("\n[3] VectorStore.pinecone():")
    try:
        store = VectorStore.pinecone(
            index_name="test",
            dimension=384,
            embedding_function=embed_func
        )
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    print("\n💡 IDE 자동완성을 활용할 수 있어서 편리!")


def demo_convenience_functions():
    """편의 함수도 동일한 패턴"""
    print("\n" + "="*60)
    print("4️⃣  편의 함수")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    docs = [
        Document(content="Test 1"),
        Document(content="Test 2"),
        Document(content="Test 3")
    ]

    # create_vector_store - 자동 선택
    print("\n[1] create_vector_store() - 자동:")
    try:
        store = create_vector_store(embedding_function=embed_func)
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # create_vector_store - 명시적
    print("\n[2] create_vector_store() - 명시:")
    try:
        store = create_vector_store(
            provider="chroma",
            embedding_function=embed_func
        )
        print(f"  ✓ {store.__class__.__name__} 생성")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # from_documents - 자동 선택
    print("\n[3] from_documents() - 자동 (가장 간단!):")
    try:
        store = from_documents(docs, embed_func)
        print(f"  ✓ {store.__class__.__name__} 생성")
        print(f"  ✓ {len(docs)}개 문서 자동 추가")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # from_documents - 명시적
    print("\n[4] from_documents() - 명시:")
    try:
        store = from_documents(docs, embed_func, provider="chroma")
        print(f"  ✓ {store.__class__.__name__} 생성")
        print(f"  ✓ {len(docs)}개 문서 자동 추가")
    except Exception as e:
        print(f"  ⚠️  {e}")

    print("\n💡 from_documents()는 provider 생략 가능!")


def demo_comparison():
    """Embedding과 동일한 패턴"""
    print("\n" + "="*60)
    print("5️⃣  Embedding과 동일한 패턴")
    print("="*60)

    print("\n【 Embedding 패턴 】")
    print("""
    # 자동 감지
    emb = Embedding(model="text-embedding-3-small")

    # 명시적 선택
    emb = Embedding(model="text-embedding-3-small", provider="openai")

    # 팩토리 메서드
    emb = Embedding.openai(model="text-embedding-3-small")
    """)

    print("\n【 VectorStore 패턴 (이제 동일!) 】")
    print("""
    # 자동 선택
    store = VectorStore(embedding_function=embed_func)

    # 명시적 선택
    store = VectorStore(provider="chroma", embedding_function=embed_func)

    # 팩토리 메서드
    store = VectorStore.chroma(embedding_function=embed_func)
    """)

    print("\n✅ 일관된 패턴으로 학습 곡선 감소!")


def demo_practical_usage():
    """실전 사용 예시"""
    print("\n" + "="*60)
    print("6️⃣  실전 사용 - RAG 파이프라인")
    print("="*60)

    from beanllm import DocumentLoader, TextSplitter
    from pathlib import Path

    # 테스트 파일
    test_file = Path("selection_test.txt")
    test_file.write_text("""
AI is transforming technology.
Machine learning learns from data.
Deep learning uses neural networks.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        print("\n[가장 간단한 방법]")
        print("="*60)

        # 파이프라인 (모두 자동!)
        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs)
        store = from_documents(chunks, embed_func)  # provider 생략!

        print(f"  ✓ 문서 로딩: {len(docs)}개")
        print(f"  ✓ 청크 분할: {len(chunks)}개")
        print(f"  ✓ Vector Store: {store.__class__.__name__}")

        results = store.similarity_search("AI", k=2)
        print(f"  ✓ 검색: {len(results)}개 결과")

        print("\n[특정 provider 사용]")
        print("="*60)

        # FAISS를 명시적으로 선택
        try:
            store = from_documents(
                chunks,
                embed_func,
                provider="faiss",  # 명시
                dimension=384
            )
            print(f"  ✓ Vector Store: {store.__class__.__name__}")
        except Exception as e:
            print(f"  ⚠️  {e}")

        print("\n💡 기본은 자동, 필요할 때만 명시!")

    finally:
        if test_file.exists():
            test_file.unlink()


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🎯 Vector Store 선택 방법 - 3가지")
    print("="*60)
    print("\nEmbedding과 동일한 패턴으로 통일!")
    print("1. 자동 선택 (provider 생략)")
    print("2. 명시적 선택 (provider 지정)")
    print("3. 팩토리 메서드 (VectorStore.chroma())")

    demo_auto_selection()
    demo_explicit_selection()
    demo_factory_methods()
    demo_convenience_functions()
    demo_comparison()
    demo_practical_usage()

    print("\n" + "="*60)
    print("🎉 Vector Store 선택 데모 완료!")
    print("="*60)
    print("\n✨ 핵심:")
    print("  1. 기본은 자동 선택 (가장 간단)")
    print("  2. 필요할 때 명시적 선택")
    print("  3. 팩토리 메서드도 여전히 사용 가능")
    print("  4. Embedding과 동일한 패턴!")
    print("\n💡 사용자가 원하는 방식으로 선택하세요!")


if __name__ == "__main__":
    main()
