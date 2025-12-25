"""
Vector Stores Demo - Fluent API
beanllm 방식: 쉽고 강력한 벡터 스토어
"""
import asyncio
from beanllm import (
    VectorStore,
    VectorStoreBuilder,
    create_vector_store,
    from_documents,
    DocumentLoader,
    TextSplitter,
    Embedding,
    Document
)


def demo_basic_usage():
    """기본 사용법"""
    print("\n" + "="*60)
    print("📦 기본 사용법")
    print("="*60)

    # 임베딩 함수 준비
    try:
        emb = Embedding(model="text-embedding-3-small")
        embed_func = emb.embed_sync
        print("\n✓ Using OpenAI embeddings")
    except:
        # API 키 없으면 더미
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]
        print("\n⚠️  Using dummy embeddings (no API key)")

    # 1. 가장 간단한 방법
    print("\n1. 가장 간단한 방법:")
    try:
        store = VectorStore.chroma(embedding_function=embed_func)

        docs = [
            Document(content="AI is transforming the world"),
            Document(content="Machine learning learns from data"),
            Document(content="Deep learning uses neural networks")
        ]

        store.add_documents(docs)
        results = store.similarity_search("artificial intelligence", k=2)

        print(f"   ✓ Found {len(results)} documents")
        for i, result in enumerate(results[:2], 1):
            print(f"   {i}. {result.document.content[:50]}... (score: {result.score:.3f})")

    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ 기본 사용법 완료!")


def demo_factory_methods():
    """팩토리 메서드"""
    print("\n" + "="*60)
    print("🏭 팩토리 메서드")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    # Chroma
    print("\n1. Chroma (로컬, 쉬움):")
    try:
        store = VectorStore.chroma(
            collection_name="demo_chroma",
            embedding_function=embed_func
        )
        print("   ✓ Chroma store created")
    except Exception as e:
        print(f"   ⚠️  {e}")

    # FAISS
    print("\n2. FAISS (로컬, 빠름):")
    try:
        store = VectorStore.faiss(
            dimension=384,
            embedding_function=embed_func
        )
        print("   ✓ FAISS store created")
    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ 팩토리 메서드 완료!")


def demo_fluent_api():
    """Fluent API"""
    print("\n" + "="*60)
    print("✨ Fluent API (Builder Pattern)")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    print("\n1. Builder 패턴:")
    try:
        store = (VectorStoreBuilder()
            .use_chroma()
            .with_embedding(embed_func)
            .with_collection("fluent_demo")
            .build())

        print("   ✓ Store built with fluent API")

        # 사용
        docs = [Document(content="Fluent API is elegant")]
        store.add_documents(docs)
        results = store.similarity_search("elegant", k=1)

        print(f"   ✓ Found: {results[0].document.content}")

    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ Fluent API 완료!")


def demo_convenience_functions():
    """편의 함수"""
    print("\n" + "="*60)
    print("⚡ 편의 함수")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    # create_vector_store()
    print("\n1. create_vector_store():")
    try:
        store = create_vector_store(
            provider="chroma",
            embedding_function=embed_func,
            collection_name="convenience_demo"
        )
        print("   ✓ Store created")
    except Exception as e:
        print(f"   ⚠️  {e}")

    # from_documents()
    print("\n2. from_documents() - 가장 편리!")
    try:
        docs = [
            Document(content="Quick document 1"),
            Document(content="Quick document 2"),
            Document(content="Quick document 3")
        ]

        store = from_documents(
            docs,
            embedding_function=embed_func,
            provider="chroma",
            collection_name="from_docs_demo"
        )

        print(f"   ✓ Store created with {len(docs)} documents")

        results = store.similarity_search("quick", k=2)
        print(f"   ✓ Search: {len(results)} results")

    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ 편의 함수 완료!")


async def demo_full_rag_pipeline():
    """전체 RAG 파이프라인"""
    print("\n" + "="*60)
    print("🚀 전체 RAG 파이프라인")
    print("="*60)

    from pathlib import Path

    # 테스트 파일 생성
    test_file = Path("rag_demo.txt")
    test_file.write_text("""
Artificial Intelligence is revolutionizing technology.
Machine learning algorithms learn patterns from data.
Deep learning uses multi-layer neural networks.
Natural language processing understands human language.
Computer vision enables machines to see and interpret images.
    """.strip(), encoding="utf-8")

    try:
        # 1. 문서 로딩
        print("\n1. 문서 로딩:")
        docs = DocumentLoader.load(test_file)
        print(f"   ✓ Loaded {len(docs)} document(s)")

        # 2. 텍스트 분할
        print("\n2. 텍스트 분할:")
        chunks = TextSplitter.split(docs, chunk_size=100)
        print(f"   ✓ Split into {len(chunks)} chunks")

        # 3. 임베딩 준비
        print("\n3. 임베딩 준비:")
        try:
            from beanllm import embed_sync
            embed_func = lambda texts: embed_sync(texts)
            print("   ✓ Using OpenAI embeddings")
        except:
            import random
            embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]
            print("   ⚠️  Using dummy embeddings")

        # 4. Vector Store 생성 및 문서 추가
        print("\n4. Vector Store 생성:")
        store = from_documents(
            chunks,
            embedding_function=embed_func,
            provider="chroma",
            collection_name="rag_pipeline"
        )
        print(f"   ✓ Created vector store with {len(chunks)} chunks")

        # 5. 검색
        print("\n5. 질의 검색:")
        query = "What is machine learning?"
        results = store.similarity_search(query, k=3)

        print(f"   Query: \"{query}\"")
        print(f"   Found {len(results)} relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.document.content[:60]}...")
            print(f"      Score: {result.score:.3f}")

        print("\n✓ 전체 RAG 파이프라인 완료!")

    except Exception as e:
        print(f"   ⚠️  {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 정리
        if test_file.exists():
            test_file.unlink()


async def demo_async_operations():
    """비동기 작업"""
    print("\n" + "="*60)
    print("⚡ 비동기 작업")
    print("="*60)

    import random
    embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

    try:
        # Store 생성
        store = VectorStore.chroma(
            collection_name="async_demo",
            embedding_function=embed_func
        )

        # 문서 추가 (동기)
        docs = [
            Document(content="Async document 1"),
            Document(content="Async document 2"),
            Document(content="Async document 3")
        ]
        store.add_documents(docs)
        print("\n✓ Documents added")

        # 비동기 검색
        print("\n비동기 검색:")
        results = await store.asimilarity_search("async", k=2)
        print(f"   ✓ Found {len(results)} results")

        print("\n✓ 비동기 작업 완료!")

    except Exception as e:
        print(f"   ⚠️  {e}")


def demo_provider_selection():
    """Provider 선택"""
    print("\n" + "="*60)
    print("🔍 Provider 선택")
    print("="*60)

    # 사용 가능한 provider 확인
    print("\n1. 사용 가능한 providers:")
    available = VectorStore.list_available_providers()
    print(f"   Available: {available}")

    # 기본 provider
    default = VectorStore.get_default_provider()
    print(f"   Default: {default}")

    # 각 provider 특징
    print("\n2. Provider 특징:")
    print("   • Chroma: 로컬, 사용하기 쉬움, 빠른 시작")
    print("   • FAISS: 로컬, 매우 빠름, 대용량 데이터")
    print("   • Pinecone: 클라우드, 확장 가능, 프로덕션")
    print("   • Qdrant: 클라우드/로컬, 모던, 필터링 강력")
    print("   • Weaviate: 엔터프라이즈, GraphQL, 복잡한 쿼리")

    print("\n✓ Provider 선택 완료!")


def demo_comparison():
    """LangChain vs beanllm 비교"""
    print("\n" + "="*60)
    print("📊 LangChain vs beanllm 비교")
    print("="*60)

    print("\n【 LangChain 방식 】")
    print("""
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    # 여러 import 필요
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./db"
    )
    """)

    print("\n【 beanllm 방식 】")
    print("""
    from beanllm import from_documents, Embedding

    # 간단하고 직관적
    embed_func = Embedding.openai().embed_sync
    store = from_documents(docs, embed_func, provider="chroma")
    """)

    print("\n✅ beanllm: 더 간단하고 직관적!")
    print("✅ 통합 인터페이스로 provider 전환 쉬움")
    print("✅ Fluent API로 가독성 향상")


async def main():
    """모든 데모 실행"""
    print("="*60)
    print("🎯 Vector Stores 데모")
    print("="*60)
    print("\nbeanllm의 철학:")
    print("  1. 통합 인터페이스 (모든 vector store 동일한 API)")
    print("  2. Fluent API (Builder 패턴)")
    print("  3. 편의 함수 (from_documents)")
    print("  4. RAG 파이프라인 완전 지원")

    demo_basic_usage()
    demo_factory_methods()
    demo_fluent_api()
    demo_convenience_functions()
    await demo_full_rag_pipeline()
    await demo_async_operations()
    demo_provider_selection()
    demo_comparison()

    print("\n" + "="*60)
    print("🎉 Vector Stores 완료!")
    print("="*60)
    print("\n✨ 주요 기능:")
    print("  1. VectorStore.chroma()  # 팩토리 메서드")
    print("  2. from_documents(docs, embed_func)  # 가장 편리")
    print("  3. VectorStoreBuilder().use_chroma()  # Fluent API")
    print("  4. 완전한 RAG 파이프라인")
    print("  5. 5개 주요 provider 지원")
    print("\n💡 Document → Chunks → Embeddings → Vector Store → Search!")


if __name__ == "__main__":
    asyncio.run(main())
