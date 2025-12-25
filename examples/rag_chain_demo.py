"""
RAG Chain - 한 줄로 완전한 RAG 시스템
가장 간단한 방법부터 고급 사용까지
"""
from pathlib import Path
from beanllm import (
    RAGChain,
    RAGBuilder,
    create_rag,
    RAG,  # 짧은 별칭
    DocumentLoader,
    TextSplitter,
    Embedding,
    from_documents,
    Client
)


def demo_simplest():
    """가장 간단한 방법 - 한 줄!"""
    print("\n" + "="*60)
    print("1️⃣  가장 간단한 방법 - create_rag()")
    print("="*60)

    # 테스트 파일
    test_file = Path("rag_test.txt")
    test_file.write_text("""
Artificial Intelligence (AI) is transforming technology.
Machine Learning (ML) is a subset of AI that learns from data.
Deep Learning (DL) uses neural networks with multiple layers.
Natural Language Processing (NLP) helps computers understand human language.
Computer Vision enables machines to interpret visual information.
    """.strip(), encoding="utf-8")

    try:
        print("\n[한 줄로 RAG 생성]")

        # 가장 간단! (API 키 필요)
        # rag = create_rag("rag_test.txt")

        # 더미 임베딩으로 테스트 (API 키 없이)
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        # 수동으로 파이프라인 구성 (데모용)
        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs, chunk_size=200)
        store = from_documents(chunks, embed_func)

        # RAG Chain 생성
        rag = RAGChain(
            vector_store=store,
            llm=Client(model="gpt-4o-mini")
        )

        print(f"  ✓ RAG 생성 완료")
        print(f"  ✓ 문서: {len(docs)}개")
        print(f"  ✓ 청크: {len(chunks)}개")

        # 질문하기 (실제로는 LLM 호출하지 않음 - API 키 필요)
        print("\n[질문 예시]")
        print("  Q: What is Machine Learning?")
        print("  A: (LLM 답변이 여기 나옴)")

        # 검색만 테스트
        print("\n[검색 테스트]")
        results = rag.retrieve("Machine Learning", k=2)
        print(f"  ✓ 검색 결과: {len(results)}개")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.document.content[:60]}...")

    except Exception as e:
        print(f"  ⚠️  {e}")

    finally:
        if test_file.exists():
            test_file.unlink()

    print("\n💡 create_rag(파일경로)만 있으면 끝!")


def demo_from_documents():
    """from_documents - 직접 구성"""
    print("\n" + "="*60)
    print("2️⃣  RAGChain.from_documents()")
    print("="*60)

    test_file = Path("rag_test2.txt")
    test_file.write_text("""
Python is a high-level programming language.
JavaScript is essential for web development.
Java is widely used in enterprise applications.
C++ offers high performance for system programming.
Go is designed for concurrent programming.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        print("\n[from_documents로 RAG 생성]")

        # 한 번에 생성 (내부에서 자동으로 모든 단계 수행)
        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs)
        store = from_documents(chunks, embed_func)

        rag = RAGChain(
            vector_store=store,
            llm=Client(model="gpt-4o-mini"),
            prompt_template="""Answer based on the context.

Context:
{context}

Question: {question}

Answer:"""
        )

        print(f"  ✓ RAG 생성 완료")
        print(f"  ✓ 커스텀 프롬프트 템플릿 사용")

        # 검색 옵션들
        print("\n[다양한 검색 옵션]")

        print("  1. 일반 검색:")
        results = rag.retrieve("Python", k=2)
        print(f"     {len(results)}개 결과")

        print("  2. MMR 검색 (다양성):")
        results = rag.retrieve("programming", k=3, mmr=True)
        print(f"     {len(results)}개 결과")

        # Hybrid search는 특정 vector store에서만 지원
        print("  3. Hybrid 검색 (벡터+키워드):")
        try:
            results = rag.retrieve("language", k=2, hybrid=True)
            print(f"     {len(results)}개 결과")
        except Exception as e:
            print(f"     ⚠️  {e}")

    except Exception as e:
        print(f"  ⚠️  {e}")

    finally:
        if test_file.exists():
            test_file.unlink()

    print("\n💡 검색 옵션: k, rerank, mmr, hybrid")


def demo_builder():
    """RAGBuilder - Fluent API"""
    print("\n" + "="*60)
    print("3️⃣  RAGBuilder - Fluent API")
    print("="*60)

    test_file = Path("rag_test3.txt")
    test_file.write_text("""
Quantum computing uses quantum mechanics principles.
Blockchain provides decentralized data storage.
IoT connects physical devices to the internet.
Cloud computing delivers services over the internet.
Edge computing processes data near the source.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        print("\n[Fluent API로 RAG 구성]")

        # Builder 패턴
        rag = (RAGBuilder()
            .load_documents(test_file)
            .split_text(chunk_size=100, chunk_overlap=20)
            .embed_with(Embedding(model="text-embedding-3-small",
                                 provider="openai"))
            .use_llm(Client(model="gpt-4o"))
            .with_prompt("""Based on the context, provide a detailed answer.

Context:
{context}

Question: {question}

Detailed Answer:""")
            .build())

        print(f"  ✓ RAG 빌드 완료")
        print(f"  ✓ 세밀한 제어 가능")

    except Exception as e:
        print(f"  ⚠️  {e}")

    finally:
        if test_file.exists():
            test_file.unlink()

    print("\n💡 Builder 패턴으로 각 단계 제어")


def demo_query_methods():
    """다양한 쿼리 메서드"""
    print("\n" + "="*60)
    print("4️⃣  다양한 쿼리 메서드")
    print("="*60)

    test_file = Path("rag_test4.txt")
    test_file.write_text("""
Data Science combines statistics and programming.
Big Data handles large volumes of data.
Data Mining extracts patterns from data.
Data Visualization presents data graphically.
Data Engineering builds data pipelines.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs)
        store = from_documents(chunks, embed_func)

        rag = RAGChain(
            vector_store=store,
            llm=Client(model="gpt-4o-mini")
        )

        print("\n[1] query() - 기본 쿼리:")
        print("  answer = rag.query('What is data science?')")
        print("  # Returns: str")

        print("\n[2] query() with sources:")
        print("  answer, sources = rag.query('question', include_sources=True)")
        print("  # Returns: (str, List[VectorSearchResult])")

        print("\n[3] stream_query() - 스트리밍:")
        print("  for chunk in rag.stream_query('question'):")
        print("      print(chunk, end='')")

        print("\n[4] batch_query() - 배치 처리:")
        print("  questions = ['Q1', 'Q2', 'Q3']")
        print("  answers = rag.batch_query(questions)")
        print("  # Returns: List[str]")

        print("\n[5] aquery() - 비동기:")
        print("  answer = await rag.aquery('question')")

        # 검색만 테스트
        print("\n[검색 결과 확인]")
        results = rag.retrieve("Data Science", k=2)
        print(f"  ✓ {len(results)}개 문서 검색됨")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.document.content[:50]}...")

    except Exception as e:
        print(f"  ⚠️  {e}")

    finally:
        if test_file.exists():
            test_file.unlink()

    print("\n💡 5가지 쿼리 메서드 제공")


def demo_advanced_retrieval():
    """고급 검색 기능"""
    print("\n" + "="*60)
    print("5️⃣  고급 검색 기능")
    print("="*60)

    test_file = Path("rag_test5.txt")
    test_file.write_text("""
Neural networks learn patterns from data.
Convolutional networks excel at image tasks.
Recurrent networks handle sequential data.
Transformer networks use attention mechanisms.
Generative networks create new content.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs)
        store = from_documents(chunks, embed_func)

        rag = RAGChain(
            vector_store=store,
            llm=Client(model="gpt-4o-mini")
        )

        print("\n[검색 전략]")

        print("  1. 기본 검색 (유사도):")
        results = rag.retrieve("neural networks", k=3)
        print(f"     {len(results)}개 결과")

        print("\n  2. MMR (다양성 고려):")
        results = rag.retrieve("networks", k=3, mmr=True)
        print(f"     {len(results)}개 결과 (중복 제거)")

        print("\n  3. Re-ranking (정확도 향상):")
        try:
            results = rag.retrieve("networks", k=3, rerank=True)
            print(f"     {len(results)}개 결과 (재순위화)")
        except ImportError:
            print("     ⚠️  sentence-transformers 필요")

        print("\n  4. Hybrid (벡터 + 키워드):")
        try:
            results = rag.retrieve("neural", k=3, hybrid=True)
            print(f"     {len(results)}개 결과")
        except Exception as e:
            print(f"     ⚠️  {e}")

        print("\n  5. 모든 옵션 조합:")
        print("     retrieve(query, k=10, rerank=True, mmr=True)")

    except Exception as e:
        print(f"  ⚠️  {e}")

    finally:
        if test_file.exists():
            test_file.unlink()

    print("\n💡 검색 품질을 높이는 다양한 전략")


def demo_aliases():
    """별칭 사용"""
    print("\n" + "="*60)
    print("6️⃣  편리한 별칭")
    print("="*60)

    print("\n[사용 가능한 방법들]")
    print("""
    # 1. RAGChain (전체 이름)
    rag = RAGChain.from_documents("doc.pdf")

    # 2. RAG (짧은 별칭)
    rag = RAG.from_documents("doc.pdf")

    # 3. create_rag (함수)
    rag = create_rag("doc.pdf")

    # 모두 동일한 결과!
    """)

    print("💡 취향에 맞게 선택하세요")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🚀 RAG Chain - 한 줄로 완전한 RAG")
    print("="*60)
    print("\n6가지 사용 방법:")
    print("  1. create_rag() - 가장 간단")
    print("  2. RAGChain.from_documents() - 직접 구성")
    print("  3. RAGBuilder - Fluent API")
    print("  4. 다양한 쿼리 메서드")
    print("  5. 고급 검색 기능")
    print("  6. 편리한 별칭")

    demo_simplest()
    demo_from_documents()
    demo_builder()
    demo_query_methods()
    demo_advanced_retrieval()
    demo_aliases()

    print("\n" + "="*60)
    print("🎉 RAG Chain 데모 완료!")
    print("="*60)
    print("\n✨ 핵심 기능:")
    print("  생성:")
    print("    • create_rag(source) - 가장 간단")
    print("    • RAGChain.from_documents(source) - 제어")
    print("    • RAGBuilder().load()....build() - Fluent")
    print("\n  쿼리:")
    print("    • query() - 기본")
    print("    • stream_query() - 스트리밍")
    print("    • batch_query() - 배치")
    print("    • aquery() - 비동기")
    print("\n  검색:")
    print("    • retrieve(k=4, rerank=True, mmr=True, hybrid=True)")
    print("\n💡 한 줄로 완전한 RAG 시스템!")


if __name__ == "__main__":
    main()
