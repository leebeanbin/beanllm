"""
고급 검색 기능 데모
- Hybrid Search (벡터 + 키워드)
- Re-ranking (Cross-encoder)
- MMR Search (다양성)
- Embedding 고급 기능
"""
from pathlib import Path
from beanllm import (
    DocumentLoader,
    TextSplitter,
    Embedding,
    from_documents,
    # Embedding 고급 기능
    find_hard_negatives,
    mmr_search as embedding_mmr,
    query_expansion,
    EmbeddingCache
)


def demo_hybrid_search():
    """Hybrid Search - 벡터 + 키워드"""
    print("\n" + "="*60)
    print("1️⃣  Hybrid Search (벡터 + 키워드)")
    print("="*60)

    # 테스트 파일
    test_file = Path("hybrid_test.txt")
    test_file.write_text("""
Python is a programming language used for web development.
JavaScript is essential for frontend development.
Machine learning requires Python and mathematics knowledge.
Deep learning uses neural networks for complex tasks.
Natural language processing helps computers understand text.
Computer vision enables image recognition and analysis.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        # 파이프라인
        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs, chunk_size=100)
        store = from_documents(chunks, embed_func)

        # 일반 벡터 검색
        print("\n[벡터 검색만]")
        results = store.similarity_search("programming", k=3)
        print(f"  결과: {len(results)}개")
        for i, r in enumerate(results[:2], 1):
            print(f"  {i}. {r.document.content[:50]}...")

        # Hybrid Search
        print("\n[Hybrid Search (벡터 + 키워드)]")
        try:
            # alpha = 0.5 (균형)
            results = store.hybrid_search("programming", k=3, alpha=0.5)
            print(f"  결과: {len(results)}개")
            for i, r in enumerate(results[:2], 1):
                print(f"  {i}. Score: {r.score:.3f}")
                print(f"      {r.document.content[:50]}...")

            # alpha = 0.8 (벡터 중심)
            print("\n  alpha=0.8 (벡터 중심):")
            results = store.hybrid_search("programming", k=3, alpha=0.8)
            print(f"  결과: {len(results)}개")

        except Exception as e:
            print(f"  ⚠️  {e}")

        print("\n💡 Hybrid Search는 벡터와 키워드를 결합!")

    finally:
        if test_file.exists():
            test_file.unlink()


def demo_reranking():
    """Re-ranking - Cross-encoder"""
    print("\n" + "="*60)
    print("2️⃣  Re-ranking (Cross-encoder)")
    print("="*60)

    # 테스트 파일
    test_file = Path("rerank_test.txt")
    test_file.write_text("""
Artificial intelligence is transforming technology.
Machine learning algorithms learn from data patterns.
Deep learning uses neural networks with multiple layers.
Natural language processing understands human language.
Computer vision interprets visual information.
Robotics combines AI with mechanical engineering.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs, chunk_size=100)
        store = from_documents(chunks, embed_func)

        # 초기 검색 (더 많이 가져옴)
        print("\n[초기 검색 - 상위 5개]")
        results = store.similarity_search("artificial intelligence", k=5)
        print(f"  결과: {len(results)}개")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. Score: {r.score:.3f}")
            print(f"      {r.document.content[:50]}...")

        # Re-ranking
        print("\n[Re-ranking 후 - 상위 3개]")
        try:
            reranked = store.rerank(
                "artificial intelligence",
                results,
                top_k=3
            )
            print(f"  결과: {len(reranked)}개")
            for i, r in enumerate(reranked, 1):
                print(f"  {i}. Score: {r.score:.3f}")
                print(f"      {r.document.content[:50]}...")

            print("\n💡 Cross-encoder로 더 정확한 순위!")

        except ImportError:
            print("  ⚠️  sentence-transformers 필요:")
            print("     pip install sentence-transformers")

    finally:
        if test_file.exists():
            test_file.unlink()


def demo_mmr_search():
    """MMR Search - 다양성 고려"""
    print("\n" + "="*60)
    print("3️⃣  MMR Search (다양성 고려)")
    print("="*60)

    # 테스트 파일 (유사한 내용들)
    test_file = Path("mmr_test.txt")
    test_file.write_text("""
Python is great for machine learning.
Python is excellent for ML tasks.
Python is perfect for ML projects.
JavaScript is used for web development.
Java is popular for enterprise applications.
C++ is fast for system programming.
    """.strip(), encoding="utf-8")

    try:
        import random
        embed_func = lambda texts: [[random.random() for _ in range(384)] for _ in texts]

        docs = DocumentLoader.load(test_file)
        chunks = TextSplitter.split(docs, chunk_size=100)
        store = from_documents(chunks, embed_func)

        # 일반 검색 (유사한 결과들)
        print("\n[일반 검색 - 유사한 결과들]")
        results = store.similarity_search("Python machine learning", k=4)
        print(f"  결과: {len(results)}개")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.document.content}")

        # MMR 검색 (다양성 고려)
        print("\n[MMR Search - 다양한 결과]")
        results = store.mmr_search(
            "Python machine learning",
            k=4,
            fetch_k=6,
            lambda_param=0.5  # 관련성 50%, 다양성 50%
        )
        print(f"  결과: {len(results)}개")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.document.content}")

        print("\n💡 MMR은 유사한 결과를 피하고 다양성 확보!")

    finally:
        if test_file.exists():
            test_file.unlink()


def demo_embedding_hard_negatives():
    """Embedding - Hard Negative Mining"""
    print("\n" + "="*60)
    print("4️⃣  Hard Negative Mining")
    print("="*60)

    try:
        embed = Embedding.openai()

        # 쿼리
        query = "고양이 사료"

        # 후보들
        candidates_text = [
            "강아지 사료",  # Hard negative (비슷하지만 다름)
            "고양이 장난감",  # Hard negative
            "자동차 부품",  # Easy negative (완전히 다름)
            "고양이 간식"   # Positive (관련있음)
        ]

        print(f"\n쿼리: '{query}'")
        print(f"후보 {len(candidates_text)}개:")
        for i, text in enumerate(candidates_text, 1):
            print(f"  {i}. {text}")

        # 임베딩
        query_vec = embed.embed_sync([query])[0]
        candidate_vecs = embed.embed_sync(candidates_text)

        # Hard Negatives 찾기
        hard_negs = find_hard_negatives(
            query_vec,
            candidate_vecs,
            threshold=0.7,  # 유사도 0.5~0.7 사이
            top_k=2
        )

        print(f"\nHard Negatives ({len(hard_negs)}개):")
        for idx, score in hard_negs:
            print(f"  • {candidates_text[idx]} (유사도: {score:.3f})")

        print("\n💡 Hard Negatives: 비슷하지만 다른 것들 (학습에 유용)")

    except Exception as e:
        print(f"⚠️  {e}")


def demo_embedding_mmr():
    """Embedding - MMR for diversity"""
    print("\n" + "="*60)
    print("5️⃣  MMR (Embedding 레벨)")
    print("="*60)

    try:
        embed = Embedding.openai()

        # 쿼리
        query = "프로그래밍 언어"

        # 후보들 (일부 중복/유사)
        candidates_text = [
            "Python은 배우기 쉬운 언어",
            "Python은 인기있는 언어",
            "JavaScript는 웹 개발에 사용",
            "Java는 엔터프라이즈급",
            "C++는 성능이 뛰어남",
            "Ruby는 생산성이 높음"
        ]

        print(f"\n쿼리: '{query}'")
        print(f"후보 {len(candidates_text)}개")

        # 임베딩
        query_vec = embed.embed_sync([query])[0]
        candidate_vecs = embed.embed_sync(candidates_text)

        # MMR로 다양한 결과 선택
        selected_indices = embedding_mmr(
            query_vec,
            candidate_vecs,
            k=3,
            lambda_param=0.6  # 관련성 60%, 다양성 40%
        )

        print(f"\nMMR 선택 결과 ({len(selected_indices)}개):")
        for rank, idx in enumerate(selected_indices, 1):
            print(f"  {rank}. {candidates_text[idx]}")

        print("\n💡 중복/유사한 결과를 피하고 다양성 확보!")

    except Exception as e:
        print(f"⚠️  {e}")


def demo_query_expansion():
    """Query Expansion"""
    print("\n" + "="*60)
    print("6️⃣  Query Expansion")
    print("="*60)

    try:
        embed = Embedding.openai()

        # 원본 쿼리
        query = "고양이"

        # 확장 후보
        expansion_candidates = [
            "냥이", "고양이과", "cat", "페르시안", "길고양이",
            "강아지", "토끼", "햄스터"  # 관련없는 것들
        ]

        print(f"\n원본 쿼리: '{query}'")
        print(f"확장 후보: {len(expansion_candidates)}개")

        # Query Expansion
        expanded = query_expansion(
            query,
            embed,
            expansion_candidates,
            top_k=3
        )

        print(f"\n확장된 쿼리 ({len(expanded)}개):")
        for term, score in expanded:
            print(f"  • {term} (유사도: {score:.3f})")

        # 확장된 쿼리로 검색 가능
        expanded_query = " ".join([term for term, _ in expanded])
        print(f"\n최종 쿼리: '{query} {expanded_query}'")

        print("\n💡 Query Expansion으로 검색 범위 확대!")

    except Exception as e:
        print(f"⚠️  {e}")


def demo_embedding_cache():
    """Embedding Cache"""
    print("\n" + "="*60)
    print("7️⃣  Embedding Cache")
    print("="*60)

    try:
        from beanllm import EmbeddingCache
        import time

        embed = Embedding.openai()
        cache = EmbeddingCache(ttl=3600, max_size=1000)

        texts = ["Python programming", "Machine learning", "Deep learning"]

        # 첫 번째 호출 (캐시 miss)
        print("\n[첫 번째 호출 - 캐시 miss]")
        start = time.time()

        # 캐시 확인
        cached_vecs = []
        for text in texts:
            vec = cache.get(text)
            if vec:
                cached_vecs.append(vec)
            else:
                # 캐시에 없으면 임베딩 생성
                vec = embed.embed_sync([text])[0]
                cache.set(text, vec)
                cached_vecs.append(vec)

        elapsed1 = time.time() - start
        print(f"  시간: {elapsed1:.3f}초")
        print(f"  캐시 상태: {cache.stats()}")

        # 두 번째 호출 (캐시 hit)
        print("\n[두 번째 호출 - 캐시 hit]")
        start = time.time()

        cached_vecs = []
        for text in texts:
            vec = cache.get(text)
            if vec:
                cached_vecs.append(vec)

        elapsed2 = time.time() - start
        print(f"  시간: {elapsed2:.3f}초")
        print(f"  캐시 상태: {cache.stats()}")

        speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
        print(f"\n  속도 향상: {speedup:.1f}x")

        print("\n💡 자주 사용하는 임베딩은 캐시로 빠르게!")

    except Exception as e:
        print(f"⚠️  {e}")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🚀 고급 검색 기능 데모")
    print("="*60)
    print("\nVectorStore 고급 검색:")
    print("  1. Hybrid Search (벡터 + 키워드)")
    print("  2. Re-ranking (Cross-encoder)")
    print("  3. MMR Search (다양성)")
    print("\nEmbedding 고급 기능:")
    print("  4. Hard Negative Mining")
    print("  5. MMR (Embedding)")
    print("  6. Query Expansion")
    print("  7. Embedding Cache")

    # VectorStore 고급 검색
    demo_hybrid_search()
    demo_reranking()
    demo_mmr_search()

    # Embedding 고급 기능
    demo_embedding_hard_negatives()
    demo_embedding_mmr()
    demo_query_expansion()
    demo_embedding_cache()

    print("\n" + "="*60)
    print("🎉 고급 검색 데모 완료!")
    print("="*60)
    print("\n✨ 핵심 기능:")
    print("  VectorStore:")
    print("    • hybrid_search() - 벡터 + 키워드")
    print("    • rerank() - Cross-encoder 재순위화")
    print("    • mmr_search() - 다양성 고려")
    print("\n  Embedding:")
    print("    • find_hard_negatives() - Hard Negative Mining")
    print("    • mmr_search() - MMR 선택")
    print("    • query_expansion() - 쿼리 확장")
    print("    • EmbeddingCache - 캐싱")
    print("\n💡 RAG 품질을 크게 향상시키는 기능들!")


if __name__ == "__main__":
    main()
