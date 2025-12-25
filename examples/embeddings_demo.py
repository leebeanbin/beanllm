"""
Embeddings Demo - 통합 인터페이스
beanllm 방식: Client와 같은 패턴
"""
import asyncio
from beanllm import Embedding, embed, embed_sync


async def demo_auto_detection():
    """자동 provider 감지"""
    print("\n" + "="*60)
    print("🤖 자동 Provider 감지")
    print("="*60)

    # OpenAI (자동 감지)
    print("\n1. OpenAI (자동 감지):")
    try:
        emb = Embedding(model="text-embedding-3-small")
        vectors = await emb.embed(["Hello", "World"])
        print(f"   ✓ OpenAI 자동 감지")
        print(f"   ✓ 2개 텍스트 → {len(vectors)} 벡터")
        print(f"   ✓ 벡터 차원: {len(vectors[0])}")
    except Exception as e:
        print(f"   ⚠️  OpenAI API 키 필요: {e}")

    # Cohere (자동 감지)
    print("\n2. Cohere (자동 감지):")
    try:
        emb = Embedding(model="embed-english-v3.0")
        vectors = await emb.embed(["Hello", "World"])
        print(f"   ✓ Cohere 자동 감지")
        print(f"   ✓ 2개 텍스트 → {len(vectors)} 벡터")
        print(f"   ✓ 벡터 차원: {len(vectors[0])}")
    except Exception as e:
        print(f"   ⚠️  Cohere API 키 필요: {e}")

    print("\n✓ 모델 이름만으로 자동 provider 감지!")


async def demo_explicit_selection():
    """명시적 provider 선택"""
    print("\n" + "="*60)
    print("🎯 명시적 Provider 선택")
    print("="*60)

    # 방법 1: provider 파라미터
    print("\n1. provider 파라미터:")
    try:
        emb = Embedding(model="text-embedding-3-small", provider="openai")
        vectors = await emb.embed(["Test"])
        print(f"   ✓ provider='openai' 명시")
        print(f"   ✓ {len(vectors)} 벡터 생성")
    except Exception as e:
        print(f"   ⚠️  {e}")

    # 방법 2: 팩토리 메서드
    print("\n2. 팩토리 메서드 (추천!):")
    try:
        emb = Embedding.openai(model="text-embedding-3-small")
        vectors = await emb.embed(["Test"])
        print(f"   ✓ Embedding.openai() 사용")
        print(f"   ✓ {len(vectors)} 벡터 생성")
    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ 자동 감지 + 명시적 선택 둘 다 가능!")


async def demo_convenience_functions():
    """편의 함수"""
    print("\n" + "="*60)
    print("⚡ 편의 함수")
    print("="*60)

    # embed() 함수
    print("\n1. embed() 함수 (비동기):")
    try:
        # 단일 텍스트
        vector = await embed("Hello world")
        print(f"   ✓ 단일 텍스트: {len(vector)} 벡터")
        print(f"   ✓ 차원: {len(vector[0])}")

        # 여러 텍스트
        vectors = await embed(["Text 1", "Text 2", "Text 3"])
        print(f"   ✓ 여러 텍스트: {len(vectors)} 벡터")
    except Exception as e:
        print(f"   ⚠️  {e}")

    # embed_sync() 함수
    print("\n2. embed_sync() 함수 (동기):")
    try:
        vectors = embed_sync(["Sync", "Embedding"])
        print(f"   ✓ 동기 버전: {len(vectors)} 벡터")
    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ 간단한 임베딩은 한 줄로!")


async def demo_batch_processing():
    """배치 처리"""
    print("\n" + "="*60)
    print("📦 배치 처리")
    print("="*60)

    texts = [
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Neural Networks",
        "Natural Language Processing"
    ]

    print(f"\n{len(texts)}개 텍스트 임베딩:")
    try:
        emb = Embedding(model="text-embedding-3-small")
        vectors = await emb.embed(texts)

        print(f"   ✓ {len(vectors)} 벡터 생성")
        print(f"   ✓ 차원: {len(vectors[0])}")

        # 벡터 미리보기
        print("\n   벡터 미리보기:")
        for i, text in enumerate(texts[:2]):
            print(f"      '{text}': [{vectors[i][0]:.4f}, {vectors[i][1]:.4f}, ...]")

    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n✓ 배치 처리로 효율적인 임베딩!")


async def demo_different_models():
    """다양한 모델"""
    print("\n" + "="*60)
    print("🔄 다양한 모델")
    print("="*60)

    models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]

    text = "Hello, embeddings!"

    print("\nOpenAI 모델 비교:")
    for model in models:
        try:
            emb = Embedding(model=model)
            vectors = await emb.embed([text])
            print(f"   ✓ {model}: 차원 {len(vectors[0])}")
        except Exception as e:
            print(f"   ⚠️  {model}: {e}")

    print("\n✓ 같은 인터페이스로 모든 모델 사용!")


async def demo_integration_with_documents():
    """문서와 통합"""
    print("\n" + "="*60)
    print("📄 문서 로딩 + 임베딩 통합")
    print("="*60)

    from beanllm import DocumentLoader, TextSplitter
    from pathlib import Path

    # 테스트 파일 생성
    test_file = Path("embedding_test.txt")
    test_file.write_text("""
Artificial Intelligence is transforming the world.
Machine learning algorithms learn from data.
Deep learning uses neural networks.
    """.strip(), encoding="utf-8")

    try:
        # 1. 문서 로딩
        docs = DocumentLoader.load(test_file)
        print(f"\n1. 문서 로딩: {len(docs)} 문서")

        # 2. 텍스트 분할
        chunks = TextSplitter.split(docs, chunk_size=100)
        print(f"2. 텍스트 분할: {len(chunks)} 청크")

        # 3. 임베딩
        texts = [chunk.content for chunk in chunks]
        try:
            vectors = await embed(texts)
            print(f"3. 임베딩: {len(vectors)} 벡터")
            print(f"   ✓ 차원: {len(vectors[0])}")

            print("\n✓ 문서 → 청크 → 임베딩 파이프라인 완성!")

        except Exception as e:
            print(f"   ⚠️  임베딩 실패: {e}")

    finally:
        # 정리
        if test_file.exists():
            test_file.unlink()


def demo_comparison():
    """LangChain vs beanllm 비교"""
    print("\n" + "="*60)
    print("📊 LangChain vs beanllm 비교")
    print("="*60)

    print("\n【 LangChain 방식 】")
    print("""
    from langchain.embeddings import OpenAIEmbeddings

    # Provider별 클래스 import
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(["text1", "text2"])
    """)

    print("\n【 beanllm 방식 】")
    print("""
    from beanllm import Embedding, embed

    # 방법 1: 자동 감지
    emb = Embedding(model="text-embedding-3-small")  # OpenAI 자동
    vectors = await emb.embed(["text1", "text2"])

    # 방법 2: 더 간단하게
    vectors = await embed(["text1", "text2"])
    """)

    print("\n✅ beanllm: 자동 감지 + 통합 인터페이스")
    print("✅ Client와 같은 패턴으로 일관성!")


async def main():
    """모든 데모 실행"""
    print("="*60)
    print("🎯 Embeddings 데모")
    print("="*60)
    print("\nbeanllm의 철학:")
    print("  1. 자동 감지 (Client와 같은 패턴)")
    print("  2. 통합 인터페이스 (일관된 API)")
    print("  3. 간단한 사용 (편의 함수)")

    await demo_auto_detection()
    await demo_explicit_selection()
    await demo_convenience_functions()
    await demo_batch_processing()
    await demo_different_models()
    await demo_integration_with_documents()
    demo_comparison()

    print("\n" + "="*60)
    print("🎉 Embeddings 완료!")
    print("="*60)
    print("\n✨ 주요 기능:")
    print("  1. Embedding(model='text-embedding-3-small')  # 자동 감지")
    print("  2. Embedding.openai()  # 명시적 선택")
    print("  3. await embed(['text1', 'text2'])  # 편의 함수")
    print("  4. 배치 처리 지원")
    print("  5. 문서 파이프라인 통합")
    print("\n💡 Client와 같은 패턴으로 쉽고 일관적!")


if __name__ == "__main__":
    asyncio.run(main())
