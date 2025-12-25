"""
RAG Demo - Document Loading & Text Splitting
beanllm 방식: 자동 감지 + 스마트 기본값
"""
import asyncio
from pathlib import Path


def demo_document_loading():
    """Document Loading 데모"""
    print("\n" + "="*60)
    print("📄 Document Loading Demo")
    print("="*60)

    from beanllm import DocumentLoader, load_documents

    # 1. 텍스트 파일 (자동 감지!)
    print("\n1. Auto-detect Text File:")
    # 테스트 파일 생성
    test_file = Path("test_doc.txt")
    test_file.write_text("This is a test document.\nWith multiple lines.\nFor testing beanllm!", encoding="utf-8")

    docs = DocumentLoader.load(test_file)
    print(f"   Loaded {len(docs)} document(s)")
    print(f"   Content: {docs[0].content[:50]}...")
    print(f"   Metadata: {docs[0].metadata}")

    # 편의 함수
    docs2 = load_documents(test_file)
    print(f"   ✓ Same result with convenience function: {len(docs2)} doc(s)")

    # 정리
    test_file.unlink()

    # 2. CSV 파일 (자동 감지!)
    print("\n2. Auto-detect CSV File:")
    csv_file = Path("test_data.csv")
    csv_file.write_text("name,age,city\nAlice,30,Seoul\nBob,25,Busan\nCharlie,35,Incheon", encoding="utf-8")

    docs = DocumentLoader.load(csv_file)
    print(f"   Loaded {len(docs)} document(s) (one per row)")
    print(f"   First row: {docs[0].content[:50]}...")

    # 특정 컬럼만
    docs_custom = DocumentLoader.load(csv_file, content_columns=["name", "city"])
    print(f"   Custom columns: {docs_custom[0].content}")

    # 정리
    csv_file.unlink()

    print("\n✓ Document Loading: AUTO-DETECTION works!")


def demo_text_splitting():
    """Text Splitting 데모"""
    print("\n" + "="*60)
    print("✂️  Text Splitting Demo")
    print("="*60)

    from beanllm import DocumentLoader, TextSplitter, split_documents, Document

    # 테스트 문서 생성
    long_text = """
# Introduction to AI

Artificial Intelligence (AI) is transforming the world. It enables machines to learn from experience.

## Machine Learning

Machine Learning is a subset of AI. It focuses on algorithms that improve through experience.

### Supervised Learning

In supervised learning, we train models on labeled data. The model learns to predict outcomes.

### Unsupervised Learning

Unsupervised learning works with unlabeled data. It finds patterns without explicit guidance.

## Deep Learning

Deep Learning uses neural networks with multiple layers. It powers modern AI applications like image recognition and natural language processing.

# Conclusion

AI continues to evolve rapidly. The future holds exciting possibilities.
    """.strip()

    docs = [Document(content=long_text, metadata={"source": "ai_intro.md"})]

    # 1. 가장 간단한 방법 (스마트 기본값!)
    print("\n1. Simplest Way (Smart Defaults):")
    chunks = TextSplitter.split(docs, chunk_size=200)
    print(f"   Split into {len(chunks)} chunks")
    print(f"   First chunk: {chunks[0].content[:80]}...")
    print(f"   Metadata preserved: {chunks[0].metadata}")

    # 2. 편의 함수
    print("\n2. Convenience Function:")
    chunks2 = split_documents(docs, chunk_size=200)
    print(f"   Same result: {len(chunks2)} chunks")

    # 3. 다양한 전략
    print("\n3. Different Strategies:")

    # Recursive (기본, 가장 권장)
    chunks_recursive = TextSplitter.split(
        docs,
        strategy="recursive",
        chunk_size=200,
        chunk_overlap=50
    )
    print(f"   Recursive: {len(chunks_recursive)} chunks (recommended)")

    # Character
    chunks_char = TextSplitter.split(
        docs,
        strategy="character",
        separator="\n\n",
        chunk_size=200
    )
    print(f"   Character: {len(chunks_char)} chunks")

    # 4. 마크다운 헤더 분할
    print("\n4. Markdown Header Splitting:")
    from beanllm import MarkdownHeaderTextSplitter

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]
    )
    md_chunks = md_splitter.split_documents(docs)
    print(f"   Split by headers: {len(md_chunks)} chunks")
    for i, chunk in enumerate(md_chunks[:3]):
        print(f"   Chunk {i+1} metadata: {chunk.metadata}")

    print("\n✓ Text Splitting: SMART DEFAULTS work!")


def demo_token_splitting():
    """Token-based Splitting 데모"""
    print("\n" + "="*60)
    print("🔢 Token-based Splitting Demo")
    print("="*60)

    try:
        import tiktoken
    except ImportError:
        print("\n⚠️  tiktoken not installed. Install with: pip install tiktoken")
        return

    from beanllm import TextSplitter, Document

    text = "AI is amazing. " * 100  # 긴 텍스트
    docs = [Document(content=text, metadata={"source": "test"})]

    # Token 기반 분할
    print("\n1. Token-based Splitting (for LLM context limits):")
    chunks = TextSplitter.split(
        docs,
        strategy="token",
        chunk_size=50,  # 토큰 단위
        chunk_overlap=10
    )
    print(f"   Split into {len(chunks)} chunks (50 tokens each)")
    print(f"   First chunk: {chunks[0].content[:50]}...")

    # 특정 모델용
    print("\n2. Model-specific (GPT-4):")
    from beanllm import TokenTextSplitter

    splitter = TokenTextSplitter(
        model_name="gpt-4",
        chunk_size=100,
        chunk_overlap=20
    )
    chunks2 = splitter.split_documents(docs)
    print(f"   Split for GPT-4: {len(chunks2)} chunks")

    print("\n✓ Token Splitting: Works great for LLM context management!")


def demo_full_pipeline():
    """전체 파이프라인 데모"""
    print("\n" + "="*60)
    print("🚀 Full RAG Pipeline Demo")
    print("="*60)

    from beanllm import DocumentLoader, TextSplitter

    # 1. 문서 로딩 (자동 감지)
    print("\n1. Load Documents (Auto-detect):")
    test_file = Path("rag_test.txt")
    test_file.write_text("""
AI and Machine Learning

Artificial Intelligence is revolutionizing technology. Machine learning algorithms learn from data.

Deep Learning

Deep learning uses neural networks. These networks have multiple layers that process information hierarchically.

Applications

AI powers many applications: voice assistants, image recognition, autonomous vehicles, and more.

Future of AI

The future of AI is bright. New breakthroughs happen constantly.
    """.strip(), encoding="utf-8")

    docs = DocumentLoader.load(test_file)
    print(f"   ✓ Loaded: {len(docs)} document(s)")
    print(f"   Content length: {len(docs[0].content)} characters")

    # 2. 텍스트 분할 (스마트 기본값)
    print("\n2. Split Text (Smart defaults):")
    chunks = TextSplitter.split(docs, chunk_size=150, chunk_overlap=30)
    print(f"   ✓ Split into: {len(chunks)} chunks")

    # 각 청크 미리보기
    print("\n   Chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"   Chunk {i+1}: {chunk.content[:60]}...")
        print(f"            Metadata: chunk={chunk.metadata.get('chunk', 'N/A')}")

    # 정리
    test_file.unlink()

    print("\n" + "="*60)
    print("🎉 beanllm RAG: Simple & Pythonic!")
    print("="*60)
    print("\nKey Features:")
    print("  ✅ Auto-detection (no manual loader selection)")
    print("  ✅ Smart defaults (optimal settings out of the box)")
    print("  ✅ One-line usage for 80% of cases")
    print("  ✅ Full customization for 20% of cases")
    print("  ✅ Pythonic and intuitive API")
    print("\nNext: Embeddings & Vector Stores!")


def main():
    """Run all demos"""
    demo_document_loading()
    demo_text_splitting()
    demo_token_splitting()
    demo_full_pipeline()


if __name__ == "__main__":
    main()
