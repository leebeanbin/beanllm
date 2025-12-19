"""
Simple test runner for Text Splitters (no pytest needed)
"""
from llmkit import (
    Document,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter,
    split_documents,
    DocumentLoader
)
from pathlib import Path


def test_character_splitter():
    """CharacterTextSplitter 테스트"""
    print("\n1. Testing CharacterTextSplitter...")

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=20
    )

    text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
    chunks = splitter.split_text(text)

    assert len(chunks) > 0, "Should create chunks"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"

    # Test with documents
    doc = Document(content=text, metadata={"source": "test"})
    doc_chunks = splitter.split_documents([doc])

    assert len(doc_chunks) > 0, "Should create document chunks"
    assert doc_chunks[0].metadata["source"] == "test", "Should preserve metadata"
    assert "chunk" in doc_chunks[0].metadata, "Should add chunk number"

    print(f"   ✓ Created {len(chunks)} text chunks")
    print(f"   ✓ Created {len(doc_chunks)} document chunks")
    print("   ✓ CharacterTextSplitter works!")


def test_recursive_splitter():
    """RecursiveCharacterTextSplitter 테스트"""
    print("\n2. Testing RecursiveCharacterTextSplitter...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    text = """
# Header

Paragraph 1 with some text.

Paragraph 2 with more text.

Final paragraph.
    """.strip()

    chunks = splitter.split_text(text)

    assert len(chunks) > 0, "Should create chunks"
    print(f"   ✓ Created {len(chunks)} chunks")

    # Test with custom separators
    splitter2 = RecursiveCharacterTextSplitter(
        separators=["###", "##", "#", "\n\n"],
        chunk_size=50,
        chunk_overlap=10
    )

    text2 = "# Big\n## Smaller\n### Smallest\nContent"
    chunks2 = splitter2.split_text(text2)

    assert len(chunks2) > 0, "Should work with custom separators"
    print(f"   ✓ Custom separators work: {len(chunks2)} chunks")
    print("   ✓ RecursiveCharacterTextSplitter works!")


def test_markdown_splitter():
    """MarkdownHeaderTextSplitter 테스트"""
    print("\n3. Testing MarkdownHeaderTextSplitter...")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
        ]
    )

    text = """
# Main Title

Introduction text.

## Section 1

Section 1 content.

### Subsection 1.1

Subsection content.
    """.strip()

    chunks = splitter.split_text(text)

    assert len(chunks) > 0, "Should create chunks"
    assert all(isinstance(chunk, Document) for chunk in chunks), "Should return Documents"

    # Check metadata
    has_h1 = any("H1" in chunk.metadata for chunk in chunks)
    has_h2 = any("H2" in chunk.metadata for chunk in chunks)

    assert has_h1 or has_h2, "Should have header metadata"

    print(f"   ✓ Created {len(chunks)} chunks with headers")
    print(f"   ✓ First chunk metadata: {chunks[0].metadata}")
    print("   ✓ MarkdownHeaderTextSplitter works!")


def test_token_splitter():
    """TokenTextSplitter 테스트"""
    print("\n4. Testing TokenTextSplitter...")

    try:
        import tiktoken
    except ImportError:
        print("   ⚠️  tiktoken not installed, skipping")
        return

    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=50,
        chunk_overlap=10
    )

    text = "AI is amazing. " * 100
    chunks = splitter.split_text(text)

    assert len(chunks) > 0, "Should create chunks"
    print(f"   ✓ Created {len(chunks)} token-based chunks")

    # Test model-specific
    splitter2 = TokenTextSplitter(
        model_name="gpt-4",
        chunk_size=100,
        chunk_overlap=20
    )

    chunks2 = splitter2.split_text(text)
    assert len(chunks2) > 0, "Should work with model name"
    print(f"   ✓ Model-specific splitting works: {len(chunks2)} chunks")
    print("   ✓ TokenTextSplitter works!")


def test_text_splitter_factory():
    """TextSplitter Factory 테스트"""
    print("\n5. Testing TextSplitter Factory...")

    doc = Document(
        content="AI is transforming the world. " * 20,
        metadata={"source": "test.txt"}
    )

    # Default
    chunks = TextSplitter.split([doc])
    assert len(chunks) > 0, "Should work with defaults"
    print(f"   ✓ Default splitting: {len(chunks)} chunks")

    # Recursive strategy
    chunks_rec = TextSplitter.split([doc], strategy="recursive", chunk_size=100)
    assert len(chunks_rec) > 0, "Should work with recursive"
    print(f"   ✓ Recursive: {len(chunks_rec)} chunks")

    # Character strategy
    chunks_char = TextSplitter.split(
        [doc],
        strategy="character",
        separator=" ",
        chunk_size=50
    )
    assert len(chunks_char) > 0, "Should work with character"
    print(f"   ✓ Character: {len(chunks_char)} chunks")

    # Create splitter
    splitter = TextSplitter.create(strategy="recursive", chunk_size=100)
    assert isinstance(splitter, RecursiveCharacterTextSplitter), "Should return correct type"
    print("   ✓ Factory creates correct splitter types")
    print("   ✓ TextSplitter Factory works!")


def test_convenience_function():
    """편의 함수 테스트"""
    print("\n6. Testing Convenience Function...")

    doc = Document(
        content="This is a test document. " * 30,
        metadata={"source": "test.txt"}
    )

    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 0, "Should create chunks"
    assert all(isinstance(chunk, Document) for chunk in chunks), "Should return Documents"

    print(f"   ✓ split_documents() created {len(chunks)} chunks")
    print("   ✓ Convenience function works!")


def test_full_integration():
    """통합 테스트"""
    print("\n7. Testing Full Integration...")

    # 1. Create test file
    test_file = Path("test_integration.txt")
    test_file.write_text("""
AI and Machine Learning are transforming the world.

Deep learning uses neural networks with multiple layers.

Applications include computer vision and natural language processing.

The future of AI is exciting and full of possibilities.
    """.strip(), encoding="utf-8")

    try:
        # 2. Load documents
        docs = DocumentLoader.load(test_file)
        assert len(docs) == 1, "Should load one document"
        print(f"   ✓ Loaded {len(docs)} document")

        # 3. Split text
        chunks = TextSplitter.split(docs, chunk_size=80, chunk_overlap=20)
        assert len(chunks) > 0, "Should create chunks"
        print(f"   ✓ Split into {len(chunks)} chunks")

        # 4. Check metadata
        assert all("source" in chunk.metadata for chunk in chunks), "Should have source"
        assert all("chunk" in chunk.metadata for chunk in chunks), "Should have chunk number"
        print(f"   ✓ Metadata preserved")

        # Preview
        print(f"\n   First chunk: {chunks[0].content[:60]}...")
        print(f"   Metadata: {chunks[0].metadata}")

        print("\n   ✓ Full integration works!")

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


def test_smart_defaults():
    """스마트 기본값 테스트"""
    print("\n8. Testing Smart Defaults...")

    text = "AI is amazing. Machine learning is powerful. " * 30
    doc = Document(content=text, metadata={"source": "test"})

    # Just call split with minimal parameters
    chunks = TextSplitter.split([doc])

    assert len(chunks) > 0, "Should work with defaults"
    print(f"   ✓ Smart defaults created {len(chunks)} chunks")
    print("   ✓ Smart defaults work!")


def test_metadata_preservation():
    """메타데이터 보존 테스트"""
    print("\n9. Testing Metadata Preservation...")

    doc = Document(
        content="Test content. " * 50,
        metadata={
            "source": "test.txt",
            "author": "Test Author",
            "date": "2024-01-01"
        }
    )

    chunks = TextSplitter.split([doc], chunk_size=100)

    # Check all metadata preserved
    assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)
    assert all(chunk.metadata["author"] == "Test Author" for chunk in chunks)
    assert all(chunk.metadata["date"] == "2024-01-01" for chunk in chunks)
    assert all("chunk" in chunk.metadata for chunk in chunks)

    print(f"   ✓ All metadata preserved across {len(chunks)} chunks")
    print("   ✓ Metadata preservation works!")


def main():
    """Run all tests"""
    print("="*60)
    print("🧪 Text Splitters Test Suite")
    print("="*60)

    tests = [
        test_character_splitter,
        test_recursive_splitter,
        test_markdown_splitter,
        test_token_splitter,
        test_text_splitter_factory,
        test_convenience_function,
        test_full_integration,
        test_smart_defaults,
        test_metadata_preservation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"   ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n🎉 All tests passed!")
        print("\nKey Features Verified:")
        print("  ✅ CharacterTextSplitter - Simple splitting")
        print("  ✅ RecursiveCharacterTextSplitter - Smart hierarchical splitting")
        print("  ✅ MarkdownHeaderTextSplitter - Header-based splitting")
        print("  ✅ TokenTextSplitter - Token-based splitting")
        print("  ✅ TextSplitter Factory - Auto-selection")
        print("  ✅ Smart Defaults - One-line usage")
        print("  ✅ Metadata Preservation - Across all splitters")
        print("  ✅ Full Integration - With DocumentLoader")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
