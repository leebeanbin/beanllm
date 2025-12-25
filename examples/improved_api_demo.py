"""
개선된 API 데모 - 사용자가 쉽게 설정하고 조정할 수 있는 방법
"""
from pathlib import Path
from beanllm import DocumentLoader, TextSplitter, Document


def demo_loader_type_selection():
    """DocumentLoader - 명시적 타입 지정"""
    print("\n" + "="*60)
    print("📂 DocumentLoader - 타입 지정 데모")
    print("="*60)

    # 테스트 파일 생성
    test_txt = Path("test.txt")
    test_txt.write_text("Text file content", encoding="utf-8")

    test_csv = Path("test.csv")
    test_csv.write_text("name,value\nAlice,100\nBob,200", encoding="utf-8")

    try:
        # 방법 1: 자동 감지 (기본)
        print("\n1. 자동 감지 (기본):")
        docs = DocumentLoader.load("test.txt")
        print(f"   ✓ 자동 감지: {len(docs)} 문서 로딩")

        # 방법 2: 명시적 타입 지정
        print("\n2. 명시적 타입 지정:")
        docs_text = DocumentLoader.load("test.txt", loader_type="text")
        print(f"   ✓ loader_type='text': {len(docs_text)} 문서")

        docs_csv = DocumentLoader.load("test.csv", loader_type="csv")
        print(f"   ✓ loader_type='csv': {len(docs_csv)} 문서")

        # 방법 3: 타입 지정 + 추가 파라미터
        print("\n3. 타입 + 파라미터:")
        docs_custom = DocumentLoader.load(
            "test.csv",
            loader_type="csv",
            content_columns=["name"]
        )
        print(f"   ✓ CSV 특정 컬럼만: {docs_custom[0].content}")

        print("\n✓ DocumentLoader: 자동 감지 + 명시적 선택 둘 다 가능!")

    finally:
        # 정리
        test_txt.unlink()
        test_csv.unlink()


def demo_splitter_strategies():
    """TextSplitter - 쉬운 전략 선택"""
    print("\n" + "="*60)
    print("✂️  TextSplitter - 전략 선택 데모")
    print("="*60)

    # 테스트 문서
    text = """
# AI Overview

Artificial Intelligence is transforming the world.

## Machine Learning

Machine learning algorithms learn from data.

## Deep Learning

Deep learning uses neural networks.
    """.strip()

    docs = [Document(content=text, metadata={"source": "test.md"})]

    # 방법 1: 가장 간단 (자동)
    print("\n1. 가장 간단 (자동 최적화):")
    chunks = TextSplitter.split(docs, chunk_size=100)
    print(f"   ✓ 자동: {len(chunks)} 청크")

    # 방법 2: 전략 팩토리 메서드 (추천!)
    print("\n2. 전략 팩토리 메서드 (쉽고 명확!):")

    # Recursive (권장)
    splitter_rec = TextSplitter.recursive(chunk_size=100)
    chunks_rec = splitter_rec.split_documents(docs)
    print(f"   ✓ TextSplitter.recursive(): {len(chunks_rec)} 청크")

    # Character (단순)
    splitter_char = TextSplitter.character(separator="\n\n")
    chunks_char = splitter_char.split_documents(docs)
    print(f"   ✓ TextSplitter.character(): {len(chunks_char)} 청크")

    # Markdown (헤더 기준)
    splitter_md = TextSplitter.markdown()
    chunks_md = splitter_md.split_documents(docs)
    print(f"   ✓ TextSplitter.markdown(): {len(chunks_md)} 청크")
    print(f"      첫 번째 청크 메타데이터: {chunks_md[0].metadata}")

    # 방법 3: 구분자만 지정 (자동 전략 선택)
    print("\n3. 구분자만 지정 (편리!):")
    chunks_sep = TextSplitter.split(docs, separator="\n\n")
    print(f"   ✓ separator='\\n\\n': {len(chunks_sep)} 청크")

    chunks_seps = TextSplitter.split(docs, separators=["##", "\n\n"])
    print(f"   ✓ separators=['##', '\\n\\n']: {len(chunks_seps)} 청크")

    # 방법 4: 전략 문자열 지정 (기존 방식)
    print("\n4. 전략 문자열 지정 (기존 방식):")
    chunks_str = TextSplitter.split(docs, strategy="recursive", chunk_size=100)
    print(f"   ✓ strategy='recursive': {len(chunks_str)} 청크")

    print("\n✓ TextSplitter: 4가지 방법 모두 사용 가능!")


def demo_advanced_customization():
    """고급 커스터마이징 예제"""
    print("\n" + "="*60)
    print("🔧 고급 커스터마이징 데모")
    print("="*60)

    text = "AI is amazing. " * 50
    docs = [Document(content=text, metadata={"source": "test"})]

    # 1. Recursive with custom separators
    print("\n1. Recursive + 커스텀 구분자:")
    splitter = TextSplitter.recursive(
        chunk_size=100,
        chunk_overlap=20,
        separators=[". ", " "]  # 문장 우선, 그 다음 단어
    )
    chunks = splitter.split_documents(docs)
    print(f"   ✓ {len(chunks)} 청크 생성")
    print(f"   ✓ 첫 번째 청크: {chunks[0].content[:50]}...")

    # 2. Character with custom separator
    print("\n2. Character + 커스텀 구분자:")
    splitter = TextSplitter.character(
        separator=". ",
        chunk_size=80,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(docs)
    print(f"   ✓ {len(chunks)} 청크 생성")

    # 3. Markdown with custom headers
    print("\n3. Markdown + 커스텀 헤더:")
    md_text = """
# Title
Content 1

## Section
Content 2

### Subsection
Content 3
    """.strip()

    md_docs = [Document(content=md_text, metadata={})]

    splitter = TextSplitter.markdown(
        headers_to_split_on=[
            ("#", "Title"),
            ("##", "Section"),
            ("###", "Subsection"),
        ]
    )
    chunks = splitter.split_documents(md_docs)
    print(f"   ✓ {len(chunks)} 청크 (헤더 기준)")
    for i, chunk in enumerate(chunks):
        print(f"      Chunk {i+1}: {chunk.metadata}")

    print("\n✓ 고급 커스터마이징: 세밀한 제어 가능!")


def demo_real_world_usage():
    """실전 사용 예제"""
    print("\n" + "="*60)
    print("🚀 실전 사용 예제")
    print("="*60)

    # 시나리오 1: PDF 문서를 작은 청크로 분할
    print("\n시나리오 1: 문서 로딩 → 분할 (간단!)")

    test_file = Path("document.txt")
    test_file.write_text("""
Introduction to AI

Artificial Intelligence (AI) is revolutionizing technology.
Machine learning is a subset of AI.

Deep Learning

Deep learning uses neural networks with multiple layers.
It powers modern AI applications.

Applications

AI is used in various fields: healthcare, finance, and more.
    """.strip(), encoding="utf-8")

    try:
        # 한 줄씩 간단하게!
        docs = DocumentLoader.load(test_file)

        # 전략을 쉽게 선택
        chunks = TextSplitter.recursive(chunk_size=100).split_documents(docs)

        print(f"   ✓ {len(docs)} 문서 → {len(chunks)} 청크")
        print(f"   ✓ 첫 번째 청크: {chunks[0].content[:50]}...")

    finally:
        test_file.unlink()

    # 시나리오 2: 특정 구분자로 분할
    print("\n시나리오 2: 특정 구분자로 분할 (편리!)")

    log_text = """
[INFO] 2024-01-01: System started
[INFO] 2024-01-01: Processing data
---
[ERROR] 2024-01-02: Connection failed
[INFO] 2024-01-02: Retrying...
---
[INFO] 2024-01-03: Success
    """.strip()

    log_docs = [Document(content=log_text, metadata={"source": "log.txt"})]

    # "---"로 분할 (간단!)
    chunks = TextSplitter.character(separator="---").split_documents(log_docs)
    print(f"   ✓ '---' 구분자로 {len(chunks)} 청크")
    for i, chunk in enumerate(chunks[:2]):
        print(f"      Chunk {i+1}: {chunk.content.strip()[:50]}...")

    # 시나리오 3: 여러 구분자로 계층적 분할
    print("\n시나리오 3: 계층적 분할 (똑똑!)")

    # separators 파라미터로 간단히!
    chunks = TextSplitter.split(
        log_docs,
        separators=["---", "\n", " "],
        chunk_size=80
    )
    print(f"   ✓ 계층적 구분자로 {len(chunks)} 청크")

    print("\n✓ 실전 사용: 간단하고 직관적!")


def demo_comparison():
    """LangChain vs beanllm 비교"""
    print("\n" + "="*60)
    print("📊 LangChain vs beanllm 비교")
    print("="*60)

    print("\n【 LangChain 방식 】(복잡)")
    print("""
    # 1. Import 여러 개
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # 2. Loader 수동 선택
    loader = TextLoader("file.txt")
    docs = loader.load()

    # 3. Splitter 수동 설정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    """)

    print("\n【 beanllm 방식 】(간단!)")
    print("""
    # 1. Import 한 번
    from beanllm import DocumentLoader, TextSplitter

    # 2. 자동 감지 로딩
    docs = DocumentLoader.load("file.txt")

    # 3. 전략 쉽게 선택
    chunks = TextSplitter.recursive().split_documents(docs)

    # 또는 더 간단하게
    chunks = TextSplitter.split(docs)
    """)

    print("\n✅ beanllm: ~10줄 → 2-3줄 (70% 감소!)")
    print("✅ 자동 감지 + 스마트 기본값 + 쉬운 커스터마이징")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("🎯 개선된 API 데모")
    print("="*60)
    print("\nbeanllm의 철학:")
    print("  1. 자동 감지 (80% 케이스)")
    print("  2. 명시적 선택 (세밀한 제어)")
    print("  3. 둘 다 가능!")

    demo_loader_type_selection()
    demo_splitter_strategies()
    demo_advanced_customization()
    demo_real_world_usage()
    demo_comparison()

    print("\n" + "="*60)
    print("🎉 개선 완료!")
    print("="*60)
    print("\n✨ 주요 개선사항:")
    print("  1. DocumentLoader.load(file, loader_type='pdf')")
    print("  2. TextSplitter.recursive(chunk_size=1000)")
    print("  3. TextSplitter.character(separator='\\n\\n')")
    print("  4. TextSplitter.split(docs, separator='---')")
    print("  5. TextSplitter.split(docs, separators=['\\n\\n', '\\n'])")
    print("\n💡 사용자가 원하는 대로 쉽게 설정하고 조정 가능!")


if __name__ == "__main__":
    main()
