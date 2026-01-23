"""
beanllm 문서 로더 테스트
지원 형식: Text, CSV, HTML, Markdown, PDF
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("/Users/leejungbin/Downloads/llmkit/src"))

from beanllm.domain.loaders.factory import DocumentLoader


def test_text_loader():
    """Text 파일 로더 테스트"""
    print("\n" + "="*60)
    print("1. Text Loader 테스트")
    print("="*60)

    try:
        documents = DocumentLoader.load("test_documents/sample.txt", loader_type="text")

        print(f"✅ 로드 성공")
        print(f"   문서 수: {len(documents)}")
        if documents:
            print(f"   첫 번째 문서 길이: {len(documents[0].content)} 글자")
            print(f"   미리보기: {documents[0].content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_csv_loader():
    """CSV 파일 로더 테스트"""
    print("\n" + "="*60)
    print("2. CSV Loader 테스트")
    print("="*60)

    try:
        documents = DocumentLoader.load("test_documents/sample.csv", loader_type="csv")

        print(f"✅ 로드 성공")
        print(f"   문서 수: {len(documents)}")
        if documents:
            print(f"   첫 번째 행 내용: {documents[0].content[:150]}...")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_html_loader():
    """HTML 파일 로더 테스트"""
    print("\n" + "="*60)
    print("3. HTML Loader 테스트")
    print("="*60)

    try:
        # HTML은 text loader로 처리될 수 있음 (beanllm에 html loader가 없으면)
        documents = DocumentLoader.load("test_documents/sample.html")

        print(f"✅ 로드 성공")
        print(f"   문서 수: {len(documents)}")
        if documents:
            print(f"   텍스트 길이: {len(documents[0].content)} 글자")
            print(f"   미리보기: {documents[0].content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_loader():
    """Markdown 파일 로더 테스트"""
    print("\n" + "="*60)
    print("4. Markdown Loader 테스트")
    print("="*60)

    try:
        # Markdown은 text loader로 처리될 수 있음
        documents = DocumentLoader.load("test_documents/sample.md", loader_type="markdown")

        print(f"✅ 로드 성공")
        print(f"   문서 수: {len(documents)}")
        if documents:
            print(f"   텍스트 길이: {len(documents[0].content)} 글자")
            print(f"   미리보기: {documents[0].content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_pdf_loader():
    """PDF 파일 로더 테스트 (샘플 PDF가 있는 경우)"""
    print("\n" + "="*60)
    print("5. PDF Loader 테스트")
    print("="*60)

    # PDF 파일이 없으면 스킵
    if not os.path.exists("test_documents/sample.pdf"):
        print("⚠️  스킵: sample.pdf 파일이 없습니다")
        return None

    try:
        documents = DocumentLoader.load("test_documents/sample.pdf", loader_type="pdf")

        print(f"✅ 로드 성공")
        print(f"   문서 수: {len(documents)}")
        if documents:
            print(f"   첫 페이지 길이: {len(documents[0].content)} 글자")
            print(f"   미리보기: {documents[0].content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_loader():
    """디렉토리 전체 로더 테스트"""
    print("\n" + "="*60)
    print("6. Directory Loader 테스트")
    print("="*60)

    try:
        documents = DocumentLoader.load("test_documents/", loader_type="directory")

        print(f"✅ 로드 성공")
        print(f"   총 문서 수: {len(documents)}")

        # 파일 타입별 카운트
        file_types = {}
        for doc in documents:
            file_path = doc.metadata.get("source", "unknown")
            ext = os.path.splitext(file_path)[1]
            file_types[ext] = file_types.get(ext, 0) + 1

        print(f"   파일 타입별 분포:")
        for ext, count in file_types.items():
            print(f"     - {ext or '(확장자 없음)'}: {count}개")

        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 문서 로더 테스트 실행"""
    print("="*60)
    print("beanllm 문서 로더 테스트")
    print("="*60)

    results = []

    # 각 로더 테스트
    results.append(("Text Loader", test_text_loader()))
    results.append(("CSV Loader", test_csv_loader()))
    results.append(("HTML Loader", test_html_loader()))
    results.append(("Markdown Loader", test_markdown_loader()))
    pdf_result = test_pdf_loader()
    if pdf_result is not None:
        results.append(("PDF Loader", pdf_result))
    results.append(("Directory Loader", test_directory_loader()))

    # 결과 요약
    print("\n" + "="*60)
    print("문서 로더 테스트 결과 요약")
    print("="*60)

    success = sum(1 for _, result in results if result is True)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)

    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP"
        else:
            status = "❌ FAIL"
        print(f"{status} - {name}")

    print("="*60)
    print(f"총 {success}/{total} 테스트 통과 ({skipped}개 스킵)")
    print("="*60)


if __name__ == "__main__":
    main()
