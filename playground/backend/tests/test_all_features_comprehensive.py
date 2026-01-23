"""
beanllm Playground 전체 기능 종합 테스트
- 모든 문서 타입 로더
- 모든 API 엔드포인트
- 샘플 파일 활용
"""
import requests
import json
import base64
import io
from pathlib import Path

BACKEND_URL = "http://localhost:8000"
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"
SAMPLE_DIR = Path("test_documents")


def encode_file_to_base64(file_path):
    """파일을 base64로 인코딩"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_document_loaders():
    """문서 로더 테스트 (직접 beanllm 사용)"""
    print("\n" + "="*60)
    print("1. Document Loaders 테스트")
    print("="*60)

    import sys
    sys.path.insert(0, "/Users/leejungbin/Downloads/llmkit/src")
    from beanllm.domain.loaders.factory import DocumentLoader

    results = []
    loaders = [
        ("Text", "sample.txt", "text"),
        ("CSV", "sample.csv", "csv"),
        ("HTML", "sample.html", None),
        ("Markdown", "sample.md", "markdown"),
        ("JSON", "sample.json", None),
        ("PDF", "sample.pdf", "pdf"),
    ]

    for name, filename, loader_type in loaders:
        file_path = SAMPLE_DIR / filename
        if not file_path.exists():
            print(f"⚠️  {name}: {filename} 파일 없음")
            results.append((name, None))
            continue

        try:
            if loader_type:
                docs = DocumentLoader.load(str(file_path), loader_type=loader_type)
            else:
                docs = DocumentLoader.load(str(file_path))

            print(f"✅ {name}: {len(docs)}개 문서 로드, {len(docs[0].content) if docs else 0}자")
            results.append((name, True))
        except Exception as e:
            print(f"❌ {name}: {str(e)[:100]}")
            results.append((name, False))

    return results


def test_rag_with_files():
    """RAG API - 파일 업로드 테스트"""
    print("\n" + "="*60)
    print("2. RAG with File Upload 테스트")
    print("="*60)

    try:
        # 파일 업로드
        files = []
        for ext in ["txt", "csv", "md"]:
            file_path = SAMPLE_DIR / f"sample.{ext}"
            if file_path.exists():
                files.append(
                    ('files', (file_path.name, open(file_path, 'rb'), 'application/octet-stream'))
                )

        data = {
            'collection_name': 'test_files',
            'model': OLLAMA_CHAT_MODEL,
            'embedding_model': 'nomic-embed-text'  # Ollama embedding
        }

        build_response = requests.post(
            f"{BACKEND_URL}/api/rag/build_from_files",
            files=files,
            data=data
        )

        # Close files
        for _, (_, f, _) in files:
            f.close()

        if build_response.status_code != 200:
            print(f"⚠️  Build 실패: {build_response.text[:200]}")
            return False

        print(f"✅ RAG Build 성공: {build_response.json()['num_documents']}개 문서")

        # Query
        query_response = requests.post(
            f"{BACKEND_URL}/api/rag/query",
            json={
                "query": "What is this about?",
                "collection_name": "test_files",
                "model": OLLAMA_CHAT_MODEL
            }
        )

        if query_response.status_code == 200:
            print(f"✅ RAG Query 성공")
            return True
        else:
            print(f"❌ Query 실패: {query_response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ 실패: {str(e)[:200]}")
        return False


def test_vision_rag_with_images():
    """Vision RAG - 실제 이미지 파일 테스트"""
    print("\n" + "="*60)
    print("3. Vision RAG with Images 테스트")
    print("="*60)

    try:
        # 이미지를 base64로 인코딩
        images = []
        for ext in ["png", "jpg"]:
            img_path = SAMPLE_DIR / f"sample.{ext}"
            if img_path.exists():
                img_base64 = encode_file_to_base64(img_path)
                images.append(f"data:image/{ext};base64,{img_base64}")

        if not images:
            print("⚠️  이미지 파일 없음")
            return None

        # Build
        build_response = requests.post(
            f"{BACKEND_URL}/api/vision_rag/build",
            json={
                "images": images,
                "texts": ["Sample image for testing", "Another test image"],
                "collection_name": "test_vision_images",
                "model": OLLAMA_CHAT_MODEL,
                "generate_captions": False
            }
        )

        if build_response.status_code != 200:
            print(f"⚠️  Build 실패: {build_response.text[:200]}")
            return False

        print(f"✅ Vision RAG Build 성공: {build_response.json()['num_images']}개 이미지")

        # Query
        query_response = requests.post(
            f"{BACKEND_URL}/api/vision_rag/query",
            json={
                "query": "What do you see in these images?",
                "collection_name": "test_vision_images"
            }
        )

        if query_response.status_code == 200:
            print(f"✅ Vision RAG Query 성공")
            return True
        else:
            print(f"❌ Query 실패: {query_response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ 실패: {str(e)[:200]}")
        return False


def test_ocr_with_image():
    """OCR - 실제 이미지로 텍스트 인식"""
    print("\n" + "="*60)
    print("4. OCR with Real Image 테스트")
    print("="*60)

    try:
        img_path = SAMPLE_DIR / "sample.png"
        if not img_path.exists():
            print("⚠️  이미지 파일 없음")
            return None

        with open(img_path, 'rb') as f:
            files = {'file': (img_path.name, f, 'image/png')}
            data = {'language': 'eng', 'engine': 'paddleocr'}

            response = requests.post(
                f"{BACKEND_URL}/api/ocr/recognize",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ OCR 성공: {result.get('text', '')[:100]}")
            return True
        else:
            print(f"⚠️  실패 (PaddleOCR 필요): {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ 실패: {str(e)[:200]}")
        return False


def test_knowledge_graph_with_docs():
    """Knowledge Graph - 문서로 그래프 구축"""
    print("\n" + "="*60)
    print("5. Knowledge Graph with Documents 테스트")
    print("="*60)

    try:
        # 텍스트 문서 읽기
        txt_path = SAMPLE_DIR / "sample.txt"
        if not txt_path.exists():
            print("⚠️  텍스트 파일 없음")
            return None

        with open(txt_path, 'r') as f:
            text = f.read()

        # Build
        build_response = requests.post(
            f"{BACKEND_URL}/api/kg/build",
            json={
                "documents": [text],
                "graph_id": "test_kg_docs",
                "model": OLLAMA_CHAT_MODEL
            }
        )

        if build_response.status_code != 200:
            print(f"⚠️  Build 실패: {build_response.text[:200]}")
            return False

        result = build_response.json()
        print(f"✅ KG Build 성공: {result.get('num_entities', 0)} entities, {result.get('num_relations', 0)} relations")

        # Query
        query_response = requests.post(
            f"{BACKEND_URL}/api/kg/query",
            json={
                "graph_id": "test_kg_docs",
                "query": "MATCH (n) RETURN n LIMIT 5"
            }
        )

        if query_response.status_code == 200:
            print(f"✅ KG Query 성공")
            return True
        else:
            print(f"❌ Query 실패: {query_response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ 실패: {str(e)[:200]}")
        return False


def test_pdf_specific():
    """PDF 전용 테스트"""
    print("\n" + "="*60)
    print("6. PDF Specific 테스트")
    print("="*60)

    import sys
    sys.path.insert(0, "/Users/leejungbin/Downloads/llmkit/src")

    try:
        from beanllm.domain.loaders import PDFLoader

        pdf_path = SAMPLE_DIR / "sample.pdf"
        if not pdf_path.exists():
            print("⚠️  PDF 파일 없음")
            return None

        loader = PDFLoader(str(pdf_path))
        docs = loader.load()

        print(f"✅ PDF 로드 성공: {len(docs)}페이지")
        for i, doc in enumerate(docs, 1):
            print(f"   Page {i}: {len(doc.content)}자")

        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)[:200]}")
        return False


def main():
    """전체 종합 테스트 실행"""
    print("="*60)
    print("beanllm Playground 전체 기능 종합 테스트")
    print("="*60)

    all_results = []

    # 1. Document Loaders
    loader_results = test_document_loaders()
    all_results.extend(loader_results)

    # 2. RAG with Files
    all_results.append(("RAG File Upload", test_rag_with_files()))

    # 3. Vision RAG
    all_results.append(("Vision RAG", test_vision_rag_with_images()))

    # 4. OCR
    all_results.append(("OCR", test_ocr_with_image()))

    # 5. Knowledge Graph
    all_results.append(("Knowledge Graph", test_knowledge_graph_with_docs()))

    # 6. PDF Specific
    all_results.append(("PDF Loader", test_pdf_specific()))

    # 결과 요약
    print("\n" + "="*60)
    print("전체 테스트 결과 요약")
    print("="*60)

    success = sum(1 for _, result in all_results if result is True)
    failed = sum(1 for _, result in all_results if result is False)
    skipped = sum(1 for _, result in all_results if result is None)
    total = len(all_results)

    for name, result in all_results:
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP"
        else:
            status = "❌ FAIL"
        print(f"{status} - {name}")

    print("="*60)
    print(f"총 {success}/{total} 성공, {failed} 실패, {skipped} 스킵")
    print("="*60)


if __name__ == "__main__":
    main()
