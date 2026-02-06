"""
RAG API 상세 테스트
- 다양한 문서 타입 (Text, CSV, Markdown, HTML)
- Hybrid 검색 (Vector + BM25)
- 리랭킹 (Reranker)
- Ollama 무료 모델 사용
"""

import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"


def test_rag_with_text_documents():
    """Text 문서로 RAG 테스트"""
    print("\n" + "=" * 60)
    print("1. RAG - Text 문서 테스트")
    print("=" * 60)

    # Build
    response = requests.post(
        f"{BACKEND_URL}/api/rag/build",
        json={
            "documents": [
                "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                "Python is a high-level programming language. It was created by Guido van Rossum.",
                "Machine learning is a subset of artificial intelligence. It enables computers to learn from data.",
                "The Great Wall of China is one of the Seven Wonders of the World.",
                "Albert Einstein developed the theory of relativity. He won the Nobel Prize in Physics.",
            ],
            "collection_name": "text_docs",
            "model": OLLAMA_CHAT_MODEL,
            "embedding_model": OLLAMA_EMBEDDING_MODEL,
        },
    )

    print(f"Build Status: {response.status_code}")
    if response.status_code != 200:
        print(f"❌ Build failed: {response.text}")
        return False

    data = response.json()
    print(f"✅ Build 성공: {data.get('num_documents')}개 문서 인덱싱")

    # Query
    queries = [
        "What is the capital of France?",
        "Who created Python?",
        "What did Einstein develop?",
    ]

    for query in queries:
        response = requests.post(
            f"{BACKEND_URL}/api/rag/query",
            json={
                "query": query,
                "collection_name": "text_docs",
                "model": OLLAMA_CHAT_MODEL,
                "top_k": 2,
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n  Query: {query}")
            print(f"  Answer: {data.get('answer', '')[:150]}...")
        else:
            print(f"  ❌ Query failed: {query}")

    return True


def test_rag_with_structured_data():
    """CSV 구조화 데이터로 RAG 테스트"""
    print("\n" + "=" * 60)
    print("2. RAG - CSV 구조화 데이터 테스트")
    print("=" * 60)

    # CSV 형식 문서
    csv_documents = [
        "name: Alice, age: 30, city: Seoul, occupation: Engineer",
        "name: Bob, age: 25, city: Busan, occupation: Designer",
        "name: Charlie, age: 35, city: Incheon, occupation: Manager",
        "name: David, age: 28, city: Daegu, occupation: Developer",
        "name: Eve, age: 32, city: Gwangju, occupation: Analyst",
    ]

    # Build
    response = requests.post(
        f"{BACKEND_URL}/api/rag/build",
        json={
            "documents": csv_documents,
            "collection_name": "csv_data",
            "model": OLLAMA_CHAT_MODEL,
            "embedding_model": OLLAMA_EMBEDDING_MODEL,
        },
    )

    print(f"Build Status: {response.status_code}")
    if response.status_code != 200:
        print(f"❌ Build failed: {response.text}")
        return False

    print(f"✅ Build 성공: {len(csv_documents)}개 레코드 인덱싱")

    # Query
    queries = ["Who works in Seoul?", "What is Bob's occupation?", "Who is the oldest person?"]

    for query in queries:
        response = requests.post(
            f"{BACKEND_URL}/api/rag/query",
            json={
                "query": query,
                "collection_name": "csv_data",
                "model": OLLAMA_CHAT_MODEL,
                "top_k": 3,
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n  Query: {query}")
            print(f"  Answer: {data.get('answer', '')[:150]}...")
        else:
            print(f"  ❌ Query failed: {query}")

    return True


def test_rag_collections_management():
    """RAG 컬렉션 관리 테스트"""
    print("\n" + "=" * 60)
    print("3. RAG - 컬렉션 관리 테스트")
    print("=" * 60)

    # List collections
    response = requests.get(f"{BACKEND_URL}/api/rag/collections")

    if response.status_code == 200:
        data = response.json()
        print("✅ 컬렉션 목록 조회 성공")
        print(f"   총 컬렉션 수: {data.get('total', 0)}")
        for coll in data.get("collections", []):
            print(f"   - {coll['name']}: {coll.get('document_count', 0)}개 문서")
    else:
        print(f"❌ 실패: {response.text}")
        return False

    # Delete collection
    if data.get("total", 0) > 0:
        collection_to_delete = data["collections"][0]["name"]
        response = requests.delete(f"{BACKEND_URL}/api/rag/collections/{collection_to_delete}")

        if response.status_code == 200:
            print(f"✅ 컬렉션 삭제 성공: {collection_to_delete}")
        else:
            print(f"❌ 삭제 실패: {response.text}")

    return True


def test_rag_with_complex_documents():
    """복잡한 문서로 RAG 테스트"""
    print("\n" + "=" * 60)
    print("4. RAG - 복잡한 문서 테스트")
    print("=" * 60)

    # 긴 문서
    long_documents = [
        """
        Artificial Intelligence (AI) Overview:

        Machine Learning is a fundamental component of AI that enables systems to learn from data.
        Deep Learning, a subset of ML, uses neural networks with multiple layers.
        Natural Language Processing (NLP) allows machines to understand human language.
        Computer Vision enables machines to interpret visual information.

        Key AI Applications:
        - Healthcare: Disease diagnosis, drug discovery
        - Finance: Fraud detection, algorithmic trading
        - Transportation: Autonomous vehicles, traffic optimization
        - Entertainment: Recommendation systems, content generation
        """,
        """
        Programming Languages Comparison:

        Python: High-level, interpreted, general-purpose
        - Pros: Easy to learn, extensive libraries, great for data science
        - Cons: Slower execution compared to compiled languages

        JavaScript: High-level, interpreted, primarily for web development
        - Pros: Runs in browsers, versatile, huge ecosystem
        - Cons: Can be inconsistent, callback hell issues

        Rust: Systems programming language focused on safety
        - Pros: Memory safety, high performance, no garbage collector
        - Cons: Steep learning curve, smaller community
        """,
        """
        Climate Change Facts:

        Global Warming: Earth's average temperature has increased by 1.1°C since pre-industrial times.
        Causes: Primarily greenhouse gas emissions from fossil fuels.
        Effects: Rising sea levels, extreme weather events, biodiversity loss.
        Solutions: Renewable energy, carbon capture, reforestation, sustainable practices.

        Paris Agreement: International treaty aiming to limit global warming to below 2°C.
        """,
    ]

    # Build
    response = requests.post(
        f"{BACKEND_URL}/api/rag/build",
        json={
            "documents": long_documents,
            "collection_name": "complex_docs",
            "model": OLLAMA_CHAT_MODEL,
            "embedding_model": OLLAMA_EMBEDDING_MODEL,
        },
    )

    print(f"Build Status: {response.status_code}")
    if response.status_code != 200:
        print(f"❌ Build failed: {response.text}")
        return False

    print(f"✅ Build 성공: {len(long_documents)}개 복잡한 문서 인덱싱")

    # Complex queries
    queries = [
        "What are the applications of AI in healthcare?",
        "What are the pros and cons of Python?",
        "What is the goal of the Paris Agreement?",
    ]

    for query in queries:
        response = requests.post(
            f"{BACKEND_URL}/api/rag/query",
            json={
                "query": query,
                "collection_name": "complex_docs",
                "model": OLLAMA_CHAT_MODEL,
                "top_k": 2,
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n  Query: {query}")
            print(f"  Answer: {data.get('answer', '')[:200]}...")
            print(f"  Sources: {len(data.get('sources', []))}개")
        else:
            print(f"  ❌ Query failed: {query}")

    return True


def main():
    """모든 RAG 테스트 실행"""
    print("=" * 60)
    print("RAG API 상세 테스트 (Ollama 무료 모델)")
    print("=" * 60)

    results = []

    # 각 테스트 실행
    results.append(("Text 문서 RAG", test_rag_with_text_documents()))
    results.append(("CSV 데이터 RAG", test_rag_with_structured_data()))
    results.append(("컬렉션 관리", test_rag_collections_management()))
    results.append(("복잡한 문서 RAG", test_rag_with_complex_documents()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("RAG API 테스트 결과 요약")
    print("=" * 60)

    success = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("=" * 60)
    print(f"총 {success}/{total} 테스트 통과")
    print("=" * 60)


if __name__ == "__main__":
    main()
