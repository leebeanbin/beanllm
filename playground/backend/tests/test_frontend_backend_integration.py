"""
프론트엔드-백엔드 통합 테스트
Ollama 무료 모델만 사용 (qwen2.5:0.5b, phi3, nomic-embed-text)
"""
import requests
import json

BACKEND_URL = "http://localhost:8000"

# Ollama 모델만 사용
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

def test_chat_api():
    """Chat API 테스트 - Ollama qwen2.5:0.5b"""
    print("\n" + "="*60)
    print("1. Chat API 테스트 (Ollama qwen2.5:0.5b)")
    print("="*60)

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "Say 'Hello' in one word"}
            ],
            "model": OLLAMA_CHAT_MODEL,
            "stream": False
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Response: {data.get('response', '')[:100]}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_rag_build():
    """RAG Build API 테스트 - Ollama 모델"""
    print("\n" + "="*60)
    print("2. RAG Build API 테스트 (Ollama)")
    print("="*60)

    response = requests.post(
        f"{BACKEND_URL}/api/rag/build",
        json={
            "documents": [
                "The capital of France is Paris.",
                "Python is a programming language.",
                "Machine learning is a subset of AI."
            ],
            "collection_name": "test_ollama_rag",
            "model": OLLAMA_CHAT_MODEL,
            "embedding_model": OLLAMA_EMBEDDING_MODEL
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Documents indexed: {data.get('num_documents', 0)}")
        print(f"✅ Collection: {data.get('collection_name', '')}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_rag_query():
    """RAG Query API 테스트 - Ollama 모델"""
    print("\n" + "="*60)
    print("3. RAG Query API 테스트 (Ollama)")
    print("="*60)

    response = requests.post(
        f"{BACKEND_URL}/api/rag/query",
        json={
            "query": "What is the capital of France?",
            "collection_name": "test_ollama_rag",
            "model": OLLAMA_CHAT_MODEL,
            "top_k": 3
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Answer: {data.get('answer', '')[:200]}")
        print(f"✅ Sources: {len(data.get('sources', []))} documents")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_agent_api():
    """Agent API 테스트 - Ollama 모델"""
    print("\n" + "="*60)
    print("4. Agent API 테스트 (Ollama)")
    print("="*60)

    response = requests.post(
        f"{BACKEND_URL}/api/agent/run",
        json={
            "task": "Calculate 5 + 3",
            "tools": ["calculator"],
            "model": OLLAMA_CHAT_MODEL,
            "max_iterations": 5
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Result: {data.get('result', '')[:200]}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_web_search():
    """Web Search API 테스트 - DuckDuckGo (무료)"""
    print("\n" + "="*60)
    print("5. Web Search API 테스트 (DuckDuckGo)")
    print("="*60)

    response = requests.post(
        f"{BACKEND_URL}/api/web/search",
        json={
            "query": "Python programming",
            "engine": "duckduckgo",
            "max_results": 3
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Results found: {len(data.get('results', []))}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def main():
    """모든 통합 테스트 실행"""
    print("="*60)
    print("프론트엔드-백엔드 통합 테스트 시작")
    print("무료 Ollama 모델 사용")
    print("="*60)

    results = []

    # 각 API 테스트
    results.append(("Chat API", test_chat_api()))
    results.append(("RAG Build API", test_rag_build()))
    results.append(("RAG Query API", test_rag_query()))
    results.append(("Agent API", test_agent_api()))
    results.append(("Web Search API", test_web_search()))

    # 결과 요약
    print("\n" + "="*60)
    print("통합 테스트 결과 요약")
    print("="*60)

    success = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("="*60)
    print(f"총 {success}/{total} 테스트 통과")
    print("="*60)

if __name__ == "__main__":
    main()
