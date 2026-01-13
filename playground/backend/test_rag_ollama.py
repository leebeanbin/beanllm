"""
Test RAG with Ollama model (free, no API key needed)
"""
import asyncio
import httpx

async def test_rag_with_ollama():
    """Test RAG build and query with Ollama model"""
    base_url = "http://localhost:8000"
    
    # Test data
    documents = [
        "beanllm\uc740 \ud1b5\ud569 LLM \uad00\ub9ac \ub3c4\uad6c\uc785\ub2c8\ub2e4.",
        "\uc8fc\uc694 \uae30\ub2a5: RAG, Agent, Knowledge Graph, Multi-Agent \ub4f1",
        "\uc9c0\uc6d0 \ud504\ub85c\ubc14\uc774\ub354: OpenAI, Claude, Gemini, DeepSeek, Ollama",
    ]
    
    print("=" * 60)
    print("Testing RAG with Ollama (qwen2.5:0.5b)")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Build RAG index with Ollama model
        print("\n[1] Building RAG index...")
        build_response = await client.post(
            f"{base_url}/api/rag/build",
            json={
                "documents": documents,
                "collection_name": "test_ollama",
                "model": "qwen2.5:0.5b"  # Free Ollama model
            }
        )
        print(f"Status: {build_response.status_code}")
        print(f"Response: {build_response.json()}")
        
        if build_response.status_code != 200:
            print(f"\nError: {build_response.text}")
            return
        
        # 2. Query RAG
        print("\n[2] Querying RAG...")
        query_response = await client.post(
            f"{base_url}/api/rag/query",
            json={
                "query": "beanllm\uc774 \ubb34\uc5c7\uc778\uac00\uc694?",
                "collection_name": "test_ollama",
                "top_k": 3
            }
        )
        print(f"Status: {query_response.status_code}")
        result = query_response.json()
        if query_response.status_code != 200:
            print(f"Error: {result.get('detail', result)}")
        else:
            print(f"\nQuery: {result.get('query')}")
            print(f"Answer: {result.get('answer')}")
            print(f"\nSources ({len(result.get('sources', []))}):")
            for i, src in enumerate(result.get('sources', []), 1):
                print(f"  {i}. {src.get('content', '')[:100]}")

if __name__ == "__main__":
    asyncio.run(test_rag_with_ollama())
