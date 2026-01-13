"""
Test RAG Debug API with Ollama model (free, no API key needed)
"""
import asyncio
import httpx

async def test_rag_debug_with_ollama():
    """Test RAG Debug API with Ollama model"""
    base_url = "http://localhost:8000"
    
    # Test data
    documents = [
        "beanllm은 통합 LLM 관리 도구입니다.",
        "주요 기능: RAG, Agent, Knowledge Graph, Multi-Agent 등",
        "지원 프로바이더: OpenAI, Claude, Gemini, DeepSeek, Ollama",
    ]
    
    print("=" * 60)
    print("Testing RAG Debug API with Ollama")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Test RAG Debug analyze endpoint
        print("\n[1] Testing RAG Debug analyze...")
        debug_response = await client.post(
            f"{base_url}/api/rag_debug/analyze",
            json={
                "query": "beanllm이 무엇인가요?",
                "documents": documents,
                "debug_mode": "full",
                "model": "qwen2.5:0.5b"  # Free Ollama model
            }
        )
        print(f"Status: {debug_response.status_code}")
        
        if debug_response.status_code != 200:
            print(f"Error: {debug_response.text}")
            return
        
        result = debug_response.json()
        print(f"\nQuery: {result.get('query')}")
        print(f"Session ID: {result.get('session_id', 'N/A')}")
        print(f"\nAnalysis:")
        analysis = result.get('analysis', {})
        print(f"  - Embedding Quality: {analysis.get('embedding_quality', 'N/A')}")
        print(f"  - Chunk Quality: {analysis.get('chunk_quality', 'N/A')}")
        print(f"  - Retrieval Quality: {analysis.get('retrieval_quality', 'N/A')}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result.get('recommendations', []), 1):
            print(f"  {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(test_rag_debug_with_ollama())
