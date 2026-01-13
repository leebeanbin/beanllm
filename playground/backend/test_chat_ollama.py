"""
Test Chat API with Ollama model (free, no API key needed)
"""
import asyncio
import httpx

async def test_chat_with_ollama():
    """Test Chat API with Ollama model"""
    base_url = "http://localhost:8000"

    print("=" * 60)
    print("Testing Chat API with Ollama (qwen2.5:0.5b)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test chat endpoint
        print("\n[1] Testing chat endpoint...")
        chat_response = await client.post(
            f"{base_url}/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "안녕하세요! beanllm이 무엇인지 간단히 설명해주세요."}
                ],
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"Status: {chat_response.status_code}")

        if chat_response.status_code != 200:
            print(f"Error: {chat_response.text}")
            return

        result = chat_response.json()
        print(f"Response: {result.get('content', result)[:200]}...")
        print(f"Model: {result.get('model', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(test_chat_with_ollama())
