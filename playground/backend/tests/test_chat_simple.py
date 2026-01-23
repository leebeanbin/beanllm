"""
Simple test for Chat API with Ollama
"""
import asyncio
import httpx

async def test_chat():
    """Test Chat API"""
    base_url = "http://localhost:8000"

    print("Testing Chat API with Ollama...")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/api/chat",
            json={
                "messages": [{"role": "user", "content": "Hello! What is 2+2?"}],
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('content', 'No content')[:200]}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_chat())
