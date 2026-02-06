"""
Simple Agent API test
"""

import asyncio

import httpx


async def test():
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/api/agent/run",
            json={
                "task": "What is 10 + 5?",
                "tools": ["calculator"],
                "model": "qwen2.5:0.5b",
                "max_iterations": 3,
            },
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")


if __name__ == "__main__":
    asyncio.run(test())
