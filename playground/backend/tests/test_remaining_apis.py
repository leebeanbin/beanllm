"""
Test remaining APIs: Web Search, Multi-Agent, Knowledge Graph
"""

import asyncio

import httpx

BASE_URL = "http://localhost:8000"


async def test_web_search():
    print("\n" + "=" * 60)
    print("Testing Web Search API")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/web/search",
            json={"query": "Python programming", "num_results": 3, "summarize": False},
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {len(result.get('results', []))} results")
        else:
            print(f"❌ Error: {response.text[:200]}")


async def test_multi_agent():
    print("\n" + "=" * 60)
    print("Testing Multi-Agent API")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/multi_agent/run",
            json={
                "task": "Calculate 5 + 3",
                "agents": [{"name": "calculator", "role": "Do math"}],
                "mode": "sequential",
                "model": "qwen2.5:0.5b",
            },
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Result: {result.get('final_result', 'No result')[:100]}")
        else:
            print(f"❌ Error: {response.text[:200]}")


async def test_kg():
    print("\n" + "=" * 60)
    print("Testing Knowledge Graph API")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/kg/build",
            json={
                "documents": ["Alice works at Google.", "Bob is friends with Alice."],
                "graph_id": "test",
                "model": "qwen2.5:0.5b",
            },
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(
                f"✅ Entities: {result.get('num_entities', 0)}, Relations: {result.get('num_relations', 0)}"
            )
        else:
            print(f"❌ Error: {response.text[:200]}")


async def main():
    await test_web_search()
    await test_multi_agent()
    await test_kg()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
