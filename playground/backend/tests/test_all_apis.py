"""
Test all major API endpoints with Ollama
"""

import asyncio

import httpx

BASE_URL = "http://localhost:8000"


async def test_agent():
    """Test Agent API"""
    print("\n" + "=" * 60)
    print("Testing Agent API")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/agent/run",
            json={
                "task": "What is the current time?",
                "tools": ["get_current_time"],
                "model": "qwen2.5:0.5b",
                "max_iterations": 3,
            },
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {result.get('result', 'No result')[:200]}")
            print(f"Iterations: {result.get('iterations', 0)}")
        else:
            print(f"Error: {response.text[:200]}")


async def test_web_search():
    """Test Web Search API"""
    print("\n" + "=" * 60)
    print("Testing Web Search API")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/web/search",
            json={"query": "beanllm python framework", "max_results": 3, "summarize": False},
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result.get('results', []))} results")
            if result.get("results"):
                print(f"First result: {result['results'][0].get('title', 'No title')}")
        else:
            print(f"Error: {response.text[:200]}")


async def test_multi_agent():
    """Test Multi-Agent API"""
    print("\n" + "=" * 60)
    print("Testing Multi-Agent API")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/multi_agent/run",
            json={
                "task": "Calculate 5 + 3 and then multiply the result by 2",
                "agents": [
                    {"name": "calculator1", "role": "Calculate 5 + 3"},
                    {"name": "calculator2", "role": "Multiply result by 2"},
                ],
                "mode": "sequential",
                "model": "qwen2.5:0.5b",
            },
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Final result: {result.get('final_result', 'No result')[:200]}")
        else:
            print(f"Error: {response.text[:200]}")


async def test_knowledge_graph():
    """Test Knowledge Graph API"""
    print("\n" + "=" * 60)
    print("Testing Knowledge Graph API (Build)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Build graph
        build_response = await client.post(
            f"{BASE_URL}/api/kg/build",
            json={
                "documents": [
                    "Alice works at Google in San Francisco.",
                    "Bob is a friend of Alice and lives in New York.",
                    "Google was founded by Larry Page and Sergey Brin.",
                ],
                "graph_id": "test_graph",
                "model": "qwen2.5:0.5b",
            },
        )
        print(f"Build Status: {build_response.status_code}")
        if build_response.status_code == 200:
            result = build_response.json()
            print(f"Entities: {result.get('num_entities', 0)}")
            print(f"Relations: {result.get('num_relations', 0)}")
        else:
            print(f"Error: {build_response.text[:200]}")


async def main():
    """Run all tests"""
    print("Starting API tests with Ollama (qwen2.5:0.5b)")

    await test_agent()
    await test_web_search()
    await test_multi_agent()
    await test_knowledge_graph()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
