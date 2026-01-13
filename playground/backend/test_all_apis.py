"""
Test all API endpoints to ensure they work without errors
"""
import asyncio
import httpx

async def test_all_apis():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Testing All API Endpoints")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Health Check
        print("\n[1] Health Check...")
        try:
            health_response = await client.get(f"{base_url}/health")
            print(f"  Status: {health_response.status_code}")
            if health_response.status_code == 200:
                print("  ✅ Health check passed")
            else:
                print(f"  ❌ Health check failed: {health_response.text}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 2. Chat API
        print("\n[2] Chat API...")
        try:
            chat_response = await client.post(
                f"{base_url}/api/chat",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "model": "qwen2.5:0.5b"
                }
            )
            print(f"  Status: {chat_response.status_code}")
            if chat_response.status_code == 200:
                print("  ✅ Chat API works")
            else:
                print(f"  ❌ Chat API failed: {chat_response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 3. RAG Build
        print("\n[3] RAG Build API...")
        try:
            build_response = await client.post(
                f"{base_url}/api/rag/build",
                json={
                    "documents": ["Test document"],
                    "model": "qwen2.5:0.5b"
                }
            )
            print(f"  Status: {build_response.status_code}")
            if build_response.status_code == 200:
                print("  ✅ RAG Build works")
            else:
                print(f"  ❌ RAG Build failed: {build_response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 4. RAG Query (if build succeeded)
        print("\n[4] RAG Query API...")
        try:
            query_response = await client.post(
                f"{base_url}/api/rag/query",
                json={
                    "query": "test",
                    "collection_name": "default"
                }
            )
            print(f"  Status: {query_response.status_code}")
            if query_response.status_code == 200:
                print("  ✅ RAG Query works")
            else:
                print(f"  ⚠️  RAG Query: {query_response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 5. RAG Debug API
        print("\n[5] RAG Debug API...")
        try:
            debug_response = await client.post(
                f"{base_url}/api/rag_debug/analyze",
                json={
                    "query": "test",
                    "documents": ["Test document"],
                    "model": "qwen2.5:0.5b"
                }
            )
            print(f"  Status: {debug_response.status_code}")
            if debug_response.status_code == 200:
                print("  ✅ RAG Debug API works")
            else:
                print(f"  ❌ RAG Debug failed: {debug_response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 6. Multi-Agent API (Sequential)
        print("\n[6] Multi-Agent API (Sequential)...")
        try:
            multi_response = await client.post(
                f"{base_url}/api/multi_agent/run",
                json={
                    "task": "What is 2+2?",
                    "num_agents": 2,
                    "strategy": "sequential",
                    "model": "qwen2.5:0.5b"
                }
            )
            print(f"  Status: {multi_response.status_code}")
            if multi_response.status_code == 200:
                print("  ✅ Multi-Agent API works")
            else:
                print(f"  ❌ Multi-Agent failed: {multi_response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 7. Agent API
        print("\n[7] Agent API...")
        try:
            agent_response = await client.post(
                f"{base_url}/api/agent/run",
                json={
                    "task": "What is 2+2?",
                    "model": "qwen2.5:0.5b"
                }
            )
            print(f"  Status: {agent_response.status_code}")
            if agent_response.status_code == 200:
                print("  ✅ Agent API works")
            else:
                print(f"  ❌ Agent API failed: {agent_response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("All API tests completed!")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all_apis())
