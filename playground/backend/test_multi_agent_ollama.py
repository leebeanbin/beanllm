"""
Test Multi-Agent API with Ollama model (free, no API key needed)
"""
import asyncio
import httpx

async def test_multi_agent_with_ollama():
    """Test Multi-Agent API with Ollama model"""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Testing Multi-Agent API with Ollama (qwen2.5:0.5b)")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        # Test 1: Sequential strategy
        print("\n[1] Testing Sequential strategy...")
        sequential_response = await client.post(
            f"{base_url}/api/multi_agent/run",
            json={
                "task": "What is 2+2? Answer in one sentence.",
                "num_agents": 2,
                "strategy": "sequential",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"Status: {sequential_response.status_code}")
        
        if sequential_response.status_code != 200:
            print(f"Error: {sequential_response.text}")
        else:
            result = sequential_response.json()
            print(f"Task: {result.get('task')}")
            print(f"Strategy: {result.get('strategy')}")
            print(f"Final Result: {result.get('final_result', '')[:100]}...")
            print(f"Agent Outputs: {len(result.get('agent_outputs', []))} agents")
        
        # Test 2: Parallel strategy
        print("\n[2] Testing Parallel strategy...")
        parallel_response = await client.post(
            f"{base_url}/api/multi_agent/run",
            json={
                "task": "What is the capital of France?",
                "num_agents": 3,
                "strategy": "parallel",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"Status: {parallel_response.status_code}")
        
        if parallel_response.status_code != 200:
            print(f"Error: {parallel_response.text}")
        else:
            result = parallel_response.json()
            print(f"Task: {result.get('task')}")
            print(f"Strategy: {result.get('strategy')}")
            print(f"Final Result: {result.get('final_result', '')[:100]}...")
            print(f"Agent Outputs: {len(result.get('agent_outputs', []))} agents")
        
        # Test 3: Hierarchical strategy (requires at least 2 agents)
        print("\n[3] Testing Hierarchical strategy...")
        hierarchical_response = await client.post(
            f"{base_url}/api/multi_agent/run",
            json={
                "task": "Explain what Python is in one sentence.",
                "num_agents": 2,
                "strategy": "hierarchical",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"Status: {hierarchical_response.status_code}")
        
        if hierarchical_response.status_code != 200:
            print(f"Error: {hierarchical_response.text}")
        else:
            result = hierarchical_response.json()
            print(f"Task: {result.get('task')}")
            print(f"Strategy: {result.get('strategy')}")
            print(f"Final Result: {result.get('final_result', '')[:100]}...")
            print(f"Agent Outputs: {len(result.get('agent_outputs', []))} agents")
            for output in result.get('agent_outputs', []):
                print(f"  - {output.get('agent_id')} ({output.get('role', 'N/A')}): {output.get('output', '')[:50]}...")
        
        # Test 4: Debate strategy
        print("\n[4] Testing Debate strategy...")
        debate_response = await client.post(
            f"{base_url}/api/multi_agent/run",
            json={
                "task": "Is AI beneficial for humanity?",
                "num_agents": 2,
                "strategy": "debate",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"Status: {debate_response.status_code}")
        
        if debate_response.status_code != 200:
            print(f"Error: {debate_response.text}")
        else:
            result = debate_response.json()
            print(f"Task: {result.get('task')}")
            print(f"Strategy: {result.get('strategy')}")
            print(f"Final Result: {result.get('final_result', '')[:100]}...")
            print(f"Agent Outputs: {len(result.get('agent_outputs', []))} agents")

if __name__ == "__main__":
    asyncio.run(test_multi_agent_with_ollama())
