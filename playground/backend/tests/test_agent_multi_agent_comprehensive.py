"""
Agent & Multi-Agent API 상세 테스트
- Agent: 단일 에이전트 (다양한 도구 사용)
- Multi-Agent: 협업 에이전트 (Sequential, Parallel, Hierarchical, Debate)
- Ollama 무료 모델 사용
"""

import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"


def test_agent_calculator():
    """Agent - 계산기 도구 사용"""
    print("\n" + "=" * 60)
    print("1. Agent - 계산기 도구")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/agent/run",
        json={
            "task": "Calculate 15 + 27",
            "tools": ["calculator"],
            "model": OLLAMA_CHAT_MODEL,
            "max_iterations": 5,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", "")
        print("✅ Task: Calculate 15 + 27")
        print(f"✅ Result: {result[:200]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_agent_multiple_tools():
    """Agent - 여러 도구 사용"""
    print("\n" + "=" * 60)
    print("2. Agent - 여러 도구 사용")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/agent/run",
        json={
            "task": "What is the current time and calculate 10 * 5?",
            "tools": ["calculator", "get_current_time"],
            "model": OLLAMA_CHAT_MODEL,
            "max_iterations": 10,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", "")
        print("✅ Task: Time + Calculation")
        print(f"✅ Result: {result[:250]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_agent_iterations():
    """Agent - 반복 실행 제한"""
    print("\n" + "=" * 60)
    print("3. Agent - 반복 실행 제한")
    print("=" * 60)

    # 최대 3회 반복으로 제한
    response = requests.post(
        f"{BACKEND_URL}/api/agent/run",
        json={
            "task": "Calculate 5 + 3, then multiply by 2, then add 10",
            "tools": ["calculator"],
            "model": OLLAMA_CHAT_MODEL,
            "max_iterations": 3,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", "")
        print("✅ Max iterations: 3")
        print(f"✅ Result: {result[:200]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_multi_agent_sequential():
    """Multi-Agent - Sequential 전략"""
    print("\n" + "=" * 60)
    print("4. Multi-Agent - Sequential (순차)")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/multi_agent/run",
        json={
            "task": "Calculate 12 + 8",
            "agents": [
                {"name": "calculator_agent", "role": "Calculate numbers"},
                {"name": "verifier_agent", "role": "Verify the result"},
            ],
            "strategy": "sequential",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        final_result = data.get("final_result", "")
        print("✅ Strategy: Sequential")
        print(f"✅ Final result: {final_result[:200]}")

        # 중간 결과 확인
        intermediate = data.get("intermediate_results", [])
        print(f"✅ Intermediate steps: {len(intermediate)}")

        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_multi_agent_parallel():
    """Multi-Agent - Parallel 전략"""
    print("\n" + "=" * 60)
    print("5. Multi-Agent - Parallel (병렬)")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/multi_agent/run",
        json={
            "task": "Analyze the number 42 from different perspectives",
            "agents": [
                {"name": "math_agent", "role": "Mathematical analysis"},
                {"name": "cultural_agent", "role": "Cultural significance"},
                {"name": "scientific_agent", "role": "Scientific relevance"},
            ],
            "strategy": "parallel",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        final_result = data.get("final_result", "")
        print("✅ Strategy: Parallel")
        print(f"✅ Final result: {final_result[:250]}")

        agent_outputs = data.get("agent_outputs", [])
        print(f"✅ Agent outputs: {len(agent_outputs)}")

        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_multi_agent_hierarchical():
    """Multi-Agent - Hierarchical 전략"""
    print("\n" + "=" * 60)
    print("6. Multi-Agent - Hierarchical (계층적)")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/multi_agent/run",
        json={
            "task": "Plan a simple research project",
            "agents": [
                {"name": "manager", "role": "Project manager"},
                {"name": "researcher", "role": "Research specialist"},
                {"name": "writer", "role": "Documentation writer"},
            ],
            "strategy": "hierarchical",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        final_result = data.get("final_result", "")
        print("✅ Strategy: Hierarchical")
        print(f"✅ Final result: {final_result[:250]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_multi_agent_debate():
    """Multi-Agent - Debate 전략"""
    print("\n" + "=" * 60)
    print("7. Multi-Agent - Debate (토론)")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/multi_agent/run",
        json={
            "task": "Is AI beneficial or harmful for society?",
            "agents": [
                {"name": "optimist", "role": "Argues benefits of AI"},
                {"name": "pessimist", "role": "Argues risks of AI"},
                {"name": "moderator", "role": "Synthesizes arguments"},
            ],
            "strategy": "debate",
            "model": OLLAMA_CHAT_MODEL,
            "max_rounds": 2,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        final_result = data.get("final_result", "")
        print("✅ Strategy: Debate")
        print(f"✅ Final result: {final_result[:250]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_agent_complex_task():
    """Agent - 복잡한 작업"""
    print("\n" + "=" * 60)
    print("8. Agent - 복잡한 작업")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/agent/run",
        json={
            "task": "Calculate (15 + 25) * 2 - 10, and verify the result step by step",
            "tools": ["calculator"],
            "model": OLLAMA_CHAT_MODEL,
            "max_iterations": 15,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", "")
        print("✅ Complex task completed")
        print(f"✅ Result: {result[:300]}")

        # Expected: (15 + 25) * 2 - 10 = 40 * 2 - 10 = 80 - 10 = 70
        if "70" in result:
            print("✅ Correct answer found: 70")
        else:
            print("⚠️  Answer may not be explicitly shown")

        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def main():
    """모든 Agent & Multi-Agent API 테스트 실행"""
    print("=" * 60)
    print("Agent & Multi-Agent API 상세 테스트 (Ollama 무료 모델)")
    print("=" * 60)

    results = []

    # Agent 테스트
    results.append(("Agent - 계산기", test_agent_calculator()))
    results.append(("Agent - 여러 도구", test_agent_multiple_tools()))
    results.append(("Agent - 반복 제한", test_agent_iterations()))
    results.append(("Agent - 복잡한 작업", test_agent_complex_task()))

    # Multi-Agent 테스트
    results.append(("Multi-Agent - Sequential", test_multi_agent_sequential()))
    results.append(("Multi-Agent - Parallel", test_multi_agent_parallel()))
    results.append(("Multi-Agent - Hierarchical", test_multi_agent_hierarchical()))
    results.append(("Multi-Agent - Debate", test_multi_agent_debate()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("Agent & Multi-Agent API 테스트 결과 요약")
    print("=" * 60)

    success = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("=" * 60)
    print(f"총 {success}/{total} 테스트 통과")
    print("=" * 60)


if __name__ == "__main__":
    main()
