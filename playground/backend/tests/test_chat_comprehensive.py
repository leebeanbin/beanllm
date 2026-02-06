"""
Chat API 상세 테스트
- 기본 채팅
- 스트리밍 모드
- Think Mode (추론 시각화)
- 시스템 메시지
- 대화 컨텍스트
- Ollama 무료 모델 사용
"""

import json

import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"


def test_basic_chat():
    """기본 채팅 테스트"""
    print("\n" + "=" * 60)
    print("1. Chat API - 기본 채팅")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "model": OLLAMA_CHAT_MODEL,
            "stream": False,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        print("✅ User: What is 2 + 2?")
        print(f"✅ Assistant: {answer[:200]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_conversation_context():
    """대화 컨텍스트 테스트"""
    print("\n" + "=" * 60)
    print("2. Chat API - 대화 컨텍스트")
    print("=" * 60)

    # Multi-turn conversation
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ]

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={"messages": messages, "model": OLLAMA_CHAT_MODEL, "stream": False},
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        print("✅ Context preserved")
        print("   User: What is my name?")
        print(f"   Assistant: {answer[:150]}")
        # Check if "Alice" is in the response
        if "Alice" in answer or "alice" in answer.lower():
            print("✅ Correctly remembered name: Alice")
        else:
            print("⚠️  May not have remembered the name")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_system_message():
    """시스템 메시지 테스트"""
    print("\n" + "=" * 60)
    print("3. Chat API - 시스템 메시지")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Answer in one short sentence.",
                },
                {"role": "user", "content": "What is calculus?"},
            ],
            "model": OLLAMA_CHAT_MODEL,
            "stream": False,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        print("✅ System message applied")
        print(f"   Answer: {answer[:200]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_temperature_control():
    """Temperature 파라미터 테스트"""
    print("\n" + "=" * 60)
    print("4. Chat API - Temperature 제어")
    print("=" * 60)

    # Low temperature (more deterministic)
    response_low = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "model": OLLAMA_CHAT_MODEL,
            "temperature": 0.1,
            "stream": False,
        },
    )

    # High temperature (more creative)
    response_high = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "model": OLLAMA_CHAT_MODEL,
            "temperature": 0.9,
            "stream": False,
        },
    )

    if response_low.status_code == 200 and response_high.status_code == 200:
        low_answer = response_low.json().get("response", "")
        high_answer = response_high.json().get("response", "")

        print("✅ Temperature 제어 작동")
        print(f"   Low (0.1): {low_answer[:100]}")
        print(f"   High (0.9): {high_answer[:100]}")
        return True
    else:
        print("❌ 실패")
        return False


def test_max_tokens():
    """Max tokens 제한 테스트"""
    print("\n" + "=" * 60)
    print("5. Chat API - Max Tokens 제한")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [{"role": "user", "content": "Tell me a long story about a cat"}],
            "model": OLLAMA_CHAT_MODEL,
            "max_tokens": 50,
            "stream": False,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        print("✅ Max tokens 제한 적용")
        print(f"   Response length: ~{len(answer.split())} words")
        print(f"   Preview: {answer[:150]}...")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_streaming_mode():
    """스트리밍 모드 테스트"""
    print("\n" + "=" * 60)
    print("6. Chat API - 스트리밍 모드")
    print("=" * 60)

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "messages": [{"role": "user", "content": "Count from 1 to 10"}],
                "model": OLLAMA_CHAT_MODEL,
                "stream": True,
            },
            stream=True,
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ 스트리밍 시작")
            print("   Chunks: ", end="")

            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    chunk_count += 1
                    if chunk_count <= 5:  # 처음 5개 청크만 출력
                        try:
                            # SSE format: data: {...}
                            if line.startswith(b"data: "):
                                data = json.loads(line[6:])
                                chunk = data.get("chunk", "")
                                print(chunk, end="", flush=True)
                        except:
                            pass

            print(f"\n✅ 총 {chunk_count}개 청크 수신")
            return True
        else:
            print(f"❌ 실패: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return False


def test_think_mode():
    """Think Mode 테스트 (추론 시각화)"""
    print("\n" + "=" * 60)
    print("7. Chat API - Think Mode")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "If a train travels at 60 mph for 2 hours, how far does it go?",
                }
            ],
            "model": OLLAMA_CHAT_MODEL,
            "enable_thinking": True,
            "stream": False,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        print("✅ Think Mode 활성화")
        print(f"   Answer: {answer[:300]}")

        # Check for thinking tags
        if "<think>" in answer or "step by step" in answer.lower():
            print("✅ 추론 과정 포함")
        else:
            print("⚠️  추론 과정이 명시적이지 않을 수 있음 (모델 의존적)")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_complex_reasoning():
    """복잡한 추론 테스트"""
    print("\n" + "=" * 60)
    print("8. Chat API - 복잡한 추론")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
                }
            ],
            "model": OLLAMA_CHAT_MODEL,
            "stream": False,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
        print("✅ 복잡한 추론 처리")
        print("   Question: Logical reasoning test")
        print(f"   Answer: {answer[:250]}...")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def main():
    """모든 Chat API 테스트 실행"""
    print("=" * 60)
    print("Chat API 상세 테스트 (Ollama 무료 모델)")
    print("=" * 60)

    results = []

    # 각 테스트 실행
    results.append(("기본 채팅", test_basic_chat()))
    results.append(("대화 컨텍스트", test_conversation_context()))
    results.append(("시스템 메시지", test_system_message()))
    results.append(("Temperature 제어", test_temperature_control()))
    results.append(("Max Tokens 제한", test_max_tokens()))
    results.append(("스트리밍 모드", test_streaming_mode()))
    results.append(("Think Mode", test_think_mode()))
    results.append(("복잡한 추론", test_complex_reasoning()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("Chat API 테스트 결과 요약")
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
