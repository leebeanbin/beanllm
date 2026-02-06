"""
Chain API 상세 테스트
- Basic Chain
- Prompt Chain
- Ollama 무료 모델 사용
"""

import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"


def test_chain_build():
    """Chain 구축 테스트"""
    print("\n" + "=" * 60)
    print("1. Chain - 체인 구축")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chain/build",
        json={
            "input": "Test input",
            "chain_id": "test_chain",
            "chain_type": "basic",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Chain 구축 성공")
        print(f"   Chain ID: {data.get('chain_id', 'N/A')}")
        print(f"   Status: {data.get('status', 'N/A')}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_basic_chain_run():
    """Basic Chain 실행 테스트"""
    print("\n" + "=" * 60)
    print("2. Chain - Basic 실행")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chain/run",
        json={"input": "What is 10 + 5?", "chain_type": "basic", "model": OLLAMA_CHAT_MODEL},
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Basic Chain 실행 완료")
        print(f"   Input: {data.get('input', '')}")
        print(f"   Output: {data.get('output', '')[:200]}")
        print(f"   Success: {data.get('success', False)}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_prompt_chain():
    """Prompt Chain 테스트"""
    print("\n" + "=" * 60)
    print("3. Chain - Prompt Chain")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chain/run",
        json={
            "input": "machine learning",
            "chain_type": "prompt",
            "template": "Explain {input} in simple terms",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Prompt Chain 실행 완료")
        print(f"   Template: Explain {input} in simple terms")
        print(f"   Output: {data.get('output', '')[:200]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_chain_with_saved_id():
    """저장된 Chain ID로 실행"""
    print("\n" + "=" * 60)
    print("4. Chain - 저장된 Chain ID 사용")
    print("=" * 60)

    # 먼저 chain 구축
    build_response = requests.post(
        f"{BACKEND_URL}/api/chain/build",
        json={
            "input": "initial",
            "chain_id": "saved_chain",
            "chain_type": "basic",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    if build_response.status_code != 200:
        print("⚠️  Chain 구축 실패, 스킵")
        return None

    # 저장된 chain으로 실행
    response = requests.post(
        f"{BACKEND_URL}/api/chain/run",
        json={"input": "Calculate 7 * 8", "chain_id": "saved_chain", "model": OLLAMA_CHAT_MODEL},
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ 저장된 Chain 실행 성공")
        print(f"   Chain ID: {data.get('chain_id', '')}")
        print(f"   Output: {data.get('output', '')[:150]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def test_chain_different_inputs():
    """다양한 입력으로 Chain 테스트"""
    print("\n" + "=" * 60)
    print("5. Chain - 다양한 입력")
    print("=" * 60)

    inputs = ["What is AI?", "Explain Python", "5 + 10 = ?"]

    all_success = True

    for inp in inputs:
        response = requests.post(
            f"{BACKEND_URL}/api/chain/run",
            json={"input": inp, "chain_type": "basic", "model": OLLAMA_CHAT_MODEL},
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Input: {inp}")
            print(f"   Output: {data.get('output', '')[:100]}")
        else:
            print(f"❌ Input: {inp} - 실패")
            all_success = False

    return all_success


def test_chain_with_template_variables():
    """템플릿 변수 사용 Chain"""
    print("\n" + "=" * 60)
    print("6. Chain - 템플릿 변수")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/chain/run",
        json={
            "input": "Python programming",
            "chain_type": "prompt",
            "template": "Write a brief introduction to {input}",
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ 템플릿 변수 사용 성공")
        print("   Template: Write a brief introduction to {input}")
        print(f"   Output: {data.get('output', '')[:200]}")
        return True
    else:
        print(f"❌ 실패: {response.text}")
        return False


def main():
    """모든 Chain API 테스트 실행"""
    print("=" * 60)
    print("Chain API 상세 테스트 (Ollama 무료 모델)")
    print("=" * 60)

    results = []

    # 각 테스트 실행
    results.append(("Chain 구축", test_chain_build()))
    results.append(("Basic Chain 실행", test_basic_chain_run()))
    results.append(("Prompt Chain", test_prompt_chain()))

    saved_result = test_chain_with_saved_id()
    if saved_result is not None:
        results.append(("저장된 Chain ID", saved_result))

    results.append(("다양한 입력", test_chain_different_inputs()))
    results.append(("템플릿 변수", test_chain_with_template_variables()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("Chain API 테스트 결과 요약")
    print("=" * 60)

    success = sum(1 for _, result in results if result is True)
    total = len(results)

    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP"
        else:
            status = "❌ FAIL"
        print(f"{status} - {name}")

    print("=" * 60)
    print(f"총 {success}/{total} 테스트 통과")
    print("=" * 60)


if __name__ == "__main__":
    main()
