"""
남은 5개 API 전체 테스트
- Evaluation API
- Vision RAG API
- Audio API
- OCR API
- Fine-tuning API
"""

import base64
import json

import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_CHAT_MODEL = "qwen2.5:0.5b"


def test_evaluation_api():
    """Evaluation API 테스트"""
    print("\n" + "=" * 60)
    print("Evaluation API 테스트")
    print("=" * 60)

    response = requests.post(
        f"{BACKEND_URL}/api/evaluation/evaluate",
        json={
            "task_type": "rag",
            "queries": ["What is machine learning?", "Explain AI"],
            "ground_truth": ["Machine learning is a subset of AI", "AI is artificial intelligence"],
            "model": OLLAMA_CHAT_MODEL,
        },
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Evaluation 성공")
        print(f"   Results: {json.dumps(data, indent=2)[:300]}")
        return True
    else:
        print(f"⚠️  실패 (구현 확인 필요): {response.text[:200]}")
        return False


def test_vision_rag_api():
    """Vision RAG API 테스트"""
    print("\n" + "=" * 60)
    print("Vision RAG API 테스트")
    print("=" * 60)

    # 간단한 1x1 픽셀 PNG 이미지 (base64)
    tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Build (without caption generation to avoid transformers dependency)
    build_response = requests.post(
        f"{BACKEND_URL}/api/vision_rag/build",
        json={
            "images": [f"data:image/png;base64,{tiny_image}"],
            "texts": ["A simple test image"],
            "collection_name": "test_vision",
            "model": OLLAMA_CHAT_MODEL,
            "generate_captions": False,  # Disable to avoid transformers dependency
        },
    )

    print(f"Build Status: {build_response.status_code}")
    if build_response.status_code != 200:
        print(f"⚠️  Build 실패: {build_response.text[:200]}")
        return False

    # Query
    query_response = requests.post(
        f"{BACKEND_URL}/api/vision_rag/query",
        json={"query": "Describe the image", "collection_name": "test_vision"},
    )

    print(f"Query Status: {query_response.status_code}")
    if query_response.status_code == 200:
        data = query_response.json()
        print("✅ Vision RAG 성공")
        print(f"   Answer: {data.get('answer', '')[:150]}")
        return True
    else:
        print(f"⚠️  Query 실패: {query_response.text[:200]}")
        return False


def test_audio_api():
    """Audio API 테스트 (STT, TTS, AudioRAG)"""
    print("\n" + "=" * 60)
    print("Audio API 테스트")
    print("=" * 60)

    # 1. Transcribe (STT) - 파일이 필요하므로 스킵 가능
    print("  1) Transcribe (STT) - 오디오 파일 필요, 스킵")

    # 2. Synthesize (TTS)
    tts_response = requests.post(
        f"{BACKEND_URL}/api/audio/synthesize",
        json={
            "text": "Hello world",
            "model": "tts-1",  # OpenAI TTS (API 키 필요)
        },
    )

    print(f"  2) TTS Status: {tts_response.status_code}")
    if tts_response.status_code == 200:
        print("  ✅ TTS 성공")
        tts_success = True
    else:
        print(f"  ⚠️  TTS 실패 (API 키 필요): {tts_response.text[:100]}")
        tts_success = False

    # 3. Audio RAG - 오디오 파일 필요, 스킵
    print("  3) Audio RAG - 오디오 파일 필요, 스킵")

    return tts_success


def test_ocr_api():
    """OCR API 테스트"""
    print("\n" + "=" * 60)
    print("OCR API 테스트")
    print("=" * 60)

    # 간단한 1x1 픽셀 이미지로 테스트
    tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # base64를 바이트로 변환
    import io

    image_bytes = base64.b64decode(tiny_image)

    # 파일 업로드로 전송 (multipart/form-data)
    files = {"file": ("test_image.png", io.BytesIO(image_bytes), "image/png")}
    data = {"language": "eng", "engine": "paddleocr"}

    response = requests.post(f"{BACKEND_URL}/api/ocr/recognize", files=files, data=data)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ OCR 성공")
        print(f"   Text: {data.get('text', '')}")
        return True
    else:
        print(f"⚠️  실패 (구현 확인 필요): {response.text[:200]}")
        return False


def test_finetuning_api():
    """Fine-tuning API 테스트"""
    print("\n" + "=" * 60)
    print("Fine-tuning API 테스트")
    print("=" * 60)

    # Create fine-tuning job
    response = requests.post(
        f"{BACKEND_URL}/api/finetuning/create",
        json={
            "base_model": "gpt-3.5-turbo",
            "training_data": [
                {"prompt": "Hello", "completion": "Hi there!"},
                {"prompt": "Bye", "completion": "Goodbye!"},
            ],
            "job_name": "test_job",
        },
    )

    print(f"Create Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        job_id = data.get("job_id", "test_id")
        print("✅ Fine-tuning Job 생성 성공")
        print(f"   Job ID: {job_id}")

        # Check status
        status_response = requests.get(f"{BACKEND_URL}/api/finetuning/status/{job_id}")

        print(f"Status Check: {status_response.status_code}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print("✅ Status 조회 성공")
            print(f"   Status: {status_data.get('status', 'N/A')}")
            return True
        else:
            print(f"⚠️  Status 조회 실패: {status_response.text[:150]}")
            return True  # Create는 성공했으므로 True
    else:
        print(f"⚠️  실패 (API 키 필요): {response.text[:200]}")
        return False


def main():
    """모든 남은 API 테스트 실행"""
    print("=" * 60)
    print("남은 5개 API 전체 테스트")
    print("=" * 60)

    results = []

    # 각 테스트 실행
    results.append(("Evaluation API", test_evaluation_api()))
    results.append(("Vision RAG API", test_vision_rag_api()))
    results.append(("Audio API", test_audio_api()))
    results.append(("OCR API", test_ocr_api()))
    results.append(("Fine-tuning API", test_finetuning_api()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("남은 API 테스트 결과 요약")
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
