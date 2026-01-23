"""
ML Tools (Audio, OCR, Evaluation) - 기존 beanllm ML 기능을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastmcp import FastMCP

# 기존 beanllm 코드 import (wrapping 대상)
from beanllm.facade.ml import AudioFacade, EvaluationFacade
from beanllm.domain.ocr import beanOCR
from beanllm.dto.request.audio import AudioRequest
from beanllm.dto.request.evaluation import EvaluationRequest
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스 생성
mcp = FastMCP("ML Tools")


# =============================================================================
# Audio Tools - 음성 인식 및 전사
# =============================================================================


@mcp.tool()
async def transcribe_audio(
    audio_path: str,
    engine: str = "whisper",
    language: Optional[str] = None,
    model_size: str = "base",
) -> dict:
    """
    음성 파일 전사 (기존 코드 재사용)

    Args:
        audio_path: 오디오 파일 경로 (.mp3, .wav, .m4a 등)
        engine: 전사 엔진 (whisper, distil-whisper, canary, etc.)
        language: 언어 코드 (None이면 자동 감지)
        model_size: 모델 크기 (tiny, base, small, medium, large)

    Returns:
        dict: 전사 텍스트, 타임스탬프, 신뢰도

    Example:
        User: "이 음성 파일 텍스트로 변환해줘"
        → transcribe_audio(
            audio_path="/path/to/audio.mp3",
            engine="whisper"
        )
    """
    try:
        # 🎯 기존 AudioFacade 사용!
        request = AudioRequest(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
        )

        facade = AudioFacade()
        result = await asyncio.to_thread(facade.transcribe, request)

        return {
            "success": True,
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "confidence": result.confidence,
            "duration_seconds": result.duration,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def batch_transcribe_audio(
    audio_paths: List[str],
    engine: str = "whisper",
    language: Optional[str] = None,
) -> dict:
    """
    여러 음성 파일 일괄 전사 (기존 코드 재사용)

    Args:
        audio_paths: 오디오 파일 경로 목록
        engine: 전사 엔진
        language: 언어 코드

    Returns:
        dict: 각 파일의 전사 결과

    Example:
        User: "이 폴더의 모든 음성 파일 텍스트로 변환해줘"
        → batch_transcribe_audio(
            audio_paths=["/path/to/audio1.mp3", "/path/to/audio2.mp3"]
        )
    """
    try:
        results = []
        facade = AudioFacade()

        for audio_path in audio_paths:
            request = AudioRequest(
                audio_path=audio_path,
                engine=engine,
                language=language,
            )
            result = await asyncio.to_thread(facade.transcribe, request)
            results.append(
                {
                    "file": audio_path,
                    "text": result.text,
                    "language": result.language,
                    "duration": result.duration,
                }
            )

        return {
            "success": True,
            "total_files": len(audio_paths),
            "results": results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# OCR Tools - 문서 이미지 인식
# =============================================================================


@mcp.tool()
async def recognize_text_ocr(
    image_path: str,
    engine: str = "tesseract",
    language: str = "eng",
    preprocess: bool = True,
) -> dict:
    """
    이미지에서 텍스트 추출 (기존 코드 재사용)

    Args:
        image_path: 이미지 파일 경로 (.png, .jpg, .jpeg, etc.)
        engine: OCR 엔진 (tesseract, easyocr, paddleocr, surya, etc.)
        language: 언어 코드 (eng, kor, etc.)
        preprocess: 전처리 활성화 (노이즈 제거, 기울기 보정 등)

    Returns:
        dict: 추출된 텍스트, 바운딩 박스, 신뢰도

    Example:
        User: "이 이미지에서 텍스트 추출해줘"
        → recognize_text_ocr(
            image_path="/path/to/image.png",
            engine="tesseract"
        )
    """
    try:
        # 🎯 기존 beanOCR 사용!
        ocr = beanOCR(engine=engine, lang=language, preprocess=preprocess)

        result = await asyncio.to_thread(ocr.extract_text, image_path)

        return {
            "success": True,
            "text": result.text,
            "boxes": result.boxes,
            "confidence": result.confidence,
            "engine": engine,
            "language": language,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def batch_recognize_text_ocr(
    image_paths: List[str],
    engine: str = "tesseract",
    language: str = "eng",
) -> dict:
    """
    여러 이미지 일괄 OCR 처리 (기존 코드 재사용)

    Args:
        image_paths: 이미지 파일 경로 목록
        engine: OCR 엔진
        language: 언어 코드

    Returns:
        dict: 각 이미지의 OCR 결과

    Example:
        User: "이 폴더의 모든 이미지에서 텍스트 추출해줘"
        → batch_recognize_text_ocr(
            image_paths=["/path/to/img1.png", "/path/to/img2.png"]
        )
    """
    try:
        results = []
        ocr = beanOCR(engine=engine, lang=language)

        for image_path in image_paths:
            result = await asyncio.to_thread(ocr.extract_text, image_path)
            results.append(
                {
                    "file": image_path,
                    "text": result.text,
                    "confidence": result.confidence,
                }
            )

        return {
            "success": True,
            "total_files": len(image_paths),
            "results": results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Evaluation Tools - 모델 평가 및 벤치마킹
# =============================================================================


@mcp.tool()
async def evaluate_model(
    model: str,
    evaluation_type: str = "answer_relevancy",
    test_data: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[List[str]] = None,
) -> dict:
    """
    LLM 모델 평가 (기존 코드 재사용)

    Args:
        model: 평가할 모델명
        evaluation_type: 평가 유형 (answer_relevancy, faithfulness, context_recall, etc.)
        test_data: 테스트 데이터 [{"question": "...", "answer": "...", "context": "..."}]
        metrics: 평가 메트릭 목록 (None이면 기본값)

    Returns:
        dict: 평가 점수, 상세 결과

    Example:
        User: "qwen2.5:0.5b 모델 성능 평가해줘"
        → evaluate_model(
            model="qwen2.5:0.5b",
            evaluation_type="answer_relevancy"
        )
    """
    try:
        # 🎯 기존 EvaluationFacade 사용!
        request = EvaluationRequest(
            model=model,
            evaluation_type=evaluation_type,
            test_data=test_data or [],
            metrics=metrics or ["answer_relevancy", "faithfulness"],
        )

        facade = EvaluationFacade()
        result = await asyncio.to_thread(facade.evaluate, request)

        return {
            "success": True,
            "model": model,
            "evaluation_type": evaluation_type,
            "overall_score": result.overall_score,
            "metric_scores": result.metric_scores,
            "details": result.details,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def benchmark_models(
    models: List[str],
    benchmark_type: str = "lm_eval_harness",
    tasks: Optional[List[str]] = None,
) -> dict:
    """
    여러 모델 벤치마킹 비교 (기존 코드 재사용)

    Args:
        models: 벤치마크할 모델 목록
        benchmark_type: 벤치마크 유형 (lm_eval_harness, ragas, deepeval, etc.)
        tasks: 태스크 목록 (None이면 기본값)

    Returns:
        dict: 각 모델의 벤치마크 점수, 순위

    Example:
        User: "qwen2.5랑 llama 모델 성능 비교해줘"
        → benchmark_models(
            models=["qwen2.5:0.5b", "llama3.2:1b"],
            benchmark_type="lm_eval_harness"
        )
    """
    try:
        results = []
        facade = EvaluationFacade()

        for model in models:
            request = EvaluationRequest(
                model=model,
                evaluation_type=benchmark_type,
                tasks=tasks or ["hellaswag", "arc_easy"],
            )
            result = await asyncio.to_thread(facade.benchmark, request)
            results.append(
                {
                    "model": model,
                    "overall_score": result.overall_score,
                    "task_scores": result.task_scores,
                }
            )

        # 점수 기준 정렬
        results_sorted = sorted(
            results, key=lambda x: x["overall_score"], reverse=True
        )

        return {
            "success": True,
            "benchmark_type": benchmark_type,
            "total_models": len(models),
            "results": results_sorted,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def compare_model_outputs(
    models: List[str],
    prompt: str,
    temperature: float = 0.7,
) -> dict:
    """
    여러 모델의 출력 비교 (기존 코드 재사용)

    Args:
        models: 비교할 모델 목록
        prompt: 동일한 프롬프트
        temperature: 생성 온도

    Returns:
        dict: 각 모델의 출력, 토큰 수, 응답 시간

    Example:
        User: "qwen2.5랑 llama로 같은 질문에 답변 비교해줘"
        → compare_model_outputs(
            models=["qwen2.5:0.5b", "llama3.2:1b"],
            prompt="AI의 미래는?"
        )
    """
    try:
        # 기존 Client 사용
        from beanllm.facade.core import Client
        import time

        results = []

        for model in models:
            client = Client(model=model)
            start_time = time.time()

            response = await client.chat(
                messages=[{"role": "user", "content": prompt}], temperature=temperature
            )

            elapsed_time = time.time() - start_time

            results.append(
                {
                    "model": model,
                    "output": response.content,
                    "tokens": response.usage.get("total_tokens", 0),
                    "response_time_seconds": round(elapsed_time, 2),
                }
            )

        return {
            "success": True,
            "prompt": prompt,
            "total_models": len(models),
            "results": results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
