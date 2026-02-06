#!/usr/bin/env python3
"""
NER 엔진 벤치마크 데모

다양한 NER 엔진(spaCy, HuggingFace, GLiNER, Flair, LLM)을 비교합니다.

사용법:
    # 기본 벤치마크 (spaCy만)
    python examples/ner_benchmark_demo.py

    # 모든 엔진 벤치마크
    python examples/ner_benchmark_demo.py --all

    # 특정 엔진만
    python examples/ner_benchmark_demo.py --engines spacy huggingface

    # 커스텀 테스트 데이터
    python examples/ner_benchmark_demo.py --data test_data.json

    # Ollama SLM 포함
    python examples/ner_benchmark_demo.py --ollama qwen2.5:0.5b

필요한 의존성:
    pip install spacy
    python -m spacy download en_core_web_sm

    pip install transformers torch

    pip install gliner

    pip install flair
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beanllm.domain.knowledge_graph.ner_engines import (
    SAMPLE_TEST_DATA,
    BaseNEREngine,
    BenchmarkSample,
    NERBenchmark,
    NEREngineFactory,
)


def create_ollama_ner_engine(model: str = "qwen2.5:0.5b"):
    """Ollama SLM 기반 NER 엔진 생성"""
    try:
        from beanllm import Client

        client = Client(model=model)

        def llm_function(prompt: str) -> str:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.content

        from beanllm.domain.knowledge_graph.ner_engines import LLMNEREngine

        return LLMNEREngine(
            llm_function=llm_function,
            labels=["PERSON", "ORGANIZATION", "LOCATION", "DATE", "PRODUCT"],
        )
    except Exception as e:
        print(f"Failed to create Ollama engine: {e}")
        return None


def load_custom_test_data(file_path: str) -> List[BenchmarkSample]:
    """커스텀 테스트 데이터 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(
            BenchmarkSample(
                text=item["text"],
                entities=item["entities"],
            )
        )
    return samples


def run_benchmark(
    engines: List[BaseNEREngine],
    test_data: List[BenchmarkSample],
    verbose: bool = True,
) -> Dict[str, Any]:
    """벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"NER Benchmark - {len(engines)} engines, {len(test_data)} samples")
    print(f"{'='*60}\n")

    benchmark = NERBenchmark(engines=engines)

    start_time = time.time()
    results = benchmark.run(test_data)
    total_time = time.time() - start_time

    # 리포트 출력
    print(benchmark.get_report())

    print(f"\nTotal benchmark time: {total_time:.2f}s")

    # 상세 결과 (verbose)
    if verbose:
        print(f"\n{'='*60}")
        print("Detailed Results")
        print(f"{'='*60}")

        for engine_name, result in results.items():
            print(f"\n## {engine_name}")
            print("   Samples with errors:")
            for detail in result.detailed_results:
                if detail["fp"] > 0 or detail["fn"] > 0:
                    print(f"   - {detail['text']}")
                    print(f"     FP: {detail['fp']}, FN: {detail['fn']}")
                    if detail["fp"] > 0:
                        extra = set(detail["predicted"]) - set(detail["gold"])
                        print(f"     Extra: {extra}")
                    if detail["fn"] > 0:
                        missed = set(detail["gold"]) - set(detail["predicted"])
                        print(f"     Missed: {missed}")

    return {
        "results": results,
        "best_f1": benchmark.get_best_engine("f1_score"),
        "best_speed": benchmark.get_best_engine("avg_latency_ms"),
        "total_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="NER Engine Benchmark")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["spacy"],
        choices=["spacy", "huggingface", "gliner", "flair", "all"],
        help="Engines to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all available engines",
    )
    parser.add_argument(
        "--ollama",
        type=str,
        default=None,
        help="Ollama model to include (e.g., qwen2.5:0.5b)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to custom test data JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed results",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="dslim/bert-base-NER",
        help="HuggingFace model to use",
    )

    args = parser.parse_args()

    # 엔진 목록 결정
    if args.all or "all" in args.engines:
        engine_types = ["spacy", "huggingface", "gliner", "flair"]
    else:
        engine_types = args.engines

    # 엔진 생성
    engines = []
    for engine_type in engine_types:
        try:
            print(f"Loading {engine_type} engine...", end=" ", flush=True)

            kwargs = {}
            if engine_type == "spacy":
                kwargs["model"] = args.spacy_model
            elif engine_type == "huggingface":
                kwargs["model"] = args.hf_model

            engine = NEREngineFactory.create(engine_type, **kwargs)
            engine.load()  # Preload
            engines.append(engine)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    # Ollama SLM 추가
    if args.ollama:
        print(f"Loading Ollama ({args.ollama})...", end=" ", flush=True)
        ollama_engine = create_ollama_ner_engine(args.ollama)
        if ollama_engine:
            ollama_engine.name = f"ollama:{args.ollama}"
            engines.append(ollama_engine)
            print("OK")
        else:
            print("FAILED")

    if not engines:
        print("\nNo engines available. Please install required dependencies.")
        print("\nRequired dependencies:")
        print("  spacy: pip install spacy && python -m spacy download en_core_web_sm")
        print("  huggingface: pip install transformers torch")
        print("  gliner: pip install gliner")
        print("  flair: pip install flair")
        sys.exit(1)

    # 테스트 데이터 로드
    if args.data:
        test_data = load_custom_test_data(args.data)
    else:
        test_data = SAMPLE_TEST_DATA

    # 벤치마크 실행
    results = run_benchmark(engines, test_data, verbose=args.verbose)

    # 결론
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"Best F1 Score: {results['best_f1']}")
    print(f"Fastest Engine: {results['best_speed']}")

    # JSON 결과 저장
    output_path = Path("ner_benchmark_results.json")
    json_results = {
        engine_name: {
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "avg_latency_ms": r.avg_latency_ms,
        }
        for engine_name, r in results["results"].items()
    }
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
