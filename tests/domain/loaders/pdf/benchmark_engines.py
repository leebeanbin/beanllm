"""
beanPDFLoader 엔진 성능 벤치마크

3-Layer 아키텍처 엔진 비교:
- Fast Layer: PyMuPDF
- Accurate Layer: pdfplumber
- ML Layer: marker-pdf (옵션)

Usage:
    python tests/domain/loaders/pdf/benchmark_engines.py
"""

import time
from pathlib import Path
from typing import Dict, List

import psutil


def get_memory_usage() -> float:
    """현재 메모리 사용량 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class EngineBenchmark:
    """엔진 벤치마크 클래스"""

    def __init__(self, pdf_paths: List[str]):
        """
        Args:
            pdf_paths: 벤치마크할 PDF 파일 경로 리스트
        """
        self.pdf_paths = pdf_paths
        self.results = {}

    def benchmark_pymupdf(self) -> Dict:
        """PyMuPDFEngine 벤치마크"""
        print("\n===== PyMuPDFEngine (Fast Layer) =====")

        try:
            from beanllm.domain.loaders.pdf.engines import PyMuPDFEngine

            engine = PyMuPDFEngine()
            config = {"extract_tables": False, "extract_images": True}

            total_time = 0
            total_pages = 0
            mem_before = get_memory_usage()

            for pdf_path in self.pdf_paths:
                start = time.time()
                result = engine.extract(pdf_path, config)
                elapsed = time.time() - start

                total_time += elapsed
                total_pages += result["metadata"]["total_pages"]

                print(
                    f"  {Path(pdf_path).name}: {elapsed:.2f}s "
                    f"({result['metadata']['total_pages']} pages)"
                )

            mem_after = get_memory_usage()

            return {
                "engine": "PyMuPDF",
                "total_time": total_time,
                "avg_time": total_time / len(self.pdf_paths),
                "total_pages": total_pages,
                "pages_per_sec": total_pages / total_time if total_time > 0 else 0,
                "memory_used_mb": mem_after - mem_before,
            }

        except Exception as e:
            print(f"  Error: {e}")
            return None

    def benchmark_pdfplumber(self) -> Dict:
        """PDFPlumberEngine 벤치마크"""
        print("\n===== PDFPlumberEngine (Accurate Layer) =====")

        try:
            from beanllm.domain.loaders.pdf.engines import PDFPlumberEngine

            engine = PDFPlumberEngine()
            config = {"extract_tables": True, "extract_images": False}

            total_time = 0
            total_pages = 0
            mem_before = get_memory_usage()

            for pdf_path in self.pdf_paths:
                start = time.time()
                result = engine.extract(pdf_path, config)
                elapsed = time.time() - start

                total_time += elapsed
                total_pages += result["metadata"]["total_pages"]

                print(
                    f"  {Path(pdf_path).name}: {elapsed:.2f}s "
                    f"({result['metadata']['total_pages']} pages)"
                )

            mem_after = get_memory_usage()

            return {
                "engine": "pdfplumber",
                "total_time": total_time,
                "avg_time": total_time / len(self.pdf_paths),
                "total_pages": total_pages,
                "pages_per_sec": total_pages / total_time if total_time > 0 else 0,
                "memory_used_mb": mem_after - mem_before,
            }

        except Exception as e:
            print(f"  Error: {e}")
            return None

    def benchmark_marker(self) -> Dict:
        """MarkerEngine 벤치마크 (ML Layer)"""
        print("\n===== MarkerEngine (ML Layer) =====")

        try:
            from beanllm.domain.loaders.pdf.engines import MarkerEngine

            engine = MarkerEngine(use_gpu=False, enable_cache=True)
            config = {
                "to_markdown": True,
                "extract_tables": True,
                "extract_images": True,
            }

            total_time = 0
            total_pages = 0
            cache_hits = 0
            mem_before = get_memory_usage()

            # 첫 번째 실행 (캐시 미스)
            print("  First run (no cache):")
            for pdf_path in self.pdf_paths:
                start = time.time()
                result = engine.extract(pdf_path, config)
                elapsed = time.time() - start

                total_time += elapsed
                total_pages += result["metadata"]["total_pages"]
                if result["metadata"].get("from_cache"):
                    cache_hits += 1

                print(
                    f"    {Path(pdf_path).name}: {elapsed:.2f}s "
                    f"({result['metadata']['total_pages']} pages)"
                )

            # 두 번째 실행 (캐시 히트)
            print("  Second run (with cache):")
            cache_time = 0
            for pdf_path in self.pdf_paths:
                start = time.time()
                result = engine.extract(pdf_path, config)
                elapsed = time.time() - start

                cache_time += elapsed
                if result["metadata"].get("from_cache"):
                    cache_hits += 1

                print(
                    f"    {Path(pdf_path).name}: {elapsed:.4f}s "
                    f"(cached: {result['metadata'].get('from_cache', False)})"
                )

            mem_after = get_memory_usage()

            # 캐시 통계
            cache_stats = engine.get_cache_stats()
            print(f"  Cache stats: {cache_stats}")

            return {
                "engine": "marker-pdf",
                "total_time": total_time,
                "avg_time": total_time / len(self.pdf_paths),
                "total_pages": total_pages,
                "pages_per_sec": total_pages / total_time if total_time > 0 else 0,
                "memory_used_mb": mem_after - mem_before,
                "cache_hits": cache_hits,
                "cache_time": cache_time,
                "speedup": total_time / cache_time if cache_time > 0 else 0,
            }

        except ImportError:
            print("  marker-pdf not installed (skip)")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

    def run_all(self) -> None:
        """모든 벤치마크 실행"""
        print("=" * 60)
        print("beanPDFLoader Engine Performance Benchmark")
        print("=" * 60)
        print(f"PDFs: {len(self.pdf_paths)}")
        print(f"Files: {[Path(p).name for p in self.pdf_paths]}")

        # PyMuPDF 벤치마크
        pymupdf_result = self.benchmark_pymupdf()
        if pymupdf_result:
            self.results["pymupdf"] = pymupdf_result

        # pdfplumber 벤치마크
        pdfplumber_result = self.benchmark_pdfplumber()
        if pdfplumber_result:
            self.results["pdfplumber"] = pdfplumber_result

        # marker-pdf 벤치마크
        marker_result = self.benchmark_marker()
        if marker_result:
            self.results["marker"] = marker_result

        # 결과 요약
        self.print_summary()

    def print_summary(self) -> None:
        """벤치마크 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)

        if not self.results:
            print("No results available")
            return

        # 표 헤더
        print(
            f"{'Engine':<12} {'Time(s)':<10} {'Avg(s)':<10} " f"{'Pages/s':<10} {'Memory(MB)':<12}"
        )
        print("-" * 60)

        # 각 엔진 결과
        for name, result in self.results.items():
            if result:
                print(
                    f"{result['engine']:<12} "
                    f"{result['total_time']:<10.2f} "
                    f"{result['avg_time']:<10.2f} "
                    f"{result['pages_per_sec']:<10.2f} "
                    f"{result['memory_used_mb']:<12.2f}"
                )

        # 캐싱 성능 (marker-pdf)
        if "marker" in self.results and self.results["marker"]:
            marker = self.results["marker"]
            if "cache_time" in marker:
                print("\nCaching Performance (marker-pdf):")
                print(f"  First run:  {marker['total_time']:.2f}s")
                print(f"  Cache hit:  {marker['cache_time']:.4f}s")
                print(f"  Speedup:    {marker['speedup']:.1f}x")

        # 속도 비교
        if len(self.results) >= 2:
            print("\nSpeed Comparison:")
            baseline = self.results.get("pymupdf")
            if baseline:
                for name, result in self.results.items():
                    if name != "pymupdf" and result:
                        ratio = result["total_time"] / baseline["total_time"]
                        print(
                            f"  {result['engine']} vs PyMuPDF: "
                            f"{ratio:.2f}x {'slower' if ratio > 1 else 'faster'}"
                        )

        print("=" * 60)


def main():
    """메인 함수"""
    # 테스트 PDF 파일 경로
    pdf_paths = [
        "tests/fixtures/pdf/simple.pdf",
        "tests/fixtures/pdf/tables.pdf",
        "tests/fixtures/pdf/images.pdf",
    ]

    # 존재하는 파일만 필터링
    existing_pdfs = [p for p in pdf_paths if Path(p).exists()]

    if not existing_pdfs:
        print("Error: No test PDF files found")
        print("Expected files:")
        for p in pdf_paths:
            print(f"  - {p}")
        return

    # 벤치마크 실행
    benchmark = EngineBenchmark(existing_pdfs)
    benchmark.run_all()


if __name__ == "__main__":
    main()
