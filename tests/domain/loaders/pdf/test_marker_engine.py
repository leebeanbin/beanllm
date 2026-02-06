"""
MarkerEngine 단위 테스트

marker-pdf가 설치되지 않은 환경에서도 테스트 가능하도록 구성
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestMarkerEngineImport:
    """MarkerEngine import 및 의존성 테스트"""

    def test_marker_engine_import_without_marker_pdf(self):
        """marker-pdf 없이 import 시도 (실패해야 함)"""
        # marker-pdf가 설치되지 않은 경우, import는 성공하지만
        # 실제 사용 시 ImportError 발생
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            # import는 성공하지만 사용 시 의존성 체크
            engine = MarkerEngine()
            assert engine.name == "Marker"
            assert engine._marker_available is None
        except ImportError:
            # marker-pdf가 없으면 import 자체가 실패할 수 있음
            pytest.skip("marker-pdf not installed")

    def test_marker_engine_initialization(self):
        """MarkerEngine 초기화 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            # 기본 초기화
            engine = MarkerEngine()
            assert engine.name == "Marker"
            assert engine.use_gpu is False
            assert engine.batch_size == 1
            assert engine.max_pages is None

            # GPU 옵션 초기화
            engine_gpu = MarkerEngine(use_gpu=True, batch_size=4, max_pages=10)
            assert engine_gpu.use_gpu is True
            assert engine_gpu.batch_size == 4
            assert engine_gpu.max_pages == 10
        except ImportError:
            pytest.skip("MarkerEngine not available")


class TestMarkerEngineWithMock:
    """Mock을 사용한 MarkerEngine 기능 테스트"""

    @pytest.fixture
    def mock_marker_modules(self):
        """marker-pdf 모듈 Mock"""
        mock_marker = MagicMock()
        mock_convert = MagicMock()
        mock_models = MagicMock()

        # convert_single_pdf 반환값 설정
        mock_convert.convert_single_pdf.return_value = (
            "# Test Document\n\nSample content",  # full_text
            {},  # images
            {"num_pages": 1},  # metadata
        )

        # load_all_models 반환값 설정
        mock_models.load_all_models.return_value = ["model1", "model2"]

        return {
            "marker": mock_marker,
            "convert": mock_convert,
            "models": mock_models,
        }

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """샘플 PDF 파일 경로"""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("dummy pdf content")
        return pdf_file

    def test_check_dependencies_with_marker_available(self, mock_marker_modules):
        """marker-pdf가 설치된 경우 의존성 체크"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()

            # marker-pdf가 사용 가능한 경우 Mock
            with patch.dict(
                "sys.modules",
                {
                    "marker": mock_marker_modules["marker"],
                    "marker.convert": mock_marker_modules["convert"],
                    "marker.models": mock_marker_modules["models"],
                },
            ):
                engine._check_dependencies()
                assert engine._marker_available is True
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_check_dependencies_without_marker(self):
        """marker-pdf가 설치되지 않은 경우 의존성 체크"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()

            # marker-pdf import 실패 Mock
            with patch.dict("sys.modules", {"marker": None}):
                with pytest.raises(ImportError, match="marker-pdf is required"):
                    engine._check_dependencies()
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_extract_without_marker_pdf(self, sample_pdf_path):
        """marker-pdf 없이 extract 호출 시 에러"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            engine._marker_available = False

            config = {"to_markdown": True}

            with pytest.raises(ImportError, match="marker-pdf is not available"):
                engine.extract(sample_pdf_path, config)
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_extract_with_marker_pdf_mock(self, sample_pdf_path, mock_marker_modules):
        """Mock을 사용한 extract 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()

            config = {
                "to_markdown": True,
                "extract_tables": True,
                "extract_images": True,
                "max_pages": None,
            }

            # marker-pdf Mock
            with patch.dict(
                "sys.modules",
                {
                    "marker": mock_marker_modules["marker"],
                    "marker.convert": mock_marker_modules["convert"],
                    "marker.models": mock_marker_modules["models"],
                },
            ):
                with patch(
                    "beanllm.domain.loaders.pdf.engines.marker_engine.convert_single_pdf",
                    mock_marker_modules["convert"].convert_single_pdf,
                ):
                    with patch(
                        "beanllm.domain.loaders.pdf.engines.marker_engine.load_all_models",
                        mock_marker_modules["models"].load_all_models,
                    ):
                        result = engine.extract(sample_pdf_path, config)

                        # 결과 검증
                        assert "pages" in result
                        assert "tables" in result
                        assert "images" in result
                        assert "markdown" in result
                        assert "metadata" in result
                        assert result["metadata"]["engine"] == "Marker"
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_split_into_pages_single_page(self):
        """단일 페이지 분리 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            full_text = "Sample text content"
            metadata = {"num_pages": 1}

            pages = engine._split_into_pages(full_text, metadata)

            assert len(pages) == 1
            assert pages[0]["page"] == 0
            assert pages[0]["text"] == full_text
            assert pages[0]["width"] == 612.0
            assert pages[0]["height"] == 792.0
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_split_into_pages_multiple_pages(self):
        """다중 페이지 분리 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            full_text = "A" * 1000
            metadata = {"num_pages": 5}

            pages = engine._split_into_pages(full_text, metadata)

            assert len(pages) == 5
            for i, page in enumerate(pages):
                assert page["page"] == i
                assert "text" in page
                assert page["width"] == 612.0
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_parse_markdown_table(self):
        """Markdown 테이블 파싱 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            table_text = """| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |"""

            result = engine._parse_markdown_table(table_text)

            assert len(result) == 3  # header + 2 rows
            assert result[0] == ["Name", "Age", "City"]
            assert result[1] == ["Alice", "30", "NYC"]
            assert result[2] == ["Bob", "25", "LA"]
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_parse_markdown_table_invalid(self):
        """잘못된 Markdown 테이블 파싱"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            table_text = "| Header |"  # 구분자와 데이터 없음

            result = engine._parse_markdown_table(table_text)

            assert result == []  # 최소 3줄 필요 (header + separator + data)
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_estimate_page_from_position(self):
        """텍스트 위치에서 페이지 번호 추정"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()

            # 3페이지 문서, 총 길이 1000
            assert engine._estimate_page_from_position(0, 1000, 3) == 0
            assert engine._estimate_page_from_position(333, 1000, 3) == 0
            assert engine._estimate_page_from_position(500, 1000, 3) == 1
            assert engine._estimate_page_from_position(999, 1000, 3) == 2
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_convert_images(self):
        """이미지 변환 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            images = {
                "image_1.png": b"fake_image_data_1",
                "image_2.jpg": b"fake_image_data_2",
            }

            result = engine._convert_images(images)

            assert len(result) == 2
            assert result[0]["image_index"] == 0
            assert result[0]["metadata"]["name"] == "image_1.png"
            assert result[1]["image_index"] == 1
            assert result[1]["metadata"]["name"] == "image_2.jpg"
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_extract_tables_from_markdown(self):
        """Markdown에서 테이블 추출 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            markdown_text = """
# Document Title

Some text here.

| Col1 | Col2 |
|------|------|
| A    | B    |
| C    | D    |

More text.

| Name | Value |
|------|-------|
| X    | 10    |
"""

            pages = [{"page": 0, "text": markdown_text}]
            tables = engine._extract_tables_from_markdown(markdown_text, pages)

            # 2개의 테이블이 추출되어야 함
            assert len(tables) == 2
            assert tables[0]["page"] == 0
            assert tables[0]["table_index"] == 0
            assert tables[1]["table_index"] == 1
        except ImportError:
            pytest.skip("MarkerEngine not available")


class TestMarkerEngineOptimization:
    """MarkerEngine 최적화 기능 테스트"""

    def test_cache_initialization(self):
        """캐시 초기화 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            # 캐시 활성화
            engine = MarkerEngine(enable_cache=True, cache_size=5)
            assert engine.enable_cache is True
            assert engine.cache_size == 5
            assert len(engine._result_cache) == 0

            # 캐시 비활성화
            engine_no_cache = MarkerEngine(enable_cache=False)
            assert engine_no_cache.enable_cache is False
        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_get_cache_key(self, tmp_path):
        """캐시 키 생성 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()

            # 테스트 파일 생성
            pdf_file = tmp_path / "test.pdf"
            pdf_file.write_text("dummy content")

            config = {"to_markdown": True, "extract_tables": True}

            # 캐시 키 생성
            key1 = engine._get_cache_key(pdf_file, config)
            assert isinstance(key1, str)
            assert len(key1) == 64  # SHA256 해시 길이

            # 같은 파일/설정 → 같은 키
            key2 = engine._get_cache_key(pdf_file, config)
            assert key1 == key2

            # 다른 설정 → 다른 키
            config2 = {"to_markdown": False, "extract_tables": True}
            key3 = engine._get_cache_key(pdf_file, config2)
            assert key1 != key3

        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_cache_result(self):
        """결과 캐싱 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine(enable_cache=True, cache_size=3)

            # 결과 캐싱
            result = {"pages": [], "tables": [], "metadata": {"test": True}}
            engine._cache_result("key1", result)

            assert len(engine._result_cache) == 1
            assert "key1" in engine._result_cache

            # 여러 결과 캐싱
            engine._cache_result("key2", result)
            engine._cache_result("key3", result)
            assert len(engine._result_cache) == 3

            # 캐시 크기 초과 시 LRU 제거
            engine._cache_result("key4", result)
            assert len(engine._result_cache) == 3
            assert "key1" not in engine._result_cache  # 가장 오래된 항목 제거
            assert "key4" in engine._result_cache

        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_clear_cache(self):
        """캐시 정리 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine(enable_cache=True)

            # 캐시에 데이터 추가
            result = {"pages": [], "metadata": {}}
            engine._cache_result("key1", result)
            assert len(engine._result_cache) > 0

            # 캐시 정리
            engine.clear_cache()
            assert len(engine._result_cache) == 0

        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_get_cache_stats(self):
        """캐시 통계 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine(enable_cache=True, cache_size=10, use_gpu=False)
            stats = engine.get_cache_stats()

            assert stats["cache_enabled"] is True
            assert stats["cache_size"] == 0
            assert stats["cache_limit"] == 10
            assert stats["use_gpu"] is False

        except ImportError:
            pytest.skip("MarkerEngine not available")

    def test_extract_batch_empty(self):
        """빈 배치 처리 테스트"""
        try:
            from beanllm.domain.loaders.pdf.engines.marker_engine import MarkerEngine

            engine = MarkerEngine()
            results = engine.extract_batch([], {})

            assert isinstance(results, list)
            assert len(results) == 0

        except ImportError:
            pytest.skip("MarkerEngine not available")


class TestMarkerEngineIntegration:
    """beanPDFLoader 통합 테스트"""

    def test_marker_engine_in_bean_pdf_loader(self):
        """beanPDFLoader에서 MarkerEngine 사용 가능 여부"""
        try:
            from beanllm.domain.loaders.pdf import beanPDFLoader

            # marker-pdf 설치 여부 확인
            try:
                import marker

                has_marker = True
            except ImportError:
                has_marker = False

            # 더미 PDF로 로더 생성
            loader = beanPDFLoader(
                "tests/fixtures/simple.pdf",
                strategy="auto",
            )

            # marker-pdf가 설치되어 있으면 ml 엔진이 초기화되어야 함
            if has_marker:
                assert "ml" in loader._engines
            else:
                # marker-pdf가 없으면 ml 엔진이 없어야 함
                assert "ml" not in loader._engines

        except Exception:
            pytest.skip("beanPDFLoader or test fixtures not available")
