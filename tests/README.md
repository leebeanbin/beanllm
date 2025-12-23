# 🧪 llmkit 테스트 가이드

## 📋 테스트 구조

```
tests/
├── __init__.py
├── conftest.py              # 공통 fixtures
├── test_import.py           # Import 테스트
├── test_config.py           # Config 테스트
├── test_registry.py         # Registry 테스트
├── test_text_splitters.py   # Text Splitter 테스트
├── test_cli.py              # CLI 테스트
├── test_domain.py           # Domain Layer 테스트
├── test_infrastructure.py   # Infrastructure Layer 테스트
├── test_facade.py           # Facade Layer 테스트
├── test_utils.py            # Utils 테스트
├── test_integration.py      # Integration 테스트
├── test_e2e.py              # End-to-End 테스트
└── run_*.py                 # 개별 테스트 실행 스크립트
```

---

## 🚀 테스트 실행

### 전체 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 상세 출력
pytest -v

# 커버리지 포함
pytest --cov=src.llmkit --cov-report=html
```

### 특정 테스트 실행

```bash
# CLI 테스트만
pytest tests/test_cli.py -v

# Domain 레이어 테스트만
pytest tests/test_domain.py -v

# 특정 테스트 함수
pytest tests/test_cli.py::TestCLIBasic::test_cli_list_command -v
```

### Makefile 사용

```bash
# 테스트 실행
make test

# 테스트 + 커버리지
make test-cov
```

---

## 📝 테스트 카테고리

### 1. Unit Tests (단위 테스트)

#### `test_import.py`
- 모듈 import 테스트
- 기본 클래스/함수 존재 확인

#### `test_config.py`
- Config 클래스 테스트
- EnvConfig 테스트

#### `test_registry.py`
- ModelRegistry 테스트
- 모델 정보 조회 테스트

#### `test_text_splitters.py`
- TextSplitter 구현체 테스트
- 다양한 전략 테스트

#### `test_domain.py`
- Domain Layer 엔티티 테스트
- Document, Embedding, VectorStore 등

#### `test_infrastructure.py`
- Infrastructure Layer 테스트
- ModelRegistry, ParameterAdapter 등

#### `test_utils.py`
- Utils 함수 테스트
- Config, Logger, Retry 등

### 2. Integration Tests (통합 테스트)

#### `test_integration.py`
- 레이어 간 통합 테스트
- Facade → Handler → Service → Domain

### 3. CLI Tests

#### `test_cli.py`
- CLI 명령어 테스트
- list, show, providers, export, summary, scan, analyze

### 4. Facade Tests

#### `test_facade.py`
- Facade API 테스트
- Client, RAGChain, Agent, Graph 등

### 5. End-to-End Tests

#### `test_e2e.py`
- 전체 워크플로우 테스트
- 실제 사용 시나리오

---

## 🔧 Fixtures

### `conftest.py`에 정의된 Fixtures

- `temp_dir`: 임시 디렉토리
- `sample_text`: 샘플 텍스트
- `sample_documents`: 샘플 Document 리스트
- `mock_env`: Mock 환경 변수
- `skip_if_no_provider`: Provider 없으면 스킵
- `mock_client`: Mock Client
- `sample_text_long`: 긴 샘플 텍스트

---

## 📊 테스트 전략

### 1. Provider 의존성 처리

Provider가 없어도 테스트가 실행되도록 처리:

```python
try:
    client = Client(model="gpt-4o-mini")
    # 테스트 코드
except (ValueError, ImportError):
    pytest.skip("Provider not available")
```

### 2. Mock 사용

외부 API 호출은 Mock으로 처리:

```python
from unittest.mock import MagicMock, patch

@patch('llmkit._source_providers.openai_provider.AsyncOpenAI')
def test_with_mock(mock_openai):
    # Mock 설정
    # 테스트 실행
```

### 3. 임시 파일 사용

`temp_dir` fixture를 사용하여 임시 파일 생성:

```python
def test_with_file(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")
    # 테스트 실행
```

---

## 🎯 테스트 커버리지 목표

- **Unit Tests**: 각 레이어별 80% 이상
- **Integration Tests**: 주요 워크플로우 100%
- **CLI Tests**: 모든 명령어 100%
- **E2E Tests**: 주요 사용 사례 100%

---

## 🐛 문제 해결

### Import 오류

```bash
# 프로젝트 루트에서 실행
cd /Users/leejungbin/Downloads/llmkit
python -m pytest tests/
```

### Provider 오류

Provider가 없어도 테스트는 실행되어야 합니다. 스킵되는 테스트는 정상입니다.

### 환경 변수 오류

`.env` 파일이 없어도 테스트는 실행되어야 합니다. Mock 환경 변수를 사용합니다.

---

## 📈 테스트 실행 예시

```bash
# 전체 테스트
$ pytest
======================== test session starts ========================
tests/test_import.py::test_import_registry PASSED
tests/test_config.py::test_env_config_exists PASSED
tests/test_cli.py::TestCLIBasic::test_cli_list_command PASSED
...
======================== 50 passed in 2.34s ========================

# 커버리지 포함
$ pytest --cov=src.llmkit --cov-report=term
======================== test session starts ========================
...
----------- coverage: platform darwin, python 3.11 -----------
Name                                    Stmts   Miss  Cover
------------------------------------------------------------
src/llmkit/__init__.py                    823     45    95%
src/llmkit/domain/__init__.py             443     12    97%
...
------------------------------------------------------------
TOTAL                                    5000    200    96%
```

---

## 🔄 CI/CD 통합

GitHub Actions에서 자동 실행:

```yaml
- name: Run tests
  run: pytest --cov=src.llmkit --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

---

**상세 가이드**: [docs/guides/TESTING_GUIDE.md](../docs/guides/TESTING_GUIDE.md) 참고

---

**최종 업데이트**: 2025-12-22

