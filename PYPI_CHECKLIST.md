# PyPI 배포 체크리스트

블로그 (https://teddylee777.github.io/python/pypi/) 기준으로 확인한 사항들입니다.

## ✅ 완료된 사항

### 1. 프로젝트 구조
- ✅ `src/` 레이아웃 사용 (`src/llmkit/`)
- ✅ `pyproject.toml` 사용 (최신 표준)
- ✅ `setup.py` 없음 (pyproject.toml로 대체)

### 2. 패키지 설정
- ✅ `[tool.setuptools.packages.find]` 사용하여 자동으로 모든 패키지 포함
- ✅ 총 42개 패키지 자동 감지
- ✅ `package-dir = {"" = "src"}` 설정

### 3. 의존성 관리
- ✅ 필수 의존성: `dependencies` 섹션
- ✅ 선택적 의존성: `[project.optional-dependencies]` 섹션
  - `openai`, `anthropic`, `gemini`, `ollama`, `all`, `dev`

### 4. 메타데이터
- ✅ `name = "llmkit"`
- ✅ `version = "0.1.0"`
- ✅ `description` 설정
- ✅ `readme = "README.md"`
- ✅ `requires-python = ">=3.11"`
- ✅ `license = {text = "MIT"}`
- ✅ `authors` 설정 (수정 필요: 실제 이름/이메일)
- ✅ `keywords` 설정
- ✅ `classifiers` 설정
- ✅ `[project.urls]` 설정 (수정 필요: 실제 GitHub URL)

### 5. CLI 진입점
- ✅ `[project.scripts]` 설정
- ✅ `llmkit = "llmkit.utils.cli.cli:main"`

### 6. 빌드 시스템
- ✅ `[build-system]` 설정
- ✅ `requires = ["setuptools>=61.0", "wheel"]`
- ✅ `build-backend = "setuptools.build_meta"`

## ⚠️ 수정 필요 사항

### 1. authors 정보
```toml
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
```
→ 실제 이름과 이메일로 변경 필요

### 2. project.urls
```toml
[project.urls]
Homepage = "https://github.com/yourusername/llmkit"
Documentation = "https://github.com/yourusername/llmkit#readme"
Repository = "https://github.com/yourusername/llmkit"
"Bug Tracker" = "https://github.com/yourusername/llmkit/issues"
```
→ 실제 GitHub 저장소 URL로 변경 필요

## 📋 배포 전 최종 확인

### 1. 빌드 테스트
```bash
# 빌드 도구 설치
python -m pip install --upgrade build twine

# 패키지 빌드
python -m build

# 빌드 결과 확인
ls -la dist/
# dist/llmkit-0.1.0.tar.gz
# dist/llmkit-0.1.0-py3-none-any.whl
```

### 2. 빌드 검증
```bash
# 빌드 파일 검증
twine check dist/*
```

### 3. 설치 테스트
```bash
# 로컬에서 설치 테스트
pip install dist/llmkit-0.1.0-py3-none-any.whl

# CLI 테스트
llmkit list

# Python에서 import 테스트
python -c "from llmkit import Client; print('OK')"
```

### 4. TestPyPI 배포 (권장)
```bash
# TestPyPI에 업로드
twine upload --repository testpypi dist/*

# TestPyPI에서 설치 테스트
pip install --index-url https://test.pypi.org/simple/ llmkit
```

### 5. PyPI 배포
```bash
# PyPI에 업로드
twine upload dist/*
```

## 🔧 블로그와의 차이점

블로그는 `setup.py`를 사용하지만, 이 프로젝트는 **최신 표준인 `pyproject.toml`**을 사용합니다.

### setup.py vs pyproject.toml

**블로그 방식 (구식):**
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="llmkit",
    version="0.1.0",
    packages=find_packages(),
    ...
)
```

**현재 프로젝트 (최신 표준):**
```toml
# pyproject.toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["llmkit*"]
```

**장점:**
- ✅ PEP 517/518 표준 준수
- ✅ 모든 빌드 도구와 호환 (setuptools, poetry, flit 등)
- ✅ 단일 파일로 모든 설정 관리
- ✅ 더 간결하고 유지보수 용이

## 📝 배포 순서

1. **pyproject.toml 수정**
   - authors 정보 업데이트
   - project.urls 업데이트

2. **빌드 및 검증**
   ```bash
   python -m build
   twine check dist/*
   ```

3. **TestPyPI 테스트 배포**
   ```bash
   twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ llmkit
   ```

4. **PyPI 배포**
   ```bash
   twine upload dist/*
   ```

5. **GitHub Release 생성** (자동 배포 사용 시)
   - GitHub에서 Release 생성
   - GitHub Actions가 자동으로 배포

## 🔗 참고 자료

- 블로그: https://teddylee777.github.io/python/pypi/
- PyPI 공식 문서: https://packaging.python.org/
- PEP 517: https://peps.python.org/pep-0517/
- PEP 518: https://peps.python.org/pep-0518/


