# PyPI 배포 가이드

이 문서는 llmkit 패키지를 PyPI에 배포하는 방법을 설명합니다.

## 사전 준비

### 1. PyPI 계정 생성

1. [PyPI](https://pypi.org/account/register/)에서 계정 생성
2. [TestPyPI](https://test.pypi.org/account/register/)에서 테스트 계정 생성 (선택사항)

### 2. API 토큰 생성

1. PyPI 로그인 후 **Account settings** → **API tokens** 이동
2. **Add API token** 클릭
3. Scope: **Entire account** 또는 **Project: llmkit** 선택
4. 토큰 복사 (한 번만 표시됨)

### 3. 환경 변수 설정

```bash
# ~/.pypirc 파일 생성 (선택사항)
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

[testpypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

또는 환경 변수로 설정:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 배포 단계

### 1. 패키지 빌드

```bash
# 빌드 도구 설치
python -m pip install --upgrade build twine

# 패키지 빌드 (source + wheel)
python -m build
```

빌드 결과물:
- `dist/llmkit-0.1.0.tar.gz` (소스 배포)
- `dist/llmkit-0.1.0-py3-none-any.whl` (wheel 배포)

### 2. 빌드 검증 (선택사항)

```bash
# 빌드 파일 검증
twine check dist/*
```

### 3. TestPyPI에 테스트 배포 (권장)

```bash
# TestPyPI에 업로드
twine upload --repository testpypi dist/*

# 테스트 설치
python -m pip install --index-url https://test.pypi.org/simple/ llmkit
```

### 4. PyPI에 배포

```bash
# PyPI에 업로드
twine upload dist/*
```

### 5. 설치 확인

```bash
# PyPI에서 설치
python -m pip install llmkit

# CLI 테스트
llmkit list
```

## 자동화 배포 (GitHub Actions)

### 1. GitHub Secrets 설정

1. GitHub 저장소 → **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret** 추가:
   - Name: `PYPI_API_TOKEN`
   - Value: PyPI API 토큰

### 2. GitHub Actions Workflow 생성

`.github/workflows/publish.yml` 파일 생성:

```yaml
name: Publish Python Package

on:
  release:
    types: [created]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Check package
        run: twine check dist/*
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### 3. 배포 프로세스

1. 버전 업데이트: `pyproject.toml`에서 `version` 수정
2. 변경사항 커밋 및 푸시
3. GitHub에서 **Release** 생성
4. GitHub Actions가 자동으로 빌드 및 배포

## 버전 관리

### 버전 형식

`pyproject.toml`에서 버전 관리:

```toml
[project]
version = "0.1.0"  # MAJOR.MINOR.PATCH
```

### 버전 업데이트 규칙

- **MAJOR**: 호환되지 않는 API 변경
- **MINOR**: 하위 호환 기능 추가
- **PATCH**: 버그 수정

### 버전 업데이트 예시

```bash
# pyproject.toml 수정
version = "0.1.1"  # 패치 버전

# 커밋 및 태그
git add pyproject.toml
git commit -m "Bump version to 0.1.1"
git tag v0.1.1
git push origin main --tags
```

## 문제 해결

### 1. 패키지 이름 충돌

PyPI에 이미 같은 이름의 패키지가 있는 경우:
- `pyproject.toml`에서 `name` 변경
- 또는 PyPI에서 패키지 이름 변경 요청

### 2. 빌드 오류

```bash
# 캐시 정리 후 재빌드
rm -rf build/ dist/ *.egg-info
python -m build
```

### 3. 업로드 오류

```bash
# 토큰 확인
echo $TWINE_PASSWORD

# 수동 인증
twine upload dist/* --verbose
```

## 참고 자료

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Documentation](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/guides/building-and-testing-python)
