# 📦 PyPI 배포 가이드 (2025년 최신)

이 문서는 beanllm 패키지를 PyPI에 배포하는 최신 방법을 설명합니다.

## 📋 목차

1. [사전 준비](#사전-준비)
2. [배포 방법](#배포-방법)
   - [방법 1: 자동 배포 스크립트 (권장)](#방법-1-자동-배포-스크립트-권장)
   - [방법 2: 수동 배포](#방법-2-수동-배포)
   - [방법 3: GitHub Actions 자동화](#방법-3-github-actions-자동화)
3. [버전 관리](#버전-관리)
4. [문제 해결](#문제-해결)

---

## 사전 준비

### 1. PyPI 계정 및 API 토큰

#### PyPI 계정 생성
1. [PyPI](https://pypi.org/account/register/)에서 계정 생성
2. [TestPyPI](https://test.pypi.org/account/register/)에서 테스트 계정 생성 (선택사항, 권장)

#### API 토큰 생성 ⚠️ 중요
**2025년 현재 username/password 방식은 deprecated되었으며, API 토큰만 지원됩니다.**

1. PyPI 로그인 → **Account settings** → **API tokens**
2. **Add API token** 클릭
3. **Scope 선택**:
   - `Entire account`: 모든 프로젝트에 사용 가능
   - `Project: beanllm`: beanllm 프로젝트만 (첫 배포 후 선택 가능)
4. 토큰 복사 (⚠️ 한 번만 표시되므로 안전하게 보관)

### 2. 로컬 환경 설정

#### `.pypirc` 파일 생성

홈 디렉토리(`~/.pypirc`)에 다음 내용으로 파일 생성:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**보안 설정** (중요):
```bash
chmod 600 ~/.pypirc
```

✅ **이미 설정 완료**: `.pypirc` 파일이 생성되어 있습니다.

### 3. 필수 도구 설치

```bash
# 최신 배포 도구 설치
pip install --upgrade build twine
```

---

## 배포 방법

### 방법 1: 자동 배포 스크립트 (권장) ⭐

프로젝트 루트에 `publish.sh` 스크립트가 준비되어 있습니다.

#### TestPyPI에 테스트 배포

```bash
# 테스트 배포
./publish.sh test

# TestPyPI에서 설치 테스트
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            beanllm
```

#### 본 PyPI에 배포

```bash
# 본 배포 (주의: 버전 되돌리기 불가)
./publish.sh prod
```

**스크립트가 자동으로 수행하는 작업:**
1. ✅ 이전 빌드 파일 정리
2. ✅ 코드 린트 체크 (ruff)
3. ✅ 테스트 실행 (선택)
4. ✅ 패키지 빌드
5. ✅ TestPyPI 또는 PyPI에 업로드
6. ✅ 설치 방법 안내

---

### 방법 2: 수동 배포

#### Step 1: 이전 빌드 정리

```bash
# 이전 빌드 파일 삭제
rm -rf dist/ build/ *.egg-info src/*.egg-info
```

#### Step 2: 패키지 빌드

```bash
# 최신 build 도구 사용 (PEP 517/518)
python -m build
```

빌드 결과물:
- `dist/beanllm-0.1.0.tar.gz` - 소스 배포 (source distribution)
- `dist/beanllm-0.1.0-py3-none-any.whl` - 휠 배포 (wheel distribution)

#### Step 3: 빌드 검증

```bash
# 빌드 파일 검증
python -m twine check dist/*
```

#### Step 4: TestPyPI 배포 (권장)

```bash
# TestPyPI에 업로드
python -m twine upload --repository testpypi dist/*

# TestPyPI에서 설치 테스트
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            beanllm[all]

# CLI 테스트
beanllm list
beanllm --version
```

#### Step 5: PyPI 배포

```bash
# 본 PyPI에 업로드
python -m twine upload dist/*

# 확인
pip install beanllm
beanllm --version
```

**배포 후 확인:**
- PyPI 페이지: https://pypi.org/project/beanllm/
- 설치 테스트: `pip install beanllm[all]`

---

### 방법 3: GitHub Actions 자동화

#### 옵션 A: Trusted Publishers (권장, API 토큰 불필요) 🆕

**2023년부터 지원되는 최신 방식으로, API 토큰 없이 배포 가능합니다.**

##### 1. PyPI에서 Trusted Publisher 설정

1. PyPI 계정 설정 → **Publishing** → **Add a new publisher**
2. 다음 정보 입력:
   - PyPI Project Name: `beanllm`
   - Owner: `leebeanbin`
   - Repository name: `beanllm`
   - Workflow name: `publish.yml`
   - Environment name: `release` (선택사항)

##### 2. GitHub Actions Workflow 생성

`.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  pypi-publish:
    name: Upload release to PyPI
    runs-on: ubuntu-latest
    environment:
      name: release
      url: https://pypi.org/project/beanllm/
    permissions:
      id-token: write  # OIDC 토큰 발급을 위해 필수

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

##### 3. 배포 프로세스

```bash
# 1. 버전 업데이트
# pyproject.toml에서 version = "0.1.1" 등으로 수정

# 2. 커밋 및 푸시
git add pyproject.toml
git commit -m "Bump version to 0.1.1"
git push origin main

# 3. GitHub Release 생성
git tag v0.1.1
git push origin v0.1.1

# 또는 GitHub 웹 UI에서 Release 생성
# → GitHub Actions가 자동으로 PyPI에 배포
```

#### 옵션 B: API 토큰 사용 (기존 방식)

##### 1. GitHub Secrets 설정

1. GitHub 저장소 → **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret**:
   - Name: `PYPI_API_TOKEN`
   - Value: PyPI API 토큰

##### 2. GitHub Actions Workflow

`.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  pypi-publish:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
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

---

## 버전 관리

### 버전 형식 (Semantic Versioning)

`pyproject.toml`에서 관리:

```toml
[project]
version = "0.1.0"  # MAJOR.MINOR.PATCH
```

### 버전 업데이트 규칙

- **MAJOR** (X.0.0): 호환되지 않는 API 변경
  - 예: `1.0.0` → `2.0.0`
- **MINOR** (0.X.0): 하위 호환 기능 추가
  - 예: `0.1.0` → `0.2.0`
- **PATCH** (0.0.X): 버그 수정
  - 예: `0.1.0` → `0.1.1`

### 개발 버전 (선택사항)

```toml
version = "0.1.0a1"  # 알파 버전
version = "0.1.0b1"  # 베타 버전
version = "0.1.0rc1" # Release Candidate
```

### 버전 업데이트 워크플로우

```bash
# 1. pyproject.toml 수정
vim pyproject.toml
# version = "0.1.1"

# 2. 변경사항 커밋
git add pyproject.toml
git commit -m "chore: bump version to 0.1.1"

# 3. 태그 생성 및 푸시
git tag v0.1.1
git push origin main --tags

# 4. GitHub Release 생성 (선택)
# GitHub UI에서 Release 생성 또는 gh CLI 사용
gh release create v0.1.1 --generate-notes
```

---

## 문제 해결

### 1. 패키지 이름 충돌

**증상**: `The name 'beanllm' is already taken`

**해결**:
- PyPI에서 패키지 이름 검색: https://pypi.org/search/?q=beanllm
- 이름이 이미 존재하면 `pyproject.toml`에서 `name` 변경

### 2. 빌드 오류

**증상**: `error: invalid command 'bdist_wheel'`

**해결**:
```bash
# 캐시 및 빌드 파일 정리
rm -rf build/ dist/ *.egg-info src/*.egg-info

# 최신 도구 재설치
pip install --upgrade build wheel setuptools

# 재빌드
python -m build
```

### 3. 업로드 인증 오류

**증상**: `403 Forbidden` 또는 `Invalid or non-existent authentication information`

**해결**:
```bash
# .pypirc 파일 확인
cat ~/.pypirc

# 파일 권한 확인
ls -la ~/.pypirc  # -rw------- (600) 이어야 함

# 토큰 확인 (username은 반드시 __token__)
# password는 pypi-로 시작해야 함

# 수동 인증으로 테스트
python -m twine upload --verbose dist/*
```

### 4. 의존성 오류

**증상**: 설치 시 의존성 충돌

**해결**:
```bash
# pyproject.toml에서 의존성 버전 확인
# 너무 엄격한 버전 제한은 피하기

# 예시 (좋음)
dependencies = [
    "httpx>=0.24.0",
    "tiktoken>=0.5.0",
]

# 예시 (나쁨 - 너무 엄격)
dependencies = [
    "httpx==0.24.0",  # 다른 패키지와 충돌 가능
]
```

### 5. README 렌더링 오류

**증상**: PyPI에서 README가 제대로 표시되지 않음

**해결**:
```bash
# README 검증
python -m twine check dist/*

# Markdown 문법 확인
# GitHub에서 제대로 보이면 대부분 PyPI에서도 정상 작동
```

### 6. 버전 업데이트 안 됨

**증상**: 새 버전을 올렸는데 이전 버전이 설치됨

**해결**:
```bash
# ⚠️ PyPI에 업로드한 버전은 삭제하거나 덮어쓸 수 없음
# 반드시 pyproject.toml의 version을 업데이트해야 함

# 캐시 정리 후 재설치
pip cache purge
pip install --upgrade --no-cache-dir beanllm
```

---

## 체크리스트

배포 전 최종 확인:

- [ ] `pyproject.toml`의 버전 업데이트
- [ ] `README.md` 최신화
- [ ] `LICENSE` 파일 존재 확인
- [ ] 테스트 통과 (`pytest`)
- [ ] 린트 체크 (`ruff check`)
- [ ] TestPyPI에서 테스트 배포
- [ ] TestPyPI에서 설치 및 동작 확인
- [ ] Git 태그 생성 및 푸시
- [ ] PyPI 배포
- [ ] PyPI에서 설치 및 동작 확인

---

## 유용한 명령어

```bash
# 현재 버전 확인
grep version pyproject.toml

# 빌드 파일 크기 확인
ls -lh dist/

# PyPI에 등록된 버전 확인
pip index versions beanllm

# 패키지 정보 확인
pip show beanllm

# 설치된 버전 업그레이드
pip install --upgrade beanllm

# 특정 버전 설치
pip install beanllm==0.1.0

# extras와 함께 설치
pip install beanllm[all]
pip install beanllm[openai,anthropic]
```

---

## 참고 자료

### 공식 문서
- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Documentation](https://pypi.org/help/)
- [PEP 517 - Build System](https://peps.python.org/pep-0517/)
- [PEP 518 - pyproject.toml](https://peps.python.org/pep-0518/)
- [Twine Documentation](https://twine.readthedocs.io/)

### 최신 기능
- [Trusted Publishers Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for PyPI](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

### 도구
- [build](https://build.pypa.io/) - 최신 빌드 도구
- [twine](https://twine.readthedocs.io/) - PyPI 업로드 도구
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) - GitHub Actions

---

## 빠른 시작

```bash
# 1. 도구 설치
pip install --upgrade build twine

# 2. 테스트 배포 (스크립트 사용)
./publish.sh test

# 3. 본 배포 (스크립트 사용)
./publish.sh prod

# 또는 수동 배포
python -m build
python -m twine upload dist/*
```

---

**마지막 업데이트**: 2025년 12월 24일
**beanllm 버전**: 0.1.0
