# Poetry Shell 활성화 가이드 (Poetry 2.0+)

Poetry 2.0부터는 `poetry shell` 대신 새로운 방법을 사용합니다.

## 방법 1: `poetry env activate` ⭐ 최신 권장 방법

Poetry 2.0+에서 공식적으로 권장하는 방법입니다:

```bash
# 프로젝트 루트에서
cd /Users/leebeanbin/Downloads/beanllm

# 가상환경 활성화
poetry env activate
```

**주의**: `poetry env activate`는 현재 셸을 활성화하지 않고, 새로운 셸을 시작합니다. 
대신 아래 방법을 사용하세요.

## 방법 2: 직접 가상환경 활성화 (가장 확실) ⭐ 실제 권장

```bash
# 프로젝트 루트에서
cd /Users/leebeanbin/Downloads/beanllm

# 가상환경 경로 확인 및 활성화
source $(poetry env info --path)/bin/activate
```

활성화되면 프롬프트에 `(beanllm-xxx)`가 표시됩니다.

## 방법 3: Poetry run 사용 (가상환경 활성화 없이) ⭐ 간편

```bash
# 프로젝트 루트에서
cd /Users/leebeanbin/Downloads/beanllm

# 백엔드 실행 (가상환경 자동 사용)
cd playground/backend
poetry run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 방법 3: alias 설정 (편의성)

`.zshrc` 또는 `.bashrc`에 다음을 추가:

```bash
# Poetry shell alias
alias poetry-shell='source $(poetry env info --path)/bin/activate'
```

사용:
```bash
poetry-shell
```

## 권장 워크플로우

### 옵션 A: 직접 가상환경 활성화 (가장 일반적) ⭐

```bash
# 1. 프로젝트 루트로 이동
cd /Users/leebeanbin/Downloads/beanllm

# 2. 의존성 설치
poetry install --with web

# 3. 가상환경 활성화
source $(poetry env info --path)/bin/activate

# 4. 백엔드 실행
cd playground/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 옵션 B: Poetry run 사용 (가장 간단)

```bash
# 1. 프로젝트 루트로 이동
cd /Users/leebeanbin/Downloads/beanllm

# 2. 의존성 설치
poetry install --with web

# 3. 백엔드 실행 (가상환경 자동 사용)
cd playground/backend
poetry run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 빠른 참조

```bash
# 가상환경 경로 확인
poetry env info --path

# 가상환경 활성화 (Poetry 2.0+ 권장 방법)
source $(poetry env info --path)/bin/activate

# Poetry run으로 실행 (활성화 불필요)
poetry run python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 가상환경 비활성화
deactivate
```

## alias 설정 (편의성)

`.zshrc` 또는 `.bashrc`에 추가:

```bash
# Poetry shell alias (Poetry 2.0+ 호환)
alias poetry-shell='source $(poetry env info --path)/bin/activate'
```

사용:
```bash
poetry-shell  # 이제 poetry shell처럼 사용 가능
```
