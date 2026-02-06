# 백엔드 실행 가이드

## 🚀 빠른 시작

### 방법 1: Docker 모드 (권장) - 자동 설정

```bash
cd playground/backend

# 1. 환경 변수 설정
cp .env.example .env

# 2. 백엔드 시작 (Docker Compose로 MongoDB/Redis 자동 시작)
./start_backend.sh
```

### 방법 2: 로컬 모드 - 로컬 서비스 사용

```bash
cd playground/backend

# 1. MongoDB와 Redis 설치 및 시작
# macOS:
brew services start mongodb-community@7.0
brew services start redis

# 2. 환경 변수 설정
cp .env.example .env

# 3. 백엔드 시작 (로컬 서비스 사용)
#    - Ollama가 없으면 자동으로 백그라운드에서 시작·추적 (PID: 프로젝트 루트 .ollama-serve.pid)
./start_backend.sh --local
```

**자세한 로컬 모드 설정은 [LOCAL_SETUP.md](./LOCAL_SETUP.md) 참고**

### Docker 모드에서 Ollama

```bash
# docker-compose up -d 시 인프라(MongoDB, Redis, Kafka) + Ollama가 함께 실행·추적됨
# 백엔드: OLLAMA_HOST=http://localhost:11434 로 접근
```

### 방법 3: 수동 실행

```bash
cd playground/backend

# 1. 환경 변수 설정
cp .env.example .env

# 2. 의존성 설치
poetry install --with web

# 3. 인프라 서비스 시작 (Docker Compose, Ollama 포함)
cd ../..
docker-compose up -d

# 4. 백엔드 서버 실행
cd playground/backend
poetry run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. 서버 확인

```bash
# 헬스 체크
curl http://localhost:8000/health

# API 문서
open http://localhost:8000/docs
```

---

## 📋 필수 환경 변수

`.env` 파일에 다음 변수들이 설정되어 있어야 합니다:

```bash
# 배포 모드 (docker 또는 local)
DEPLOYMENT_MODE=docker

# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=true

# MongoDB
# Docker 모드:
MONGODB_URI=mongodb://beanllm:beanllm_secret@localhost:27017/beanllm?authSource=admin
# 로컬 모드:
# MONGODB_URI=mongodb://localhost:27017/beanllm

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# LLM Provider API Keys (최소 1개)
OPENAI_API_KEY=sk-...
# 또는
ANTHROPIC_API_KEY=sk-...
# 또는
OLLAMA_HOST=http://localhost:11434
```

---

## 🔧 문제 해결

### MongoDB 연결 실패
```bash
# MongoDB 상태 확인
docker-compose ps mongodb

# MongoDB 재시작
docker-compose restart mongodb
```

### 포트 충돌
```bash
# 포트 사용 확인
lsof -i :8000

# 다른 포트 사용
uvicorn main:app --port 8001
```

### 의존성 오류
```bash
# Poetry 환경 재생성
poetry env remove python
poetry install
```

---

## 📝 실행 예시

### Docker 모드

```bash
# 1. 백엔드 시작 (자동으로 MongoDB/Redis 시작)
cd playground/backend
./start_backend.sh

# 2. 프론트엔드 실행 (별도 터미널)
cd playground/frontend
pnpm dev
```

### 로컬 모드

```bash
# 1. 로컬 서비스 시작
brew services start mongodb-community@7.0
brew services start redis

# 2. 백엔드 시작
cd playground/backend
./start_backend.sh --local

# 3. 프론트엔드 실행 (별도 터미널)
cd playground/frontend
pnpm dev
```

---

**참고**: 자세한 내용은 `playground/backend/README.md` 참조
