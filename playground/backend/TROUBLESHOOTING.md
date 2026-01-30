# 백엔드 문제 해결 가이드

## 경고 및 오류 해소 방법

### 1. MongoDB 연결 실패

**증상:**
```
❌ MongoDB ping failed: localhost:27017: [Errno 61] Connection refused
⚠️  MongoDB not available - chat history will not be saved
```

**해결 방법:**

#### 방법 1: Docker Compose 사용 (권장)

```bash
# 프로젝트 루트에서
cd /Users/leebeanbin/Downloads/beanllm

# MongoDB와 Redis 시작
docker-compose up -d mongodb redis

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs mongodb
```

#### 방법 2: 수동으로 MongoDB 시작

```bash
# Homebrew로 설치된 경우
brew services start mongodb-community@7.0

# 또는 직접 실행
mongod --config /usr/local/etc/mongod.conf
```

#### 방법 3: MongoDB 없이 실행 (기능 제한)

MongoDB가 없어도 백엔드는 실행되지만, 채팅 히스토리 저장 기능은 사용할 수 없습니다.

**환경 변수 설정:**
```bash
# .env 파일에 추가 (선택사항)
MONGODB_URI=mongodb://localhost:27017/beanllm
```

### 2. Redis 연결 실패

**증상:**
```
Redis client creation failed
```

**해결 방법:**

```bash
# Docker Compose로 시작
docker-compose up -d redis

# 또는 Homebrew로 시작
brew services start redis

# 연결 확인
redis-cli ping
# 응답: PONG
```

### 3. Pydantic V1 경고 (Python 3.14)

**증상:**
```
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
```

**해결 방법:**

이 경고는 기능에 영향을 주지 않습니다. Python 3.10-3.12를 사용하면 경고가 사라집니다.

```bash
# Python 3.12로 가상환경 재생성
poetry env use python3.12
poetry install --with web
```

### 4. ENCRYPTION_KEY 경고

**증상:**
```
⚠️  ENCRYPTION_KEY not set. Generating temporary key.
   For production, set ENCRYPTION_KEY in .env
```

**해결 방법:**

```bash
# .env 파일에 추가
ENCRYPTION_KEY=your-secret-encryption-key-here

# 또는 자동 생성된 키 사용 (개발 환경)
# 경고는 무시해도 됩니다.
```

### 5. ModuleNotFoundError

**증상:**
```
ModuleNotFoundError: No module named 'xxx'
```

**해결 방법:**

```bash
# 프로젝트 루트에서
cd /Users/leebeanbin/Downloads/beanllm

# 누락된 모듈 추가
poetry add --group web <missing-module>

# 예시:
# poetry add --group web motor
# poetry add --group web pymongo
# poetry add --group web cryptography

# 재설치
poetry install --with web
```

## 인프라 서비스 상태 확인

### Docker Compose 서비스 확인

```bash
# 모든 서비스 상태
docker-compose ps

# 특정 서비스 로그
docker-compose logs mongodb
docker-compose logs redis

# 서비스 재시작
docker-compose restart mongodb
docker-compose restart redis
```

### 포트 확인

```bash
# MongoDB (27017)
lsof -i :27017

# Redis (6379)
lsof -i :6379

# 백엔드 (8000)
lsof -i :8000
```

## 빠른 시작 스크립트

```bash
# 자동으로 인프라 시작 + 백엔드 실행
./playground/backend/start_backend.sh
```

이 스크립트는:
1. Docker Compose로 MongoDB, Redis 시작
2. Poetry 의존성 확인 및 설치
3. .env 파일 확인
4. 백엔드 서버 실행

## 수동 시작 순서

```bash
# 1. 인프라 서비스 시작
cd /Users/leebeanbin/Downloads/beanllm
docker-compose up -d mongodb redis

# 2. 의존성 설치
poetry install --with web

# 3. 가상환경 활성화
source $(poetry env info --path)/bin/activate

# 4. 백엔드 실행
cd playground/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
