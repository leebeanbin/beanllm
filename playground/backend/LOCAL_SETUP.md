# 로컬 모드 설정 가이드

beanllm Playground를 Docker 없이 로컬에 설치된 MongoDB와 Redis를 사용하여 실행하는 방법입니다.

## 📋 사전 요구사항

- Python 3.10 이상
- Poetry
- MongoDB 7.0 이상
- Redis 7.0 이상

## 🚀 빠른 시작

### 1. MongoDB 설치 및 시작 (macOS)

**중요**: MongoDB를 설치하기 전에 tap을 먼저 추가해야 합니다!

```bash
# 1. MongoDB tap 추가 (필수! 이 단계를 먼저 실행)
brew tap mongodb/brew

# 2. MongoDB Community Edition 설치
brew install mongodb-community@7.0

# 3. 서비스 시작
brew services start mongodb-community@7.0

# 또는 수동 시작
# Apple Silicon (M1/M2/M3):
mongod --config /opt/homebrew/etc/mongod.conf
# Intel Mac:
mongod --config /usr/local/etc/mongod.conf
```

**설정 파일 위치**:
- Apple Silicon (M1/M2/M3) Mac: `/opt/homebrew/etc/mongod.conf`
- Intel Mac: `/usr/local/etc/mongod.conf`

### 2. Redis 설치 및 시작 (macOS)

```bash
# Homebrew로 설치
brew install redis

# 서비스 시작
brew services start redis

# 또는 수동 시작
redis-server
```

### 3. 백엔드 실행 (로컬 모드)

```bash
cd playground/backend

# 로컬 모드로 실행
./start_backend.sh --local

# 또는 환경 변수로 지정
DEPLOYMENT_MODE=local ./start_backend.sh
```

## ⚙️ 환경 변수 설정

`.env` 파일에서 로컬 모드 설정:

```bash
# 배포 모드
DEPLOYMENT_MODE=local

# MongoDB (로컬)
MONGODB_URI=mongodb://localhost:27017/beanllm
MONGODB_DATABASE=beanllm

# Redis (로컬)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

## 🔍 연결 확인

### MongoDB 연결 확인

```bash
# mongosh 사용
mongosh --eval "db.adminCommand('ping')"

# 또는 mongo 사용 (구버전)
mongo --eval "db.adminCommand('ping')"
```

### Redis 연결 확인

```bash
redis-cli ping
# 응답: PONG
```

## 🐛 문제 해결

### MongoDB가 시작되지 않는 경우

```bash
# 서비스 상태 확인
brew services list

# 로그 확인
# Apple Silicon:
tail -f /opt/homebrew/var/log/mongodb/mongo.log
# Intel:
tail -f /usr/local/var/log/mongodb/mongo.log

# 포트 확인
lsof -i :27017

# MongoDB 프로세스 확인
ps aux | grep mongod
```

### Redis가 시작되지 않는 경우

```bash
# 서비스 상태 확인
brew services list

# 로그 확인
tail -f /usr/local/var/log/redis.log

# 포트 확인
lsof -i :6379
```

### 포트 충돌

다른 애플리케이션이 포트를 사용 중인 경우:

```bash
# MongoDB 포트 변경
# /usr/local/etc/mongod.conf 수정
net:
  port: 27018

# Redis 포트 변경
# /usr/local/etc/redis.conf 수정
port 6380
```

그리고 `.env` 파일에서 포트 번호 업데이트:

```bash
MONGODB_URI=mongodb://localhost:27018/beanllm
REDIS_PORT=6380
```

## 📝 Docker 모드와 로컬 모드 비교

| 항목 | Docker 모드 | 로컬 모드 |
|------|------------|----------|
| MongoDB | Docker 컨테이너 | 로컬 설치 |
| Redis | Docker 컨테이너 | 로컬 설치 |
| 설정 | 자동 | 수동 설치 필요 |
| 성능 | 약간 느림 | 더 빠름 |
| 리소스 | Docker 사용 | 직접 실행 |

## 🔄 모드 전환

### Docker 모드로 전환

```bash
# Docker Compose로 MongoDB/Redis 시작
docker-compose up -d mongodb redis

# Docker 모드로 백엔드 실행
./start_backend.sh --docker
```

### 로컬 모드로 전환

```bash
# Docker 서비스 중지
docker-compose down

# 로컬 서비스 시작
brew services start mongodb-community@7.0
brew services start redis

# 로컬 모드로 백엔드 실행
./start_backend.sh --local
```

## 💡 팁

- 로컬 모드는 개발 시 더 빠른 반응 속도를 제공합니다
- 프로덕션 환경에서는 Docker 모드를 권장합니다
- 두 모드를 동시에 사용할 수 없으므로, 한 번에 하나만 실행하세요
