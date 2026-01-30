#!/bin/bash
# beanllm Playground Backend 시작 스크립트
# MongoDB, Redis 등 인프라 서비스 시작 및 백엔드 실행
#
# Usage:
#   ./start_backend.sh              # Docker 모드 (기본값)
#   ./start_backend.sh --local      # 로컬 모드 (로컬 MongoDB/Redis 사용)
#   ./start_backend.sh --docker     # Docker 모드 (명시적)
#   DEPLOYMENT_MODE=local ./start_backend.sh  # 환경 변수로 모드 지정

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 배포 모드 결정 (명령줄 인자 > 환경 변수 > 기본값)
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-docker}"
if [ "$1" == "--local" ]; then
    DEPLOYMENT_MODE="local"
elif [ "$1" == "--docker" ]; then
    DEPLOYMENT_MODE="docker"
fi

echo -e "${BLUE}🚀 beanllm Playground Backend 시작${NC}"
echo -e "${YELLOW}   모드: ${DEPLOYMENT_MODE}${NC}"
echo ""

# 1. Docker Compose로 인프라 서비스 시작
echo -e "${YELLOW}📦 인프라 서비스 시작 중...${NC}"

# Docker 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker가 설치되어 있지 않습니다.${NC}"
    echo "   Docker Desktop을 설치하세요: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# docker-compose 또는 docker compose 확인 및 설치
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
    echo -e "${GREEN}  ✅ docker compose 발견 (하이픈 없이)${NC}"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
    echo -e "${GREEN}  ✅ docker-compose 발견 (하이픈 있이)${NC}"
else
    # docker-compose가 없으면 설치
    echo -e "${YELLOW}  ⚠️  docker-compose가 없습니다. 설치 중...${NC}"
    
    # Homebrew로 설치 시도
    if command -v brew &> /dev/null; then
        echo -e "${BLUE}  → Homebrew로 docker-compose 설치 중...${NC}"
        if brew install docker-compose &> /dev/null 2>&1; then
            DOCKER_COMPOSE_CMD="docker-compose"
            echo -e "${GREEN}  ✅ docker-compose 설치 완료${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Homebrew 설치 실패, pip로 시도...${NC}"
        fi
    fi
    
    # pip로 설치 시도 (Homebrew 실패 시 또는 Homebrew가 없는 경우)
    if [ -z "$DOCKER_COMPOSE_CMD" ]; then
        if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
            PIP_CMD=$(command -v pip3 || command -v pip)
            echo -e "${BLUE}  → pip로 docker-compose 설치 중...${NC}"
            if $PIP_CMD install --user docker-compose &> /dev/null 2>&1; then
                DOCKER_COMPOSE_CMD="docker-compose"
                echo -e "${GREEN}  ✅ docker-compose 설치 완료${NC}"
                
                # PATH에 추가 (pip user install 경로)
                USER_BIN="$HOME/.local/bin"
                if [ -d "$USER_BIN" ] && [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
                    export PATH="$USER_BIN:$PATH"
                    echo -e "${BLUE}  → PATH에 $USER_BIN 추가됨 (현재 세션)${NC}"
                    
                    # .zshrc 또는 .bashrc에 영구적으로 추가
                    SHELL_RC=""
                    if [ -f "$HOME/.zshrc" ]; then
                        SHELL_RC="$HOME/.zshrc"
                    elif [ -f "$HOME/.bashrc" ]; then
                        SHELL_RC="$HOME/.bashrc"
                    elif [ -f "$HOME/.bash_profile" ]; then
                        SHELL_RC="$HOME/.bash_profile"
                    fi
                    
                    if [ -n "$SHELL_RC" ]; then
                        if ! grep -q "$USER_BIN" "$SHELL_RC"; then
                            echo "" >> "$SHELL_RC"
                            echo "# docker-compose PATH (added by beanllm)" >> "$SHELL_RC"
                            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
                            echo -e "${GREEN}  ✅ $SHELL_RC에 PATH 추가됨 (영구 설정)${NC}"
                        fi
                    fi
                fi
            else
                echo -e "${RED}  ❌ docker-compose 설치 실패${NC}"
            fi
        fi
    fi
    
    # 여전히 없으면 수동 설치 안내
    if [ -z "$DOCKER_COMPOSE_CMD" ]; then
        echo -e "${RED}  ❌ docker-compose 설치에 실패했습니다.${NC}"
        echo ""
        echo -e "${YELLOW}  수동 설치 방법:${NC}"
        echo -e "${BLUE}    1. Homebrew: brew install docker-compose${NC}"
        echo -e "${BLUE}    2. pip: pip3 install docker-compose${NC}"
        echo -e "${BLUE}    3. 또는 Docker Desktop 설치 (docker compose 포함)${NC}"
        echo ""
        echo -e "${YELLOW}  인프라 없이 백엔드만 시작합니다...${NC}"
    fi
fi

# 인프라 서비스 시작 (모드에 따라)
if [ "$DEPLOYMENT_MODE" == "docker" ]; then
    # Docker 모드: Docker Compose로 MongoDB와 Redis 시작
    if [ -n "$DOCKER_COMPOSE_CMD" ]; then
        echo -e "${BLUE}  → Docker Compose로 MongoDB와 Redis 시작...${NC}"
        $DOCKER_COMPOSE_CMD up -d mongodb redis

        # 서비스가 준비될 때까지 대기
        echo -e "${YELLOW}  ⏳ 서비스 준비 대기 중...${NC}"
        sleep 5

        # MongoDB 연결 확인
        echo -e "${BLUE}  → MongoDB 연결 확인...${NC}"
        if $DOCKER_COMPOSE_CMD exec -T mongodb mongosh --eval "db.adminCommand('ping')" &> /dev/null; then
            echo -e "${GREEN}  ✅ MongoDB 연결 성공${NC}"
        else
            echo -e "${YELLOW}  ⚠️  MongoDB 연결 확인 실패 (서비스가 아직 시작 중일 수 있습니다)${NC}"
        fi

        # Redis 연결 확인
        echo -e "${BLUE}  → Redis 연결 확인...${NC}"
        if $DOCKER_COMPOSE_CMD exec -T redis redis-cli ping | grep -q "PONG"; then
            echo -e "${GREEN}  ✅ Redis 연결 성공${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Redis 연결 확인 실패 (서비스가 아직 시작 중일 수 있습니다)${NC}"
        fi
    else
        echo -e "${RED}  ❌ Docker Compose를 사용할 수 없습니다.${NC}"
        echo -e "${YELLOW}     로컬 모드로 전환하거나 Docker를 설치하세요.${NC}"
        DEPLOYMENT_MODE="local"
    fi
else
    # 로컬 모드: 로컬 MongoDB와 Redis 사용
    echo -e "${BLUE}  → 로컬 모드: 로컬 MongoDB와 Redis 사용${NC}"
    
    # MongoDB 연결 확인
    echo -e "${BLUE}  → MongoDB 연결 확인 (localhost:27017)...${NC}"
    if command -v mongosh &> /dev/null; then
        if mongosh --quiet --eval "db.adminCommand('ping')" &> /dev/null; then
            echo -e "${GREEN}  ✅ MongoDB 연결 성공${NC}"
        else
            echo -e "${YELLOW}  ⚠️  MongoDB가 실행 중이지 않습니다.${NC}"
            echo -e "${YELLOW}     시작 방법: brew services start mongodb-community@7.0${NC}"
        fi
    elif command -v mongo &> /dev/null; then
        if mongo --quiet --eval "db.adminCommand('ping')" &> /dev/null; then
            echo -e "${GREEN}  ✅ MongoDB 연결 성공${NC}"
        else
            echo -e "${YELLOW}  ⚠️  MongoDB가 실행 중이지 않습니다.${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠️  MongoDB 클라이언트가 설치되어 있지 않습니다.${NC}"
        echo -e "${YELLOW}     설치 방법:${NC}"
        echo -e "${BLUE}       1. brew tap mongodb/brew${NC}"
        echo -e "${BLUE}       2. brew install mongodb-community@7.0${NC}"
        echo -e "${BLUE}       3. brew services start mongodb-community@7.0${NC}"
    fi
    
    # Redis 연결 확인
    echo -e "${BLUE}  → Redis 연결 확인 (localhost:6379)...${NC}"
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping 2>/dev/null | grep -q "PONG"; then
            echo -e "${GREEN}  ✅ Redis 연결 성공${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Redis가 실행 중이지 않습니다.${NC}"
            echo -e "${YELLOW}     시작 방법: brew services start redis${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠️  Redis 클라이언트가 설치되어 있지 않습니다.${NC}"
        echo -e "${YELLOW}     설치: brew install redis${NC}"
    fi
fi

echo ""

# 2. Poetry 의존성 확인
echo -e "${YELLOW}📦 Poetry 의존성 확인 중...${NC}"
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry가 설치되어 있지 않습니다.${NC}"
    echo "   설치: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# poetry.lock이 오래되었으면 업데이트
if [ -f "poetry.lock" ]; then
    LOCK_AGE=$(find poetry.lock -mtime +1 2>/dev/null || echo "new")
    if [ "$LOCK_AGE" != "new" ]; then
        echo -e "${BLUE}  → poetry.lock 업데이트 중...${NC}"
        # Poetry 2.0+에서는 --no-update 옵션이 없음
        poetry lock 2>/dev/null || true
    fi
fi

# 의존성 설치 확인
if ! poetry show motor &> /dev/null; then
    echo -e "${BLUE}  → Web 그룹 의존성 설치 중...${NC}"
    poetry install --with web
else
    echo -e "${GREEN}  ✅ 의존성 이미 설치됨${NC}"
fi

echo ""

# 3. .env 파일 확인 및 ENCRYPTION_KEY 생성
echo -e "${YELLOW}⚙️  환경 변수 확인 중...${NC}"
cd "$SCRIPT_DIR"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}  ⚠️  .env 파일이 없습니다. .env.example을 복사합니다...${NC}"
        cp .env.example .env
        echo -e "${GREEN}  ✅ .env 파일 생성됨${NC}"
    else
        echo -e "${YELLOW}  ⚠️  .env 파일이 없습니다. (선택사항)${NC}"
    fi
else
    echo -e "${GREEN}  ✅ .env 파일 존재${NC}"
fi

# 배포 모드에 따라 환경 변수 설정
if [ "$DEPLOYMENT_MODE" == "local" ]; then
    echo -e "${BLUE}  → 로컬 모드 설정 적용 중...${NC}"
    if [ -f ".env" ]; then
        # MongoDB URI를 localhost로 설정
        if grep -q "^MONGODB_URI=" .env; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' 's|^MONGODB_URI=.*|MONGODB_URI=mongodb://localhost:27017/beanllm|' .env
            else
                sed -i 's|^MONGODB_URI=.*|MONGODB_URI=mongodb://localhost:27017/beanllm|' .env
            fi
        else
            echo "MONGODB_URI=mongodb://localhost:27017/beanllm" >> .env
        fi
        
        # Redis Host를 localhost로 설정
        if grep -q "^REDIS_HOST=" .env; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' 's|^REDIS_HOST=.*|REDIS_HOST=localhost|' .env
            else
                sed -i 's|^REDIS_HOST=.*|REDIS_HOST=localhost|' .env
            fi
        else
            echo "REDIS_HOST=localhost" >> .env
        fi
        
        echo -e "${GREEN}  ✅ 로컬 모드 설정 적용 완료${NC}"
    fi
elif [ "$DEPLOYMENT_MODE" == "docker" ]; then
    echo -e "${BLUE}  → Docker 모드 설정 적용 중...${NC}"
    if [ -f ".env" ]; then
        # MongoDB URI를 Docker 서비스명으로 설정
        if grep -q "^MONGODB_URI=" .env; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' 's|^MONGODB_URI=.*|MONGODB_URI=mongodb://beanllm:beanllm_secret@localhost:27017/beanllm?authSource=admin|' .env
            else
                sed -i 's|^MONGODB_URI=.*|MONGODB_URI=mongodb://beanllm:beanllm_secret@localhost:27017/beanllm?authSource=admin|' .env
            fi
        else
            echo "MONGODB_URI=mongodb://beanllm:beanllm_secret@localhost:27017/beanllm?authSource=admin" >> .env
        fi
        
        # Redis Host를 localhost로 설정 (포트 포워딩 사용)
        if grep -q "^REDIS_HOST=" .env; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' 's|^REDIS_HOST=.*|REDIS_HOST=localhost|' .env
            else
                sed -i 's|^REDIS_HOST=.*|REDIS_HOST=localhost|' .env
            fi
        else
            echo "REDIS_HOST=localhost" >> .env
        fi
        
        echo -e "${GREEN}  ✅ Docker 모드 설정 적용 완료${NC}"
    fi
fi

# ENCRYPTION_KEY가 없으면 자동 생성
if [ -f ".env" ]; then
    if ! grep -q "^ENCRYPTION_KEY=.*[^[:space:]]" .env || grep -q "^ENCRYPTION_KEY=$" .env; then
        echo -e "${BLUE}  → ENCRYPTION_KEY 생성 중...${NC}"
        # Python으로 안전한 랜덤 키 생성
        ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)
        
        # .env 파일에 ENCRYPTION_KEY 추가 또는 업데이트
        if grep -q "^ENCRYPTION_KEY=" .env; then
            # 기존 ENCRYPTION_KEY= 라인을 업데이트
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                sed -i '' "s/^ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env
            else
                # Linux
                sed -i "s/^ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env
            fi
        else
            # ENCRYPTION_KEY가 없으면 추가
            echo "" >> .env
            echo "# Encryption key (auto-generated)" >> .env
            echo "ENCRYPTION_KEY=$ENCRYPTION_KEY" >> .env
        fi
        echo -e "${GREEN}  ✅ ENCRYPTION_KEY 자동 생성 및 설정 완료${NC}"
    else
        echo -e "${GREEN}  ✅ ENCRYPTION_KEY 이미 설정됨${NC}"
    fi
fi

echo ""

# 4. 백엔드 실행
echo -e "${GREEN}🚀 백엔드 서버 시작 중...${NC}"
echo -e "${BLUE}   URL: http://localhost:8000${NC}"
echo -e "${BLUE}   API Docs: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}   종료: Ctrl+C${NC}"
echo ""

# Poetry run을 사용하여 백엔드 실행
cd "$SCRIPT_DIR"
poetry run python -m uvicorn main:app --host 0.0.0.0 --port 8000
