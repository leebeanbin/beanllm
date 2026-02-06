#!/bin/bash

# ===========================================
# beanllm Environment Check Script
# ===========================================
# This script checks if all required services and
# environment variables are properly configured.
#
# Usage:
#   ./scripts/check-env.sh
#   ./scripts/check-env.sh --verbose
#   ./scripts/check-env.sh --fix (auto-create .env files)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
VERBOSE=false
FIX=false

for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE=true
            ;;
        --fix|-f)
            FIX=true
            ;;
    esac
done

echo ""
echo "=========================================="
echo "  beanllm Environment Check"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# ===========================================
# Helper Functions
# ===========================================

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is not installed"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

check_service() {
    local name=$1
    local host=$2
    local port=$3

    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $name is running on $host:$port"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not running on $host:$port"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

check_env_var() {
    local var_name=$1
    local required=$2
    local var_value="${!var_name}"

    if [ -n "$var_value" ]; then
        if [ "$VERBOSE" = true ]; then
            # Mask sensitive values
            if [[ "$var_name" == *"KEY"* ]] || [[ "$var_name" == *"SECRET"* ]] || [[ "$var_name" == *"PASSWORD"* ]]; then
                echo -e "${GREEN}✓${NC} $var_name is set (****${var_value: -4})"
            else
                echo -e "${GREEN}✓${NC} $var_name = $var_value"
            fi
        else
            echo -e "${GREEN}✓${NC} $var_name is set"
        fi
        return 0
    else
        if [ "$required" = "required" ]; then
            echo -e "${RED}✗${NC} $var_name is not set (required)"
            ERRORS=$((ERRORS + 1))
        else
            echo -e "${YELLOW}!${NC} $var_name is not set (optional)"
            WARNINGS=$((WARNINGS + 1))
        fi
        return 1
    fi
}

# ===========================================
# 1. Check Required Commands
# ===========================================

echo -e "${BLUE}[1/5] Checking required commands...${NC}"
echo ""

check_command "docker"
check_command "docker-compose" || check_command "docker compose"
check_command "python3" || check_command "python"
check_command "node"
check_command "pnpm" || check_command "npm"
check_command "nc"

echo ""

# ===========================================
# 2. Check .env Files
# ===========================================

echo -e "${BLUE}[2/5] Checking .env files...${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Backend .env
BACKEND_ENV="$PROJECT_ROOT/playground/backend/.env"
BACKEND_ENV_EXAMPLE="$PROJECT_ROOT/playground/backend/.env.example"

if [ -f "$BACKEND_ENV" ]; then
    echo -e "${GREEN}✓${NC} Backend .env exists"
    # Load backend env
    set -a
    source "$BACKEND_ENV"
    set +a
else
    echo -e "${RED}✗${NC} Backend .env does not exist"
    if [ "$FIX" = true ] && [ -f "$BACKEND_ENV_EXAMPLE" ]; then
        echo -e "${YELLOW}  Creating from .env.example...${NC}"
        cp "$BACKEND_ENV_EXAMPLE" "$BACKEND_ENV"
        echo -e "${GREEN}  Created $BACKEND_ENV${NC}"
    else
        echo "  Run: cp playground/backend/.env.example playground/backend/.env"
    fi
    ERRORS=$((ERRORS + 1))
fi

# Frontend .env.local
FRONTEND_ENV="$PROJECT_ROOT/playground/frontend/.env.local"
FRONTEND_ENV_EXAMPLE="$PROJECT_ROOT/playground/frontend/.env.local.example"

if [ -f "$FRONTEND_ENV" ]; then
    echo -e "${GREEN}✓${NC} Frontend .env.local exists"
else
    echo -e "${RED}✗${NC} Frontend .env.local does not exist"
    if [ "$FIX" = true ] && [ -f "$FRONTEND_ENV_EXAMPLE" ]; then
        echo -e "${YELLOW}  Creating from .env.local.example...${NC}"
        cp "$FRONTEND_ENV_EXAMPLE" "$FRONTEND_ENV"
        echo -e "${GREEN}  Created $FRONTEND_ENV${NC}"
    else
        echo "  Run: cp playground/frontend/.env.local.example playground/frontend/.env.local"
    fi
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ===========================================
# 3. Check Infrastructure Services
# ===========================================

echo -e "${BLUE}[3/5] Checking infrastructure services...${NC}"
echo ""

# MongoDB
MONGO_HOST="${MONGODB_URI:-localhost}"
if [[ "$MONGO_HOST" == mongodb://* ]]; then
    # Extract host from URI
    MONGO_HOST=$(echo "$MONGO_HOST" | sed -E 's|mongodb://[^@]*@([^:/]+).*|\1|')
fi
check_service "MongoDB" "${MONGO_HOST:-localhost}" 27017

# Redis
check_service "Redis" "${REDIS_HOST:-localhost}" "${REDIS_PORT:-6379}"

# Kafka (optional)
if [ "${USE_DISTRIBUTED:-false}" = "true" ]; then
    KAFKA_HOST=$(echo "${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}" | cut -d: -f1)
    KAFKA_PORT=$(echo "${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}" | cut -d: -f2)
    check_service "Kafka" "$KAFKA_HOST" "$KAFKA_PORT"
else
    echo -e "${YELLOW}!${NC} Kafka check skipped (USE_DISTRIBUTED=false)"
    WARNINGS=$((WARNINGS + 1))
fi

# Ollama (optional but common)
if check_service "Ollama" "localhost" 11434 2>/dev/null; then
    :
else
    echo -e "${YELLOW}!${NC} Ollama is not running (optional for local models)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""

# ===========================================
# 4. Check Environment Variables
# ===========================================

echo -e "${BLUE}[4/5] Checking environment variables...${NC}"
echo ""

# Required
echo "Required variables:"
check_env_var "MONGODB_URI" "required" || check_env_var "MONGODB_DATABASE" "required"

echo ""
echo "LLM Provider Keys (at least one required):"
HAS_LLM_KEY=false

if [ -n "$OPENAI_API_KEY" ]; then
    check_env_var "OPENAI_API_KEY" "optional"
    HAS_LLM_KEY=true
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    check_env_var "ANTHROPIC_API_KEY" "optional"
    HAS_LLM_KEY=true
fi

if [ -n "$GOOGLE_API_KEY" ] || [ -n "$GEMINI_API_KEY" ]; then
    check_env_var "GOOGLE_API_KEY" "optional" || check_env_var "GEMINI_API_KEY" "optional"
    HAS_LLM_KEY=true
fi

# Check Ollama as LLM option
if nc -z localhost 11434 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is available (local LLM)"
    HAS_LLM_KEY=true
fi

if [ "$HAS_LLM_KEY" = false ]; then
    echo -e "${RED}✗${NC} No LLM provider configured!"
    echo "  Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
    echo "  Or start Ollama:"
    echo "    - 로컬: ollama serve"
    echo "    - Docker: docker-compose up -d (Ollama 포함)"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "Optional variables:"
check_env_var "ENCRYPTION_KEY" "optional"
check_env_var "GOOGLE_CLIENT_ID" "optional"
check_env_var "GOOGLE_CLIENT_SECRET" "optional"
check_env_var "TAVILY_API_KEY" "optional"
check_env_var "PINECONE_API_KEY" "optional"

echo ""

# ===========================================
# 5. Check Python & Node Dependencies
# ===========================================

echo -e "${BLUE}[5/5] Checking dependencies...${NC}"
echo ""

# Check if beanllm is installed
if python3 -c "import beanllm" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} beanllm package is installed"
else
    echo -e "${YELLOW}!${NC} beanllm package is not installed"
    echo "  Run: pip install -e ."
    WARNINGS=$((WARNINGS + 1))
fi

# Check backend dependencies
if [ -f "$PROJECT_ROOT/playground/backend/requirements.txt" ]; then
    echo -e "${GREEN}✓${NC} Backend requirements.txt exists"
else
    echo -e "${RED}✗${NC} Backend requirements.txt not found"
    ERRORS=$((ERRORS + 1))
fi

# Check frontend dependencies
if [ -d "$PROJECT_ROOT/playground/frontend/node_modules" ]; then
    echo -e "${GREEN}✓${NC} Frontend node_modules exists"
else
    echo -e "${YELLOW}!${NC} Frontend node_modules not found"
    echo "  Run: cd playground/frontend && pnpm install"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""

# ===========================================
# Summary
# ===========================================

echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "You can now start the services:"
    echo "  1. Start infrastructure: docker-compose up -d"
    echo "  2. Start backend: cd playground/backend && uvicorn main:app --reload"
    echo "  3. Start frontend: cd playground/frontend && pnpm dev"
    echo "  4. Start monitoring: cd playground/backend && streamlit run monitoring_dashboard.py"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Checks completed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "The system should work, but some features may be limited."
    exit 0
else
    echo -e "${RED}Checks failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before starting the services."
    echo ""
    echo "Quick fixes:"
    echo "  1. Start infrastructure: docker-compose up -d"
    echo "  2. Create .env files: ./scripts/check-env.sh --fix"
    echo "  3. Install dependencies: pip install -e . && cd playground/frontend && pnpm install"
    exit 1
fi
