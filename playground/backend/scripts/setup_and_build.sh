#!/bin/bash

# Complete Setup and Build Script
# This installs everything needed and runs full tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}beanllm Playground - Complete Setup & Build${NC}"
echo -e "${BLUE}============================================================${NC}"

# Step 1: Install system dependencies
echo -e "\n${YELLOW}Step 1: Checking system dependencies...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo "Install Python 3.11+: https://www.python.org/downloads/"
    exit 1
fi
echo -e "${GREEN}✅ Python: $(python3 --version)${NC}"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pip3 installed${NC}"

# Step 2: Install MongoDB if not exists
echo -e "\n${YELLOW}Step 2: Checking MongoDB...${NC}"

if command -v mongod &> /dev/null; then
    echo -e "${GREEN}✅ MongoDB already installed${NC}"
else
    echo -e "${YELLOW}MongoDB not found. Installing...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing MongoDB via Homebrew..."
            brew tap mongodb/brew
            brew install mongodb-community@7.0
            echo -e "${GREEN}✅ MongoDB installed${NC}"
        else
            echo -e "${RED}❌ Homebrew not found. Install from: https://brew.sh${NC}"
            echo "Or install MongoDB manually: https://www.mongodb.com/try/download/community"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo -e "${YELLOW}Please install MongoDB manually:${NC}"
        echo "Ubuntu: sudo apt-get install -y mongodb-org"
        echo "CentOS: sudo yum install -y mongodb-org"
        exit 1
    else
        echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
        exit 1
    fi
fi

# Step 3: Start MongoDB
echo -e "\n${YELLOW}Step 3: Starting MongoDB...${NC}"

if pgrep -x mongod > /dev/null; then
    echo -e "${GREEN}✅ MongoDB already running${NC}"
else
    echo "Starting MongoDB..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start mongodb-community@7.0
    else
        sudo systemctl start mongod
    fi
    sleep 3
    echo -e "${GREEN}✅ MongoDB started${NC}"
fi

# Step 4: Install Python dependencies
echo -e "\n${YELLOW}Step 4: Installing Python dependencies...${NC}"

echo "Installing backend dependencies..."
pip3 install -q motor pymongo httpx fastapi uvicorn python-dotenv pydantic redis kafka-python

echo "Installing beanllm package..."
cd ../..
pip3 install -q -e ".[all,dev]"
cd playground/backend

echo -e "${GREEN}✅ All Python dependencies installed${NC}"

# Step 5: Verify .env file
echo -e "\n${YELLOW}Step 5: Verifying .env file...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found (should have been created)${NC}"
    exit 1
fi

# Update MongoDB URI to local
if grep -q "MONGODB_URI=mongodb://localhost" .env; then
    echo -e "${GREEN}✅ MongoDB URI already configured for localhost${NC}"
else
    echo "Updating MongoDB URI..."
    sed -i.bak 's|MONGODB_URI=.*|MONGODB_URI=mongodb://localhost:27017/beanllm|g' .env
    echo -e "${GREEN}✅ MongoDB URI updated${NC}"
fi

echo ""
echo "Current .env configuration:"
echo "  MongoDB: $(grep MONGODB_URI .env | cut -d'=' -f2)"
echo "  Ollama: $(grep OLLAMA_HOST .env | cut -d'=' -f2)"

# Step 6: Test imports
echo -e "\n${YELLOW}Step 6: Testing Python imports...${NC}"

python3 -c "
import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '.')

print('Testing imports...')
errors = []

try:
    from database import get_mongodb_client, ping_mongodb
except Exception as e:
    errors.append(f'database.py: {e}')

try:
    from schemas.database import ChatSession, ChatMessage
except Exception as e:
    errors.append(f'models.py: {e}')

try:
    from routers.history_router import router
except Exception as e:
    errors.append(f'chat_history.py: {e}')

try:
    from common import get_client, get_kg, get_multi_agent
except Exception as e:
    errors.append(f'common.py: {e}')

if errors:
    print('❌ Import errors found:')
    for err in errors:
        print(f'  - {err}')
    sys.exit(1)
else:
    print('✅ All imports successful')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Import test failed${NC}"
    exit 1
fi

# Step 7: Start backend
echo -e "\n${YELLOW}Step 7: Starting backend server...${NC}"

# Kill existing process
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

# Start backend
echo "Starting FastAPI backend on http://localhost:8000"
python3 main.py > backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    echo "Last 30 lines of backend.log:"
    tail -30 backend.log
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Step 8: Run tests
echo -e "\n${YELLOW}Step 8: Running API tests...${NC}"
echo ""

python3 tests/test_chat_history_api.py
TEST_RESULT=$?

# Step 9: Cleanup
echo -e "\n${YELLOW}Step 9: Cleaning up...${NC}"
kill $BACKEND_PID 2>/dev/null || true
echo -e "${GREEN}✅ Backend stopped${NC}"

# Final summary
echo ""
echo -e "${BLUE}============================================================${NC}"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Setup and tests completed successfully!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo "Your environment is ready!"
    echo ""
    echo "To start developing:"
    echo -e "  ${YELLOW}1. Start backend:${NC}  cd playground/backend && python3 main.py"
    echo -e "  ${YELLOW}2. Start frontend:${NC} cd playground/frontend && npm run dev"
    echo ""
    echo "API Documentation:"
    echo "  http://localhost:8000/docs"
    echo ""
    echo "Backend logs:"
    echo "  tail -f playground/backend/backend.log"
    echo ""
else
    echo -e "${RED}❌ Tests failed${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo "Check logs for details:"
    echo "  tail -50 backend.log"
    echo ""
    echo "Common issues:"
    echo "  - MongoDB not running: brew services start mongodb-community"
    echo "  - Port 8000 in use: lsof -ti:8000 | xargs kill -9"
    echo ""
    exit 1
fi

exit 0
