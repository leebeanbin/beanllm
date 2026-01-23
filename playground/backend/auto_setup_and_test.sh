#!/bin/bash

# Auto Setup & Test Script for beanllm Playground
# This script sets up MongoDB and tests the entire backend

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "beanllm Playground - Auto Setup & Test"
echo "============================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check dependencies
echo -e "\n${YELLOW}Step 1: Checking dependencies...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python: $(python --version)${NC}"

if ! command -v pip &> /dev/null; then
    echo -e "${RED}❌ pip not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pip installed${NC}"

# Step 2: Install Python dependencies
echo -e "\n${YELLOW}Step 2: Installing Python dependencies...${NC}"
pip install -q motor pymongo httpx fastapi uvicorn python-dotenv pydantic
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 3: Check MongoDB setup
echo -e "\n${YELLOW}Step 3: Checking MongoDB setup...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env and add your MONGODB_URI${NC}"
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  1. MongoDB Atlas (Free): https://www.mongodb.com/cloud/atlas"
    echo -e "  2. Local MongoDB: mongodb://localhost:27017/beanllm"
    echo -e "  3. Skip MongoDB (test will fail but API will work)"
    echo ""
    read -p "Press Enter after setting MONGODB_URI (or press Ctrl+C to exit)..."
fi

# Check if MONGODB_URI is set
if grep -q "MONGODB_URI=mongodb" .env; then
    MONGODB_URI=$(grep MONGODB_URI .env | cut -d '=' -f 2-)
    if [ ! -z "$MONGODB_URI" ] && [ "$MONGODB_URI" != "mongodb+srv://..." ]; then
        echo -e "${GREEN}✅ MONGODB_URI found in .env${NC}"
    else
        echo -e "${YELLOW}⚠️  MONGODB_URI not configured properly${NC}"
        echo -e "${YELLOW}Tests will continue but session storage will not work${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  MONGODB_URI not found in .env${NC}"
fi

# Step 4: Test imports
echo -e "\n${YELLOW}Step 4: Testing Python imports...${NC}"
python -c "
import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '.')

try:
    from database import get_mongodb_client
    from models import ChatSession
    from chat_history import router
    from common import get_client
    print('${GREEN}✅ All imports successful${NC}')
except Exception as e:
    print(f'${RED}❌ Import error: {e}${NC}')
    sys.exit(1)
"

# Step 5: Start backend (background)
echo -e "\n${YELLOW}Step 5: Starting backend server...${NC}"
echo -e "${YELLOW}Backend will run on http://localhost:8000${NC}"

# Kill any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start backend in background
python main.py > backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
echo "Waiting for backend to be ready..."
sleep 5

# Check if backend is running
if ! ps -p $BACKEND_PID > /dev/null; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    echo "Last 20 lines of backend.log:"
    tail -20 backend.log
    exit 1
fi

# Step 6: Test health endpoint
echo -e "\n${YELLOW}Step 6: Testing health endpoint...${NC}"
if curl -s http://localhost:8000/health | grep -q "healthy\|unhealthy"; then
    echo -e "${GREEN}✅ Health endpoint responded${NC}"
    curl -s http://localhost:8000/health | python -m json.tool
else
    echo -e "${RED}❌ Health endpoint failed${NC}"
    kill $BACKEND_PID
    exit 1
fi

# Step 7: Run API tests
echo -e "\n${YELLOW}Step 7: Running API tests...${NC}"
python tests/test_chat_history_api.py

TEST_RESULT=$?

# Cleanup
echo -e "\n${YELLOW}Cleaning up...${NC}"
kill $BACKEND_PID
echo -e "${GREEN}✅ Backend stopped${NC}"

# Final result
echo ""
echo "============================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Check backend.log for details"
    echo "  2. Visit http://localhost:8000/docs for API documentation"
    echo "  3. Run 'python main.py' to start backend manually"
    echo "  4. Run frontend: cd ../frontend && npm run dev"
    exit 0
else
    echo -e "${RED}❌ Tests failed${NC}"
    echo "============================================================"
    echo ""
    echo "Check backend.log for error details:"
    echo "  tail -50 backend.log"
    exit 1
fi
