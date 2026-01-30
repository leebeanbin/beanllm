#!/bin/bash

# Quick Test - Just run the test without MongoDB setup
# Use this if you already have MongoDB configured

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Starting backend..."
python main.py > /dev/null 2>&1 &
BACKEND_PID=$!

sleep 5

echo "Running tests..."
python tests/test_chat_history_api.py

kill $BACKEND_PID
echo "Done"
