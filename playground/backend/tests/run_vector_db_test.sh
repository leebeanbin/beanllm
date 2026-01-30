#!/bin/bash
# Vector DB 성능 테스트 실행 스크립트

set -e

echo "🚀 Vector DB 성능 테스트 실행"
echo "================================"

# 프로젝트 루트로 이동
cd "$(dirname "$0")/../.."

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src:$(pwd)/playground/backend"

# 환경 변수 설정 (기본값)
export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017/beanllm_test}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "📋 환경 설정:"
echo "   MONGODB_URI: $MONGODB_URI"
echo "   OLLAMA_BASE_URL: $OLLAMA_BASE_URL"
echo ""

# 테스트 실행
python3 playground/backend/tests/test_vector_db_performance.py

echo ""
echo "✅ 테스트 완료!"
echo "📊 결과 파일: playground/backend/tests/vector_db_test_results.json"
