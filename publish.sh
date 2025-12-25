#!/bin/bash

# llmkit PyPI 배포 스크립트
# 사용법: ./publish.sh [test|prod]

set -e  # 에러 발생시 중단

echo "🚀 llmkit PyPI 배포 스크립트"
echo "=============================="

# 인자 확인
MODE=${1:-test}

if [[ "$MODE" != "test" && "$MODE" != "prod" ]]; then
    echo "❌ 잘못된 인자입니다. 'test' 또는 'prod'를 사용하세요."
    echo "사용법: ./publish.sh [test|prod]"
    exit 1
fi

# 1. 이전 빌드 파일 삭제
echo ""
echo "📁 Step 1: 이전 빌드 파일 정리..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# 2. 린트 체크 (선택사항)
echo ""
echo "🔍 Step 2: 코드 품질 체크..."
if command -v ruff &> /dev/null; then
    echo "  - Ruff 린트 실행 중..."
    ruff check src/llmkit --fix || echo "  ⚠️  경고가 있지만 계속 진행합니다."
else
    echo "  ⚠️  Ruff가 설치되어 있지 않습니다. 건너뜁니다."
fi

# 3. 테스트 실행 (선택사항)
echo ""
echo "🧪 Step 3: 테스트 실행..."
read -p "테스트를 실행하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || {
            echo "❌ 테스트 실패! 계속 진행하시겠습니까?"
            read -p "(y/N): " -n 1 -r
            echo
            [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
        }
    else
        echo "  ⚠️  pytest가 설치되어 있지 않습니다."
    fi
else
    echo "  ⏭️  테스트를 건너뜁니다."
fi

# 4. 빌드
echo ""
echo "📦 Step 4: 패키지 빌드 중..."
python -m build

# 5. 빌드 결과 확인
echo ""
echo "✅ 빌드 완료!"
echo "생성된 파일:"
ls -lh dist/

# 6. 업로드
echo ""
if [ "$MODE" = "test" ]; then
    echo "🧪 Step 5: TestPyPI에 업로드 중..."
    echo "  TestPyPI: https://test.pypi.org/project/llmkit/"
    python -m twine upload --repository testpypi dist/*

    echo ""
    echo "✅ TestPyPI 업로드 완료!"
    echo ""
    echo "테스트 설치 방법:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmkit"

elif [ "$MODE" = "prod" ]; then
    echo "🚀 Step 5: PyPI에 업로드 중..."
    echo ""
    echo "⚠️  주의: 본 PyPI에 배포하면 버전을 되돌릴 수 없습니다!"
    read -p "정말 배포하시겠습니까? (yes/no): " -r
    echo

    if [[ $REPLY = "yes" ]]; then
        python -m twine upload dist/*

        echo ""
        echo "✅ PyPI 업로드 완료!"
        echo ""
        echo "설치 방법:"
        echo "  pip install llmkit"
        echo ""
        echo "PyPI 페이지: https://pypi.org/project/llmkit/"
    else
        echo "❌ 배포가 취소되었습니다."
        exit 1
    fi
fi

echo ""
echo "🎉 모든 작업이 완료되었습니다!"
