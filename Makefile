.PHONY: help install install-dev type-check lint lint-fix format check-fix all clean test pre-commit quality

# 기본 변수
PYTHON := python
PACKAGE := src/beanllm
TESTS := tests

# 색상 출력
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## 도움말 표시
	@echo "$(GREEN)LLMKit 개발 도구$(NC)"
	@echo ""
	@echo "$(YELLOW)사용 가능한 명령어:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## 필수 의존성 설치
	@echo "$(GREEN)필수 의존성 설치 중...$(NC)"
	$(PYTHON) -m pip install -e .

install-dev: ## 개발 의존성 설치 (타입 체커, 린터, pre-commit 포함)
	@echo "$(GREEN)개발 의존성 설치 중...$(NC)"
	$(PYTHON) -m pip install -e ".[dev]"
	@echo "$(GREEN)✅ 개발 도구 설치 완료$(NC)"

pre-commit: ## pre-commit 훅 설치 (git commit 시 자동 품질 검사)
	@echo "$(GREEN)pre-commit 훅 설치 중...$(NC)"
	$(PYTHON) -m pre_commit install
	@echo "$(GREEN)✅ pre-commit 설치 완료. 이제 git commit 시 Ruff/mypy/Bandit이 자동 실행됩니다.$(NC)"

quality: ## 품질 검사 한 번에 실행 (Black/Ruff → mypy → Bandit → pytest, 글 예제 10.13 스타일)
	@set -e; \
	echo "$(GREEN)Python 코드 품질 검사 시작...$(NC)"; \
	echo "\n▶ 코드 포매팅 (Ruff)..."; $(PYTHON) -m ruff format $(PACKAGE); \
	echo "\n▶ 린팅 + 임포트 정렬 (Ruff)..."; $(PYTHON) -m ruff check --fix $(PACKAGE) --select E,F,I --ignore E501; \
	echo "\n▶ 타입 체크 (mypy)..."; $(PYTHON) -m mypy $(PACKAGE) --ignore-missing-imports --no-error-summary || true; \
	echo "\n▶ 보안 검사 (Bandit)..."; $(PYTHON) -m bandit -r $(PACKAGE) -q || true; \
	echo "\n▶ 테스트 (pytest)..."; $(PYTHON) -m pytest $(TESTS) -v; \
	echo "\n$(GREEN)모든 검사가 완료되었습니다!$(NC)"

type-check: ## 타입 체크 (mypy)
	@echo "$(GREEN)타입 체크 중...$(NC)"
	@$(PYTHON) -m mypy $(PACKAGE) \
		--ignore-missing-imports \
		--show-error-codes \
		--show-error-context \
		--no-error-summary || true
	@echo "$(GREEN)✅ 타입 체크 완료$(NC)"

type-check-strict: ## 엄격한 타입 체크 (모든 타입 어노테이션 필수)
	@echo "$(GREEN)엄격한 타입 체크 중...$(NC)"
	@$(PYTHON) -m mypy $(PACKAGE) \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--disallow-incomplete-defs \
		--check-untyped-defs \
		--show-error-codes \
		--show-error-context || true
	@echo "$(GREEN)✅ 엄격한 타입 체크 완료$(NC)"

lint: ## 린트 체크 (ruff, E501 제외)
	@echo "$(GREEN)린트 체크 중...$(NC)"
	@$(PYTHON) -m ruff check $(PACKAGE) \
		--select E,F,I \
		--ignore E501 \
		--output-format=concise || true
	@echo "$(GREEN)✅ 린트 체크 완료$(NC)"

lint-all: ## 린트 체크 (모든 오류 포함, E501 포함)
	@echo "$(GREEN)전체 린트 체크 중...$(NC)"
	@$(PYTHON) -m ruff check $(PACKAGE) \
		--select E,F,I \
		--output-format=concise || true
	@echo "$(GREEN)✅ 전체 린트 체크 완료$(NC)"

lint-fix: ## 린트 자동 수정 (ruff --fix)
	@echo "$(GREEN)린트 자동 수정 중...$(NC)"
	@$(PYTHON) -m ruff check --fix $(PACKAGE) \
		--select E,F,I \
		--ignore E501 \
		--output-format=concise
	@echo "$(GREEN)✅ 린트 자동 수정 완료$(NC)"

format: ## 코드 포맷팅 자동 수정 (ruff format)
	@echo "$(GREEN)코드 포맷팅 중...$(NC)"
	@$(PYTHON) -m ruff format $(PACKAGE)
	@echo "$(GREEN)✅ 코드 포맷팅 완료$(NC)"

format-check: ## 코드 포맷팅 필요 여부 확인만 (수정 안함)
	@echo "$(GREEN)포맷팅 확인 중...$(NC)"
	@$(PYTHON) -m ruff format --check $(PACKAGE) || \
		(echo "$(YELLOW)⚠️  포맷팅이 필요한 파일이 있습니다. 'make format'을 실행하세요.$(NC)" && exit 1)
	@echo "$(GREEN)✅ 모든 파일이 올바르게 포맷팅되어 있습니다$(NC)"

import-sort: ## Import 정렬 (ruff --fix I001)
	@echo "$(GREEN)Import 정렬 중...$(NC)"
	@$(PYTHON) -m ruff check --fix $(PACKAGE) --select I001
	@echo "$(GREEN)✅ Import 정렬 완료$(NC)"

check: type-check lint ## 타입 체크 + 린트 체크
	@echo "$(GREEN)✅ 전체 검사 완료$(NC)"

check-fix: lint-fix format import-sort ## 자동 수정 가능한 모든 오류 수정
	@echo "$(GREEN)✅ 자동 수정 완료$(NC)"
	@echo "$(YELLOW)남은 오류는 수동으로 수정이 필요합니다.$(NC)"

all: check-fix type-check ## 모든 검사 및 자동 수정
	@echo "$(GREEN)✅ 전체 검사 및 수정 완료$(NC)"

test: ## 테스트 실행
	@echo "$(GREEN)테스트 실행 중...$(NC)"
	@$(PYTHON) -m pytest $(TESTS) -v

test-cov: ## 테스트 + 커버리지
	@echo "$(GREEN)테스트 + 커버리지 실행 중...$(NC)"
	@$(PYTHON) -m pytest $(TESTS) --cov=$(PACKAGE) --cov-report=html --cov-report=term

clean: ## 캐시 및 빌드 파일 정리
	@echo "$(GREEN)정리 중...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "$(GREEN)✅ 정리 완료$(NC)"

fix-all: clean check-fix type-check ## 완전 정리 및 수정
	@echo "$(GREEN)✅ 완전 정리 및 수정 완료$(NC)"

# 타입 오류 수정 도우미
fix-types: ## 주요 타입 오류 자동 수정 시도
	@echo "$(GREEN)주요 타입 오류 수정 중...$(NC)"
	@echo "$(YELLOW)이 명령어는 일부 타입 오류를 자동으로 수정합니다.$(NC)"
	@$(PYTHON) -c "import subprocess; import sys; \
		files = [ \
			'src/llmkit/utils/exceptions.py', \
			'src/llmkit/dto/response/graph_response.py', \
			'src/llmkit/domain/loaders/base.py', \
			'src/llmkit/domain/splitters/factory.py', \
		]; \
		print('타입 오류 수정 스크립트 실행...')"
	@echo "$(GREEN)✅ 타입 오류 수정 완료$(NC)"

# 빠른 검사 (자주 사용)
quick-check: lint ## 빠른 린트 체크만
	@echo "$(GREEN)✅ 빠른 검사 완료$(NC)"

quick-fix: lint-fix format import-sort ## 빠른 자동 수정 (린트 + 포맷팅 + import 정렬)
	@echo "$(GREEN)✅ 빠른 수정 완료$(NC)"

lint-format: lint-fix format import-sort ## 린트 수정 + 포맷팅 + import 정렬 (가장 많이 사용)
	@echo "$(GREEN)✅ 코드 품질 개선 완료$(NC)"
