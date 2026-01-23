# Git Workflow & Commit Rules

**우선순위**: HIGH
**적용 범위**: 모든 커밋, PR

## 커밋 메시지 규칙

### 기본 포맷

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

**필수 타입**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링 (기능 변경 없음)
- `docs`: 문서 변경
- `test`: 테스트 추가/수정
- `perf`: 성능 개선
- `chore`: 빌드, 설정 변경
- `style`: 코드 스타일 (포매팅, 세미콜론 등)

### Scope

**beanllm 레이어**:
- `facade`, `handler`, `service`, `domain`, `infrastructure`

**기능 영역**:
- `rag`, `agent`, `multi-agent`, `kg`, `vision`, `audio`, `eval`
- `chat`, `chain`, `graph`, `optimizer`, `orchestrator`

**기타**:
- `playground`, `docs`, `ci`, `deps`

### Subject (제목)

```
# ✅ Good
feat(rag): Add HyDE query expansion
fix(chat): Handle rate limit errors correctly
refactor(service): Extract common cache logic to decorator

# ❌ Bad
feat: add new feature
fix: bug fix
update code
```

**규칙**:
- 50자 이내
- 동사 원형으로 시작 (Add, Fix, Update, Remove)
- 마침표 없음
- 명령문 형식 (과거형 X)

### Body (본문)

**선택 사항**이지만 권장:

```
# ✅ Good
feat(rag): Add HyDE query expansion

Implement Hypothetical Document Embeddings for improved retrieval:
- Generate hypothetical answers for queries
- Embed hypothetical answers instead of raw queries
- 20% improvement in retrieval accuracy
- Added unit tests with 85% coverage

Tested on 1,000 documents with 50 queries.
```

**규칙**:
- 72자마다 줄바꿈
- **왜** 변경했는지 설명 (무엇이 아닌)
- 영향, 테스트 결과 포함

### Footer (푸터)

**Breaking changes**:
```
BREAKING CHANGE: RAGChain.from_documents() now requires vector_store parameter
```

**Issue 참조**:
```
Closes #123
Fixes #456
Related to #789
```

### Co-Authored-By

모든 커밋에 Claude Code 크레딧 추가:

```
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## 커밋 예시

### Feature 추가

```bash
git commit -m "$(cat <<'EOF'
feat(rag): Add HyDE query expansion

Implement Hypothetical Document Embeddings:
- Generate hypothetical answers for queries
- Embed hypothetical answers instead of raw queries
- 20% improvement in retrieval accuracy
- Added unit tests with 85% coverage

Benchmarked on 1,000 documents with 50 queries.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

### 버그 수정

```bash
git commit -m "$(cat <<'EOF'
fix(chat): Handle rate limit errors correctly

- Add exponential backoff retry logic
- Max 3 retries with 1s, 2s, 4s delays
- Log rate limit errors with sanitized messages

Fixes #234

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

### 리팩토링

```bash
git commit -m "$(cat <<'EOF'
refactor(service): Extract cache logic to decorator

Replace 456 lines of duplicate caching code with @with_cache decorator:
- 92% code reduction (456 → 40 lines)
- Consistent caching behavior across all services
- Easier to maintain and test

No functional changes. All 624 tests pass.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

### 성능 개선

```bash
git commit -m "$(cat <<'EOF'
perf(retrieval): Optimize similarity search (O(n log n) → O(n log k))

Replace sorted() with heapq.nlargest():
- 5.7× faster for k=5, n=10,000
- Reduced from 0.523s to 0.092s
- Memory usage unchanged

Benchmarked with pytest-benchmark.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

## PR (Pull Request) 프로세스

### 1. PR 생성 전 체크리스트

```bash
# 1. 브랜치 최신화
git pull origin main

# 2. 테스트 통과 확인
pytest --cov=src/beanllm --cov-report=term

# 3. 코드 품질 확인
make quick-fix  # Black, Ruff 자동 수정
make type-check  # MyPy
make lint        # Ruff 검사

# 4. Clean Architecture 검증
/arch-check

# 5. 커밋 메시지 확인
git log --oneline -5
```

### 2. PR 제목 및 설명

**제목 포맷**:
```
<type>(<scope>): <subject>

예:
feat(rag): Add HyDE query expansion
fix(chat): Handle rate limit errors
refactor(arch): Extract service layer
```

**PR 설명 템플릿**:

```markdown
## Summary
- Implement HyDE (Hypothetical Document Embeddings) for RAG
- 20% improvement in retrieval accuracy
- Fully tested with 85% coverage

## Changes
- Added `HyDEQueryExpander` class in `domain/retrieval/`
- Updated `RAGServiceImpl` to use HyDE
- Added 12 unit tests, 5 integration tests

## Test Plan
- [x] Unit tests pass (85% coverage)
- [x] Integration tests with Ollama pass
- [x] Benchmarked on 1,000 documents
- [x] No performance regression (<5% latency increase)

## Breaking Changes
- None

## Related Issues
- Closes #123
- Related to #456

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### 3. PR 생성 (gh CLI)

```bash
# 브랜치 생성 및 푸시
git checkout -b feat/rag-hyde
git add .
git commit -m "feat(rag): Add HyDE query expansion"
git push -u origin feat/rag-hyde

# PR 생성
gh pr create --title "feat(rag): Add HyDE query expansion" --body "$(cat <<'EOF'
## Summary
- Implement HyDE for improved retrieval accuracy
- 20% improvement on benchmark dataset

## Test plan
- [x] Unit tests (85% coverage)
- [x] Integration tests pass
- [x] Benchmarked

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 4. 코드 리뷰 대응

```bash
# 리뷰어 피드백 반영
git add .
git commit -m "fix(rag): Address code review feedback

- Rename HyDEExpander → HyDEQueryExpander
- Add type hints to _generate_hypothetical_answer()
- Extract magic number 3 to MAX_HYPOTHETICAL_ANSWERS constant

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
"

git push
```

### 5. Merge 전 최종 확인

```bash
# 1. Main 브랜치와 동기화
git checkout main
git pull
git checkout feat/rag-hyde
git merge main

# 2. 충돌 해결 (있는 경우)

# 3. 테스트 재실행
pytest

# 4. PR merge
gh pr merge --merge  # Merge commit (기본)
# 또는
gh pr merge --squash  # Squash merge (커밋 하나로 압축)
# 또는
gh pr merge --rebase  # Rebase merge (커밋 히스토리 유지)
```

## 브랜치 전략

### 브랜치 명명 규칙

```
<type>/<scope>-<short-description>

예:
feat/rag-hyde
feat/multi-agent-debate
fix/chat-rate-limit
refactor/service-layer
docs/api-reference
```

### 브랜치 수명

- **Feature 브랜치**: PR merge 후 즉시 삭제
- **Release 브랜치**: 태그 생성 후 유지
- **Hotfix 브랜치**: Merge 후 삭제

### Main 브랜치 보호

```bash
# Main 브랜치 직접 푸시 금지
# 모든 변경은 PR을 통해서만
```

## 커밋 규칙

### DO ✅

```bash
# 1. 작고 집중된 커밋
git add src/beanllm/domain/retrieval/hyde.py
git add tests/domain/retrieval/test_hyde.py
git commit -m "feat(rag): Add HyDE query expander"

# 2. 의미있는 단위로 커밋
git add src/beanllm/service/impl/core/rag_service_impl.py
git commit -m "feat(rag): Integrate HyDE into RAG service"

# 3. 테스트와 함께 커밋
git add src/beanllm/domain/retrieval/hyde.py
git add tests/domain/retrieval/test_hyde.py
git commit -m "feat(rag): Add HyDE with tests"
```

### DON'T ❌

```bash
# 1. 여러 기능을 한 커밋에 - 금지
git add .
git commit -m "add features and fix bugs"

# 2. WIP 커밋 - 금지 (스쿼시하거나 리베이스)
git commit -m "wip"
git commit -m "fix"
git commit -m "update"

# 3. 깨진 테스트 커밋 - 금지
# 모든 커밋은 테스트가 통과해야 함
```

## Git Hooks (선택)

### pre-commit

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit checks..."

# 1. 테스트 실행
pytest tests/ -x --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi

# 2. Clean Architecture 검증
/arch-check
if [ $? -ne 0 ]; then
    echo "❌ Clean Architecture violations found. Commit aborted."
    exit 1
fi

# 3. 코드 포매팅
black src/beanllm/
ruff check --fix src/beanllm/

echo "✅ All pre-commit checks passed"
```

### commit-msg

```bash
# .git/hooks/commit-msg
#!/bin/bash

# 커밋 메시지 포맷 검증
commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")

# <type>(<scope>): <subject> 포맷 확인
if ! echo "$commit_msg" | grep -qE "^(feat|fix|refactor|docs|test|perf|chore|style)(\(.+\))?: .+"; then
    echo "❌ Invalid commit message format"
    echo "Expected: <type>(<scope>): <subject>"
    echo "Example: feat(rag): Add HyDE query expansion"
    exit 1
fi

# 제목 50자 제한 확인
subject=$(echo "$commit_msg" | head -n 1)
if [ ${#subject} -gt 72 ]; then
    echo "❌ Commit subject too long (max 72 chars)"
    echo "Current: ${#subject} chars"
    exit 1
fi

echo "✅ Commit message format valid"
```

## Tag & Release

### Semantic Versioning

```
v<major>.<minor>.<patch>

v0.2.2 (current)
v0.3.0 (next minor)
v1.0.0 (first stable)
```

### Tag 생성

```bash
git tag -a v0.3.0 -m "$(cat <<'EOF'
beanllm v0.3.0 - Enhanced RAG Features

New Features:
- HyDE query expansion (20% accuracy improvement)
- Multi-agent debate pattern
- Knowledge graph RAG integration

Improvements:
- 92% code deduplication via decorators
- 5.7× faster similarity search
- 80% test coverage achieved

See CHANGELOG.md for details.
EOF
)"

git push origin v0.3.0
```

### GitHub Release

```bash
gh release create v0.3.0 \
  --title "beanllm v0.3.0 - Enhanced RAG" \
  --notes "$(cat RELEASE_NOTES.md)" \
  dist/beanllm-0.3.0-py3-none-any.whl \
  dist/beanllm-0.3.0.tar.gz
```

## 참고 문서

- **Conventional Commits**: https://www.conventionalcommits.org/
- **Semantic Versioning**: https://semver.org/
- **GitHub Flow**: https://guides.github.com/introduction/flow/
- `CLAUDE.md` - 프로젝트 컨텍스트
