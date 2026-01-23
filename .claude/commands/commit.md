# /commit - Intelligent Commit Splitter

**트리거**: `/commit`
**모델**: sonnet
**설명**: 변경된 파일을 분석하여 도메인별로 자동 분할 커밋

## Command Description

변경된 모든 파일을 읽고 분석하여 **도메인/연관성 기준으로 자동으로 여러 개의 작은 커밋으로 분할**합니다.

**핵심 철학**:
- ❌ 모든 변경사항을 하나의 긴 메시지로 커밋 (X)
- ✅ 연관된 파일들끼리 묶어서 여러 개의 짧은 커밋 (O)

## Usage

```bash
# 모든 변경사항 분석 후 자동 분할 커밋
/commit

# 분할 제안만 보기 (커밋 안함)
/commit --dry-run

# 모든 변경사항 한번에 커밋 (비추천)
/commit --all
```

## How It Works

### Step 1: Read All Changed Files

```bash
# Unstaged + Staged 모든 변경 확인
git status --short

# 각 파일의 실제 변경 내용 읽기
git diff HEAD
```

### Step 2: Analyze & Group by Domain

**그룹화 기준**:

1. **파일 경로** (가장 중요)
   ```python
   # 같은 도메인끼리 그룹화
   groups = {
       "rag": [
           "src/beanllm/domain/retrieval/hyde.py",
           "src/beanllm/service/impl/core/rag_service_impl.py",
           "tests/domain/retrieval/test_hyde.py"
       ],
       "multi-agent": [
           "src/beanllm/domain/multi_agent/debate.py",
           "tests/domain/multi_agent/test_debate.py"
       ],
       "docs": [
           "README.md",
           "docs/API_REFERENCE.md"
       ]
   }
   ```

2. **변경 유형**
   - 구현 + 테스트 → 함께 커밋
   - 문서만 → 별도 커밋
   - 설정 파일 → 별도 커밋

3. **논리적 연관성**
   - 같은 클래스/함수 수정 → 함께 커밋
   - 의존 관계 (A가 B를 사용) → 함께 커밋

### Step 3: Generate Short Commit Messages

**형식**: `<type>(<scope>): <short subject>` (50자 이내)

```bash
# ✅ Good (짧고 명확)
feat(rag): Add HyDE query expansion
fix(chat): Handle rate limits
test(agent): Add debate pattern tests
docs(rag): Update HyDE usage

# ❌ Bad (너무 김)
feat(rag): Add HyDE (Hypothetical Document Embeddings) query expansion with 20% accuracy improvement and comprehensive unit tests
```

### Step 4: Execute Multiple Commits

```bash
# Commit 1: RAG domain
git add src/beanllm/domain/retrieval/hyde.py \
        tests/domain/retrieval/test_hyde.py
git commit -m "feat(rag): Add HyDE query expansion

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Commit 2: RAG service integration
git add src/beanllm/service/impl/core/rag_service_impl.py
git commit -m "feat(rag): Integrate HyDE into service

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Commit 3: Multi-agent domain
git add src/beanllm/domain/multi_agent/debate.py \
        tests/domain/multi_agent/test_debate.py
git commit -m "feat(agent): Add debate pattern

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Commit 4: Documentation
git add README.md docs/API_REFERENCE.md
git commit -m "docs: Update RAG and multi-agent docs

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

## Example Workflow

### Before: Bad (1 huge commit)

```bash
$ git status
Modified:
  src/beanllm/domain/retrieval/hyde.py
  src/beanllm/domain/retrieval/reranker.py
  src/beanllm/domain/multi_agent/debate.py
  src/beanllm/service/impl/core/rag_service_impl.py
  src/beanllm/service/impl/advanced/multi_agent_service_impl.py
  tests/domain/retrieval/test_hyde.py
  tests/domain/retrieval/test_reranker.py
  tests/domain/multi_agent/test_debate.py
  README.md
  docs/API_REFERENCE.md
  CHANGELOG.md

# ❌ Bad: One big commit
$ git add .
$ git commit -m "Add HyDE, reranker, debate pattern, update services, add tests, update docs"

# Problems:
# - 11 files in one commit
# - Can't rollback partially
# - Unclear what each change does
```

### After: Good (6 small commits)

```bash
$ /commit

# Output:
"""
📝 Analyzing 11 changed files...

📊 Grouping by domain:

Group 1: RAG - HyDE feature (3 files)
  ├─ src/beanllm/domain/retrieval/hyde.py
  ├─ tests/domain/retrieval/test_hyde.py
  └─ src/beanllm/service/impl/core/rag_service_impl.py (integration)

Group 2: RAG - Reranker feature (2 files)
  ├─ src/beanllm/domain/retrieval/reranker.py
  └─ tests/domain/retrieval/test_reranker.py

Group 3: Multi-Agent - Debate pattern (3 files)
  ├─ src/beanllm/domain/multi_agent/debate.py
  ├─ tests/domain/multi_agent/test_debate.py
  └─ src/beanllm/service/impl/advanced/multi_agent_service_impl.py

Group 4: Documentation (3 files)
  ├─ README.md
  ├─ docs/API_REFERENCE.md
  └─ CHANGELOG.md

💡 Suggested commits: 4 atomic commits

Proceed? (yes/no/edit)
"""

> yes

# Executes:

[1/4] Committing RAG - HyDE...
[main 3a1b4c7] feat(rag): Add HyDE query expansion
 3 files changed, 78 insertions(+)

[2/4] Committing RAG - Reranker...
[main 4b2c5d8] feat(rag): Add cross-encoder reranker
 2 files changed, 56 insertions(+)

[3/4] Committing Multi-Agent - Debate...
[main 5c3d6e9] feat(agent): Add debate pattern
 3 files changed, 89 insertions(+)

[4/4] Committing Documentation...
[main 6d4e7f0] docs: Update RAG and agent docs
 3 files changed, 45 insertions(+)

✅ Created 4 commits successfully!

📊 Summary:
  - RAG domain: 2 commits
  - Multi-agent domain: 1 commit
  - Documentation: 1 commit

🎯 Each commit is focused and can be reviewed/rolled back independently.
```

## Domain Grouping Rules

### Priority 1: beanllm Layers

```python
LAYER_MAP = {
    "facade": ["src/beanllm/facade/"],
    "handler": ["src/beanllm/handler/"],
    "service": ["src/beanllm/service/impl/"],
    "domain": ["src/beanllm/domain/"],
    "infrastructure": ["src/beanllm/infrastructure/", "src/beanllm/providers/"]
}
```

### Priority 2: Feature Domains

```python
FEATURE_MAP = {
    "rag": [
        "retrieval", "splitters", "loaders", "embeddings",
        "vector_stores", "rag_service"
    ],
    "agent": [
        "multi_agent", "agent", "communication", "strategies"
    ],
    "kg": [
        "knowledge_graph", "entity_extractor", "relation_extractor",
        "graph_builder", "neo4j"
    ],
    "vision": [
        "vision", "ocr", "florence", "sam", "yolo"
    ],
    "audio": [
        "audio", "transcription", "speech"
    ],
    "eval": [
        "evaluation", "metrics", "benchmarker"
    ]
}
```

### Priority 3: File Type

```python
FILE_TYPE_MAP = {
    "test": ["tests/"],
    "docs": [".md", ".rst", "docs/"],
    "config": [
        "pyproject.toml", ".env", "requirements.txt",
        ".gitignore", ".github/"
    ],
    "playground": ["playground/"]
}
```

## Commit Message Format

### Short & Sweet (Default)

```bash
<type>(<scope>): <subject>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**예시**:
```
feat(rag): Add HyDE query expansion
fix(chat): Handle rate limit errors
test(agent): Add debate pattern tests
docs(rag): Update retrieval docs
refactor(service): Extract cache decorator
perf(retrieval): Optimize similarity search
```

### With Body (Only if needed)

**언제 Body 추가?**:
- Breaking change
- 성능 개선 (수치 포함)
- 복잡한 버그 수정

```bash
feat(rag): Add HyDE query expansion

20% improvement in retrieval accuracy.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Interactive Mode

```bash
$ /commit

# Output:
"""
📝 Found 7 changed files

📊 Suggested grouping:

[1] feat(rag): Add HyDE query expansion (3 files)
    ├─ domain/retrieval/hyde.py
    ├─ tests/domain/retrieval/test_hyde.py
    └─ service/impl/core/rag_service_impl.py

[2] docs: Update RAG documentation (2 files)
    ├─ README.md
    └─ docs/API_REFERENCE.md

[3] chore: Update dependencies (2 files)
    ├─ pyproject.toml
    └─ requirements.txt

💡 Options:
  a) Auto-commit all 3 groups (recommended)
  e) Edit grouping
  s) Skip some groups
  c) Cancel

Choose: [a/e/s/c]
"""

> a

# Executes 3 commits automatically
```

## Edge Cases

### Case 1: Single File Change

```bash
# Only 1 file changed
$ git status
Modified: src/beanllm/domain/retrieval/hyde.py

$ /commit

# Output:
"""
📝 Found 1 changed file

🎯 Commit message:
feat(rag): Update HyDE query expander

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

Commit? (yes/no)
"""
```

### Case 2: Unrelated Files

```bash
# Completely unrelated changes
$ git status
Modified:
  src/beanllm/domain/retrieval/hyde.py
  playground/frontend/app/page.tsx
  docs/CONTRIBUTING.md

$ /commit

# Output:
"""
📝 Found 3 changed files

📊 Grouping by domain:

[1] feat(rag): Update HyDE query expander (1 file)
[2] feat(playground): Update frontend home page (1 file)
[3] docs: Update contribution guide (1 file)

💡 3 separate commits recommended (files are unrelated)
"""
```

### Case 3: Tests Without Implementation

```bash
# Test file only
$ git status
Modified: tests/domain/retrieval/test_hyde.py

$ /commit

# Output:
"""
⚠️  Warning: Test file changed without implementation

📝 Commit message:
test(rag): Add HyDE query expansion tests

💡 Suggestion: Implement feature in next commit

Commit? (yes/no)
"""
```

## Integration with Workflow

```bash
# TDD workflow with automatic commits

# 1. Write test
"Create test file"
/commit
# → "test(rag): Add HyDE tests"

# 2. Implement
"Implement feature"
/commit
# → "feat(rag): Implement HyDE expander"

# 3. Integrate
"Add to service"
/commit
# → "feat(rag): Integrate HyDE into service"

# 4. Document
/update-docs
/commit
# → "docs(rag): Add HyDE documentation"

# Result: 4 clear, focused commits
```

## Benefits

### ✅ Advantages

1. **자동 분할**: 도메인별로 자동 그룹화
2. **짧은 메시지**: 50자 이내, 명확한 제목
3. **쉬운 롤백**: 문제 있는 커밋만 되돌리기
4. **명확한 히스토리**: 각 커밋의 목적이 분명
5. **빠른 리뷰**: 작은 단위로 리뷰 가능

### ❌ Avoids

1. **거대한 커밋**: 모든 변경사항 한번에
2. **긴 메시지**: 3줄 이상의 긴 설명
3. **혼재된 변경**: 무관한 파일들 함께 커밋
4. **모호한 제목**: "update code", "fix bug" 같은 메시지

## Comparison

### ❌ Without `/commit`

```bash
# Manual grouping (time-consuming)
git add src/beanllm/domain/retrieval/hyde.py tests/domain/retrieval/test_hyde.py
git commit -m "Add HyDE"

git add src/beanllm/domain/retrieval/reranker.py tests/domain/retrieval/test_reranker.py
git commit -m "Add reranker"

# Easy to forget files or mix unrelated changes
```

### ✅ With `/commit`

```bash
# One command, automatic grouping
/commit

# Result: Perfect atomic commits automatically
```

## Configuration

Edit `.claude/commands/commit.md` to customize:

```python
# Minimum files per commit
MIN_FILES_PER_COMMIT = 1

# Maximum files per commit
MAX_FILES_PER_COMMIT = 5

# Always group tests with implementation
GROUP_TESTS_WITH_IMPL = True

# Separate docs into own commits
SEPARATE_DOCS = True
```

## Related Commands

- `/arch-check` - Run before committing to verify architecture
- `/code-review` - Run before committing large changes
- `/pr` - Create PR after commits

## Quick Reference

| Command | Behavior |
|---------|----------|
| `/commit` | Auto-analyze and split into multiple commits |
| `/commit --dry-run` | Show grouping without committing |
| `/commit --all` | Force single commit (not recommended) |

---

**💡 Philosophy**: Many small commits > One big commit

**🎯 Goal**: Each commit = One logical change in one domain
