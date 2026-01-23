# /pr - Smart Pull Request Creator

**트리거**: `/pr`
**모델**: sonnet
**설명**: 커밋 히스토리 분석하여 PR 자동 생성

## Command Description

현재 브랜치의 커밋들을 분석하여 GitHub Pull Request를 자동으로 생성합니다.

## Usage

```bash
# 기본: main 브랜치로 PR 생성
/pr

# 특정 베이스 브랜치로 PR
/pr --base develop

# Draft PR 생성
/pr --draft

# PR 제목/설명만 생성 (실제 PR 생성 안함)
/pr --dry-run
```

## How It Works

### Step 1: Analyze Commits

```bash
# 현재 브랜치와 main 차이 확인
git log main..HEAD --oneline

# 예시 출력:
# 6d4e7f0 docs: Update RAG and agent docs
# 5c3d6e9 feat(agent): Add debate pattern
# 4b2c5d8 feat(rag): Add cross-encoder reranker
# 3a1b4c7 feat(rag): Add HyDE query expansion
```

### Step 2: Group by Feature

```python
# 커밋 메시지에서 feature 추출
commits = [
    "feat(rag): Add HyDE query expansion",
    "feat(rag): Add cross-encoder reranker",
    "feat(agent): Add debate pattern",
    "docs: Update RAG and agent docs"
]

# 그룹화
groups = {
    "rag": [
        "Add HyDE query expansion",
        "Add cross-encoder reranker"
    ],
    "agent": [
        "Add debate pattern"
    ],
    "docs": [
        "Update RAG and agent docs"
    ]
}
```

### Step 3: Generate PR Title

**Format**: `<type>(<scope>): <summary>`

**Rules**:
- 가장 중요한 변경사항 기준
- 여러 feature면 가장 큰 scope 선택
- 50자 이내

```bash
# ✅ Good
feat(rag): Add HyDE and reranker
feat(agent): Add debate pattern
fix(chat): Handle rate limits and retries

# ❌ Bad
Add new features
Update code
feat(rag): Add HyDE (Hypothetical Document Embeddings) query expansion and cross-encoder reranker with 20% accuracy improvement
```

### Step 4: Generate PR Description

**Template**:

```markdown
## Summary
- Feature 1
- Feature 2
- Feature 3

## Changes
- Detailed change 1
- Detailed change 2

## Test Plan
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### Step 5: Create PR

```bash
# gh CLI 사용
gh pr create \
  --title "feat(rag): Add HyDE and reranker" \
  --body "$(cat pr_description.md)" \
  --base main
```

## Example: Simple Feature PR

**Commits**:
```bash
$ git log main..HEAD --oneline
3a1b4c7 feat(rag): Add HyDE query expansion
2b1c3d8 test(rag): Add HyDE tests
```

**Generated PR**:

```bash
$ /pr

# Output:
"""
📝 Analyzing commits...
  Found 2 commits on branch: feat/rag-hyde

🎯 PR Title:
feat(rag): Add HyDE query expansion

📋 PR Description:
───────────────────────────────────
## Summary
- Add HyDE (Hypothetical Document Embeddings) query expansion
- 20% improvement in retrieval accuracy

## Changes
- Implement HyDEQueryExpander class
- Integrate HyDE into RAG service
- Add comprehensive unit tests (95% coverage)

## Test Plan
- [x] Unit tests pass (10/10)
- [x] Integration tests pass (3/3)
- [x] Benchmarked on 1,000 documents

🤖 Generated with [Claude Code](https://claude.com/claude-code)
───────────────────────────────────

🚀 Create PR? (yes/no/edit)
"""

> yes

# Executes:
gh pr create --title "feat(rag): Add HyDE query expansion" --body "..."

# Result:
✅ PR created: https://github.com/yourname/llmkit/pull/123
```

## Example: Multi-Feature PR

**Commits**:
```bash
$ git log main..HEAD --oneline
6d4e7f0 docs: Update RAG and agent docs
5c3d6e9 feat(agent): Add debate pattern
4b2c5d8 feat(rag): Add cross-encoder reranker
3a1b4c7 feat(rag): Add HyDE query expansion
```

**Generated PR**:

```bash
$ /pr

# Output:
"""
📝 Analyzing commits...
  Found 4 commits on branch: feat/rag-improvements

🎯 PR Title:
feat(rag): Add HyDE and reranker

📋 PR Description:
───────────────────────────────────
## Summary
- Add HyDE query expansion (20% accuracy ↑)
- Add cross-encoder reranker
- Add multi-agent debate pattern
- Update documentation

## RAG Features
- **HyDE**: Generate hypothetical answers for better retrieval
- **Reranker**: Re-rank results with cross-encoder model

## Multi-Agent Features
- **Debate Pattern**: Multiple agents discuss and reach consensus

## Test Plan
- [x] All unit tests pass (45/45)
- [x] Integration tests pass (12/12)
- [x] RAG accuracy improved by 20%
- [x] Multi-agent debate tested with 3 agents

🤖 Generated with [Claude Code](https://claude.com/claude-code)
───────────────────────────────────

🚀 Create PR? (yes/no/edit)
"""
```

## PR Description Sections

### 1. Summary (Required)

**Format**: Bullet points of main changes

```markdown
## Summary
- Add HyDE query expansion
- Add cross-encoder reranker
- Improve retrieval accuracy by 20%
```

### 2. Changes (Auto-generated)

**Source**: From commit messages and git diff

```markdown
## Changes

### RAG Domain
- Implement `HyDEQueryExpander` class
- Add `CrossEncoderReranker` class
- Update `RAGServiceImpl` to use HyDE

### Tests
- Add 15 unit tests for HyDE
- Add 8 unit tests for reranker
- 95% coverage achieved
```

### 3. Test Plan (Checklist)

**Auto-detect**:
- Unit tests존재 → "Unit tests pass"
- Integration tests 존재 → "Integration tests pass"
- 성능 개선 언급 → "Performance benchmarked"

```markdown
## Test Plan
- [x] Unit tests pass (23/23)
- [x] Integration tests pass (5/5)
- [x] Manual testing on 1,000 documents
- [x] Performance improved by 20%
```

### 4. Breaking Changes (If any)

```markdown
## ⚠️ Breaking Changes
- `RAGChain.from_documents()` now requires `vector_store` parameter
- Migration guide: See docs/MIGRATION.md
```

### 5. Related Issues

**Auto-detect** from commit messages:

```markdown
## Related Issues
Closes #123
Fixes #234
Related to #345
```

## Draft PR Mode

```bash
# Create draft PR (not ready for review)
/pr --draft

# Output:
"""
📝 Creating DRAFT Pull Request...

✅ Draft PR created: https://github.com/yourname/llmkit/pull/123

💡 Mark as ready for review:
   gh pr ready 123
"""
```

**Use case**:
- Work in progress
- Need CI feedback
- Want to show progress

## Dry Run Mode

```bash
# Generate PR title/description only
/pr --dry-run

# Output:
"""
🎯 PR Title:
feat(rag): Add HyDE query expansion

📋 PR Description:
[Full description here]

💡 To create PR, run:
   /pr
"""
```

**Use case**:
- Preview before creating
- Copy description manually
- Generate template

## Integration with Workflow

```bash
# Complete feature development

# 1. Plan
/plan "Add HyDE to RAG"

# 2. TDD + Multiple commits
/tdd
# [write code...]
/commit  # → feat(rag): Add HyDE expander
/commit  # → feat(rag): Integrate HyDE into service
/commit  # → test(rag): Add HyDE tests

# 3. Review
/arch-check
/code-review

# 4. Update docs
/update-docs
/commit  # → docs(rag): Add HyDE documentation

# 5. Push
git push -u origin feat/rag-hyde

# 6. Create PR
/pr

# Result: Clean PR with 4 atomic commits
```

## PR Template Customization

Edit `.github/pull_request_template.md`:

```markdown
## Summary
<!-- Brief description of changes -->

## Changes
<!-- Detailed list of changes -->

## Test Plan
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows Clean Architecture
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## GitHub CLI Setup

```bash
# Install gh CLI
brew install gh

# Authenticate
gh auth login

# Set default repo
gh repo set-default yourname/llmkit

# Verify
gh pr list
```

## Best Practices

### ✅ Good PR Practices

1. **Small PRs**: < 400 lines changed
2. **Focused**: One feature per PR
3. **Atomic commits**: Each commit builds/tests
4. **Clear description**: What, why, how
5. **Test plan**: How to verify

### ❌ Bad PR Practices

1. **Giant PRs**: > 1000 lines
2. **Mixed features**: 3+ unrelated features
3. **Broken commits**: Intermediate commits don't build
4. **Vague description**: "update code"
5. **No tests**: "tests coming in next PR"

## Size Guidelines

| Lines Changed | Review Time | Merge Speed |
|---------------|-------------|-------------|
| < 100 | 15 min | Fast ✅ |
| 100-400 | 1 hour | Medium 🟡 |
| 400-1000 | 2-4 hours | Slow 🟠 |
| > 1000 | 1+ days | Very Slow 🔴 |

**Recommendation**: Keep PRs < 400 lines

## Related Commands

- `/commit` - Create atomic commits before PR
- `/code-review` - Review before creating PR
- `/update-docs` - Update docs before PR

## Quick Reference

| Command | Purpose |
|---------|---------|
| `/pr` | Create PR to main |
| `/pr --base develop` | Create PR to develop |
| `/pr --draft` | Create draft PR |
| `/pr --dry-run` | Preview PR without creating |

---

**💡 Pro Tip**: Use `/commit` for many small commits, then `/pr` to create a well-structured PR

**🎯 Goal**: Clear PR that tells a story of what changed and why
