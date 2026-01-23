# 📝 Commit Workflow Guide - Many Small Commits

This guide shows how to use `/commit` and `/pr` for efficient code management with **many tiny, atomic commits**.

---

## 🎯 Philosophy: Many Small Commits

### Why Small Commits?

```bash
# ❌ Bad: One giant commit
[main 3a1b4c7] Add new features
  15 files changed, 523 insertions(+), 89 deletions(-)

# Problems:
# - Can't rollback specific changes
# - Hard to review
# - Unclear what each change does
# - Difficult to debug with git bisect

# ✅ Good: Many atomic commits
[main 3a1b4c7] feat(rag): Add HyDE query expander
  2 files changed, 45 insertions(+)

[main 4b2c5d8] test(rag): Add HyDE tests
  1 file changed, 32 insertions(+)

[main 5c3d6e9] feat(rag): Integrate HyDE into service
  1 file changed, 23 insertions(+)

[main 6d4e7f0] docs(rag): Add HyDE documentation
  2 files changed, 15 insertions(+)

# Benefits:
# ✅ Can rollback only commit 3 if integration has issues
# ✅ Easy to review each commit
# ✅ Clear progression of changes
# ✅ Easy to find when bug was introduced
```

---

## 🚀 Complete Workflow Example

### Scenario: Add HyDE Query Expansion to RAG

```bash
# You edited 11 files:
# - src/beanllm/domain/retrieval/hyde.py (new)
# - src/beanllm/domain/retrieval/reranker.py (new)
# - src/beanllm/service/impl/core/rag_service_impl.py (modified)
# - tests/domain/retrieval/test_hyde.py (new)
# - tests/domain/retrieval/test_reranker.py (new)
# - README.md (modified)
# - docs/API_REFERENCE.md (modified)
# - CHANGELOG.md (modified)
# - playground/frontend/app/page.tsx (modified)
# - playground/backend/main.py (modified)
# - pyproject.toml (modified)

# Instead of manually grouping, use /commit:
/commit

# Output:
"""
📝 Analyzing 11 changed files...

📊 Automatic grouping by domain:

Group 1: RAG Domain - HyDE Feature (3 files)
  ├─ src/beanllm/domain/retrieval/hyde.py
  ├─ tests/domain/retrieval/test_hyde.py
  └─ src/beanllm/service/impl/core/rag_service_impl.py (integration)

Group 2: RAG Domain - Reranker Feature (2 files)
  ├─ src/beanllm/domain/retrieval/reranker.py
  └─ tests/domain/retrieval/test_reranker.py

Group 3: Documentation (3 files)
  ├─ README.md
  ├─ docs/API_REFERENCE.md
  └─ CHANGELOG.md

Group 4: Playground Backend (1 file)
  └─ playground/backend/main.py

Group 5: Playground Frontend (1 file)
  └─ playground/frontend/app/page.tsx

Group 6: Dependencies (1 file)
  └─ pyproject.toml

💡 Suggested: 6 atomic commits

Proceed? (yes/no/edit)
"""

> yes

# Executes automatically:

[1/6] Committing RAG - HyDE...
[main 3a1b4c7] feat(rag): Add HyDE query expansion
 3 files changed, 78 insertions(+)

[2/6] Committing RAG - Reranker...
[main 4b2c5d8] feat(rag): Add cross-encoder reranker
 2 files changed, 56 insertions(+)

[3/6] Committing Documentation...
[main 5c3d6e9] docs(rag): Update RAG documentation
 3 files changed, 45 insertions(+)

[4/6] Committing Playground Backend...
[main 6d4e7f0] feat(playground): Add RAG endpoints
 1 file changed, 23 insertions(+)

[5/6] Committing Playground Frontend...
[main 7e5f8g1] feat(playground): Add RAG UI
 1 file changed, 34 insertions(+)

[6/6] Committing Dependencies...
[main 8f6g9h2] chore: Update dependencies
 1 file changed, 2 insertions(+)

✅ Created 6 commits successfully!

📊 Summary:
  - RAG domain: 2 commits
  - Documentation: 1 commit
  - Playground: 2 commits
  - Dependencies: 1 commit

🎯 Each commit is focused and independent
```

---

## 📋 How `/commit` Works

### 1. Read All Changed Files

```bash
# Checks both staged and unstaged
git status --short

# Reads actual diff content
git diff HEAD
```

### 2. Analyze & Group by Domain

**Grouping Priority**:

1. **Feature Domain** (highest priority)
   - RAG: retrieval, embeddings, loaders
   - Multi-Agent: multi_agent, strategies, communication
   - Knowledge Graph: knowledge_graph, neo4j
   - Vision: vision, ocr, florence
   - Audio: audio, transcription

2. **beanllm Layer**
   - domain/ → "domain"
   - service/ → "service"
   - handler/ → "handler"
   - facade/ → "facade"

3. **File Type**
   - tests/ → Group with implementation
   - docs/ → Separate commit
   - playground/ → Separate by backend/frontend
   - config files → Separate commit

### 3. Generate Short Messages

**Format**: `<type>(<scope>): <subject>` (50 chars max)

```bash
# ✅ Good (short & clear)
feat(rag): Add HyDE query expansion
fix(chat): Handle rate limit errors
test(agent): Add debate pattern tests
docs(rag): Update retrieval docs

# ❌ Bad (too long)
feat(rag): Add HyDE (Hypothetical Document Embeddings) query expansion with 20% accuracy improvement
```

### 4. Create Multiple Commits

Each group becomes one commit automatically.

---

## 🎨 Real Examples

### Example 1: Mixed Changes → 4 Commits

**Your edits**:
```bash
# You modified:
- src/beanllm/domain/retrieval/hyde.py
- src/beanllm/domain/multi_agent/debate.py
- tests/domain/retrieval/test_hyde.py
- tests/domain/multi_agent/test_debate.py
- README.md
```

**`/commit` does**:
```bash
# Commit 1: RAG domain
[main abc123] feat(rag): Add HyDE query expander
  src/beanllm/domain/retrieval/hyde.py
  tests/domain/retrieval/test_hyde.py

# Commit 2: Multi-Agent domain
[main def456] feat(agent): Add debate pattern
  src/beanllm/domain/multi_agent/debate.py
  tests/domain/multi_agent/test_debate.py

# Commit 3: Documentation
[main ghi789] docs: Update RAG and agent docs
  README.md
```

### Example 2: Tests Only → 1 Commit

**Your edits**:
```bash
# You modified:
- tests/domain/retrieval/test_hyde.py
```

**`/commit` does**:
```bash
[main abc123] test(rag): Add HyDE tests
  tests/domain/retrieval/test_hyde.py
```

### Example 3: Frontend + Backend → 2 Commits

**Your edits**:
```bash
# You modified:
- playground/backend/main.py
- playground/frontend/app/page.tsx
```

**`/commit` does**:
```bash
# Commit 1: Backend
[main abc123] feat(playground): Add RAG endpoint
  playground/backend/main.py

# Commit 2: Frontend
[main def456] feat(playground): Add RAG UI
  playground/frontend/app/page.tsx
```

---

## 🔄 Integration with `/pr`

After many small commits, create a PR:

```bash
# You made 6 commits with /commit
git log --oneline
8f6g9h2 chore: Update dependencies
7e5f8g1 feat(playground): Add RAG UI
6d4e7f0 feat(playground): Add RAG endpoints
5c3d6e9 docs(rag): Update RAG documentation
4b2c5d8 feat(rag): Add cross-encoder reranker
3a1b4c7 feat(rag): Add HyDE query expansion

# Push
git push -u origin feat/rag-improvements

# Create PR (analyzes all 6 commits)
/pr

# Output:
"""
📝 Analyzing 6 commits...

🎯 PR Title:
feat(rag): Add HyDE and reranker

📋 PR Description:
───────────────────────────────────
## Summary
- Add HyDE query expansion
- Add cross-encoder reranker
- Update playground with RAG features

## RAG Features
- **HyDE**: Hypothetical Document Embeddings
- **Reranker**: Cross-encoder reranking

## Changes
- 6 files added/modified
- 238 lines added
- 95% test coverage

## Test Plan
- [x] Unit tests pass (23/23)
- [x] Integration tests pass (5/5)
- [x] Manual testing completed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
───────────────────────────────────

Create PR? (yes/no)
"""

> yes

✅ PR created: https://github.com/yourname/llmkit/pull/123
```

---

## 💡 Best Practices

### ✅ Do This

1. **Edit multiple files freely**
   - Don't worry about staging
   - Just code naturally

2. **Use `/commit` when done**
   - Automatically groups by domain
   - Creates multiple atomic commits

3. **Keep commits small**
   - Each commit = one logical change
   - Easy to review and rollback

4. **Use `/pr` for GitHub**
   - Analyzes all commits
   - Generates structured PR description

### ❌ Don't Do This

1. **Manual `git add` for each group**
   - Time-consuming
   - Error-prone
   - Let `/commit` do it

2. **One giant commit**
   - `git add .`
   - `git commit -m "update code"`
   - Hard to review/rollback

3. **Long commit messages**
   - Keep subject < 50 chars
   - Body is optional

---

## 🎯 Daily Workflow

```bash
# Morning: Start feature
/plan "Add caching to embeddings"

# Code for 2 hours, edit 10+ files

# Lunch break: Commit progress
/commit
# → Creates 3-5 atomic commits

# Afternoon: Continue coding

# End of day: Commit again
/commit
# → Creates another 3-5 commits

# Feature complete: Create PR
git push -u origin feat/caching
/pr
# → Analyzes all 6-10 commits, creates structured PR
```

---

## 📊 Benefits Comparison

| Approach | Review Time | Rollback | Debugging | History Clarity |
|----------|-------------|----------|-----------|-----------------|
| **One big commit** | 2+ hours | Hard 🔴 | Hard 🔴 | Poor 🔴 |
| **Manual grouping** | 1 hour | Medium 🟡 | Medium 🟡 | Good 🟢 |
| **`/commit` auto** | 30 min ✅ | Easy ✅ | Easy ✅ | Excellent ✅ |

---

## 🔧 Commands Summary

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/commit` | Auto-split into atomic commits | After editing multiple files |
| `/commit --dry-run` | Preview grouping | Before committing |
| `/pr` | Create GitHub PR | After pushing commits |
| `/pr --draft` | Create draft PR | Work in progress |

---

## 📚 Related Docs

- `.claude/commands/commit.md` - Full `/commit` documentation
- `.claude/commands/pr.md` - Full `/pr` documentation
- `.claude/rules/git-workflow.md` - Commit message format rules

---

**💡 Remember**: Code freely, commit smartly with `/commit`, PR automatically with `/pr`

**🎯 Goal**: Clear git history with many small, focused commits
