# 🚀 Getting Started with Claude Code for beanllm

This guide will help you **activate and use** your Claude Code configuration.

---

## ✅ Setup Checklist

### 1. Verify Claude Code Installation

```bash
# Check Claude Code version
claude --version

# Check if .claude directory is recognized
cd /Users/leejungbin/Downloads/llmkit
ls -la .claude/

# Expected output:
# drwx------  10 leejungbin  staff   320 Jan 19 15:27 .claude
```

**✅ DONE**: `.claude/` directory exists with all configurations

---

### 2. ⚠️ TODO: Install Required Python Tools

The automation hooks need these tools:

```bash
# Install development tools
pip install black ruff mypy pytest pytest-cov

# Verify installation
black --version
ruff --version
mypy --version
pytest --version
```

**Why?**
- `black` - Auto-formatting (PostToolUse hook)
- `ruff` - Linting (PostToolUse hook)
- `mypy` - Type checking (build-fix command)
- `pytest` - Testing (TDD workflow)

---

### 3. ⚠️ TODO: Enable Anthropic Agent Skills (Frontend Only)

The frontend testing/design skills require marketplace access:

```bash
# Check if marketplace is installed
ls ~/.claude/plugins/marketplaces/anthropic-agent-skills/

# If not installed, add marketplace
claude marketplace add anthropics/skills

# Verify
ls ~/.claude/plugins/marketplaces/anthropic-agent-skills/skills/
# Should see: webapp-testing, frontend-design, etc.
```

**✅ DONE**: You already added the marketplace (`/plugin marketplace add anthropics/skills`)

---

### 4. ⚠️ TODO: Test Hooks Activation

Hooks are configured in `.claude/settings.json` but need testing:

```bash
# Start Claude Code in project
cd /Users/leejungbin/Downloads/llmkit
claude

# Test PostToolUse hook (auto-formatting)
# In Claude Code session:
# 1. Edit a Python file
# 2. Check if Black/Ruff runs automatically
```

**Expected behavior**:
```
You: "Edit src/beanllm/facade/core/client_facade.py"
Claude: [Makes edit]
Hook: 🎨 Auto-formatting Python file: src/beanllm/facade/core/client_facade.py
       1 file reformatted
```

---

### 5. ⚠️ TODO: Test Commands

Try each command to verify it works:

```bash
# In Claude Code session:

# Test architecture check
/arch-check

# Expected output:
# 🏗️ Checking Clean Architecture...
# ✅ No violations found

# Test TDD workflow
/tdd

# Expected: Red-Green-Refactor guide appears

# Test code review delegation
/code-review --path src/beanllm/facade/core/client_facade.py

# Expected: Opus agent starts comprehensive review
```

---

### 6. ⚠️ TODO: Configure Optional MCP Servers

MCP servers are **disabled by default** in `.claude/settings.json`. Enable if needed:

```json
// .claude/settings.json
{
  "mcpServers": {
    "github": {
      "_enabled": false,  // ← Change to true if you want GitHub integration
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": ""  // ← Add your token
      }
    },
    "postgres": {
      "_enabled": false,  // ← Change to true for pgvector
      "env": {
        "POSTGRES_CONNECTION_STRING": ""  // ← Add your connection string
      }
    }
  }
}
```

**When to enable**:
- `github`: If you want Claude to read/create issues, PRs
- `postgres`: If using pgvector for RAG
- `filesystem`: **Keep disabled** for security (already restricted to project dir)

---

## 🎯 How to Use in Your Codebase

### Scenario 1: Adding a New Feature (Complete Workflow)

**Goal**: Add HyDE query expansion to RAG pipeline

```bash
# Step 1: Start planning
/plan "Add HyDE (Hypothetical Document Embeddings) query expansion to RAG"

# Claude will:
# - Analyze existing RAG implementation
# - Create step-by-step plan
# - Identify files to modify
# - Estimate complexity

# Output example:
"""
## Implementation Plan: HyDE Query Expansion

### Steps:
1. Create `src/beanllm/domain/retrieval/hyde.py`
   - HyDEQueryExpander class
   - generate_hypothetical() method
   - expand_query() method

2. Modify `src/beanllm/service/impl/core/rag_service_impl.py`
   - Add use_hyde parameter
   - Integrate HyDE expander

3. Update `src/beanllm/facade/core/rag_facade.py`
   - Expose use_hyde option

4. Add tests in `tests/domain/retrieval/test_hyde.py`

### Estimated Files: 4
### Complexity: Medium
"""

# Step 2: Start TDD cycle
/tdd

# Claude will guide you through:
"""
🔴 RED Phase:
1. Write failing test in tests/domain/retrieval/test_hyde.py
2. Run: pytest tests/domain/retrieval/test_hyde.py -v
3. Expected: ❌ All tests fail (not implemented yet)

🟢 GREEN Phase:
1. Implement minimum code to pass tests
2. Run: pytest tests/domain/retrieval/test_hyde.py -v
3. Expected: ✅ All tests pass

🔵 REFACTOR Phase:
1. Run /dedup to find duplicate code
2. Extract common patterns
3. Run tests again to ensure still passing
"""

# Step 3: Write failing test first (RED)
"Create tests/domain/retrieval/test_hyde.py with failing tests"

# Step 4: Implement feature (GREEN)
"Implement src/beanllm/domain/retrieval/hyde.py"

# PostToolUse Hook automatically runs:
# 🎨 Auto-formatting Python file...
# ✅ black src/beanllm/domain/retrieval/hyde.py
# ✅ ruff check --fix src/beanllm/domain/retrieval/hyde.py

# Step 5: Remove duplication
/dedup

# Output:
"""
🔍 Found duplicate code patterns:

1. Cache logic (Lines 45-67 in hyde.py, Lines 89-111 in embedder.py)
   Suggestion: Extract to @with_cache decorator
   Potential reduction: 92% (40 lines → 3 lines)

2. Error handling (Lines 123-145 in hyde.py, Lines 234-256 in embedder.py)
   Suggestion: Use common error_handler() utility
   Potential reduction: 85% (23 lines → 3 lines)
"""

# Step 6: Verify architecture
/arch-check

# Output:
"""
🏗️ Verifying Clean Architecture...

✅ Facade → Handler dependencies: OK
✅ Handler → Service interface: OK
✅ Service → Domain: OK
✅ No circular dependencies: OK
✅ Absolute imports only: OK

✅ Clean Architecture verified
"""

# Step 7: Comprehensive review
/code-review

# Delegates to code-reviewer agent (Opus):
"""
# Code Review Report

## Summary
- ✅ Overall: APPROVED
- 🎯 Quality Score: 91/100
- ⚠️ Warnings: 1
- 💡 Suggestions: 3

## Clean Architecture: ✅ PASS
## Security: ✅ PASS
## Performance: ✅ GOOD
## Test Coverage: ✅ EXCELLENT (95%)

## Action Items:
1. Add docstring example to HyDEQueryExpander class
"""

# Step 8: Update documentation
/update-docs

# Output:
"""
📝 Detecting code changes...
✅ Found 3 changed files in domain/retrieval/

📚 Updating API Reference...
  - Added HyDEQueryExpander class documentation

📖 Updating README.md...
  - Added HyDE feature to Features section

📋 Updating CHANGELOG.md...
  - Added [0.3.0] section with HyDE feature

✅ Documentation updated successfully!
"""

# Step 9: Commit
git add .
git commit -m "feat(rag): Add HyDE query expansion

Implement Hypothetical Document Embeddings for improved retrieval:
- 20% accuracy improvement on benchmark dataset
- Generates hypothetical answers and embeds them instead of raw queries
- Added comprehensive tests with 95% coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Scenario 2: Fixing a Bug

**Goal**: Fix rate limit error handling in OpenAI provider

```bash
# Step 1: Reproduce with test (TDD)
/tdd

"Write failing test for rate limit handling"

# Step 2: Fix the bug
"Implement exponential backoff retry logic"

# PostToolUse Hook runs automatically:
# 🎨 Auto-formatting...

# Step 3: Verify architecture (quick check)
/arch-check

# Step 4: Update docs
/update-docs --scope changelog

# Output:
"""
📋 Updating CHANGELOG.md...
  - Added fix to [0.2.3] section

### Fixed
- Fixed rate limit handling in OpenAI provider with exponential backoff
"""

# Step 5: Commit
git add .
git commit -m "fix(provider): Handle rate limits with exponential backoff

Fixes #234

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Scenario 3: Large Refactoring

**Goal**: Extract duplicate cache logic across all services

```bash
# Step 1: Find all duplicates
/dedup

# Output:
"""
🔍 Scanning for duplicate code...

Found 15 instances of cache logic across:
- src/beanllm/service/impl/core/rag_service_impl.py (5 methods)
- src/beanllm/service/impl/core/chat_service_impl.py (4 methods)
- src/beanllm/service/impl/ml/vision_rag_service_impl.py (6 methods)

Total duplicate lines: 456
Potential reduction: 92% (456 → 40 lines)

Suggested refactoring:
1. Create @with_distributed_features decorator
2. Replace all cache logic with decorator
3. Estimated time: 2-3 hours
"""

# Step 2: Create plan
/plan "Extract cache logic to @with_distributed_features decorator"

# Step 3: Implement with TDD
/tdd

"Write tests for decorator first"
"Implement decorator"
"Replace old code with decorator"

# Step 4: Verify no architecture violations
/arch-check

# Step 5: Comprehensive review (use Opus for large refactoring)
/code-review --full

# Step 6: Update docs
/update-docs

# Step 7: Commit
git add .
git commit -m "refactor(service): Extract cache logic to decorator

Replace 456 lines of duplicate caching code with @with_distributed_features:
- 92% code reduction (456 → 40 lines)
- Consistent caching behavior across all services
- Easier to maintain and test

No functional changes. All 624 tests pass.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Scenario 4: Frontend Development (Playground)

**Goal**: Create chat UI component for playground/frontend

```bash
# This auto-activates frontend-design skill!

"Create a chat message component for playground/frontend"

# frontend-design skill activates automatically:
"""
🎨 Using Technical Elegance design system:

**Colors**: Deep Tech Green (Emerald), Data Amber, Insight Blue
**Typography**: JetBrains Mono (headers), Work Sans (body)
**Avoiding**: Inter font, purple gradients (AI slop)
"""

# Claude creates:
"""
// components/ChatMessage.tsx
'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

export function ChatMessage({ role, content, isStreaming }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'rounded-2xl p-6 backdrop-blur-sm',
        'border border-neutral-800/50',
        role === 'user'
          ? 'bg-gradient-to-br from-primary-900/20 to-primary-800/10'
          : 'bg-gradient-to-br from-neutral-900/80 to-neutral-800/40'
      )}
    >
      {content}
      {isStreaming && (
        <motion.span
          animate={{ opacity: [1, 0, 1] }}
          className="ml-1 inline-block h-4 w-1.5 bg-primary-500"
        />
      )}
    </motion.div>
  );
}
"""

# Test with E2E (webapp-testing auto-activates)

"Create E2E test for chat UI using Playwright"

# webapp-testing skill activates:
"""
// tests/e2e/chat.spec.ts
import { test, expect } from '@playwright/test';

test('chat message displays and streams', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.fill('textarea', 'Hello');
  await page.click('button:has-text("Send")');

  // Check streaming cursor appears
  await expect(page.locator('.message-assistant .cursor')).toBeVisible();

  // Check message content
  await expect(page.locator('.message-assistant')).toContainText('Hello');
});
"""
```

---

## 🔧 Advanced Usage

### Custom Triggers

Add your own keywords to auto-activate skills in `.claude/settings.json`:

```json
{
  "skills": {
    "webapp-testing": {
      "triggers": [
        "E2E 테스트",
        "integration test",
        "Playwright",
        "내 커스텀 키워드"  // ← Add your own
      ]
    }
  }
}
```

### Enable/Disable Skills

```json
{
  "skills": {
    "webapp-testing": {
      "enabled": false  // ← Completely disable
    },
    "frontend-design": {
      "enabled": true,
      "autoActivate": false  // ← Manual mode (need to explicitly request)
    }
  }
}
```

### Custom Hooks

Add your own automation in `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "tool == \"Edit\" && tool_input.file_path matches \"test_.*\\.py$\"",
        "hooks": [
          {
            "type": "command",
            "command": "pytest '${tool_input.file_path}' -v"
          }
        ]
      }
    ]
  }
}
```

---

## 📊 Cost Optimization Tips

### Use Right Model for Each Task

| Task | Command | Model | Cost | Rationale |
|------|---------|-------|------|-----------|
| Planning | `/plan` | Sonnet | $0.08 | Medium complexity |
| TDD Guide | `/tdd` | Sonnet | $0.05 | Straightforward |
| Architecture Check | `/arch-check` | Sonnet | $0.03 | Pattern matching |
| Deduplication | `/dedup` | Sonnet | $0.05 | Code analysis |
| **Code Review** | `/code-review` | **Opus** | **$0.50** | **Deep analysis** |
| Documentation | `/update-docs` | Haiku | $0.01 | Simple task |
| Bug Fix | Regular chat | Sonnet | $0.08 | General coding |

**Monthly Estimate** (with optimization):
- Daily development: $0.50/day × 20 days = **$10**
- Weekly comprehensive review (Opus): $2 × 4 = **$8**
- Documentation updates: $0.05 × 20 = **$1**
- **Total: ~$20/month**

### When to Use Opus (Expensive but Worth It)

✅ **Use Opus**:
- Before production deployment
- Major refactoring (1000+ lines)
- Security-critical code
- Public API changes
- Weekly comprehensive reviews

❌ **Don't Use Opus**:
- Simple bug fixes
- Documentation updates
- Quick feature additions
- Daily code checks

---

## 🐛 Troubleshooting

### Issue 1: Hooks Not Running

**Symptom**: Black/Ruff doesn't run after editing Python files

**Solution**:
```bash
# Check if tools are installed
black --version
ruff --version

# Check settings.json syntax
cat .claude/settings.json | python -m json.tool

# Restart Claude Code session
```

### Issue 2: Commands Not Found

**Symptom**: `/arch-check` shows "command not found"

**Solution**:
```bash
# Check if command files exist
ls -la .claude/commands/

# Verify file names match (no typos)
# Should be: arch-check.md (not arch_check.md)
```

### Issue 3: Skills Not Auto-Activating

**Symptom**: frontend-design doesn't activate when saying "UI 디자인"

**Solution**:
```json
// .claude/settings.json - check triggers
{
  "skills": {
    "frontend-design": {
      "enabled": true,  // ← Must be true
      "autoActivate": true,  // ← Must be true
      "triggers": [
        "UI 디자인"  // ← Must include this exact keyword
      ]
    }
  }
}
```

### Issue 4: Agents Not Delegating

**Symptom**: `/code-review` doesn't call Opus agent

**Solution**:
```bash
# Check if agent file exists
ls -la .claude/agents/code-reviewer.md

# Check if command delegates properly
cat .claude/commands/code-review.md | grep "code-reviewer"
```

---

## ✅ Final Checklist

Before starting development, ensure:

- [ ] ✅ Claude Code installed and updated
- [ ] ⚠️ Black, Ruff, MyPy, pytest installed
- [ ] ✅ Anthropic Agent Skills marketplace added
- [ ] ⚠️ Tested at least one command (`/arch-check`)
- [ ] ⚠️ Tested PostToolUse hook (auto-formatting)
- [ ] 📖 Read `.claude/README.md`
- [ ] 📖 Read `CLAUDE.md` workflow section
- [ ] 🎯 Ready to code!

---

## 🎉 You're All Set!

Try your first workflow:

```bash
cd /Users/leejungbin/Downloads/llmkit
claude

# In Claude Code session:
/plan "Add caching to embeddings"
```

**Questions?**
- Check `.claude/README.md` for reference
- Check `CLAUDE.md` for detailed workflows
- Check individual command files in `.claude/commands/`

**Happy Coding! 🫘**
