# ⚠️ TODO: Setup Tasks

Quick checklist of what you need to do to fully activate Claude Code.

---

## 🔴 CRITICAL (Do First)

### 1. Install Python Development Tools

```bash
# These are required for automation hooks
pip install black ruff mypy pytest pytest-cov

# Verify
black --version   # Should show version
ruff --version    # Should show version
mypy --version    # Should show version
pytest --version  # Should show version
```

**Why**: PostToolUse hooks auto-format Python files with Black/Ruff.

**Status**: ⚠️ **NOT DONE**

---

## 🟡 IMPORTANT (Do Soon)

### 2. Test Commands

Try each command to make sure it works:

```bash
# In Claude Code session (run 'claude' in project root):

# Test 1: Architecture check
/arch-check
# Expected: ✅ No violations found

# Test 2: TDD guide
/tdd
# Expected: Red-Green-Refactor guide appears

# Test 3: Code review
/code-review --path src/beanllm/facade/core/client_facade.py
# Expected: Opus agent starts review

# Test 4: Documentation update
/update-docs
# Expected: Docs updated successfully
```

**Status**: ⚠️ **NOT DONE**

---

### 3. Test Hooks

Verify automation works:

```bash
# Start Claude Code
claude

# Edit a Python file
"Edit src/beanllm/facade/core/client_facade.py - add a comment"

# Expected output:
# 🎨 Auto-formatting Python file: src/beanllm/facade/core/client_facade.py
# 1 file reformatted
```

**Status**: ⚠️ **NOT DONE**

---

## 🟢 OPTIONAL (Nice to Have)

### 4. Enable MCP Servers (If Needed)

Only enable if you need these integrations:

**GitHub MCP** (for reading issues, creating PRs):
```bash
# Edit .claude/settings.json
# Change "_enabled": false → "_enabled": true
# Add your GITHUB_PERSONAL_ACCESS_TOKEN
```

**Postgres MCP** (for pgvector RAG):
```bash
# Edit .claude/settings.json
# Change "_enabled": false → "_enabled": true
# Add your POSTGRES_CONNECTION_STRING
```

**Status**: ⚠️ **NOT DONE** (but okay, it's optional)

---

### 5. Test Marketplace Skills (Frontend Only)

Only needed for playground/frontend development:

```bash
# Test webapp-testing skill
"Create E2E test for playground/frontend using Playwright"
# Expected: webapp-testing skill auto-activates

# Test frontend-design skill
"Design a chat message component for playground/frontend"
# Expected: frontend-design skill auto-activates with Technical Elegance system
```

**Status**: ⚠️ **NOT DONE** (but okay, only needed for frontend work)

---

## 📋 Quick Verification Script

Run this to check everything at once:

```bash
#!/bin/bash
echo "🔍 Checking Claude Code setup..."

# Check Python tools
echo "1. Python tools:"
command -v black >/dev/null && echo "  ✅ black" || echo "  ❌ black (run: pip install black)"
command -v ruff >/dev/null && echo "  ✅ ruff" || echo "  ❌ ruff (run: pip install ruff)"
command -v mypy >/dev/null && echo "  ✅ mypy" || echo "  ❌ mypy (run: pip install mypy)"
command -v pytest >/dev/null && echo "  ✅ pytest" || echo "  ❌ pytest (run: pip install pytest)"

# Check .claude directory
echo "2. Claude Code config:"
[ -d ".claude" ] && echo "  ✅ .claude directory" || echo "  ❌ .claude directory missing"
[ -f ".claude/settings.json" ] && echo "  ✅ settings.json" || echo "  ❌ settings.json missing"
[ -d ".claude/commands" ] && echo "  ✅ commands/" || echo "  ❌ commands/ missing"
[ -d ".claude/agents" ] && echo "  ✅ agents/" || echo "  ❌ agents/ missing"

# Count files
echo "3. Configuration files:"
echo "  Rules: $(ls .claude/rules/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  Skills: $(ls .claude/skills/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  Commands: $(ls .claude/commands/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  Agents: $(ls .claude/agents/ 2>/dev/null | wc -l | tr -d ' ')"

# Check marketplace
echo "4. Anthropic Agent Skills:"
if [ -d "$HOME/.claude/plugins/marketplaces/anthropic-agent-skills" ]; then
    echo "  ✅ Marketplace installed"
else
    echo "  ⚠️  Marketplace not installed (run: claude marketplace add anthropics/skills)"
fi

echo ""
echo "📊 Summary:"
echo "  Total config files: $(($(ls .claude/rules/ 2>/dev/null | wc -l) + $(ls .claude/skills/ 2>/dev/null | wc -l) + $(ls .claude/commands/ 2>/dev/null | wc -l) + $(ls .claude/agents/ 2>/dev/null | wc -l)))"
echo ""
echo "Next steps:"
echo "1. Install missing Python tools (see above)"
echo "2. Run 'claude' in this directory to start"
echo "3. Try '/arch-check' to test commands"
```

Save as `.claude/check-setup.sh` and run:
```bash
chmod +x .claude/check-setup.sh
./.claude/check-setup.sh
```

---

## 🎯 Priority Order

Do these in order:

1. **Install Python tools** (black, ruff, mypy, pytest) - 5 minutes
2. **Test one command** (`/arch-check`) - 2 minutes
3. **Test auto-formatting hook** (edit a file) - 3 minutes
4. **Read GETTING_STARTED.md** - 10 minutes
5. **Try first workflow** (see below) - 30 minutes

---

## 🚀 Your First Workflow (After Setup)

Once everything is installed, try this:

```bash
# 1. Start Claude Code
cd /Users/leejungbin/Downloads/llmkit
claude

# 2. Create a simple plan
/plan "Add a hello_world() function to utils"

# 3. Use TDD
/tdd

# 4. Write test first
"Create tests/utils/test_hello.py with a failing test"

# 5. Implement
"Implement src/beanllm/utils/hello.py with hello_world() function"

# 6. Check architecture
/arch-check

# 7. Review
/code-review --path src/beanllm/utils/hello.py

# 8. Done! Now you know how it works.
```

---

## ✅ Completion Criteria

You're ready when:

- [ ] Black/Ruff auto-format after editing Python files
- [ ] `/arch-check` runs without errors
- [ ] `/code-review` successfully delegates to Opus agent
- [ ] You've completed your first feature with the full workflow

---

## 📚 Documentation References

- **`.claude/GETTING_STARTED.md`** ← Detailed guide with examples (READ THIS)
- **`.claude/README.md`** ← Quick reference
- **`CLAUDE.md`** ← Full project context
- **`.claude/commands/*.md`** ← Individual command docs

---

## 🆘 Need Help?

If stuck:

1. Check `.claude/GETTING_STARTED.md` Troubleshooting section
2. Run verification script above
3. Check individual command files for details
4. Ask Claude: "Why isn't /arch-check working?"

---

**Updated**: 2026-01-20
**Status**: ⚠️ Setup incomplete - follow checklist above
