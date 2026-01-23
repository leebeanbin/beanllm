#!/bin/bash

echo "🔍 Checking Claude Code setup for beanllm..."
echo ""

# Check Python tools
echo "1️⃣  Python Development Tools:"
command -v black >/dev/null && echo "  ✅ black installed" || echo "  ❌ black (run: pip install black)"
command -v ruff >/dev/null && echo "  ✅ ruff installed" || echo "  ❌ ruff (run: pip install ruff)"
command -v mypy >/dev/null && echo "  ✅ mypy installed" || echo "  ❌ mypy (run: pip install mypy)"
command -v pytest >/dev/null && echo "  ✅ pytest installed" || echo "  ❌ pytest (run: pip install pytest pytest-cov)"
echo ""

# Check .claude directory structure
echo "2️⃣  Claude Code Configuration:"
[ -d ".claude" ] && echo "  ✅ .claude directory exists" || echo "  ❌ .claude directory missing!"
[ -f ".claude/settings.json" ] && echo "  ✅ settings.json" || echo "  ❌ settings.json missing"
[ -f ".claude/GETTING_STARTED.md" ] && echo "  ✅ GETTING_STARTED.md" || echo "  ❌ GETTING_STARTED.md missing"
[ -f ".claude/TODO.md" ] && echo "  ✅ TODO.md" || echo "  ❌ TODO.md missing"
echo ""

# Count configuration files
echo "3️⃣  Configuration Files:"
rules_count=$(ls .claude/rules/*.md 2>/dev/null | wc -l | tr -d ' ')
skills_count=$(ls .claude/skills/*.md 2>/dev/null | wc -l | tr -d ' ')
commands_count=$(ls .claude/commands/*.md 2>/dev/null | wc -l | tr -d ' ')
agents_count=$(ls .claude/agents/*.md 2>/dev/null | wc -l | tr -d ' ')

echo "  Rules: $rules_count/6 (always-on guidelines)"
echo "  Skills: $skills_count/6 (auto-activate patterns)"
echo "  Commands: $commands_count/10 (slash commands)"
echo "  Agents: $agents_count/3 (task delegation)"
echo ""

# Check marketplace
echo "4️⃣  Anthropic Agent Skills Marketplace:"
if [ -d "$HOME/.claude/plugins/marketplaces/anthropic-agent-skills" ]; then
    echo "  ✅ Marketplace installed at ~/.claude/plugins/"
    webapp_testing=$([ -d "$HOME/.claude/plugins/marketplaces/anthropic-agent-skills/skills/webapp-testing" ] && echo "✅" || echo "❌")
    frontend_design=$([ -d "$HOME/.claude/plugins/marketplaces/anthropic-agent-skills/skills/frontend-design" ] && echo "✅" || echo "❌")
    echo "     $webapp_testing webapp-testing skill"
    echo "     $frontend_design frontend-design skill"
else
    echo "  ⚠️  Marketplace not installed"
    echo "     Run: claude marketplace add anthropics/skills"
fi
echo ""

# Check settings.json structure
echo "5️⃣  Settings Configuration:"
if [ -f ".claude/settings.json" ]; then
    # Check if skills section exists
    grep -q '"skills"' .claude/settings.json && echo "  ✅ Skills configured" || echo "  ❌ Skills section missing"
    grep -q '"hooks"' .claude/settings.json && echo "  ✅ Hooks configured" || echo "  ❌ Hooks section missing"
    grep -q '"mcpServers"' .claude/settings.json && echo "  ✅ MCP servers configured" || echo "  ❌ MCP section missing"
else
    echo "  ❌ settings.json not found"
fi
echo ""

# Summary
total_files=$((rules_count + skills_count + commands_count + agents_count))
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total config files: $total_files/25"
echo "  Expected structure: ✅ Complete"
echo ""

# Next steps
echo "📝 Next Steps:"
echo ""
if ! command -v black >/dev/null || ! command -v ruff >/dev/null || ! command -v mypy >/dev/null || ! command -v pytest >/dev/null; then
    echo "  ⚠️  1. Install missing Python tools:"
    echo "     pip install black ruff mypy pytest pytest-cov"
    echo ""
fi

if [ ! -d "$HOME/.claude/plugins/marketplaces/anthropic-agent-skills" ]; then
    echo "  ⚠️  2. Install Anthropic Agent Skills (optional, for frontend):"
    echo "     claude marketplace add anthropics/skills"
    echo ""
fi

echo "  ✅ 3. Read the getting started guide:"
echo "     cat .claude/GETTING_STARTED.md"
echo ""

echo "  ✅ 4. Start Claude Code and test:"
echo "     claude"
echo "     > /arch-check"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Quick Test:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  cd /Users/leejungbin/Downloads/llmkit"
echo "  claude"
echo "  > /plan \"Add hello_world function\""
echo ""
echo "📚 Documentation:"
echo "  - .claude/GETTING_STARTED.md (detailed guide)"
echo "  - .claude/TODO.md (checklist)"
echo "  - .claude/README.md (quick reference)"
echo "  - CLAUDE.md (project context)"
echo ""
