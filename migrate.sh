#!/bin/bash

cd "$(dirname "$0")"

echo "🔄 Starting migration: llm-model-manager → llmkit"
echo ""

# 1. Rename directory
echo "Step 1: Renaming llm_model_manager → llmkit..."
if [ -d "llm_model_manager" ]; then
    mv llm_model_manager llmkit
    echo "   ✅ Directory renamed"
else
    echo "   ⚠️  llm_model_manager not found (maybe already renamed?)"
    if [ -d "llmkit" ]; then
        echo "   ✅ llmkit directory already exists"
    fi
fi
echo ""

# 2. Clean old cache
echo "Step 2: Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "   ✅ Cache cleaned"
echo ""

# 3. Uninstall old package
echo "Step 3: Uninstalling old packages..."
pip uninstall -y llm-model-manager 2>/dev/null || echo "   (not installed)"
pip uninstall -y llmkit 2>/dev/null || echo "   (not installed)"
echo ""

# 4. Install new package
echo "Step 4: Installing llmkit in development mode..."
pip install -e .
echo ""

# 5. Verify installation
echo "Step 5: Verifying installation..."
python -c "from llmkit import get_registry; r = get_registry(); print(f'   ✅ Import works! {len(r.get_available_models())} models found')"
echo ""

# 6. Test CLI
echo "Step 6: Testing CLI..."
llmkit summary
echo ""

echo "🎉 Migration complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Run tests: pytest"
echo "   2. Try examples: python examples/basic_usage.py"
echo "   3. Build package: python -m build"
echo ""
echo "📚 See README.md for usage guide"
