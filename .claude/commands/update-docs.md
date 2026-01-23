# /update-docs - Documentation Synchronization

**트리거**: `/update-docs`
**모델**: haiku (fast & cost-effective)
**설명**: 코드 변경사항을 문서에 자동 반영

## Command Description

코드 변경 후 README, API 문서, CHANGELOG, Docstring을 자동으로 업데이트합니다.

## Usage

```bash
/update-docs
/update-docs --scope api
/update-docs --scope readme
/update-docs --scope changelog
/update-docs --all
```

## What Gets Updated

### 1. API Reference (docs/API_REFERENCE.md)

**검사**:
```bash
# Public API 변경사항 확인
git diff HEAD~1 src/beanllm/facade/core/client_facade.py
git diff HEAD~1 src/beanllm/facade/core/rag_facade.py
```

**업데이트**:
- 새 메서드 추가 → API 문서에 추가
- 메서드 시그니처 변경 → 문서 업데이트
- Docstring 변경 → 예제 코드 업데이트
- Deprecated 메서드 → 경고 추가

**예시**:
```python
# src/beanllm/facade/core/client_facade.py 변경
class Client:
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7  # ← 새 파라미터 추가
    ) -> ChatResponse:
        """Chat with LLM."""
        pass
```

→ `docs/API_REFERENCE.md` 업데이트:
```markdown
## Client.chat()

Chat with LLM using various providers.

**Parameters**:
- `messages` (List[Dict[str, str]]): Chat messages
- `model` (str, optional): Model name (default: "gpt-4o")
- `temperature` (float, optional): Sampling temperature 0.0-2.0 (default: 0.7) ← 추가

**Returns**: `ChatResponse`

**Example**:
```python
client = Client(model="gpt-4o")
response = await client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7  # ← 업데이트
)
```
```

### 2. README.md

**검사**:
```bash
# 주요 기능 변경
git diff HEAD~1 src/beanllm/facade/
git diff HEAD~1 README.md
```

**업데이트**:
- 새 기능 추가 → Features 섹션 업데이트
- API 변경 → Quick Start 예제 업데이트
- 설치 요구사항 변경 → Installation 섹션 업데이트

**예시**:
```markdown
## Features

- ✅ **7 LLM Providers**: OpenAI, Claude, Gemini, DeepSeek, Perplexity, Ollama, Meta
- ✅ **RAG Pipeline**: Document loading, chunking, embedding, retrieval
- ✅ **Multi-Agent**: Debate, Sequential, Hierarchical patterns
- ✅ **Knowledge Graph**: Neo4j integration, entity/relation extraction
- ✅ **HyDE Query Expansion**: Hypothetical Document Embeddings for 20% accuracy improvement ← 추가
```

### 3. CHANGELOG.md

**자동 생성**:
```bash
# Git 커밋 메시지에서 CHANGELOG 생성
git log --oneline --since="2024-01-01" | grep -E "^[a-f0-9]+ (feat|fix|refactor|perf)"
```

**포맷**:
```markdown
# Changelog

## [0.3.0] - 2026-01-20

### Added
- HyDE (Hypothetical Document Embeddings) query expansion for RAG
- Multi-agent debate pattern support
- Knowledge graph RAG integration

### Changed
- RAG pipeline now supports query expansion strategies
- Improved embedding caching performance (5.7× faster)

### Fixed
- Fixed rate limit handling in OpenAI provider
- Fixed memory leak in vector store cleanup

### Deprecated
- `RAGChain.from_documents(vector_store=None)` - use `vector_store` parameter

### Performance
- Similarity search optimized from O(n log n) to O(n log k)
- Reduced code duplication by 92% using decorators
```

### 4. Docstrings (In-Code Documentation)

**검사**:
```bash
# Docstring 누락 확인
pydocstyle src/beanllm/ --count
```

**업데이트**:
- Public 메서드에 Docstring 없음 → 자동 생성
- 파라미터 변경 → Docstring Args 업데이트
- 예제 코드 outdated → 최신 API로 업데이트

**예시**:
```python
# Before: Docstring 없음
def expand_query(self, query: str) -> List[float]:
    pass

# After: Docstring 추가
def expand_query(self, query: str) -> List[float]:
    """
    Expand query using HyDE (Hypothetical Document Embeddings).

    Generates a hypothetical document that would answer the query,
    then embeds the hypothetical document instead of the raw query.
    This improves retrieval accuracy by 20% on average.

    Args:
        query: User query to expand

    Returns:
        Expanded query embedding (1536-dim for OpenAI)

    Raises:
        ValueError: If query is empty
        APIError: If LLM call fails

    Example:
        >>> expander = HyDEQueryExpander(model="gpt-4o")
        >>> embedding = await expander.expand_query("What is RAG?")
        >>> len(embedding)
        1536
    """
    pass
```

### 5. Tutorial & Guides (docs/)

**업데이트 대상**:
- `docs/QUICKSTART.md` - 빠른 시작 가이드
- `docs/TUTORIAL.md` - 단계별 튜토리얼
- `docs/ADVANCED.md` - 고급 기능 가이드
- `docs/MIGRATION.md` - 마이그레이션 가이드 (Breaking changes 시)

## Execution Steps

### Step 1: Detect Changes

```bash
echo "📝 Detecting code changes..."

# 변경된 Public API 파일
changed_files=$(git diff --name-only HEAD~1 src/beanllm/facade/)

# 새로 추가된 Public 메서드
git diff HEAD~1 src/beanllm/facade/ | grep "^+\s*def " | sed 's/^+\s*//'
```

### Step 2: Extract API Signatures

```python
import ast
from pathlib import Path

def extract_public_api(file_path: str):
    """Extract public methods from a Python file."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    public_methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'):  # Public method
                # Extract signature, docstring, type hints
                public_methods.append({
                    'name': node.name,
                    'signature': ast.unparse(node.args),
                    'docstring': ast.get_docstring(node),
                    'returns': node.returns
                })

    return public_methods
```

### Step 3: Generate Documentation

```python
def generate_api_doc(method_info):
    """Generate markdown documentation for a method."""
    doc = f"## {method_info['name']}()\n\n"
    doc += f"{method_info['docstring']}\n\n"
    doc += f"**Signature**: `{method_info['signature']}`\n\n"
    doc += f"**Returns**: `{method_info['returns']}`\n\n"
    return doc
```

### Step 4: Update Files

```bash
# README.md 업데이트
echo "Updating README.md..."

# API_REFERENCE.md 업데이트
echo "Updating docs/API_REFERENCE.md..."

# CHANGELOG.md 업데이트
echo "Updating CHANGELOG.md..."
```

### Step 5: Verify Consistency

```bash
# 모든 Public API가 문서화되었는지 확인
python scripts/verify_docs.py

# Docstring 스타일 검사
pydocstyle src/beanllm/facade/

# 깨진 링크 확인
markdown-link-check docs/*.md
```

## Integration with Workflow

```bash
# Complete workflow with docs
/plan "Add HyDE to RAG"       # 1. Plan
/tdd                           # 2. TDD cycle
# [Write code]
/dedup                         # 3. Remove duplication
/arch-check                    # 4. Verify architecture
/code-review                   # 5. Comprehensive review
/update-docs                   # 6. Update documentation ⭐
git add . && git commit        # 7. Commit (docs included)
```

## Automation with Hooks

`.claude/settings.json`에 PostToolUse 훅 추가 (선택사항):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "tool == \"Edit\" && tool_input.file_path matches \"facade.*\\.py$\"",
        "hooks": [
          {
            "type": "command",
            "command": "/update-docs --scope api"
          }
        ]
      }
    ]
  }
}
```

→ Facade 파일 수정 시 자동으로 API 문서 업데이트

## Cost Optimization

**Model**: Haiku (빠르고 저렴)
- Cost: ~$0.01-0.02 per run
- Speed: ~5-10 seconds
- Quality: Sufficient for documentation

**왜 Haiku?**:
- 문서 업데이트는 단순 작업
- 고품질 추론 불필요
- 빠른 피드백 중요

## Example Run

```bash
$ /update-docs

📝 Detecting code changes...
✅ Found 3 changed files in facade/

📚 Updating API Reference...
  - Added Client.chat() temperature parameter
  - Updated RAGChain.from_documents() signature

📖 Updating README.md...
  - Added HyDE feature to Features section
  - Updated Quick Start example

📋 Updating CHANGELOG.md...
  - Added [0.3.0] section
  - 5 new features, 2 fixes, 1 performance improvement

✨ Generating missing Docstrings...
  - Added docstring to HyDEQueryExpander.expand_query()
  - Updated RAGServiceImpl.query() examples

✅ Documentation updated successfully!

📊 Documentation Coverage:
  - Public API: 98% documented (2 methods missing)
  - Docstrings: 95% complete
  - Examples: 87% up-to-date

💡 Next steps:
  1. Review docs/API_REFERENCE.md
  2. git add docs/ README.md CHANGELOG.md
  3. git commit -m "docs: Update for v0.3.0"
```

## Quality Checks

### Before Update
```bash
# Check current doc coverage
python scripts/doc_coverage.py
# Output: 85% documented
```

### After Update
```bash
# Verify all Public APIs documented
python scripts/verify_docs.py
# Output: ✅ All Public APIs documented

# Check docstring quality
pydocstyle src/beanllm/facade/
# Output: No issues found

# Validate markdown
markdownlint docs/*.md README.md
# Output: ✅ All markdown valid
```

## Related Documents

- `CLAUDE.md` - Documentation workflow
- `.claude/rules/coding-standards.md` - Docstring standards
- `docs/CONTRIBUTING.md` - Documentation contribution guide

## Quick Reference

| Scope | Command | Updates |
|-------|---------|---------|
| API docs | `/update-docs --scope api` | docs/API_REFERENCE.md |
| README | `/update-docs --scope readme` | README.md, Features |
| Changelog | `/update-docs --scope changelog` | CHANGELOG.md |
| All | `/update-docs --all` | All documentation |

---

**💡 Pro Tip**: Run `/update-docs` after every feature implementation to keep docs in sync!

**🎯 Goal**: 95%+ documentation coverage for Public APIs
