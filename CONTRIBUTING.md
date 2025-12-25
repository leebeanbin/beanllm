# Contributing to beanllm

Thank you for your interest in contributing to beanllm! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/beanllm.git
cd beanllm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev,all]"
```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting
- **MyPy**: Type checking

Run all checks before submitting:
```bash
black src/
ruff check src/
mypy src/beanllm
```

## Testing

All new features should include tests:

```bash
pytest tests/ -v --cov=beanllm
```

Test coverage should remain above 80%.

## Commit Messages

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test updates
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example:
```
feat: Add support for streaming responses in RAG

Add streaming capability to RAGChain for real-time response generation.
Includes tests and documentation updates.
```

## Pull Request Process

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: Your feature description"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request on GitHub with:
   - Clear description of changes
   - Link to related issues
   - Test results
   - Documentation updates

## Documentation

- Update docstrings for all public APIs
- Add examples to `examples/` directory
- Update relevant markdown files in `docs/`
- Follow NumPy docstring style

## Adding New Features

### New Provider Support

1. Create provider implementation in `src/beanllm/providers/`
2. Add provider to registry in `src/beanllm/registry.py`
3. Add tests in `tests/test_providers/`
4. Update documentation and examples

### New Tools or Features

1. Implement in appropriate module under `src/beanllm/`
2. Add comprehensive tests
3. Create tutorial in `docs/tutorials/`
4. Add theory documentation if complex
5. Update README.md with usage examples

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Welcome newcomers and help them learn

## Questions?

- Open an issue for discussion
- Check existing issues and PRs
- Read the documentation thoroughly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
