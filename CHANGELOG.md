# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-19

### Added

#### Core Infrastructure
- Model registry with automatic provider detection
- Unified client interface supporting OpenAI, Anthropic, Google, and Ollama
- Intelligent adapters for seamless provider switching
- Response streaming with callback support
- Distributed tracing integration (OpenTelemetry)
- Configuration management with environment variable support

#### Document Processing & RAG
- 10+ document loaders (PDF, Word, Markdown, CSV, JSON, HTML, etc.)
- Intelligent text splitters (recursive, semantic, token-based)
- Complete RAG pipeline with vector store integration
- Support for 5 vector stores (Chroma, FAISS, Pinecone, Weaviate, Qdrant)
- Embeddings support (OpenAI, Sentence Transformers, custom)
- RAG debugging and evaluation tools
- Document chunking with overlap and metadata preservation

#### Advanced LLM Features
- Agent framework with ReAct and function calling
- Tool integration system with built-in and custom tools
- Conversation memory (buffer, summary, vector-based)
- Chain of Thought prompting
- Sequential and parallel chains
- Router chains for dynamic routing
- MapReduce chains for document processing

#### Graph & Multi-Agent Systems
- StateGraph for complex workflows
- Conditional branching and routing
- Multi-agent collaboration framework
- Supervisor agents for coordination
- Hierarchical agent structures
- Graph persistence and checkpointing

#### Multimodal AI
- Vision API integration (GPT-4V, Claude 3, Gemini)
- Image analysis and description
- OCR and document understanding
- Vision-Language Model (VLM) support
- ML model integration (scikit-learn, PyTorch, TensorFlow)
- Model deployment and serving utilities

#### Web & Audio Processing
- Web search integration (Tavily, SerpAPI, DuckDuckGo)
- Web scraping with BeautifulSoup and Playwright
- Audio transcription (Whisper API)
- Text-to-speech generation
- Audio file processing
- Web content extraction and parsing

#### Production Features
- Token counting with tiktoken
- Cost estimation for 50+ models
- Cost optimization recommendations
- Prompt templates (few-shot, chat, chain-of-thought)
- Evaluation metrics (BLEU, ROUGE, semantic similarity)
- LLM-as-Judge evaluation framework
- Fine-tuning data preparation and API integration
- Error handling (retry, circuit breaker, rate limiting)
- Production monitoring and logging

#### Developer Experience
- Rich CLI interface with interactive commands
- Comprehensive documentation (900+ lines theory, 600+ lines tutorials)
- 16-week learning curriculum
- 50+ code examples
- Type hints throughout codebase
- Async/await support
- Extensive test coverage

#### CI/CD & Infrastructure
- GitHub Actions workflows for testing (multi-OS, multi-Python)
- Automated PyPI publishing
- CodeQL security scanning
- Dependabot for dependency updates
- Documentation deployment to GitHub Pages
- Issue and PR templates
- Contributing guidelines

### Documentation
- Complete API reference
- 9 theory documents covering graduate-level concepts
- 9 hands-on tutorials with real-world examples
- Learning path from basics to advanced topics
- 50+ practical examples
- Migration guides and best practices

### Dependencies
- Core: `httpx`, `python-dotenv`, `openai`, `anthropic`, `rich`
- Optional: `google-generativeai` (Gemini), `ollama` (local models)
- Development: `pytest`, `black`, `ruff`, `mypy`, `pytest-cov`

### Notes
- Python 3.11+ required
- Supports macOS, Linux, and Windows
- Modular design allows installing only needed providers
- Comprehensive test coverage with pytest
- Production-ready with error handling and monitoring

## [Unreleased]

### Planned
- Additional vector store integrations
- More evaluation metrics
- Enhanced multi-agent collaboration patterns
- Streaming support for all providers
- Plugin system for extensions
- GUI dashboard for monitoring

---

For detailed information about each feature, see the [documentation](docs/).

[0.1.0]: https://github.com/leebeanbin/beanllm/releases/tag/v0.1.0
