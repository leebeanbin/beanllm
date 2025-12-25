# beanllm v0.1.0 Release Notes

**Release Date:** December 19, 2024

We're excited to announce the first release of **beanllm** - a unified, production-ready toolkit for managing and using multiple LLM providers with advanced features for RAG, agents, multi-modal AI, and production deployment.

## 🎯 Overview

beanllm v0.1.0 is a comprehensive LLM toolkit that brings together the best features from multiple providers (OpenAI, Anthropic, Google, Ollama) with a unified interface. This release includes everything needed to build production-grade AI applications, from basic completions to complex multi-agent systems.

## ✨ Highlights

### 🤖 Unified Multi-Provider Interface
- **Single API** for OpenAI, Anthropic, Google Gemini, and Ollama
- **Automatic provider detection** from model names
- **Seamless switching** between providers without code changes
- **Streaming support** with real-time callbacks

### 📚 Production-Ready RAG
- **One-line RAG**: `RAGChain.from_documents("docs/")`
- **10+ document loaders** (PDF, DOCX, CSV, JSON, HTML, etc.)
- **5 vector stores** (Chroma, FAISS, Pinecone, Weaviate, Qdrant)
- **Intelligent text splitting** with semantic and token-based strategies
- **RAG debugging tools** for retrieval analysis

### 🧠 Advanced Agent Systems
- **ReAct agents** with function calling
- **Tool integration** with 20+ built-in tools
- **Multi-agent collaboration** with supervisor patterns
- **Graph workflows** for complex decision trees
- **Memory systems** (buffer, summary, vector-based)

### 🎨 Multimodal AI
- **Vision APIs** (GPT-4V, Claude 3, Gemini Vision)
- **Image analysis** and OCR
- **Audio processing** (Whisper transcription, TTS)
- **Web search** integration (Tavily, SerpAPI, DuckDuckGo)
- **ML model** integration (scikit-learn, PyTorch, TensorFlow)

### 💰 Cost Optimization
- **Token counting** with tiktoken for accurate estimates
- **Cost calculation** for 50+ models
- **Model recommendations** based on cost and performance
- **Usage tracking** and budget monitoring

### 🎓 Comprehensive Documentation
- **900+ lines** of graduate-level theory
- **600+ lines** of hands-on tutorials
- **16-week curriculum** from basics to advanced
- **50+ code examples** for common use cases
- **Best practices** for production deployment

## 🚀 Getting Started

### Installation

```bash
# Basic installation (OpenAI + Anthropic)
pip install beanllm

# With all providers
pip install beanllm[all]

# Development installation
pip install beanllm[dev]
```

### Quick Start

```python
from beanllm import Client

# Basic usage
client = Client(model="gpt-4o")
response = client.chat("Explain quantum computing")
print(response.content)

# RAG in one line
from beanllm import RAGChain
rag = RAGChain.from_documents("docs/")
answer = rag.query("What is the main topic?")

# Cost optimization
from beanllm import estimate_cost, get_cheapest_model
cost = estimate_cost(
    input_text="Your prompt",
    output_text="Expected response",
    model="gpt-4o"
)
```

## 📦 What's Included

### Core Modules (14 total)

1. **beanllm.client** - Unified LLM interface
2. **beanllm.registry** - Model and provider management
3. **beanllm.adapters** - Provider-specific implementations
4. **beanllm.document_loaders** - Document ingestion
5. **beanllm.text_splitters** - Intelligent chunking
6. **beanllm.embeddings** - Vector embedding generation
7. **beanllm.vector_stores** - Vector database integration
8. **beanllm.rag** - Complete RAG pipeline
9. **beanllm.agents** - Agent framework
10. **beanllm.tools** - Tool integration system
11. **beanllm.memory** - Conversation memory
12. **beanllm.chains** - Chain of thought and workflows
13. **beanllm.graphs** - Graph-based workflows
14. **beanllm.multi_agent** - Multi-agent systems

### Production Features

- **Token counting** (`beanllm.token_counter`)
- **Cost estimation** (`beanllm.cost_estimator`)
- **Prompt templates** (`beanllm.prompts`)
- **Evaluation metrics** (`beanllm.evaluation`)
- **Error handling** (`beanllm.error_handling`)
- **Fine-tuning** (`beanllm.finetuning`)

### Developer Tools

- **CLI interface** with rich formatting
- **Streaming utilities** for real-time processing
- **Tracing integration** with OpenTelemetry
- **Debugging tools** for RAG and agents
- **Testing utilities** with pytest integration

## 🔧 Technical Details

### Supported Models

**OpenAI:**
- GPT-4 Turbo, GPT-4o, GPT-4o-mini
- GPT-3.5 Turbo variants
- Embedding models (text-embedding-3-small/large)

**Anthropic:**
- Claude 3.5 Sonnet, Claude 3 Opus
- Claude 3 Sonnet, Claude 3 Haiku

**Google:**
- Gemini 1.5 Pro, Gemini 1.5 Flash
- Gemini 1.0 Pro

**Ollama:**
- Llama 3/3.1, Mistral, Mixtral
- CodeLlama, Phi-3, and more

### System Requirements

- **Python:** 3.11 or higher
- **OS:** macOS, Linux, Windows
- **Memory:** 4GB minimum (8GB+ recommended for vector stores)
- **Storage:** 500MB for package + models (varies by provider)

### Performance

- **Streaming:** Real-time token streaming for all providers
- **Async support:** Full async/await compatibility
- **Batch processing:** Efficient batch operations
- **Caching:** Built-in response caching
- **Rate limiting:** Automatic rate limit handling

## 📖 Documentation

- **Theory Docs:** 9 comprehensive guides with mathematical foundations
- **Tutorials:** 9 hands-on tutorials with real code
- **Learning Path:** 16-week curriculum (3 hours/week)
- **Examples:** 50+ code examples for common tasks
- **API Reference:** Complete API documentation

Access docs at: [docs/](docs/)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas for contribution:
- New provider integrations
- Additional vector store support
- More evaluation metrics
- Enhanced multi-agent patterns
- Documentation improvements

## 🐛 Known Issues

- Some vector stores require additional system dependencies
- Async support varies by provider
- Fine-tuning only supports OpenAI API currently

See [GitHub Issues](https://github.com/leebeanbin/beanllm/issues) for full list.

## 🗺️ Roadmap

### v0.2.0 (Q1 2025)
- Additional vector store integrations
- Enhanced streaming for all providers
- GUI dashboard for monitoring
- More evaluation metrics

### v0.3.0 (Q2 2025)
- Plugin system for extensions
- Advanced multi-agent patterns
- Model fine-tuning enhancements
- Performance optimizations

### Future
- Cloud deployment templates
- Kubernetes operators
- Enterprise features
- Advanced security features

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

Built with support from:
- OpenAI for GPT models and API
- Anthropic for Claude models
- Google for Gemini models
- Ollama for local model support
- The open-source community

## 📞 Support

- **Documentation:** [GitHub README](README.md)
- **Issues:** [GitHub Issues](https://github.com/leebeanbin/beanllm/issues)
- **Discussions:** [GitHub Discussions](https://github.com/leebeanbin/beanllm/discussions)

## 🎉 Get Started Today

```bash
pip install beanllm
```

Start building production-grade AI applications with beanllm!

---

**Full Changelog:** https://github.com/leebeanbin/beanllm/blob/main/CHANGELOG.md
