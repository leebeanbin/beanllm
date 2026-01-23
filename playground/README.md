# 🫘 beanllm Playground

Complete web interface for beanllm framework with **all AI features**.

## ✨ Features

### Core Features

1. **💬 Chat** - General LLM conversation with Think Mode support
2. **🔍 RAG** - Retrieval-Augmented Generation
3. **🤖 Agent** - Autonomous task execution with tools
4. **👥 Multi-Agent** - Collaborative multi-agent systems
5. **🔀 Chain** - LLM chains and workflows
6. **🕸️ Knowledge Graph** - Entity/relation extraction and graph reasoning
7. **🖼️ Vision RAG** - Image-based RAG
8. **🎵 Audio** - Audio transcription and synthesis
9. **📊 Evaluation** - Model evaluation tools
10. **🔧 Fine-tuning** - Model fine-tuning
11. **📄 OCR** - Optical Character Recognition
12. **🌐 Web Search** - Multi-engine web search

### UI Features

- 🎨 KRDS design system with pastel colors
- 🌓 Dark mode support
- 📱 Fully responsive
- ⚡ Real-time streaming
- 🧠 Think Mode visualization
- 📥 Model download with progress tracking
- 🎯 Interactive onboarding guide

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Node.js 18+ & pnpm
node --version
pnpm --version
```

### 1. Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd playground/backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional for open-source models)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=...
export GEMINI_API_KEY=...
export PERPLEXITY_API_KEY=...

# Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Backend will start on**: http://localhost:8000

**API Documentation**: http://localhost:8000/docs

### 2. Frontend Setup (Next.js)

```bash
# Navigate to frontend directory
cd playground/frontend

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

**Frontend will start on**: http://localhost:3000

### 3. Start Using

1. Open http://localhost:3000
2. Select a model from the dropdown
3. Start chatting or using any feature!

---

## 📁 Project Structure

```
playground/
├── backend/                 # FastAPI backend
│   ├── main.py             # All-in-one server
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app directory (pages)
│   │   ├── components/    # React components
│   │   ├── lib/           # API client & utilities
│   │   └── providers/     # Context providers
│   ├── package.json       # Node dependencies
│   └── tailwind.config.js # Tailwind configuration
│
└── README.md              # This file
```

---

## 🔌 API Endpoints

### Health Check ✅
- `GET /` - Service status
- `GET /health` - Detailed health check

### Chat ✅
- `POST /api/chat` - General conversation
  - Supports text and multimodal (image + text)
  - Think Mode support (`enable_thinking: true`)

### Models ✅
- `GET /api/models` - List all available models grouped by provider
- `GET /api/models/{model_name}/parameters` - Get model parameters
- `POST /api/models/{model_name}/pull` - Download Ollama model (SSE streaming)

### RAG ✅
- `POST /api/rag/build` - Build RAG index from documents
- `POST /api/rag/query` - Query RAG system
- `GET /api/rag/collections` - List RAG collections
- `DELETE /api/rag/collections/{name}` - Delete collection
- `POST /api/rag/build_from_files` - Build from uploaded files

### Knowledge Graph ✅
- `POST /api/kg/build` - Build knowledge graph from documents
- `POST /api/kg/query` - Query graph (Cypher)
- `POST /api/kg/graph_rag` - Graph RAG query
- `GET /api/kg/visualize/{graph_id}` - Visualize graph

### Agent ✅
- `POST /api/agent/run` - Run autonomous agent task

### Multi-Agent ✅
- `POST /api/multi_agent/run` - Run multi-agent system
  - `mode`: "sequential", "parallel", "hierarchical", "debate"

### Chain ✅
- `POST /api/chain/build` - Build chain
- `POST /api/chain/run` - Run chain

### Web Search ✅
- `POST /api/web/search` - Search the web
  - `summarize`: Use LLM to summarize results
  - `model`: Model for summarization

### Evaluation ✅
- `POST /api/evaluation/evaluate` - Evaluate model responses

### Vision RAG 🔧
- `POST /api/vision_rag/build` - Build VisionRAG index from images
- `POST /api/vision_rag/query` - Query VisionRAG
- Note: Requires additional dependencies

### Audio 🔧
- `POST /api/audio/transcribe` - Transcribe audio (requires audio file)
- `POST /api/audio/synthesize` - Synthesize speech (requires OpenAI API key)

### OCR 🔧
- `POST /api/ocr/recognize` - Recognize text from images (requires PaddleOCR)

### Fine-tuning 🔧
- `POST /api/finetuning/create` - Create fine-tuning job (requires OpenAI API key)
- `GET /api/finetuning/status/{job_id}` - Get job status

**Legend**: ✅ Fully working | 🔧 Requires dependencies or API keys

---

## 🎯 Model Support

### Supported Providers

- **OpenAI**: GPT-5, GPT-4o, GPT-4.1, O1, O3 series
- **Anthropic**: Claude 3.5, Claude 4, Claude 4.5 series
- **Google**: Gemini 1.5, Gemini 2.0, Gemini 2.5, Gemini 3.0 series
- **DeepSeek**: DeepSeek Chat, DeepSeek V3, DeepSeek R1 (API or Ollama)
- **Perplexity**: Sonar, Sonar Pro, Sonar Reasoning Pro
- **Ollama**: Local open-source models (Qwen, Llama, Phi, etc.)

### Smart Provider Detection

The backend automatically detects the best provider for each model:

1. **Registry Check**: Checks if model is registered in the model registry
2. **Ollama Check**: For open-source models, checks if installed locally in Ollama
3. **Pattern Detection**: Falls back to pattern-based detection

**Open-source models** (DeepSeek, Mistral, Gemma, etc.) are automatically checked in Ollama first. If found, they use the local Ollama provider (no API key needed). Otherwise, they fall back to API providers.

### Model Download

Ollama models can be downloaded directly from the UI:
- Click the download button next to any Ollama model
- Track download progress in real-time
- Cancel downloads if needed
- Downloads persist across page navigation

---

## 🧠 Think Mode

The backend supports thinking/reasoning mode for different model types:

### Native Thinking Models
- **Claude models**: Uses native `thinking` parameter
- **OpenAI reasoning models** (o1, o3, gpt-5): Automatic thinking mode
- **DeepSeek R1**: Native reasoning mode

### Prompt-Based Thinking
- **Other models** (including open-source like Qwen, Llama, Phi): Uses system prompt to encourage step-by-step reasoning

Enable thinking mode by toggling the Think Mode button in the UI or setting `enable_thinking: true` in the API request.

---

## 🛠️ Development

### Backend Development

```bash
cd playground/backend

# Run with auto-reload
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or simply
python main.py
```

### Frontend Development

```bash
cd playground/frontend

# Development server
pnpm dev

# Production build
pnpm build

# Start production server
pnpm start
```

### Environment Variables

**Backend** (environment variables or `.env`):
```bash
# Required for API providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...
PERPLEXITY_API_KEY=...
```

**Frontend** (`.env.local`):
```bash
# Optional - defaults to http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🎨 UI Features

### Design System
- **KRDS Design System**: Government UI/UX design system
- **Pastel Colors**: Soft, modern color palette
- **Pretendard GOV Font**: Recommended Korean font
- **Accessibility**: ARIA attributes, keyboard navigation

### Components
- **Model Selector**: Easy model selection with download support
- **Settings Panel**: Dynamic model parameters with tooltips
- **Think Mode Toggle**: Visual reasoning process display
- **Onboarding Guide**: Interactive step-by-step guide
- **File Upload**: Drag-and-drop file upload for RAG

---

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`
**Solution**:
```bash
cd playground/backend
pip install -r requirements.txt
```

**Problem**: `OPENAI_API_KEY not set`
**Solution**: Set environment variables or use Ollama models (no API key needed)

**Problem**: Ollama models not found
**Solution**: 
1. Install Ollama: https://ollama.ai
2. Download models using the UI or `ollama pull <model-name>`

### Frontend Issues

**Problem**: `Cannot find module 'next'`
**Solution**:
```bash
cd playground/frontend
pnpm install
```

**Problem**: Build fails with type errors
**Solution**: Clear cache and rebuild
```bash
rm -rf .next
pnpm build
```

### Connection Issues

**Problem**: Frontend can't connect to backend
**Solution**:
1. Ensure backend is running on http://localhost:8000
2. Check CORS settings in `main.py`
3. Verify `NEXT_PUBLIC_API_URL` in `.env.local`

---

## 📚 Documentation

### Interactive API Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Testing

**종합 테스트 결과**: 7/11 테스트 통과 (64%)

✅ **완벽 작동** (7개):

- Document Loaders: Text, CSV, HTML, Markdown, JSON, PDF (6/6)
- PDF Multi-page 지원

⚠️ **의존성 필요** (4개):

- RAG File Upload (Ollama embedding 설정)
- Vision RAG (아키텍처 수정 필요)
- OCR (PaddleOCR 설치)
- Knowledge Graph Query (Ollama 모델 사용)

**테스트 스크립트**:

```bash
cd backend

# 샘플 파일 생성
python create_sample_files.py

# 문서 로더 테스트
python test_document_loaders.py

# 전체 기능 종합 테스트
python test_all_features_comprehensive.py
```

상세 결과: `backend/TEST_SUMMARY.md`

---

## 🔧 Technology Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Request/response validation
- **WebSockets** - Real-time communication
- **beanllm** - Core LLM framework

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **Radix UI** - Accessible components
- **Lucide React** - Icons
- **sonner** - Toast notifications

---

## 📝 License

Same as beanllm framework.

---

## 🎯 Next Steps

### Try It Out
1. Set up backend and frontend (see Quick Start above)
2. Open http://localhost:3000
3. Try all features
4. Download and test Ollama models

### Customize
1. Modify `backend/main.py` to add custom endpoints
2. Update frontend components in `frontend/src/components`
3. Enhance UI in `frontend/src/app`

### Deploy
1. Set up production environment variables
2. Build frontend: `cd frontend && pnpm build`
3. Deploy backend (Docker, Cloud Run, etc.)
4. Deploy frontend (Vercel, Netlify, etc.)

---

**🫘 Built with beanllm - The unified LLM framework**
