# 🫘 beanllm MCP Server

**Model Context Protocol** 서버로 기존 beanllm 코드를 함수화하여 Claude Desktop, Cursor, ChatGPT 등에서 자연어로 호출할 수 있습니다.

## 🎯 핵심 컨셉

### Before: 9개 분리된 페이지 (RAG, Multi-Agent, KG, OCR, Audio, Evaluation, Dashboard)
- 각 페이지마다 UI/UX 개발 필요
- 중복된 로직 (39 files, 6,000 lines)
- 사용자가 여러 페이지 이동

### After: 단일 Chat UI + MCP 서버 (70% 코드 감소)
- **모든 기능이 자연어로 호출 가능**
- 기존 beanllm 코드를 wrapping만 (14 files, 1,800 lines)
- Chat 하나로 모든 기능 접근

### 사용자 경험 변화

**Before**:
```
"RAG 시스템 만들고 싶어"
→ RAG 페이지 이동
→ 파일 업로드 UI에서 PDF 선택
→ 설정 폼 작성 (chunk_size, overlap, etc.)
→ 빌드 버튼 클릭
→ 쿼리 페이지 이동
→ 질문 입력
(7단계)
```

**After**:
```
User: "이 폴더의 PDF로 RAG 시스템 만들어줘"
→ MCP가 build_rag_system() 자동 호출
→ 진행 상황 채팅에 실시간 표시
→ "완료! 무엇을 도와드릴까요?"

User: "beanllm이 뭐야?"
→ MCP가 query_rag_system() 자동 호출
→ 답변과 출처가 채팅에 표시
(1단계)
```

## 📦 설치

### 1. FastMCP 설치

```bash
# beanllm with MCP support
pip install -e ".[mcp]"

# 또는 개별 설치
pip install fastmcp>=2.0.0 sse-starlette>=1.6.0
```

### 2. Ollama 설치 (로컬 모델용)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

### 3. 환경 변수 설정

`.env` 파일 생성:

```bash
# Ollama (기본)
OLLAMA_HOST=http://localhost:11434

# 선택적: 다른 LLM Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# 선택적: 세션 관리 (MongoDB + Redis)
MONGODB_URI=mongodb+srv://...
REDIS_URL=rediss://...

# 선택적: Google Workspace 연동
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# MCP Server 설정
MCP_HOST=127.0.0.1
MCP_PORT=8765

# 기본 모델
DEFAULT_CHAT_MODEL=qwen2.5:0.5b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text:latest

# RAG 설정
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_TOP_K=5
```

## 🚀 사용 방법

### Claude Desktop에서 사용

#### 1. Claude Desktop 설정 파일 편집

**macOS**: `~/.config/claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "beanllm": {
      "command": "python",
      "args": ["/absolute/path/to/llmkit/mcp_server/run.py"]
    }
  }
}
```

> **중요**: 절대 경로 사용! `~/` 대신 `/Users/username/` 형식

#### 2. Claude Desktop 재시작

#### 3. MCP 서버가 연결되었는지 확인

Claude Desktop 하단에 🔌 아이콘이 나타나면 성공!

#### 4. 자연어로 beanllm 기능 사용

```
You: "이 폴더의 PDF 파일들로 RAG 시스템 만들어줘: /Users/me/documents"

Claude: [build_rag_system() 도구를 호출합니다...]
✅ RAG 시스템 구축 완료
- 문서 수: 15개
- 청크 수: 234개
- 컬렉션 이름: default

You: "beanllm이 뭐야?"

Claude: [query_rag_system() 도구를 호출합니다...]
📚 beanllm은 Clean Architecture로 구축된 프로덕션 레벨의 LLM 통합 툴킷입니다...

출처:
1. README.md (유사도: 0.92)
2. ARCHITECTURE.md (유사도: 0.87)
```

### Playground Chat UI에서 사용

**TODO**: Phase 2에서 구현 예정
- Next.js 15 + React 19 Chat UI
- SSE streaming으로 MCP 서버 연결
- Tool call 진행 상황 실시간 표시

## 📚 사용 가능한 기능 (33 Tools)

### RAG Tools (5)
- `build_rag_system()` - RAG 시스템 구축
- `query_rag_system()` - RAG 질의
- `get_rag_stats()` - RAG 통계
- `list_rag_systems()` - RAG 시스템 목록
- `delete_rag_system()` - RAG 시스템 삭제

### Multi-Agent Tools (6)
- `create_multiagent_system()` - 다중 에이전트 시스템 생성
- `run_multiagent_task()` - 다중 에이전트 작업 실행
- `get_multiagent_stats()` - 다중 에이전트 통계
- `list_multiagent_systems()` - 시스템 목록
- `delete_multiagent_system()` - 시스템 삭제

### Knowledge Graph Tools (7)
- `build_knowledge_graph()` - 지식 그래프 구축
- `query_knowledge_graph()` - 지식 그래프 질의
- `get_kg_stats()` - 지식 그래프 통계
- `visualize_knowledge_graph()` - 지식 그래프 시각화
- `list_knowledge_graphs()` - 그래프 목록
- `delete_knowledge_graph()` - 그래프 삭제

### ML Tools (9)
**Audio**:
- `transcribe_audio()` - 음성 파일 전사
- `batch_transcribe_audio()` - 일괄 음성 전사

**OCR**:
- `recognize_text_ocr()` - 이미지 텍스트 인식
- `batch_recognize_text_ocr()` - 일괄 OCR 처리

**Evaluation**:
- `evaluate_model()` - 모델 평가
- `benchmark_models()` - 모델 벤치마킹
- `compare_model_outputs()` - 모델 출력 비교

### Google Workspace Tools (6)
- `export_to_google_docs()` - Google Docs 내보내기
- `save_to_google_drive()` - Google Drive 저장
- `share_via_gmail()` - Gmail 공유
- `get_google_export_statistics()` - 통계 조회 (관리자)
- `list_google_drive_files()` - Drive 파일 목록

## 🎨 사용 예시

### 예시 1: RAG 시스템 구축 및 질의

```
You: "내 문서 폴더(/Users/me/docs)로 RAG 시스템 만들어줘"

Claude: [build_rag_system() 호출...]
✅ RAG 시스템 구축 완료
- 문서 15개, 청크 234개

You: "주요 내용 요약해줘"

Claude: [query_rag_system() 호출...]
📚 주요 내용은...
```

### 예시 2: 다중 에이전트 토론

```
You: "AI의 미래에 대해 낙관론자, 비판론자, 실용주의자 3명의 에이전트가 토론하게 해줘"

Claude: [create_multiagent_system() 호출...]
✅ 에이전트 시스템 생성 완료

[run_multiagent_task() 호출...]
🤖 낙관론자: AI는 인류의 문제를 해결할 것입니다...
😐 비판론자: 하지만 윤리적 문제와 일자리 감소가...
🔧 실용주의자: 현실적으로는 규제와 교육이 필요...

💡 최종 결론: ...
```

### 예시 3: 지식 그래프 구축 및 탐색

```
You: "이 논문들로 지식 그래프 만들고 'Transformer 아키텍처'와 관련된 개념 찾아줘"

Claude: [build_knowledge_graph() 호출...]
✅ 지식 그래프 구축 완료
- 엔티티 127개, 관계 253개

[query_knowledge_graph() 호출...]
🔍 Transformer 아키텍처와 관련된 개념:
1. Self-Attention (유사도: 0.95)
2. Multi-Head Attention (유사도: 0.92)
3. Positional Encoding (유사도: 0.88)
...

[visualize_knowledge_graph() 호출...]
📊 시각화 저장: /path/to/graph.html
```

### 예시 4: 음성 파일 전사

```
You: "이 폴더의 모든 .mp3 파일 텍스트로 변환해줘: /Users/me/audio"

Claude: [batch_transcribe_audio() 호출...]
🎙️ 전사 완료:
- audio1.mp3: "안녕하세요, 오늘은..."
- audio2.mp3: "AI 기술의 발전으로..."
- audio3.mp3: "향후 계획은..."

총 3개 파일, 평균 신뢰도 94%
```

### 예시 5: 모델 성능 비교

```
You: "qwen2.5:0.5b랑 llama3.2:1b 모델 비교해줘. 프롬프트는 'AI의 미래는?'"

Claude: [compare_model_outputs() 호출...]
📊 모델 비교 결과:

1. qwen2.5:0.5b:
   - 응답: "AI의 미래는 밝습니다..."
   - 토큰: 127
   - 응답 시간: 1.2초

2. llama3.2:1b:
   - 응답: "AI 기술은 계속 발전할 것이며..."
   - 토큰: 156
   - 응답 시간: 2.3초

💡 분석:
- qwen2.5:0.5b가 1.9배 빠름
- llama3.2:1b가 더 상세한 답변
```

### 예시 6: Google Workspace 연동

```
You: "이 채팅 내역을 Google Docs로 저장해줘"

Claude: [export_to_google_docs() 호출...]
📝 Google Docs 저장 완료
- 문서 ID: 1a2b3c...
- URL: https://docs.google.com/document/d/1a2b3c.../edit
```

## 🔧 개발 모드 실행

### 직접 실행 (개발/테스트)

```bash
# MCP 서버 실행
python mcp_server/run.py

# 또는 uvicorn으로 실행
uvicorn mcp_server.run:mcp --host 127.0.0.1 --port 8765
```

### 로그 확인

MCP 서버 실행 시 터미널에서 로그 확인 가능:

```
🚀 Loading beanllm MCP Server...
✅ Tools loaded:
  - RAG Tools (5 tools)
  - Multi-Agent Tools (6 tools)
  - Knowledge Graph Tools (7 tools)
  - ML Tools (9 tools: audio, ocr, evaluation)
  - Google Workspace Tools (6 tools)
  Total: 33 tools
✅ Resources loaded:
  - Session Resources (7 resources)
✅ Prompts loaded:
  - Prompt Templates (8 templates)

============================================================
🫘 beanllm-mcp-server v0.1.0
============================================================
Host: 127.0.0.1
Port: 8765
Default Chat Model: qwen2.5:0.5b
Default Embedding Model: nomic-embed-text:latest
============================================================

🎯 MCP Server is ready!
```

## 📋 MCP Resources (7)

Resources는 Claude가 읽을 수 있는 데이터 소스입니다.

- `session://stats/google_exports` - Google 서비스 사용 통계
- `session://stats/security_events` - 보안 이벤트 통계
- `session://config/server` - 서버 설정 정보
- `session://info/rag_systems` - RAG 시스템 정보
- `session://info/multiagent_systems` - Multi-Agent 시스템 정보
- `session://info/knowledge_graphs` - 지식 그래프 정보

### 사용 예시

```
You: "session://stats/google_exports?hours=24" 리소스 읽어줘

Claude: [Resource를 읽습니다...]
📊 지난 24시간 Google 서비스 사용 통계:
- 총 내보내기: 123건
- Docs: 45건, Drive: 38건, Gmail: 40건
- 상위 사용자: user123 (34건), user456 (28건)
```

## 🎭 MCP Prompts (8)

Prompts는 재사용 가능한 워크플로우 템플릿입니다.

- `rag_system_builder` - RAG 시스템 구축 워크플로우
- `multiagent_debate` - 다중 에이전트 토론
- `knowledge_graph_explorer` - 지식 그래프 탐색
- `audio_transcription_batch` - 음성 파일 일괄 전사
- `model_comparison` - 모델 성능 비교
- `google_workspace_exporter` - Google Workspace 내보내기
- `rag_optimization` - RAG 시스템 최적화

### 사용 예시

```
You: "rag_system_builder" 프롬프트 사용해줘. 문서 경로는 /Users/me/docs

Claude: [프롬프트를 실행합니다...]
📚 RAG 시스템 구축을 시작합니다...

1단계: build_rag_system() 호출...
✅ 문서 15개, 청크 234개

2단계: 테스트 질의...
Q: 이 문서들의 주제는 무엇인가요?
A: ...

3단계: 결과 요약...
```

## 🔐 보안

### API 키 관리

- `.env` 파일 사용 (Git에 커밋하지 않음)
- 환경 변수로 관리
- 민감한 정보는 로그에 출력하지 않음

### Google OAuth

- 사용자별 액세스 토큰 관리
- Incremental authorization 사용
- Secret masking 적용 (2025년 6월부터)

## 🐛 트러블슈팅

### 1. "Module not found" 에러

```bash
# beanllm 설치 확인
pip install -e ".[mcp]"

# 또는
cd /path/to/llmkit
pip install -e .
pip install fastmcp>=2.0.0
```

### 2. Claude Desktop에서 MCP 서버 연결 안 됨

- 절대 경로 사용 확인 (`~` 대신 `/Users/username/`)
- Claude Desktop 재시작
- 로그 확인: `~/Library/Logs/Claude/mcp-server-beanllm.log` (macOS)

### 3. Ollama 연결 실패

```bash
# Ollama 실행 확인
ollama list

# 모델 다운로드
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

### 4. Tool 호출 실패

- `.env` 파일에 필수 환경 변수 설정 확인
- Ollama 서비스 실행 확인
- 파일 경로는 절대 경로 사용

## 📚 참고 문서

- **MCP 사양**: https://spec.modelcontextprotocol.io/
- **FastMCP 문서**: https://github.com/jlowin/fastmcp
- **Claude Desktop MCP 가이드**: https://docs.anthropic.com/claude/docs/model-context-protocol
- **beanllm 아키텍처**: `/CLAUDE.md`, `/ARCHITECTURE.md`

## 🤝 기여

이슈나 개선 사항이 있으면 GitHub Issues에 올려주세요!

---

**Built with ❤️ by beanllm team**

🎯 **핵심**: 기존 코드를 새로 만들지 않고 wrapping만 해서 70% 코드 감소!
