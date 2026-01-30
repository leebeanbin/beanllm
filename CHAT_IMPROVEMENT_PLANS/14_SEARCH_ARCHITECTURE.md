# 검색 아키텍처 정리

## 📋 개요

검색 시스템 구조 및 Playground 기능 활용 현황

**관련 문서**: [00_INDEX.md](./00_INDEX.md) - Phase 4 참고

---

## 🎯 목표

검색엔진과 내부 DB 검색의 역할을 명확히 구분하고, playground의 기능 활용 상태를 정리

---

## 📊 검색 시스템 구조

### 1. 내부 DB 검색 (이미 구현됨 ✅)

**목적**: 사용자의 채팅 세션 및 메시지 검색

**구현 위치**:
- `playground/backend/services/session_search_service.py`: 세션 검색
- `playground/backend/services/message_vector_store.py`: 메시지 검색

**기술 스택**:
- **MongoDB**: 세션 메타데이터 (제목, 날짜, 필터링)
- **Vector DB (Chroma)**: 의미 기반 검색 (세션 내용, 메시지 내용)

**사용 케이스**:
```python
# 세션 검색
GET /api/sessions?query="AI에 대해 논의한 대화"
# → MongoDB + Vector DB 하이브리드 검색

# 메시지 검색
message_vector_store.search_messages(
    query="RAG 시스템",
    session_id="session_123",
    k=20
)
# → Vector DB 의미 기반 검색
```

**특징**:
- ✅ 이미 구현되어 있음
- ✅ 하이브리드 검색 (MongoDB 필터링 + Vector DB 의미 검색)
- ✅ 세션별, 전체 메시지 검색 지원

---

### 2. 외부 검색엔진 통합 (계획됨 ⚠️)

**목적**: 웹 검색 결과를 인덱싱하고 캐싱하여 AI 검색 강화

**구현 위치**:
- `CHAT_IMPROVEMENT_PLANS/09_SEARCH_ENGINE.md` (계획 문서)
- 아직 구현 안 됨

**기술 스택**:
- **Meilisearch** 또는 **Algolia**: 검색 결과 인덱싱
- **Redis**: 검색 결과 캐싱

**사용 케이스**:
```python
# 웹 검색 실행
orchestrator._handle_web_search() 
# → 웹 검색 API 호출 (DuckDuckGo, Google 등)

# 검색 결과를 Meilisearch에 인덱싱
search_engine_service.index_search_result(
    session_id="session_123",
    query="최신 AI 뉴스",
    results=[...],  # 웹 검색 결과
    ai_summary="AI 뉴스 요약..."
)
# → 나중에 "AI 뉴스"로 검색하면 캐시된 결과 사용
```

**특징**:
- ⚠️ 아직 구현 안 됨 (계획 단계)
- 외부 웹 검색 결과 관리용
- 내부 DB 검색과는 **별개**

---

## 🔍 검색 시스템 비교

| 구분 | 내부 DB 검색 | 외부 검색엔진 |
|------|------------|--------------|
| **목적** | 채팅 세션/메시지 검색 | 웹 검색 결과 관리 |
| **데이터 소스** | MongoDB + Vector DB | 웹 검색 API 결과 |
| **검색 대상** | 사용자의 대화 내용 | 인터넷 정보 |
| **구현 상태** | ✅ 구현됨 | ⚠️ 계획됨 |
| **사용 위치** | `session_search_service` | `orchestrator._handle_web_search` |
| **관련 문서** | `13_DB_OPTIMIZATION.md` | `09_SEARCH_ENGINE.md` |

---

## 🎯 Playground 기능 활용 상태

### 현재 구조 (2025-01-24 업데이트)

```
Playground Backend
├── main.py                    # 1,161줄 (정리 완료) ✅
├── routers/                   # 18개 라우터 (정리 완료) ✅
│   ├── history_router.py      # 이동됨 ✅
│   └── ...
├── schemas/                   # 스키마 (정리 완료) ✅
│   ├── database.py            # 이동됨 ✅
│   └── ...
├── services/
│   ├── orchestrator.py (Agentic 모드)
│   │   ├── Intent 분류
│   │   ├── Tool 선택
│   │   └── beanllm Facade 직접 호출 ❌ (MCP 통합 필요)
│   │
│   ├── tool_registry.py
│   │   └── beanllm 기능을 Tool로 등록 ✅
│   │
│   └── ...
├── scripts/                   # 스크립트 정리 완료 ✅
├── docs/                      # 문서 정리 완료 ✅
└── mcp_streaming.py (레거시)
    └── MCP 서버와 통신 (사용 안 함 ⚠️)
```

### MCP 서버 상태

```
mcp_server/
├── run.py (FastMCP 서버)
├── tools/
│   ├── rag_tools.py ✅
│   ├── agent_tools.py ✅
│   ├── kg_tools.py ✅
│   ├── ml_tools.py ✅
│   └── google_tools.py ✅
└── 총 33개 tools 등록됨 ✅
```

**중요**: 
- MCP 서버는 **Claude Desktop, Cursor 등 외부 클라이언트용**
- Playground는 MCP 서버를 **직접 호출하지 않음**
- Playground는 `orchestrator`를 통해 **beanllm Facade를 직접 사용**

---

## ✅ Playground 기능 활용 현황

### 최근 개선 사항 (2025-01-24)
- ✅ 코드 정리 완료: 중복 엔드포인트 제거, import 정리
- ✅ 구조 개선 완료: 파일 이동, 디렉토리 정리
- ✅ 문서화 완료: README.md 생성
- ⏳ MCP 통합 대기: MCP Client Service 생성 필요

### 1. Core 기능 (✅ 모두 활용 가능)

| 기능 | Facade | Orchestrator 핸들러 | 상태 |
|------|--------|-------------------|------|
| Chat | `Client` | `_handle_chat` | ✅ |
| RAG | `RAGChain` | `_handle_rag` | ✅ (Facade 직접 호출) |
| Agent | `Agent` | `_handle_agent` | ⚠️ TODO (MCP tool 사용 필요) |
| Multi-Agent | `MultiAgentFacade` | `_handle_multi_agent` | ⚠️ TODO (MCP tool 사용 필요) |
| Knowledge Graph | `KnowledgeGraphFacade` | `_handle_kg` | ✅ |
| Web Search | - | `_handle_web_search` | ✅ |

### 2. ML 기능 (✅ 모두 활용 가능)

| 기능 | Facade | Orchestrator 핸들러 | 상태 |
|------|--------|-------------------|------|
| Audio | `AudioFacade` | `_handle_audio` | ⚠️ TODO (MCP tool 사용 필요) |
| OCR | `beanOCR` | `_handle_ocr` | ⚠️ TODO (MCP tool 사용 필요) |
| Evaluation | `EvaluationFacade` | `_handle_evaluation` | ⚠️ TODO (MCP tool 사용 필요) |

### 3. Google Workspace (✅ 활용 가능)

| 기능 | Facade | Orchestrator 핸들러 | 상태 |
|------|--------|-------------------|------|
| Google Docs | `google_tools` | `_handle_google_docs` | ✅ |
| Google Drive | `google_tools` | `_handle_google_drive` | ✅ |
| Gmail | `google_tools` | `_handle_gmail` | ✅ |

### 4. 세션 관리 (✅ 구현됨)

| 기능 | 구현 위치 | 상태 |
|------|---------|------|
| 세션 생성/조회 | `routers/history_router.py` | ✅ (이동됨) |
| 메시지 저장 | `message_vector_store.py` | ✅ |
| 세션 검색 | `session_search_service.py` | ✅ |
| 세션 캐싱 | `session_cache.py` | ✅ |

---

## 🔄 데이터 흐름

### Agentic Chat 흐름

```
사용자 질의
    ↓
Intent Classifier (의도 분류)
    ↓
Tool Registry (도구 선택)
    ↓
Orchestrator (도구 실행)
    ↓
beanllm Facade 직접 호출 ✅
    ↓
결과 스트리밍 (SSE)
```

### MCP 서버 흐름 (외부 클라이언트용)

```
Claude Desktop / Cursor
    ↓
MCP Server (mcp_server/run.py)
    ↓
beanllm Facade 호출
    ↓
결과 반환
```

**중요**: Playground는 MCP 서버를 거치지 않고 직접 Facade 호출

---

## 📋 정리

### 검색 시스템

1. **내부 DB 검색** (✅ 구현됨)
   - 목적: 채팅 세션/메시지 검색
   - 기술: MongoDB + Vector DB
   - 위치: `session_search_service`, `message_vector_store`

2. **외부 검색엔진** (⚠️ 계획됨)
   - 목적: 웹 검색 결과 인덱싱/캐싱
   - 기술: Meilisearch/Algolia
   - 위치: `09_SEARCH_ENGINE.md` (계획 문서)
   - **내부 DB 검색과 별개**

### Playground 기능 활용

1. **모든 beanllm 기능 활용 가능** ✅
   - Core: Chat, RAG, Agent, Multi-Agent, KG
   - ML: Audio, OCR, Evaluation
   - Google: Docs, Drive, Gmail

2. **MCP 서버는 별도** ⚠️
   - MCP 서버는 외부 클라이언트용 (Claude Desktop 등)
   - Playground는 MCP 서버를 직접 호출하지 않음
   - Playground는 `orchestrator`를 통해 beanllm Facade 직접 사용

3. **Agentic 모드** ✅
   - Intent 자동 분류
   - Tool 자동 선택
   - 결과 스트리밍

---

## 💡 결론

1. **검색엔진 vs 내부 DB 검색**: **별개**
   - 검색엔진: 외부 웹 검색 결과 관리 (계획됨)
   - 내부 DB 검색: 채팅 세션/메시지 검색 (구현됨)

2. **Playground 기능 활용**: **모두 가능** ✅
   - 모든 beanllm 기능을 orchestrator를 통해 사용
   - MCP 서버는 외부 클라이언트용 (playground와 별개)

3. **MCP 통합**: **완료됨** ✅
   - MCP 서버에 33개 tools 등록됨
   - Playground는 MCP 서버를 거치지 않고 직접 Facade 사용
