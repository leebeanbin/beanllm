# 커밋 그룹 계획 (Commit Plan)

작업 기록(DEVELOPMENT_LOG.md, CHAT_IMPROVEMENT_PLANS/00_INDEX.md)을 반영하여 그룹별로 커밋 메시지를 분리한 계획입니다.

---

## 그룹 1: docs – 계획 문서 및 개발 로그

**커밋 메시지:**
```
docs: Add chat improvement plans and development log

- CHAT_IMPROVEMENT_PLANS/: Phase 0–10 개선 계획 인덱스 및 가이드
- DEVELOPMENT_LOG.md: Playground 백엔드/프론트 단계별 작업 기록
- COMMITTABLE_EXCLUDE_MD.md: 커밋 제외 문서 정리
- Remove .claude/GETTING_STARTED.md, MODEL_SELECTION_GUIDE.md (obsolete)
```

**스테이징 대상:**
- `CHAT_IMPROVEMENT_PLANS/` (전체)
- `COMMITTABLE_EXCLUDE_MD.md`
- `COMMIT_PLAN.md` (이 문서)
- `DEVELOPMENT_LOG.md`
- `.claude/GETTING_STARTED.md` (삭제)
- `.claude/MODEL_SELECTION_GUIDE.md` (삭제)

---

## 그룹 2: chore – 인프라 및 스크립트

**커밋 메시지:**
```
chore: Add Docker Compose and env check scripts

- docker-compose.yml: MongoDB, Redis, Kafka, optional UIs
- scripts/mongo-init.js: MongoDB 컬렉션/인덱스 초기화
- scripts/check-env.sh: 환경 검증 (--fix 지원)
- .gitignore, Makefile 업데이트
```

**스테이징 대상:**
- `docker-compose.yml`
- `scripts/check-env.sh`
- `scripts/mongo-init.js`
- `.gitignore`
- `Makefile`

---

## 그룹 3: refactor(playground/backend) – 구조 및 스키마

**커밋 메시지:**
```
refactor(playground): Restructure backend (schemas, routers, services)

Structure:
- schemas/: Pydantic 모델 (database, chat, rag, agent, etc.)
- scripts/: setup_and_build, auto_setup_and_test, quick_test
- docs/: CLEANUP_ANALYSIS, MCP_INTEGRATION_ANALYSIS, STRUCTURE_ANALYSIS
- monitoring/: middleware, dashboard

Removed:
- chat_history.py → history_router
- models.py → schemas/database
- mcp_streaming.py, monitoring.py, monitoring_dashboard.py
- requirements.txt (Poetry 사용)
- ml_router.py (빈 파일)
- setup_and_build.sh, auto_setup_and_test.sh, quick_test.sh → scripts/

Updated: main.py, database.py, routers/__init__.py, agent/config/chat/kg/models/rag
Added: history_router, monitoring_router, audio/chain/evaluation/finetuning/
       google_auth/ocr/optimizer/vision/web routers
Ref: DEVELOPMENT_LOG Phase 7–8, CHAT_IMPROVEMENT_PLANS Phase 10.5–10.6
```

**스테이징 대상:**
- `playground/backend/schemas/` (전체)
- `playground/backend/scripts/` (전체)
- `playground/backend/docs/` (전체)
- `playground/backend/monitoring/` (전체)
- `playground/backend/.env.example`
- `playground/backend/README.md`
- `playground/backend/LOCAL_SETUP.md`
- `playground/backend/POETRY_SHELL.md`
- `playground/backend/START_GUIDE.md`
- `playground/backend/TROUBLESHOOTING.md`
- `playground/backend/start_backend.sh`
- `playground/backend/tests/run_vector_db_test.sh`
- `playground/backend/tests/test_vector_db_performance.py`
- `playground/backend/tests/vector_db_test_results.json`
- 삭제: `chat_history.py`, `models.py`, `mcp_streaming.py`, `monitoring.py`, `monitoring_dashboard.py`, `requirements.txt`, `ml_router.py`, `auto_setup_and_test.sh`, `quick_test.sh`, `setup_and_build.sh`
- 수정: `database.py`, `main.py`, `routers/__init__.py`, `routers/agent_router.py`, `routers/chat_router.py`, `routers/config_router.py`, `routers/kg_router.py`, `routers/models_router.py`, `routers/rag_router.py`
- 신규: `routers/audio_router.py`, `chain_router.py`, `evaluation_router.py`, `finetuning_router.py`, `google_auth_router.py`, `history_router.py`, `monitoring_router.py`, `ocr_router.py`, `optimizer_router.py`, `vision_router.py`, `web_router.py`

---

## 그룹 4: feat(playground/backend) – 서비스 및 동적 설정

**커밋 메시지:**
```
feat(playground): Add config, agentic, Google OAuth, monitoring services

Services:
- config_service, encryption_service, key_validator
- intent_classifier, tool_registry, orchestrator (Agentic)
- google_oauth_service
- message_vector_store, session_search_service, session_cache

Features:
- Dynamic API key management (ConfigService, KeyValidator)
- Agentic chat (intent classification, tool registry, SSE)
- Google OAuth (Drive, Docs, Gmail, Calendar, Sheets)
- Monitoring router + middleware (health, metrics, dashboard)
Ref: DEVELOPMENT_LOG Phase 2–5, Phase 7
```

**스테이징 대상:**
- `playground/backend/services/` (전체) — 그룹 3에서 이미 포함 시 스킵 가능. 그룹 3과 한 번에 커밋해도 됨.

※ 그룹 3과 4를 **한 커밋**으로 합쳐도 됨: `refactor(playground): Restructure backend and add services/routers`

---

## 그룹 5: refactor(playground/frontend) – UI 정리 및 새 컴포넌트

**커밋 메시지:**
```
refactor(playground): Clean Chat UI, Settings, Monitoring pages

Removed:
- thread/ (ContentBlocksPreview, MultimodalPreview, artifact, history, messages, etc.)
- AssistantInputPanel, AssistantSelector, ModelSelector, ModelSettingsPanel
- OnboardingGuide, ParameterTooltip, page_with_sessions
- ChatIcon, github, langgraph icons
- useMediaQuery, Thread provider (Stream 유지)

Added:
- ApiKeyModal, FeatureBadge, GoogleOAuthCard, GoogleConnectModal
- InfoPanel, PackageInstallModal, ProviderWarning
- ReasoningNarrative, ReasoningStepper, StreamingText, TypingIndicator
- UI: alert-dialog, alert, checkbox, dialog, dropdown-menu, popover, scroll-area, select, slider
- app/settings/, app/monitoring/

Updated: chat/page.tsx, Navigation, PageLayout, FeatureSelector, ModelSelectorSimple
        GoogleServiceSelector, ThinkMode, ToolCallDisplay, Visualization
        globals.css, layout.tsx, page.tsx, types/chat.ts, tooltip.tsx
Docs: CHANGELOG_UI, CLEANUP_*, REDESIGN_SUMMARY, UI_WORK_SUMMARY, etc.
Ref: DEVELOPMENT_LOG Phase 6–7
```

**스테이징 대상:**
- `playground/frontend/` 아래 삭제/수정/추가된 모든 파일 (package.json, pnpm-lock.yaml, tsconfig.json, tsconfig.tsbuildinfo 포함)
- `playground/frontend/.env.local.example`
- `playground/frontend/e2e/`
- `playground/frontend/*.md` (CHANGELOG_UI, CLEANUP_*, 등)

---

## 그룹 6: fix(beanllm) / refactor(beanllm) – 코어 및 MCP

**커밋 메시지:**
```
fix(beanllm): Update RAG handler, Neo4j adapter, Ollama provider

- handler/core/rag_handler.py: 수정 사항 반영
- domain/knowledge_graph/neo4j_adapter.py: 수정 사항 반영
- providers/ollama_provider.py: 수정 사항 반영
- service/impl: rag_service_impl, __init__ 수정
- mcp_server/tools: google_tools, rag_tools 수정
- mcp_server/services/: 신규 서비스 (있는 경우)
- pyproject.toml, poetry.lock (의존성 정리)
```

**스테이징 대상:**
- `mcp_server/tools/google_tools.py`
- `mcp_server/tools/rag_tools.py`
- `mcp_server/services/` (전체)
- `src/beanllm/domain/knowledge_graph/neo4j_adapter.py`
- `src/beanllm/handler/core/rag_handler.py`
- `src/beanllm/providers/ollama_provider.py`
- `src/beanllm/service/impl/__init__.py`
- `src/beanllm/service/impl/core/rag_service_impl.py`
- `pyproject.toml`
- `poetry.lock`

---

## 그룹 7: docs – README 최신화

**커밋 메시지:**
```
docs: Update README with Playground frontend and current state

- Documentation: Playground Backend 링크 유지, Frontend/Monitoring/Settings 언급
- Playground 섹션: 통합 Chat UI, Agentic 모드, Google OAuth, 모니터링 대시보드
```

**스테이징 대상:**
- `README.md` (루트)

---

## 실행 순서 (권장)

1. **그룹 1** – docs (계획/로그)
2. **그룹 2** – chore (인프라)
3. **그룹 3 + 4** – playground/backend 한 번에 또는 3 → 4
4. **그룹 5** – playground/frontend
5. **그룹 6** – beanllm core / MCP
6. **그룹 7** – README

각 커밋 후 `git status`로 남은 변경이 그룹 계획과 일치하는지 확인하면 좋습니다.

---

## 참고: 이미 수행한 작업 기록

- **CHAT_IMPROVEMENT_PLANS/00_INDEX.md**: Phase 10 (MCP), 10.5 (코드 정리), 10.6 (스키마 분리), Phase 0 (현재 구현 활용) 완료 표시
- **DEVELOPMENT_LOG.md**: Phase 1–8, Frontend Phase 6–7, Backend 구조/서비스/모니터링/Agentic/Google OAuth 등 단계별 완료 내역

이 문서는 위 기록을 바탕으로 커밋만 그룹별로 나누기 위한 체크리스트입니다.
