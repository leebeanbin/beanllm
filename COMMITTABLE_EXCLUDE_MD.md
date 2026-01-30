# MD 제외 커밋 가능 목록

> "md 관련된 거 빼고" 커밋할 수 있는 것들만 정리함.

---

## 1. 제외하는 것 (MD 관련)

### 수정/삭제 대상에서 제외 (스테이징 시 제외 권장)
- `README.md` (modified)
- `.claude/GETTING_STARTED.md` (deleted)
- `.claude/MODEL_SELECTION_GUIDE.md` (deleted)

### Untracked에서 제외 (add 하지 않음)
- `DEVELOPMENT_LOG.md`
- `CHAT_IMPROVEMENT_PLANS/` (전부 .md)
- `playground/backend/LOCAL_SETUP.md`
- `playground/backend/POETRY_SHELL.md`
- `playground/backend/README.md`
- `playground/backend/START_GUIDE.md`
- `playground/backend/TROUBLESHOOTING.md`
- `playground/backend/docs/` (전부 .md)
- `playground/frontend/CHANGELOG_UI.md`
- `playground/frontend/CLEANUP_ANALYSIS.md`
- `playground/frontend/CLEANUP_COMPLETE.md`
- `playground/frontend/CLEANUP_FINAL.md`
- `playground/frontend/DELETED_FILES_REVIEW.md`
- `playground/frontend/DESIGN_BENCHMARK_2025.md`
- `playground/frontend/ENHANCEMENT_COMPLETE.md`
- `playground/frontend/REDESIGN_SUMMARY.md`
- `playground/frontend/RESTORED_FILES.md`
- `playground/frontend/UI_WORK_SUMMARY.md`

---

## 2. 커밋 가능 — 이미 추적 중인 변경/삭제 (MD 제외)

아래는 `git add` 후 커밋해도 되는 항목. (위 MD 3건은 add 생략)

```
.gitignore
Makefile
mcp_server/tools/google_tools.py
mcp_server/tools/rag_tools.py
playground/backend/database.py
playground/backend/main.py
playground/backend/routers/__init__.py
playground/backend/routers/agent_router.py
playground/backend/routers/chat_router.py
playground/backend/routers/config_router.py
playground/backend/routers/kg_router.py
playground/backend/routers/models_router.py
playground/backend/routers/rag_router.py
playground/frontend/package.json
playground/frontend/pnpm-lock.yaml
playground/frontend/src/app/chat/page.tsx
playground/frontend/src/app/globals.css
playground/frontend/src/app/layout.tsx
playground/frontend/src/app/page.tsx
playground/frontend/src/components/FeatureSelector.tsx
playground/frontend/src/components/GoogleServiceSelector.tsx
playground/frontend/src/components/ModelSelectorSimple.tsx
playground/frontend/src/components/Navigation.tsx
playground/frontend/src/components/PageLayout.tsx
playground/frontend/src/components/ThinkMode.tsx
playground/frontend/src/components/ToolCallDisplay.tsx
playground/frontend/src/components/Visualization.tsx
playground/frontend/src/components/ui/tooltip.tsx
playground/frontend/src/types/chat.ts
playground/frontend/tsconfig.json
pyproject.toml
src/beanllm/handler/core/rag_handler.py
src/beanllm/providers/ollama_provider.py
src/beanllm/service/impl/__init__.py
src/beanllm/service/impl/core/rag_service_impl.py
```

**삭제된 파일 (MD 아님, 스테이징 가능)**
```
.claude/GETTING_STARTED.md          → MD라 제외
.claude/MODEL_SELECTION_GUIDE.md    → MD라 제외
playground/backend/auto_setup_and_test.sh
playground/backend/chat_history.py
playground/backend/mcp_streaming.py
playground/backend/models.py
playground/backend/monitoring.py
playground/backend/monitoring_dashboard.py
playground/backend/quick_test.sh
playground/backend/requirements.txt
playground/backend/routers/ml_router.py
playground/backend/setup_and_build.sh
playground/frontend/src/app/chat/page_with_sessions.tsx
playground/frontend/src/components/AssistantInputPanel.tsx
playground/frontend/src/components/AssistantSelector.tsx
playground/frontend/src/components/ModelSelector.tsx
playground/frontend/src/components/ModelSettingsPanel.tsx
playground/frontend/src/components/OnboardingGuide.tsx
playground/frontend/src/components/ParameterTooltip.tsx
playground/frontend/src/components/icons/ChatIcon.tsx
playground/frontend/src/components/icons/github.tsx
playground/frontend/src/components/icons/langgraph.tsx
playground/frontend/src/components/thread/* (전체 삭제분)
playground/frontend/src/hooks/useMediaQuery.tsx
playground/frontend/src/providers/Stream.tsx
playground/frontend/src/providers/Thread.tsx
```

---

## 3. 커밋 가능 — Untracked (MD·MD 전용 디렉터리 제외)

아래만 add 하면 “MD 제외” 조건에 맞음.

```
docker-compose.yml
mcp_server/services/
playground/backend/.env.example
playground/backend/monitoring/
playground/backend/routers/audio_router.py
playground/backend/routers/chain_router.py
playground/backend/routers/evaluation_router.py
playground/backend/routers/finetuning_router.py
playground/backend/routers/google_auth_router.py
playground/backend/routers/history_router.py
playground/backend/routers/monitoring_router.py
playground/backend/routers/ocr_router.py
playground/backend/routers/optimizer_router.py
playground/backend/routers/vision_router.py
playground/backend/routers/web_router.py
playground/backend/schemas/
playground/backend/scripts/
playground/backend/services/
playground/backend/start_backend.sh
playground/backend/tests/run_vector_db_test.sh
playground/backend/tests/test_vector_db_performance.py
playground/backend/tests/vector_db_test_results.json
playground/frontend/.env.local.example
playground/frontend/e2e/
playground/frontend/src/app/monitoring/
playground/frontend/src/app/settings/
playground/frontend/src/components/ApiKeyModal.tsx
playground/frontend/src/components/BrowserTabs.tsx
playground/frontend/src/components/FeatureBadge.tsx
playground/frontend/src/components/GoogleConnectModal.tsx
playground/frontend/src/components/GoogleOAuthCard.tsx
playground/frontend/src/components/InfoPanel.tsx
playground/frontend/src/components/PackageInstallModal.tsx
playground/frontend/src/components/ProviderWarning.tsx
playground/frontend/src/components/StreamingText.tsx
playground/frontend/src/components/TypingIndicator.tsx
playground/frontend/src/components/ui/alert-dialog.tsx
playground/frontend/src/components/ui/alert.tsx
playground/frontend/src/components/ui/checkbox.tsx
playground/frontend/src/components/ui/dialog.tsx
playground/frontend/src/components/ui/dropdown-menu.tsx
playground/frontend/src/components/ui/popover.tsx
playground/frontend/src/components/ui/scroll-area.tsx
playground/frontend/src/components/ui/select.tsx
playground/frontend/src/components/ui/slider.tsx
playground/frontend/tsconfig.tsbuildinfo
poetry.lock
scripts/check-env.sh
scripts/mongo-init.js
```

**의도적으로 제외한 항목**
- `DEVELOPMENT_LOG.md`, `CHAT_IMPROVEMENT_PLANS/`, `playground/backend/docs/`
- `playground/backend/*.md`, `playground/frontend/*.md` 전부

---

## 4. 한 번에 스테이징 (MD 제외) — 예시 명령

```bash
# 1) 수정/삭제된 파일 (MD 3건 제외)
git add .gitignore Makefile mcp_server/tools/google_tools.py mcp_server/tools/rag_tools.py
git add playground/backend/database.py playground/backend/main.py playground/backend/routers/
git add playground/frontend/package.json playground/frontend/pnpm-lock.yaml
git add playground/frontend/src/app/chat/page.tsx playground/frontend/src/app/globals.css
git add playground/frontend/src/app/layout.tsx playground/frontend/src/app/page.tsx
git add playground/frontend/src/components/FeatureSelector.tsx playground/frontend/src/components/GoogleServiceSelector.tsx
git add playground/frontend/src/components/ModelSelectorSimple.tsx playground/frontend/src/components/Navigation.tsx
git add playground/frontend/src/components/PageLayout.tsx playground/frontend/src/components/ThinkMode.tsx
git add playground/frontend/src/components/ToolCallDisplay.tsx playground/frontend/src/components/Visualization.tsx
git add playground/frontend/src/components/ui/tooltip.tsx playground/frontend/src/types/chat.ts playground/frontend/tsconfig.json
git add pyproject.toml src/beanllm/handler/core/rag_handler.py src/beanllm/providers/ollama_provider.py
git add src/beanllm/service/impl/__init__.py src/beanllm/service/impl/core/rag_service_impl.py
git add playground/backend/auto_setup_and_test.sh playground/backend/chat_history.py playground/backend/mcp_streaming.py
git add playground/backend/models.py playground/backend/monitoring.py playground/backend/monitoring_dashboard.py
git add playground/backend/quick_test.sh playground/backend/requirements.txt playground/backend/routers/ml_router.py
git add playground/backend/setup_and_build.sh
git add playground/frontend/src/app/chat/page_with_sessions.tsx
git add playground/frontend/src/components/AssistantInputPanel.tsx playground/frontend/src/components/AssistantSelector.tsx
git add playground/frontend/src/components/ModelSelector.tsx playground/frontend/src/components/ModelSettingsPanel.tsx
git add playground/frontend/src/components/OnboardingGuide.tsx playground/frontend/src/components/ParameterTooltip.tsx
git add playground/frontend/src/components/icons/ playground/frontend/src/components/thread/
git add playground/frontend/src/hooks/useMediaQuery.tsx playground/frontend/src/providers/Stream.tsx playground/frontend/src/providers/Thread.tsx

# 2) Untracked (MD·MD전용 디렉터리 제외)
git add docker-compose.yml mcp_server/services/ playground/backend/.env.example playground/backend/monitoring/
git add playground/backend/routers/audio_router.py playground/backend/routers/chain_router.py
git add playground/backend/routers/evaluation_router.py playground/backend/routers/finetuning_router.py
git add playground/backend/routers/google_auth_router.py playground/backend/routers/history_router.py
git add playground/backend/routers/monitoring_router.py playground/backend/routers/ocr_router.py
git add playground/backend/routers/optimizer_router.py playground/backend/routers/vision_router.py
git add playground/backend/routers/web_router.py playground/backend/schemas/ playground/backend/scripts/
git add playground/backend/services/ playground/backend/start_backend.sh
git add playground/backend/tests/run_vector_db_test.sh playground/backend/tests/test_vector_db_performance.py
git add playground/backend/tests/vector_db_test_results.json
git add playground/frontend/.env.local.example playground/frontend/e2e/
git add playground/frontend/src/app/monitoring/ playground/frontend/src/app/settings/
git add playground/frontend/src/components/ApiKeyModal.tsx playground/frontend/src/components/BrowserTabs.tsx
git add playground/frontend/src/components/FeatureBadge.tsx playground/frontend/src/components/GoogleConnectModal.tsx
git add playground/frontend/src/components/GoogleOAuthCard.tsx playground/frontend/src/components/InfoPanel.tsx
git add playground/frontend/src/components/PackageInstallModal.tsx playground/frontend/src/components/ProviderWarning.tsx
git add playground/frontend/src/components/StreamingText.tsx playground/frontend/src/components/TypingIndicator.tsx
git add playground/frontend/src/components/ui/alert-dialog.tsx playground/frontend/src/components/ui/alert.tsx
git add playground/frontend/src/components/ui/checkbox.tsx playground/frontend/src/components/ui/dialog.tsx
git add playground/frontend/src/components/ui/dropdown-menu.tsx playground/frontend/src/components/ui/popover.tsx
git add playground/frontend/src/components/ui/scroll-area.tsx playground/frontend/src/components/ui/select.tsx
git add playground/frontend/src/components/ui/slider.tsx
git add playground/frontend/tsconfig.tsbuildinfo poetry.lock scripts/check-env.sh scripts/mongo-init.js
```

또는 “트랙된 변경만, MD 3건 제외” 후 “Untracked 중 MD 제외”를 스크립트로 한 번에:

```bash
# 트랙된 변경 전체 add 후, MD 3건만 unstage
git add -u
git reset HEAD README.md .claude/GETTING_STARTED.md .claude/MODEL_SELECTION_GUIDE.md 2>/dev/null || true

# Untracked: MD·CHAT_IMPROVEMENT_PLANS·backend/docs·각종 *.md 제외
git add docker-compose.yml mcp_server/services/ playground/backend/.env.example
git add playground/backend/monitoring/ playground/backend/routers/ playground/backend/schemas/
git add playground/backend/scripts/ playground/backend/services/ playground/backend/start_backend.sh
git add playground/backend/tests/ poetry.lock scripts/
git add playground/frontend/.env.local.example playground/frontend/e2e/
git add playground/frontend/src/app/monitoring/ playground/frontend/src/app/settings/
git add playground/frontend/src/components/ playground/frontend/tsconfig.tsbuildinfo
```

---

## 5. 요약

| 구분 | MD 포함 시 | MD 제외 시 (커밋 가능) |
|------|------------|-------------------------|
| 수정/삭제 (tracked) | 76개 | **73개** (README.md + .claude/*.md 2개 제외) |
| Untracked | 50+ | **45개** (상단 MD·MD전용 디렉터리 제외) |

이 파일(`COMMITTABLE_EXCLUDE_MD.md`)은 정리용이므로, 커밋할 때 같이 넣을지 말지는 본인이 결정하면 됩니다.
