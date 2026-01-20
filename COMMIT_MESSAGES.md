# 커밋 메시지 가이드

이 문서는 209개 파일의 변경 사항을 논리적으로 그룹핑한 커밋 메시지입니다.

---

## 커밋 1: 불필요한 모듈 및 문서 파일 제거

```
refactor: remove unused integrations and models modules

- Remove integrations module (langgraph, llamaindex bridges)
  - Removed 7 files (-1,111 lines)
  - Simplifies architecture by removing external library integrations
- Remove models module (duplicate code)
  - Removed llm_provider.py and model_config.py (-385 lines)
  - Functionality moved to domain layer
- Clean up temporary documentation files in playground/backend
  - Removed 8 temporary markdown files (-2,481 lines)
  - Removed 6 duplicate test files (-324 lines)

Total: -4,301 lines removed
Files: 20 files deleted
```

**파일 목록:**
- `src/beanllm/integrations/**` (7개 파일)
- `src/beanllm/models/**` (2개 파일)
- `playground/backend/*.md` (8개 파일)
- `playground/backend/test_*_ollama.py` (6개 파일)

---

## 커밋 2: Domain 레이어에 Protocol 패턴 적용 (Infrastructure 의존성 제거)

```
refactor(domain): apply protocol pattern to remove infrastructure dependencies

Apply Dependency Inversion Principle by replacing direct infrastructure
imports with protocol-based dependency injection in domain layer.

Major changes:
- Replace direct infrastructure imports with Protocol interfaces
- Add dependency injection for distributed system features
- Enable graceful degradation when protocols are not injected

Affected modules:
- embeddings: LockManagerProtocol injection
- evaluation: RateLimiterProtocol, ConcurrencyControllerProtocol
- ocr: 5 protocols (Cache, RateLimiter, EventLogger, LockManager, Config)
- vector_stores: EventLoggerProtocol, LockManagerProtocol
- multi_agent: EventBusProtocol, EventLoggerProtocol
- loaders: BatchProcessorProtocol
- vision: CacheProtocol, RateLimiterProtocol, EventLoggerProtocol
- graph: CacheProtocol
- prompts: CacheProtocol

Files changed: 20+ files
Lines changed: ~800 lines
```

**주요 변경 파일:**
- `src/beanllm/domain/embeddings/base.py` (34줄)
- `src/beanllm/domain/embeddings/local/local_embeddings.py` (39줄)
- `src/beanllm/domain/embeddings/utils/cache.py` (37줄)
- `src/beanllm/domain/evaluation/evaluator.py` (38줄)
- `src/beanllm/domain/ocr/bean_ocr.py` (244줄)
- `src/beanllm/domain/vector_stores/factory.py` (107줄)
- `src/beanllm/domain/vector_stores/base.py` (27줄)
- `src/beanllm/domain/vector_stores/local/chroma.py` (26줄)
- `src/beanllm/domain/multi_agent/communication.py` (78줄)
- `src/beanllm/domain/loaders/core/directory.py` (33줄)
- `src/beanllm/domain/vision/embeddings.py` (82줄)
- `src/beanllm/domain/graph/node_cache.py` (44줄)
- `src/beanllm/domain/prompts/cache.py` (46줄)

---

## 커밋 3: Facade 레이어에 AsyncHelperMixin 추가 및 비동기 처리 개선

```
feat(facade): add AsyncHelperMixin and improve async handling

- Add AsyncHelperMixin to all facade classes
- Replace asyncio.run() with run_async_in_sync() to prevent event loop conflicts
- Add async versions of methods (evaluate_async, batch_evaluate_async)
- Fix indentation errors in import statements
- Convert relative imports to absolute imports

Changes:
- evaluation_facade: Add AsyncHelperMixin, async methods, fix imports
- finetuning_facade: Add AsyncHelperMixin, fix import indentation
- vision_rag_facade: Add AsyncHelperMixin, fix import indentation
- web_search_facade: Add AsyncHelperMixin, fix import indentation
- client_facade: Improve provider detection logic, add Ollama pattern matching
- rag_facade: Improve async handling, fix imports
- audio_facade: Improve async handling, fix imports

Files changed: 9 files
Lines changed: ~300 lines
```

**주요 변경 파일:**
- `src/beanllm/facade/ml/evaluation_facade.py` (97줄)
- `src/beanllm/facade/ml/vision_rag_facade.py` (53줄)
- `src/beanllm/facade/core/client_facade.py` (47줄)
- `src/beanllm/facade/core/rag_facade.py` (45줄)
- `src/beanllm/facade/ml/finetuning_facade.py` (23줄)
- `src/beanllm/facade/ml/audio_facade.py` (27줄)
- `src/beanllm/facade/ml/web_search_facade.py` (12줄)

---

## 커밋 4: Import 경로 정리 (상대 경로 → 절대 경로)

```
refactor: convert relative imports to absolute imports

Convert all relative imports (.., ...) to absolute imports (beanllm.*)
for better code clarity, refactoring safety, and import error prevention.

Affected modules:
- decorators (3 files)
- domain/audio (8 files)
- domain/evaluation (10 files)
- domain/knowledge_graph (6 files)
- domain/loaders (8 files)
- domain/memory (2 files)
- domain/ocr (12 files)
- domain/optimizer (6 files)
- domain/orchestrator (3 files)
- domain/parsers (1 file)
- domain/rag_debug (6 files)
- domain/retrieval (3 files)
- domain/splitters (3 files)
- domain/tools (2 files)
- domain/vision (5 files)
- handler (4 files)
- service (35 files)
- dto (1 file)
- facade (2 files)

Total: 150+ files
Lines changed: ~300 lines (mostly 2-line changes per file)
```

**변경 패턴:**
```python
# Before
from ..utils.logging import get_logger
from ...domain.evaluation.results import BatchEvaluationResult

# After
from beanllm.utils.logging import get_logger
from beanllm.domain.evaluation.results import BatchEvaluationResult
```

---

## 커밋 5: Provider 및 Infrastructure 개선

```
feat(providers,infrastructure): improve error handling and model detection

Providers:
- ollama_provider: Improve list_models() to handle various response types
  - Support dict, list, and object responses
  - Add comprehensive logging for debugging
  - Improve error handling with traceback
- openai_provider: Fix import paths
- provider_factory: Improve import paths and logic

Infrastructure:
- distributed/messaging: Add detailed error logging
  - Replace silent exceptions with debug logging
  - Improve Redis client error handling
- distributed/kafka: Improve error handling
- distributed/redis: Improve error handling

Files changed: 18 files
Lines changed: ~100 lines
```

**주요 변경 파일:**
- `src/beanllm/providers/ollama_provider.py` (75줄)
- `src/beanllm/providers/openai_provider.py` (6줄)
- `src/beanllm/providers/provider_factory.py` (8줄)
- `src/beanllm/infrastructure/distributed/messaging.py` (20줄)
- `src/beanllm/infrastructure/distributed/kafka/**` (2개 파일)
- `src/beanllm/infrastructure/distributed/redis/**` (3개 파일)
- `src/beanllm/infrastructure/distributed/in_memory/**` (6개 파일)

---

## 커밋 6: Utils 모듈 확장 및 기타 개선

```
feat(utils): export async helpers and improve utilities

- Export AsyncHelperMixin and related utilities from utils.__init__
  - AsyncHelperMixin
  - run_async_in_sync
  - log_event_sync
  - get_cached_sync
  - set_cache_sync
- Improve DI container logic
- Improve error handling utilities
- Improve cache utilities
- Update pyproject.toml dependencies
- Update main __init__.py exports

Files changed: 12 files
Lines changed: ~50 lines
```

**주요 변경 파일:**
- `src/beanllm/utils/__init__.py` (+15줄)
- `src/beanllm/utils/core/di_container.py` (18줄)
- `src/beanllm/utils/integration/error_handling.py` (10줄)
- `src/beanllm/utils/core/cache.py` (11줄)
- `pyproject.toml` (+8줄)
- `src/beanllm/__init__.py` (6줄)

---

## 커밋 7: Playground/Backend 업데이트

```
feat(playground): update backend API and test files

- Expand main.py with new API endpoints and features (+1,559 lines)
- Refactor test_all_apis.py for better test structure
- Remove duplicate test files (consolidated into test_all_apis.py)

Files changed: 2 files
Lines changed: +1,559, -255
```

**주요 변경 파일:**
- `playground/backend/main.py` (+1,559줄)
- `playground/backend/test_all_apis.py` (+255/-255줄)

---

## 전체 커밋 순서 요약

1. **불필요한 모듈 및 문서 파일 제거** (20개 파일 삭제)
2. **Domain 레이어 Protocol 패턴 적용** (20+ 파일, ~800줄)
3. **Facade 레이어 AsyncHelperMixin 추가** (9개 파일, ~300줄)
4. **Import 경로 정리** (150+ 파일, ~300줄)
5. **Provider 및 Infrastructure 개선** (18개 파일, ~100줄)
6. **Utils 모듈 확장** (12개 파일, ~50줄)
7. **Playground/Backend 업데이트** (2개 파일, +1,559줄)

---

## 커밋 실행 방법

각 커밋을 순서대로 실행하려면:

```bash
# 1. 파일 삭제 커밋
git add -A
git commit -m "refactor: remove unused integrations and models modules

- Remove integrations module (langgraph, llamaindex bridges)
- Remove models module (duplicate code)
- Clean up temporary documentation files in playground/backend

Total: -4,301 lines removed, 20 files deleted"

# 2. Protocol 패턴 적용 커밋
git add src/beanllm/domain/embeddings/ src/beanllm/domain/evaluation/ \
        src/beanllm/domain/ocr/ src/beanllm/domain/vector_stores/ \
        src/beanllm/domain/multi_agent/ src/beanllm/domain/loaders/ \
        src/beanllm/domain/vision/ src/beanllm/domain/graph/ \
        src/beanllm/domain/prompts/
git commit -m "refactor(domain): apply protocol pattern to remove infrastructure dependencies

Apply Dependency Inversion Principle by replacing direct infrastructure
imports with protocol-based dependency injection in domain layer.

Files changed: 20+ files, ~800 lines"

# 3. Facade 레이어 개선 커밋
git add src/beanllm/facade/
git commit -m "feat(facade): add AsyncHelperMixin and improve async handling

- Add AsyncHelperMixin to all facade classes
- Replace asyncio.run() with run_async_in_sync()
- Add async versions of methods
- Fix indentation errors in import statements

Files changed: 9 files, ~300 lines"

# 4. Import 경로 정리 커밋
git add src/beanllm/decorators/ src/beanllm/domain/ src/beanllm/handler/ \
        src/beanllm/service/ src/beanllm/dto/
git commit -m "refactor: convert relative imports to absolute imports

Convert all relative imports to absolute imports for better code clarity.

Files changed: 150+ files, ~300 lines"

# 5. Provider 및 Infrastructure 개선 커밋
git add src/beanllm/providers/ src/beanllm/infrastructure/
git commit -m "feat(providers,infrastructure): improve error handling and model detection

- Improve ollama_provider list_models() to handle various response types
- Add comprehensive logging for debugging
- Improve error handling in infrastructure layer

Files changed: 18 files, ~100 lines"

# 6. Utils 모듈 확장 커밋
git add src/beanllm/utils/ pyproject.toml src/beanllm/__init__.py
git commit -m "feat(utils): export async helpers and improve utilities

- Export AsyncHelperMixin and related utilities
- Improve DI container logic
- Update dependencies

Files changed: 12 files, ~50 lines"

# 7. Playground/Backend 업데이트 커밋
git add playground/backend/
git commit -m "feat(playground): update backend API and test files

- Expand main.py with new API endpoints (+1,559 lines)
- Refactor test_all_apis.py

Files changed: 2 files"
```

---

## 대안: 단일 커밋 (모든 변경 사항)

모든 변경 사항을 하나의 커밋으로 만들려면:

```bash
git add -A
git commit -m "refactor: major architecture improvements and code cleanup

Architecture Improvements:
- Apply Protocol pattern to Domain layer (remove infrastructure dependencies)
- Add AsyncHelperMixin to Facade layer for consistent async handling
- Convert all relative imports to absolute imports (150+ files)

Code Quality:
- Improve error handling in providers and infrastructure
- Add comprehensive logging for debugging
- Fix indentation errors in import statements

Code Cleanup:
- Remove unused integrations module (-1,111 lines)
- Remove duplicate models module (-385 lines)
- Clean up temporary documentation files (-2,481 lines)

Feature Improvements:
- Improve Ollama provider model detection
- Add async method versions to facades
- Export async helpers from utils module
- Expand playground backend API

Statistics:
- Files changed: 209 files
- Lines added: +2,863
- Lines removed: -5,083
- Net change: -2,220 lines (code simplification)"
```

---

**작성일**: 2026-01-21  
**총 커밋 수**: 7개 (또는 1개 단일 커밋)
