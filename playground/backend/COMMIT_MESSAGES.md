# 커밋 메시지 가이드 (전체 변경사항 반영)

이 문서는 논리적으로 분리된 커밋 메시지들을 제공합니다.

---

## 커밋 1: 아키텍처 리팩토링 - Facade 레이어 재구성

```
refactor(arch): Facade 레이어를 하위 디렉토리로 재구성

Clean Architecture 준수를 위한 Facade 레이어 구조화

Changes:
- facade/core/: 핵심 Facade (Client, RAGChain, Agent, Chain)
- facade/advanced/: 고급 Facade (KnowledgeGraph, MultiAgent, Orchestrator, Optimizer, RAGDebug, StateGraph)
- facade/ml/: ML Facade (VisionRAG, Audio, Evaluation, Fine-tuning, WebSearch)
- 기존 facade/*.py 파일들을 적절한 하위 디렉토리로 이동
- __init__.py를 통한 하위 호환성 유지

Breaking Changes:
- Import 경로 변경: `from beanllm.facade.rag_facade import RAGChain` 
  → `from beanllm.facade.core.rag_facade import RAGChain`
- 하위 호환성을 위해 facade/__init__.py에서 re-export 제공

Files:
- src/beanllm/facade/core/* (새 디렉토리)
- src/beanllm/facade/advanced/* (새 디렉토리)
- src/beanllm/facade/ml/* (새 디렉토리)
- src/beanllm/facade/__init__.py (re-export)
- src/beanllm/facade/*.py (삭제됨)
```

---

## 커밋 2: 아키텍처 리팩토링 - Handler 레이어 재구성

```
refactor(arch): Handler 레이어를 하위 디렉토리로 재구성

Clean Architecture 준수를 위한 Handler 레이어 구조화

Changes:
- handler/core/: 핵심 Handler (Chat, RAG, Agent, Chain)
- handler/advanced/: 고급 Handler (KnowledgeGraph, MultiAgent, Orchestrator, Optimizer, RAGDebug, StateGraph)
- handler/ml/: ML Handler (VisionRAG, Audio, Evaluation, Fine-tuning, WebSearch)
- handler/base_handler.py: 공통 BaseHandler 클래스
- handler/factory.py: HandlerFactory 업데이트

Files:
- src/beanllm/handler/core/* (새 디렉토리)
- src/beanllm/handler/advanced/* (새 디렉토리)
- src/beanllm/handler/ml/* (새 디렉토리)
- src/beanllm/handler/base_handler.py
- src/beanllm/handler/factory.py
- src/beanllm/handler/*.py (삭제됨)
```

---

## 커밋 3: 아키텍처 리팩토링 - Service 레이어 재구성

```
refactor(arch): Service 레이어를 하위 디렉토리로 재구성

Clean Architecture 준수를 위한 Service 레이어 구조화

Changes:
- service/impl/core/: 핵심 Service 구현 (Chat, RAG, Agent, Chain)
- service/impl/advanced/: 고급 Service 구현 (KnowledgeGraph, MultiAgent, Orchestrator, Optimizer, RAGDebug, StateGraph)
- service/impl/ml/: ML Service 구현 (VisionRAG, Audio, Evaluation, Fine-tuning, WebSearch)
- service/impl/base_service.py: 공통 BaseService 클래스
- service/factory.py: ServiceFactory 업데이트

Files:
- src/beanllm/service/impl/core/* (새 디렉토리)
- src/beanllm/service/impl/advanced/* (새 디렉토리)
- src/beanllm/service/impl/ml/* (새 디렉토리)
- src/beanllm/service/impl/base_service.py
- src/beanllm/service/factory.py
- src/beanllm/service/impl/*.py (삭제됨)
```

---

## 커밋 4: 아키텍처 리팩토링 - DTO 레이어 재구성

```
refactor(arch): DTO 레이어를 하위 디렉토리로 재구성

Clean Architecture 준수를 위한 DTO 레이어 구조화

Changes:
- dto/request/core/: 핵심 Request DTO
- dto/request/advanced/: 고급 Request DTO
- dto/request/ml/: ML Request DTO
- dto/request/graph/: Graph Request DTO
- dto/request/web/: Web Request DTO
- dto/response/core/: 핵심 Response DTO
- dto/response/advanced/: 고급 Response DTO
- dto/response/ml/: ML Response DTO
- dto/response/graph/: Graph Response DTO
- dto/response/web/: Web Response DTO

Files:
- src/beanllm/dto/request/core/* (새 디렉토리)
- src/beanllm/dto/request/advanced/* (새 디렉토리)
- src/beanllm/dto/request/ml/* (새 디렉토리)
- src/beanllm/dto/request/graph/* (새 디렉토리)
- src/beanllm/dto/request/web/* (새 디렉토리)
- src/beanllm/dto/response/core/* (새 디렉토리)
- src/beanllm/dto/response/advanced/* (새 디렉토리)
- src/beanllm/dto/response/ml/* (새 디렉토리)
- src/beanllm/dto/response/graph/* (새 디렉토리)
- src/beanllm/dto/response/web/* (새 디렉토리)
- src/beanllm/dto/request/*.py (삭제됨)
- src/beanllm/dto/response/*.py (삭제됨)
```

---

## 커밋 5: 도메인 레이어 리팩토링 - Embeddings 재구성

```
refactor(domain): Embeddings 도메인 재구성

Embeddings 모듈 구조화 및 최적화

Changes:
- domain/embeddings/api/: API 기반 Embeddings
- domain/embeddings/local/: 로컬 Embeddings
- domain/embeddings/utils/: Embeddings 유틸리티
- 중복 코드 제거 및 모듈화
- 기존 advanced.py, api_embeddings.py, local_embeddings.py 등 통합

Files:
- src/beanllm/domain/embeddings/api/* (새 디렉토리)
- src/beanllm/domain/embeddings/local/* (새 디렉토리)
- src/beanllm/domain/embeddings/utils/* (새 디렉토리)
- src/beanllm/domain/embeddings/__init__.py
- src/beanllm/domain/embeddings/base.py
- src/beanllm/domain/embeddings/factory.py
- src/beanllm/domain/embeddings/*.py (삭제됨)
```

---

## 커밋 6: 도메인 레이어 리팩토링 - Loaders 재구성

```
refactor(domain): Loaders 도메인 재구성

Loaders 모듈 구조화 및 최적화

Changes:
- domain/loaders/core/: 핵심 Loader (Text, PDF, Directory 등)
- domain/loaders/advanced/: 고급 Loader (Docling 등)
- 중복 코드 제거 및 모듈화
- 기존 loaders.py의 God Class 분해

Files:
- src/beanllm/domain/loaders/core/* (새 디렉토리)
- src/beanllm/domain/loaders/advanced/* (새 디렉토리)
- src/beanllm/domain/loaders/__init__.py
- src/beanllm/domain/loaders/factory.py
- src/beanllm/domain/loaders/loaders.py
- src/beanllm/domain/loaders/*.py (삭제됨)
```

---

## 커밋 7: 도메인 레이어 리팩토링 - Vector Stores 재구성

```
refactor(domain): Vector Stores 도메인 재구성

Vector Stores 모듈 구조화 및 최적화

Changes:
- domain/vector_stores/cloud/: 클라우드 Vector Store (Pinecone, Qdrant, Weaviate 등)
- domain/vector_stores/local/: 로컬 Vector Store (Chroma, FAISS, LanceDB 등)
- 중복 코드 제거 및 모듈화
- 기존 implementations.py의 God Class 분해

Files:
- src/beanllm/domain/vector_stores/cloud/* (새 디렉토리)
- src/beanllm/domain/vector_stores/local/* (새 디렉토리)
- src/beanllm/domain/vector_stores/__init__.py
- src/beanllm/domain/vector_stores/base.py
- src/beanllm/domain/vector_stores/factory.py
- src/beanllm/domain/vector_stores/implementations.py
- src/beanllm/domain/vector_stores/*.py (삭제됨)
```

---

## 커밋 8: 인프라 레이어 리팩토링

```
refactor(infra): Infrastructure 레이어 재구성

Infrastructure 모듈 구조화 및 최적화

Changes:
- infrastructure/distributed/: 분산 시스템 지원
- infrastructure/integrations/: 외부 시스템 통합
- infrastructure/routing/: 라우팅 로직
- infrastructure/streaming/: 스트리밍 지원
- infrastructure/models/: 모델 관리
- 기존 provider/ 디렉토리 통합

Files:
- src/beanllm/infrastructure/distributed/* (새 디렉토리)
- src/beanllm/infrastructure/integrations/* (새 디렉토리)
- src/beanllm/infrastructure/routing/* (새 디렉토리)
- src/beanllm/infrastructure/streaming/* (새 디렉토리)
- src/beanllm/infrastructure/models/* (업데이트)
- src/beanllm/infrastructure/provider/* (삭제됨)
```

---

## 커밋 9: 유틸리티 레이어 리팩토링

```
refactor(utils): Utils 레이어 재구성

Utils 모듈 구조화 및 최적화

Changes:
- utils/core/: 핵심 유틸리티 (DI Container, Error Handling 등)
- utils/integration/: 통합 유틸리티
- utils/logging/: 로깅 유틸리티
- utils/streaming/: 스트리밍 유틸리티
- 중복 코드 제거 및 모듈화

Files:
- src/beanllm/utils/core/* (새 디렉토리)
- src/beanllm/utils/integration/* (새 디렉토리)
- src/beanllm/utils/logging/* (새 디렉토리)
- src/beanllm/utils/streaming/* (새 디렉토리)
- src/beanllm/utils/*.py (삭제됨)
```

---

## 커밋 10: Playground Backend - Phase 1 API 수정

```
fix(playground): RAG Debug 및 Multi-Agent API 수정

Phase 1 필수 수정 작업 완료

Changes:
- RAG Debug API: collection_name 필드 추가 및 vector_store 파라미터 처리
- Multi-Agent API: agent_configs 필드 추가 및 실제 MultiAgentCoordinator 사용
- 4가지 전략(sequential, parallel, hierarchical, debate) 모두 구현
- 에러 처리 및 검증 로직 추가

Files:
- playground/backend/main.py (RAGDebugRequest, MultiAgentRequest 모델 수정)
- playground/backend/main.py (rag_debug_analyze, multi_agent_run 엔드포인트)
```

---

## 커밋 11: Playground Backend - Phase 2 API 수정

```
fix(playground): Orchestrator 및 Optimizer API 메서드 시그니처 수정

Phase 2 기존 API 수정 작업 완료

Changes:
- Orchestrator API: quick_research_write, quick_parallel_consensus, quick_debate 메서드에 agent 파라미터 추가
- Orchestrator API: 동적으로 Agent 인스턴스 생성 로직 구현
- Optimizer API: quick_optimize 메서드 시그니처에 맞게 수정
- Optimizer API: top_k_range, threshold_range 파라미터 추가

Files:
- playground/backend/main.py (WorkflowRequest 모델에 num_agents 필드 추가)
- playground/backend/main.py (orchestrator_run 엔드포인트 수정)
- playground/backend/main.py (OptimizeRequest 모델 수정)
- playground/backend/main.py (optimize 엔드포인트 수정)
```

---

## 커밋 12: Playground Backend - Chain API 구현

```
feat(playground): Chain API 엔드포인트 추가

Chain API 구현 완료

Changes:
- Chain API: /api/chain/run 엔드포인트 추가 (체인 실행)
- Chain API: /api/chain/build 엔드포인트 추가 (체인 빌드)
- ChainRequest 모델 추가
- ChainBuilder를 사용한 체인 생성 지원
- PromptChain 및 기본 Chain 지원

Files:
- playground/backend/main.py (ChainRequest 모델)
- playground/backend/main.py (chain_run, chain_build 엔드포인트)
- playground/backend/main.py (Chain, ChainBuilder, PromptChain import)
- playground/backend/main.py (_chains 전역 변수 추가)
```

---

## 커밋 13: Playground Backend - VisionRAG API 구현

```
feat(playground): VisionRAG API 엔드포인트 추가

VisionRAG API 구현 완료

Changes:
- VisionRAG API: /api/vision_rag/build 엔드포인트 추가
- VisionRAG API: /api/vision_rag/query 엔드포인트 추가
- VisionRAGBuildRequest, VisionRAGQueryRequest 모델 추가
- 이미지 처리 로직 (base64 디코딩, 임시 디렉토리 사용)
- VisionRAG.from_images 클래스 메서드 활용

Files:
- playground/backend/main.py (VisionRAGBuildRequest, VisionRAGQueryRequest 모델)
- playground/backend/main.py (vision_rag_build, vision_rag_query 엔드포인트)
- playground/backend/main.py (VisionRAG, MultimodalRAG import)
- playground/backend/main.py (_vision_rag 전역 변수 추가)
```

---

## 커밋 14: Playground Backend - Audio API 구현

```
feat(playground): Audio API 엔드포인트 추가

Audio API 구현 완료

Changes:
- Audio API: /api/audio/transcribe 엔드포인트 추가 (음성 → 텍스트)
- Audio API: /api/audio/synthesize 엔드포인트 추가 (텍스트 → 음성)
- Audio API: /api/audio/rag 엔드포인트 추가 (Audio RAG 쿼리)
- AudioTranscribeRequest, AudioSynthesizeRequest, AudioRAGRequest 모델 추가
- WhisperSTT, TextToSpeech, AudioRAG facade 활용
- Base64 인코딩을 통한 오디오 응답 처리

Files:
- playground/backend/main.py (AudioTranscribeRequest, AudioSynthesizeRequest, AudioRAGRequest 모델)
- playground/backend/main.py (audio_transcribe, audio_synthesize, audio_rag 엔드포인트)
- playground/backend/main.py (WhisperSTT, TextToSpeech, AudioRAG import)
- playground/backend/main.py (_audio_rag 전역 변수 추가)
```

---

## 커밋 15: Playground Backend - Evaluation API 구현

```
feat(playground): Evaluation API 엔드포인트 추가

Evaluation API 구현 완료

Changes:
- Evaluation API: /api/evaluation/evaluate 엔드포인트 추가
- EvaluationRequest 모델 추가
- 단일 평가 및 배치 평가 모두 지원
- EvaluatorFacade.batch_evaluate 메서드 활용
- 메트릭 집계 및 요약 기능

Files:
- playground/backend/main.py (EvaluationRequest 모델)
- playground/backend/main.py (evaluation_evaluate 엔드포인트)
- playground/backend/main.py (EvaluatorFacade import)
- playground/backend/main.py (_evaluator 전역 변수 추가)
```

---

## 커밋 16: Playground Backend - Fine-tuning API 구현

```
feat(playground): Fine-tuning API 엔드포인트 추가

Fine-tuning API 구현 완료

Changes:
- Fine-tuning API: /api/finetuning/create 엔드포인트 추가
- Fine-tuning API: /api/finetuning/status/{job_id} 엔드포인트 추가
- FineTuningCreateRequest, FineTuningStatusRequest 모델 추가
- OpenAIFineTuningProvider 사용
- JSONL 형식으로 훈련 데이터 변환
- FineTuningManagerFacade.start_training 및 get_training_progress 활용

Files:
- playground/backend/main.py (FineTuningCreateRequest, FineTuningStatusRequest 모델)
- playground/backend/main.py (finetuning_create, finetuning_status 엔드포인트)
- playground/backend/main.py (FineTuningManagerFacade import)
- playground/backend/main.py (_finetuning 전역 변수 추가)
```

---

## 커밋 17: Playground 테스트 파일 추가

```
test(playground): Phase 1 API 테스트 파일 추가

테스트 파일 작성 완료

Changes:
- test_rag_debug_ollama.py: RAG Debug API 테스트
- test_multi_agent_ollama.py: Multi-Agent API 테스트 (4가지 전략)
- test_all_apis.py: 모든 주요 API 통합 테스트

Files:
- playground/backend/test_rag_debug_ollama.py
- playground/backend/test_multi_agent_ollama.py
- playground/backend/test_all_apis.py
```

---

## 커밋 18: 문서 및 설정 업데이트

```
docs: 프로젝트 문서 및 설정 업데이트

문서화 및 설정 파일 업데이트

Changes:
- README.md: 프로젝트 구조 및 변경사항 반영
- CHANGELOG.md: 버전별 변경사항 기록
- ARCHITECTURE.md: 아키텍처 문서 업데이트
- QUICK_START.md: 빠른 시작 가이드 업데이트
- pyproject.toml: 패키지 설정 업데이트
- src/beanllm/__init__.py: Public API 업데이트
- playground/backend/IMPLEMENTATION_PROGRESS.md: 구현 진행 상황
- playground/backend/COMPLETION_SUMMARY.md: 완료 요약
- playground/backend/MISSING_FEATURES.md: 누락 기능 계획
- playground/backend/VERIFICATION_SUMMARY.md: 검증 요약
- playground/backend/COMMIT_MESSAGES.md: 커밋 메시지 가이드

Files:
- README.md
- CHANGELOG.md
- ARCHITECTURE.md
- QUICK_START.md
- pyproject.toml
- src/beanllm/__init__.py
- playground/backend/*.md
```

---

## 커밋 19: Tutorial 및 예제 추가

```
docs: Tutorial 및 예제 추가

Jupyter Notebook 튜토리얼 및 예제 코드 추가

Changes:
- 17개의 Jupyter Notebook 튜토리얼 추가
- 예제 코드 추가 (model_router_example.py, streaming_example.py)
- REPL UI 추가 (repl_shell.py, common_commands.py 등)

Files:
- docs/tutorials/01_setup_and_installation.ipynb
- docs/tutorials/02_core_client.ipynb
- docs/tutorials/03_rag_and_documents.ipynb
- docs/tutorials/04_embeddings_vector_stores.ipynb
- docs/tutorials/05_agent_and_tools.ipynb
- docs/tutorials/06_chain_and_graph.ipynb
- docs/tutorials/07_multi_agent.ipynb
- docs/tutorials/08_vision_rag.ipynb
- docs/tutorials/09_ocr.ipynb
- docs/tutorials/10_audio_processing.ipynb
- docs/tutorials/11_web_search.ipynb
- docs/tutorials/12_evaluation.ipynb
- docs/tutorials/13_finetuning.ipynb
- docs/tutorials/14_production_features.ipynb
- docs/tutorials/15_distributed_architecture.ipynb
- docs/tutorials/16_performance_benchmarks.ipynb
- docs/tutorials/17_practical_knowledge_graph.ipynb
- examples/model_router_example.py
- examples/streaming_example.py
- src/beanllm/ui/repl/*.py
```

---

## 커밋 20: 기타 도메인 레이어 개선

```
refactor(domain): 기타 도메인 레이어 개선

도메인 레이어 전반적인 개선 및 최적화

Changes:
- evaluation/: 평가 메트릭 및 래퍼 개선
- finetuning/: Fine-tuning 프로바이더 개선
- graph/: 그래프 노드 및 캐시 개선
- knowledge_graph/: Knowledge Graph 빌더 및 쿼리어 개선
- memory/: 메모리 구현 개선
- multi_agent/: 멀티 에이전트 통신 및 전략 개선
- ocr/: OCR 엔진 및 모델 개선
- optimizer/: 최적화 엔진 및 벤치마커 개선
- orchestrator/: 워크플로우 분석 및 모니터링 개선
- parsers/: 파서 개선
- prompts/: 프롬프트 캐시 개선
- rag_debug/: RAG 디버깅 도구 개선
- retrieval/: 하이브리드 검색 및 리랭킹 개선
- splitters/: 텍스트 스플리터 개선
- tools/: 도구 레지스트리 개선
- vision/: 비전 모델 및 임베딩 개선

Files:
- src/beanllm/domain/evaluation/*
- src/beanllm/domain/finetuning/*
- src/beanllm/domain/graph/*
- src/beanllm/domain/knowledge_graph/*
- src/beanllm/domain/memory/*
- src/beanllm/domain/multi_agent/*
- src/beanllm/domain/ocr/*
- src/beanllm/domain/optimizer/*
- src/beanllm/domain/orchestrator/*
- src/beanllm/domain/parsers/*
- src/beanllm/domain/prompts/*
- src/beanllm/domain/rag_debug/*
- src/beanllm/domain/retrieval/*
- src/beanllm/domain/splitters/*
- src/beanllm/domain/tools/*
- src/beanllm/domain/vision/*
```

---

## 커밋 21: Provider 및 Infrastructure 개선

```
refactor(infra): Provider 및 Infrastructure 개선

Provider 및 Infrastructure 레이어 전반적인 개선

Changes:
- providers/: 모든 Provider 개선 (OpenAI, Claude, Gemini, Ollama, DeepSeek, Perplexity)
- infrastructure/adapter/: 파라미터 어댑터 개선
- infrastructure/hybrid/: 하이브리드 매니저 개선
- infrastructure/inferrer/: 메타데이터 추론 개선
- infrastructure/scanner/: 모델 스캐너 개선
- infrastructure/security/: 보안 설정 개선
- decorators/: 데코레이터 개선 (error_handler, logger, provider_error_handler)

Files:
- src/beanllm/providers/*
- src/beanllm/infrastructure/adapter/*
- src/beanllm/infrastructure/hybrid/*
- src/beanllm/infrastructure/inferrer/*
- src/beanllm/infrastructure/scanner/*
- src/beanllm/infrastructure/security/*
- src/beanllm/decorators/*
```

---

## 전체 커밋 순서 (권장)

### 아키텍처 리팩토링 (1-9)
1. `refactor(arch): Facade 레이어를 하위 디렉토리로 재구성`
2. `refactor(arch): Handler 레이어를 하위 디렉토리로 재구성`
3. `refactor(arch): Service 레이어를 하위 디렉토리로 재구성`
4. `refactor(arch): DTO 레이어를 하위 디렉토리로 재구성`
5. `refactor(domain): Embeddings 도메인 재구성`
6. `refactor(domain): Loaders 도메인 재구성`
7. `refactor(domain): Vector Stores 도메인 재구성`
8. `refactor(infra): Infrastructure 레이어 재구성`
9. `refactor(utils): Utils 레이어 재구성`

### Playground Backend (10-16)
10. `fix(playground): RAG Debug 및 Multi-Agent API 수정`
11. `fix(playground): Orchestrator 및 Optimizer API 메서드 시그니처 수정`
12. `feat(playground): Chain API 엔드포인트 추가`
13. `feat(playground): VisionRAG API 엔드포인트 추가`
14. `feat(playground): Audio API 엔드포인트 추가`
15. `feat(playground): Evaluation API 엔드포인트 추가`
16. `feat(playground): Fine-tuning API 엔드포인트 추가`

### 테스트 및 문서 (17-19)
17. `test(playground): Phase 1 API 테스트 파일 추가`
18. `docs: 프로젝트 문서 및 설정 업데이트`
19. `docs: Tutorial 및 예제 추가`

### 기타 개선 (20-21)
20. `refactor(domain): 기타 도메인 레이어 개선`
21. `refactor(infra): Provider 및 Infrastructure 개선`

---

## 대안: 더 큰 단위로 커밋

만약 커밋 수를 줄이고 싶다면:

### 옵션 A: 3개 커밋
1. `refactor(arch): 전체 아키텍처 리팩토링 (Facade, Handler, Service, DTO 재구성)`
2. `feat(playground): Playground Backend API 구현 및 수정`
3. `docs: 문서, 튜토리얼 및 설정 업데이트`

### 옵션 B: 5개 커밋
1. `refactor(arch): 레이어 구조 재구성 (Facade, Handler, Service, DTO)`
2. `refactor(domain): 도메인 레이어 재구성 (Embeddings, Loaders, Vector Stores 등)`
3. `refactor(infra): Infrastructure 및 Provider 개선`
4. `feat(playground): Playground Backend API 구현 및 수정`
5. `docs: 문서, 튜토리얼 및 설정 업데이트`

---

## 사용 방법

각 커밋은 독립적으로 적용 가능하도록 설계되었습니다. 
프로젝트 규모와 팀 정책에 따라 적절한 커밋 단위를 선택하세요.

```bash
# 예시: 첫 번째 커밋
git add src/beanllm/facade/
git commit -m "refactor(arch): Facade 레이어를 하위 디렉토리로 재구성

Clean Architecture 준수를 위한 Facade 레이어 구조화

Changes:
- facade/core/: 핵심 Facade (Client, RAGChain, Agent, Chain)
- facade/advanced/: 고급 Facade (KnowledgeGraph, MultiAgent, Orchestrator, Optimizer, RAGDebug, StateGraph)
- facade/ml/: ML Facade (VisionRAG, Audio, Evaluation, Fine-tuning, WebSearch)
- 기존 facade/*.py 파일들을 적절한 하위 디렉토리로 이동
- __init__.py를 통한 하위 호환성 유지"
```
