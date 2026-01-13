# 누락된 기능 및 구현 계획

**날짜**: 2026-01-13
**현재 상태**: Phase 1 완료, 추가 기능 구현 필요

---

## 📊 현재 구현 상태

### ✅ 구현된 Backend API (9개)

1. ✅ **Chat API** - `/api/chat`
2. ✅ **Knowledge Graph API** - `/api/kg/*` (Graph 포함)
3. ✅ **RAG API** - `/api/rag/*`
4. ✅ **Agent API** - `/api/agent/run`
5. ✅ **Web Search API** - `/api/web/search`
6. ✅ **RAG Debug API** - `/api/rag_debug/analyze`
7. ✅ **Optimizer API** - `/api/optimizer/optimize`
8. ✅ **Multi-Agent API** - `/api/multi_agent/run`
9. ✅ **Orchestrator API** - `/api/orchestrator/run`

### ❌ 누락된 Backend API (5개)

1. ❌ **Chain API** - `/api/chain/*`
2. ❌ **VisionRAG API** - `/api/vision_rag/*`
3. ❌ **Audio API** - `/api/audio/*`
4. ❌ **Evaluation API** - `/api/evaluation/*`
5. ❌ **Fine-tuning API** - `/api/finetuning/*`

### ❌ 프론트엔드 UI

- ❌ 각 기능별 UI 페이지 미구현
- ✅ 기본 구조만 존재 (Next.js + shadcn/ui)

---

## 🔍 Facade 확인 결과

### 사용 가능한 Facade

1. **Chain** - `beanllm.facade.core.chain_facade`
   - `Chain`, `ChainBuilder`, `PromptChain` 등

2. **VisionRAG** - `beanllm.facade.ml.vision_rag_facade`
   - `VisionRAG`, `MultimodalRAG` 등

3. **Audio** - `beanllm.facade.ml.audio_facade`
   - `WhisperSTT`, `TextToSpeech`, `AudioRAG` 등

4. **Evaluation** - `beanllm.facade.ml.evaluation_facade`
   - `EvaluatorFacade` 등

5. **Fine-tuning** - `beanllm.facade.ml.finetuning_facade`
   - `FineTuningManagerFacade` 등

---

## 📋 구현 계획

### Phase 4: 누락된 Backend API 구현

#### Task 4.1: Chain API 구현
- **엔드포인트**: 
  - `POST /api/chain/run` - 기본 체인 실행
  - `POST /api/chain/prompt` - 프롬프트 템플릿 체인
  - `POST /api/chain/build` - 체인 빌더 사용
- **Facade**: `Chain`, `ChainBuilder`, `PromptChain`
- **예상 시간**: 1-2시간

#### Task 4.2: VisionRAG API 구현
- **엔드포인트**:
  - `POST /api/vision_rag/build` - Vision RAG 빌드
  - `POST /api/vision_rag/query` - Vision RAG 쿼리
  - `POST /api/vision_rag/upload` - 이미지 업로드
- **Facade**: `VisionRAG`, `MultimodalRAG`
- **예상 시간**: 2-3시간

#### Task 4.3: Audio API 구현
- **엔드포인트**:
  - `POST /api/audio/transcribe` - 음성 → 텍스트
  - `POST /api/audio/synthesize` - 텍스트 → 음성
  - `POST /api/audio/rag` - Audio RAG
- **Facade**: `WhisperSTT`, `TextToSpeech`, `AudioRAG`
- **예상 시간**: 2-3시간

#### Task 4.4: Evaluation API 구현
- **엔드포인트**:
  - `POST /api/evaluation/evaluate` - 평가 실행
  - `POST /api/evaluation/benchmark` - 벤치마크
  - `GET /api/evaluation/results` - 결과 조회
- **Facade**: `EvaluatorFacade`
- **예상 시간**: 2-3시간

#### Task 4.5: Fine-tuning API 구현
- **엔드포인트**:
  - `POST /api/finetuning/create` - Fine-tuning 작업 생성
  - `GET /api/finetuning/status/{job_id}` - 작업 상태 조회
  - `POST /api/finetuning/upload` - 데이터 업로드
- **Facade**: `FineTuningManagerFacade`
- **예상 시간**: 2-3시간

### Phase 5: 프론트엔드 UI 구현

#### Task 5.1: 기본 레이아웃 및 라우팅
- Next.js App Router 설정
- 각 기능별 페이지 생성
- 네비게이션 구성

#### Task 5.2: 각 기능별 UI 페이지
1. **Chat UI** - `/chat`
2. **RAG UI** - `/rag`
3. **Agent UI** - `/agent`
4. **Multi-Agent UI** - `/multi-agent`
5. **Knowledge Graph UI** - `/knowledge-graph`
6. **VisionRAG UI** - `/vision-rag`
7. **Audio UI** - `/audio`
8. **Evaluation UI** - `/evaluation`
9. **Fine-tuning UI** - `/finetuning`
10. **Chain UI** - `/chain`
11. **Orchestrator UI** - `/orchestrator`
12. **Optimizer UI** - `/optimizer`
13. **RAG Debug UI** - `/rag-debug`
14. **Web Search UI** - `/web-search`

각 페이지는:
- API 호출 로직
- 폼 입력
- 결과 표시
- 에러 처리
- 로딩 상태

---

## 📊 우선순위

### 높음 (즉시 구현)
1. **Chain API** - 기본 기능, 다른 기능의 기반
2. **프론트엔드 기본 구조** - 사용자 경험 향상

### 중간
3. **VisionRAG API** - 멀티모달 기능
4. **Audio API** - 음성 처리 기능
5. **Evaluation API** - 품질 평가

### 낮음
6. **Fine-tuning API** - 고급 기능

---

## 🎯 완료 기준

### Backend
- [ ] 모든 API 엔드포인트 구현
- [ ] 모든 API 테스트 통과
- [ ] 에러 처리 완료
- [ ] 문서화 완료

### Frontend
- [ ] 모든 기능별 페이지 구현
- [ ] API 연동 완료
- [ ] 반응형 디자인
- [ ] 에러 처리 및 로딩 상태
- [ ] 사용자 가이드

---

## 📝 다음 단계

1. **즉시 시작**: Chain API 구현
2. **그 다음**: 프론트엔드 기본 구조 및 Chat/RAG UI
3. **순차적으로**: 나머지 API 및 UI 구현

---

## 📚 참고 자료

- `DETAILED_IMPLEMENTATION_PLAN.md` - 기존 구현 계획
- `IMPLEMENTATION_PROGRESS.md` - 진행 상황
- `src/beanllm/facade/` - Facade 구현 확인
