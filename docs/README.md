# 📚 llmkit 문서 가이드

## 📋 목차

1. [문서 구조](#문서-구조)
2. [문서 유형별 설명](#문서-유형별-설명)
3. [사용자별 추천 경로](#사용자별-추천-경로)
4. [주제별 문서 읽기 순서](#주제별-문서-읽기-순서)
5. [빠른 검색](#빠른-검색)

---

## 문서 구조

```
docs/
├── README.md              # 이 파일 (문서 가이드)
│
├── theory/                # 이론 문서 (주제별 폴더)
│   ├── embeddings/        # 임베딩 관련 문서
│   │   ├── 00_overview.md           # 종합 이론
│   │   ├── 01_vector_space_foundations.md  # 벡터 공간 기초
│   │   ├── 02_cosine_similarity_deep_dive.md  # 코사인 유사도 심화
│   │   ├── 03_euclidean_distance_and_norms.md  # 유클리드 거리
│   │   ├── 04_contrastive_learning_and_hard_negatives.md  # 대조 학습
│   │   ├── 05_mmr_maximal_marginal_relevance.md  # MMR 알고리즘
│   │   ├── practice_01_embeddings_usage.md  # 실무 활용
│   │   └── study_01_embeddings_learning.md  # 학습 가이드
│   │
│   ├── rag/               # RAG 관련 문서
│   │   ├── 00_overview.md           # 종합 이론
│   │   ├── 01_rag_probabilistic_model.md  # RAG 확률 모델
│   │   ├── 02_vector_search_and_ann.md  # 벡터 검색 및 ANN
│   │   ├── 03_hybrid_search_and_rrf.md  # 하이브리드 검색 및 RRF
│   │   ├── 04_reranking_cross_encoder.md  # 리랭킹 및 Cross-Encoder
│   │   ├── 05_chunking_strategies.md  # 청킹 전략
│   │   ├── 06_context_injection.md  # 컨텍스트 주입
│   │   ├── practice_01_rag_usage.md  # 실무 활용
│   │   └── study_01_rag_learning.md  # 학습 가이드
│   │
│   ├── graph/             # Graph Workflows
│   │   ├── 00_overview.md
│   │   ├── 01_directed_graphs_and_state_transitions.md
│   │   ├── 02_conditional_routing_and_cycles.md
│   │   ├── 03_node_caching_and_checkpointing.md
│   │   ├── practice_01_graph_usage.md
│   │   └── study_01_graph_learning.md
│   │
│   ├── multi_agent/       # Multi-Agent Systems
│   │   ├── 00_overview.md
│   │   ├── 01_message_passing_models.md
│   │   ├── 02_coordination_strategies.md
│   │   ├── practice_01_multi_agent_usage.md
│   │   └── study_01_multi_agent_learning.md
│   │
│   ├── vision/            # Vision RAG
│   │   ├── 00_overview.md
│   │   ├── 01_clip_architecture_and_contrastive_learning.md
│   │   ├── 02_cross_modal_retrieval.md
│   │   ├── practice_01_vision_rag_usage.md
│   │   └── study_01_vision_rag_learning.md
│   │
│   ├── tools/             # Tool Calling
│   │   ├── 00_overview.md
│   │   ├── 01_tool_schemas_and_type_systems.md
│   │   ├── 02_react_pattern.md
│   │   ├── practice_01_tools_usage.md
│   │   └── study_01_tools_learning.md
│   │
│   ├── web_search/        # Web Search
│   │   ├── 00_overview.md
│   │   ├── 01_tf_idf_and_bm25.md
│   │   ├── 02_pagerank_algorithm.md
│   │   ├── practice_01_web_search_usage.md
│   │   └── study_01_web_search_learning.md
│   │
│   ├── audio/             # Audio Processing
│   │   ├── 00_overview.md
│   │   ├── 01_fourier_transform_and_stft.md
│   │   ├── 02_whisper_and_ctc.md
│   │   ├── practice_01_audio_usage.md
│   │   └── study_01_audio_learning.md
│   │
│   ├── ml_models/         # ML Models Integration
│   │   ├── 00_overview.md
│   │   ├── 01_unified_interface_design.md
│   │   ├── practice_01_ml_models_usage.md
│   │   └── study_01_ml_models_learning.md
│   │
│   ├── production/        # Production Features
│   │   ├── 00_overview.md
│   │   ├── 01_caching_lru_and_ttl.md
│   │   ├── 02_rate_limiting_token_bucket.md
│   │   ├── practice_01_production_usage.md
│   │   └── study_01_production_learning.md
│   │
│   ├── 01_cs_foundations_for_ai.md  # CS 기초 학습 가이드
│   └── 02_ai_engineering_roadmap.md  # AI 엔지니어링 로드맵
│
└── tutorials/             # 튜토리얼 코드
    ├── 01_embeddings_tutorial.py
    ├── 03_graph_tutorial.py
    ├── 03_vision_rag_tutorial.py
    ├── 04_multi_agent_tutorial.py
    ├── 05_ml_models_tutorial.py
    ├── 06_tool_calling_tutorial.py
    ├── 07_web_search_tutorial.py
    ├── 08_audio_speech_tutorial.py
    └── 09_production_features_tutorial.py
```

---

## 문서 유형별 설명

### 1. 이론 문서 (Theory)

**위치**: `theory/{주제}/`

**종류:**
- `00_overview.md`: 종합 이론 문서 (전체 개요)
- `01_*.md`, `02_*.md`, ...: 세부 이론 문서 (수학적, 학술적)

**특징:**
- 석사 수준의 수학적 엄밀성
- 정리와 증명 포함
- CS 관점의 알고리즘 분석
- 다양한 수식과 시각적 표현

**대상**: 연구자, 석사 이상 학습자

---

### 2. 실무 문서 (Practice)

**위치**: `theory/{주제}/practice_*.md`

**특징:**
- 실제 사용 예시
- 베스트 프랙티스
- 성능 최적화
- 트러블슈팅

**대상**: AI 엔지니어, 백엔드 개발자

---

### 3. 학습 가이드 (Study)

**위치**: `theory/{주제}/study_*.md`

**특징:**
- 단계별 학습 로드맵
- 필수 지식 영역
- 실무 프로젝트 추천
- 학습 자료 정리

**대상**: AI 엔지니어 지망생, 전환 개발자

---

### 4. 튜토리얼 (Tutorials)

**위치**: `tutorials/`

**특징:**
- 실행 가능한 Python 코드
- 단계별 설명
- 실제 사용 사례
- 성능 벤치마킹

**대상**: 모든 사용자

---

### 5. 일반 학습 가이드

**위치**: `theory/01_cs_foundations_for_ai.md`, `theory/02_ai_engineering_roadmap.md`

**내용:**
- CS 기초 (데이터 구조, 알고리즘, 시스템 설계)
- AI 엔지니어링 전체 로드맵

---

## 사용자별 추천 경로

### 🎓 초보자

1. **빠른 시작**: [`../QUICK_START.md`](../QUICK_START.md)
2. **학습 로드맵**: `theory/02_ai_engineering_roadmap.md`
3. **CS 기초**: `theory/01_cs_foundations_for_ai.md` (선택)
4. **주제별 학습 가이드**: `theory/{주제}/study_*.md`
5. **튜토리얼 실행**: `tutorials/`
6. **실무 가이드**: `theory/{주제}/practice_*.md`

### 💼 실무자

1. **빠른 시작**: [`../QUICK_START.md`](../QUICK_START.md)
2. **실무 문서 우선**: `theory/{주제}/practice_*.md`
3. **이론 개요**: `theory/{주제}/00_overview.md` (필요시)
4. **튜토리얼**: `tutorials/`
5. **세부 이론**: `theory/{주제}/01_*.md` (필요시)

### 🔬 연구자/학생

1. **종합 이론**: `theory/{주제}/00_overview.md`
2. **세부 이론**: `theory/{주제}/01_*.md` 깊이 있게 학습
3. **학습 가이드**: `theory/{주제}/study_*.md` 참고
4. **구현 확인**: `tutorials/`
5. **실무 적용**: `theory/{주제}/practice_*.md`

---

## 주제별 문서 읽기 순서

### 📊 Embeddings (임베딩)

1. `theory/01_cs_foundations_for_ai.md` - CS 기초 (선택)
2. `theory/embeddings/study_01_embeddings_learning.md` - 학습 가이드
3. `theory/embeddings/00_overview.md` - 종합 이론
4. `theory/embeddings/01_vector_space_foundations.md` - 벡터 공간 이론
5. `theory/embeddings/02_cosine_similarity_deep_dive.md` - 코사인 유사도
6. `theory/embeddings/03_euclidean_distance_and_norms.md` - 유클리드 거리
7. `theory/embeddings/04_contrastive_learning_and_hard_negatives.md` - 대조 학습
8. `theory/embeddings/05_mmr_maximal_marginal_relevance.md` - MMR 알고리즘
9. `theory/embeddings/practice_01_embeddings_usage.md` - 실무 활용
10. `tutorials/01_embeddings_tutorial.py` - 실습

### 🔍 RAG (Retrieval-Augmented Generation)

1. `theory/rag/study_01_rag_learning.md` - 학습 가이드
2. `theory/rag/00_overview.md` - 종합 이론
3. `theory/rag/01_rag_probabilistic_model.md` - RAG 확률 모델
4. `theory/rag/02_vector_search_and_ann.md` - 벡터 검색 및 ANN
5. `theory/rag/03_hybrid_search_and_rrf.md` - 하이브리드 검색 및 RRF
6. `theory/rag/04_reranking_cross_encoder.md` - 리랭킹 및 Cross-Encoder
7. `theory/rag/05_chunking_strategies.md` - 청킹 전략
8. `theory/rag/06_context_injection.md` - 컨텍스트 주입
9. `theory/rag/practice_01_rag_usage.md` - 실무 가이드
10. `tutorials/02_rag_tutorial.py` - 실습

### 🕸️ Graph Workflows

1. `theory/graph/study_01_graph_learning.md` - 학습 가이드
2. `theory/graph/00_overview.md` - 종합 이론
3. `theory/graph/01_directed_graphs_and_state_transitions.md` - 방향 그래프 및 상태 전이
4. `theory/graph/02_conditional_routing_and_cycles.md` - 조건부 라우팅 및 사이클
5. `theory/graph/03_node_caching_and_checkpointing.md` - 노드 캐싱 및 체크포인팅
6. `theory/graph/practice_01_graph_usage.md` - 실무 가이드
7. `tutorials/03_graph_tutorial.py` - 실습

### 👥 Multi-Agent Systems

1. `theory/multi_agent/study_01_multi_agent_learning.md` - 학습 가이드
2. `theory/multi_agent/00_overview.md` - 종합 이론
3. `theory/multi_agent/01_message_passing_models.md` - 메시지 전달 모델
4. `theory/multi_agent/02_coordination_strategies.md` - 조정 전략
5. `theory/multi_agent/practice_01_multi_agent_usage.md` - 실무 가이드
6. `tutorials/04_multi_agent_tutorial.py` - 실습

### 🖼️ Vision RAG

1. `theory/vision/study_01_vision_rag_learning.md` - 학습 가이드
2. `theory/vision/00_overview.md` - 종합 이론
3. `theory/vision/01_clip_architecture_and_contrastive_learning.md` - CLIP 아키텍처 및 대조 학습
4. `theory/vision/02_cross_modal_retrieval.md` - 교차 모달 검색
5. `theory/vision/practice_01_vision_rag_usage.md` - 실무 가이드
6. `tutorials/03_vision_rag_tutorial.py` - 실습

### 🛠️ Tools & Agents

1. `theory/tools/study_01_tools_learning.md` - 학습 가이드
2. `theory/tools/00_overview.md` - 종합 이론
3. `theory/tools/01_tool_schemas_and_type_systems.md` - 도구 스키마 및 타입 시스템
4. `theory/tools/02_react_pattern.md` - ReAct 패턴
5. `theory/tools/practice_01_tools_usage.md` - 실무 가이드
6. `tutorials/06_tool_calling_tutorial.py` - 실습

### 🌐 Web Search

1. `theory/web_search/study_01_web_search_learning.md` - 학습 가이드
2. `theory/web_search/00_overview.md` - 종합 이론
3. `theory/web_search/01_tf_idf_and_bm25.md` - TF-IDF 및 BM25
4. `theory/web_search/02_pagerank_algorithm.md` - PageRank 알고리즘
5. `theory/web_search/practice_01_web_search_usage.md` - 실무 가이드
6. `tutorials/07_web_search_tutorial.py` - 실습

### 🎙️ Audio Processing

1. `theory/audio/study_01_audio_learning.md` - 학습 가이드
2. `theory/audio/00_overview.md` - 종합 이론
3. `theory/audio/01_fourier_transform_and_stft.md` - 푸리에 변환 및 STFT
4. `theory/audio/02_whisper_and_ctc.md` - Whisper 및 CTC
5. `theory/audio/practice_01_audio_usage.md` - 실무 가이드
6. `tutorials/08_audio_speech_tutorial.py` - 실습

### 🏭 Production Features

1. `theory/production/study_01_production_learning.md` - 학습 가이드
2. `theory/production/00_overview.md` - 종합 이론
3. `theory/production/01_caching_lru_and_ttl.md` - 캐싱 (LRU 및 TTL)
4. `theory/production/02_rate_limiting_token_bucket.md` - Rate Limiting (Token Bucket)
5. `theory/production/practice_01_production_usage.md` - 실무 가이드
6. `tutorials/09_production_features_tutorial.py` - 실습

---

## 빠른 검색

### 주제별 문서 찾기

- **임베딩**: `theory/embeddings/`
- **RAG**: `theory/rag/`
- **그래프**: `theory/graph/`
- **Vision RAG**: `theory/vision/`
- **멀티 에이전트**: `theory/multi_agent/`
- **Tool Calling**: `theory/tools/`
- **웹 검색**: `theory/web_search/`
- **ML 모델**: `theory/ml_models/`
- **오디오**: `theory/audio/`
- **프로덕션**: `theory/production/`

### 문서 타입별 찾기

- **이론 (종합)**: `theory/{주제}/00_overview.md`
- **이론 (세부)**: `theory/{주제}/01_*.md`, `02_*.md`, ...
- **실무**: `theory/{주제}/practice_*.md`
- **학습**: `theory/{주제}/study_*.md`
- **튜토리얼**: `tutorials/`

---

## 📖 추가 자료

### 프로젝트 문서

- **[README.md](../README.md)**: 프로젝트 개요 및 주요 기능
- **[QUICK_START.md](../QUICK_START.md)**: 빠른 시작 가이드
- **[ARCHITECTURE.md](../ARCHITECTURE.md)**: 아키텍처 상세 설명
- **[guides/IMPLEMENTATION_ROADMAP_FINAL.md](guides/IMPLEMENTATION_ROADMAP_FINAL.md)**: 최종 구현 로드맵

### 개발 가이드

- **[guides/](guides/)**: 개발 가이드 문서
  - 평가 시스템 분석
  - 벤치마크 구현 계획
  - 테스트 가이드
  - 코드 리뷰 노트

### 예제 코드

- **[examples/](../examples/)**: 다양한 사용 예시
- **[tutorials/](tutorials/)**: 단계별 튜토리얼

---

## 📝 문서 기여

문서를 개선하거나 추가하고 싶으시면:

1. 해당 주제 폴더에 문서 작성
2. 이 README 업데이트
3. Pull Request 제출

---

**최종 업데이트**: 2025-12-22
