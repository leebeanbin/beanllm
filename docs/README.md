# llmkit 문서 가이드

이 디렉토리는 llmkit의 모든 문서를 체계적으로 정리한 곳입니다.

---

## 📁 문서 구조

```
docs/
├── theory/          # 모든 이론 문서 (주제별 폴더)
│   ├── embeddings/  # 임베딩 관련 문서
│   │   ├── 00_overview.md (종합 이론)
│   │   ├── 01_vector_space_foundations.md (이론)
│   │   ├── 02_cosine_similarity_deep_dive.md (이론)
│   │   ├── 03_euclidean_distance_and_norms.md (이론)
│   │   ├── 04_contrastive_learning_and_hard_negatives.md (이론)
│   │   ├── 05_mmr_maximal_marginal_relevance.md (이론)
│   │   ├── practice_01_embeddings_usage.md (실무)
│   │   └── study_01_embeddings_learning.md (학습)
│   │
│   ├── rag/         # RAG 관련 문서
│   │   ├── 00_overview.md (종합 이론)
│   │   ├── 01_rag_probabilistic_model.md (이론)
│   │   ├── practice_01_rag_usage.md (실무)
│   │   └── study_01_rag_learning.md (학습)
│   │
│   ├── graph/       # 그래프 워크플로우
│   ├── vision/      # Vision RAG
│   ├── multi_agent/ # 멀티 에이전트
│   ├── ml_models/   # ML 모델 통합
│   ├── tools/       # Tool Calling
│   ├── web_search/  # 웹 검색
│   ├── audio/       # 오디오 처리
│   ├── production/  # 프로덕션 기능
│   │
│   ├── 01_cs_foundations_for_ai.md (CS 기초 학습 가이드)
│   └── 02_ai_engineering_roadmap.md (AI 엔지니어링 로드맵)
│
└── tutorials/       # 튜토리얼 코드
    ├── 01_embeddings_tutorial.py
    ├── 02_rag_tutorial.py
    └── ...
```

---

## 📚 문서 유형별 설명

### 1. 이론 문서 (Theory)

**위치**: `theory/{주제}/`

**종류:**
- `00_overview.md`: 종합 이론 문서 (기존 통합 문서)
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

### 4. 일반 학습 가이드

**위치**: `theory/01_cs_foundations_for_ai.md`, `theory/02_ai_engineering_roadmap.md`

**내용:**
- CS 기초 (데이터 구조, 알고리즘, 시스템 설계)
- AI 엔지니어링 전체 로드맵

---

## 🎯 사용자별 추천 경로

### 초보자
1. `theory/02_ai_engineering_roadmap.md` - 학습 로드맵 확인
2. `theory/01_cs_foundations_for_ai.md` - CS 기초 학습
3. `theory/{주제}/study_*.md` - 주제별 학습 가이드
4. `tutorials/` - 튜토리얼 코드 실행
5. `theory/{주제}/practice_*.md` - 실무 가이드 참고

### 실무자
1. `theory/{주제}/practice_*.md` - 실무 문서 우선
2. `theory/{주제}/00_overview.md` - 필요시 이론 개요
3. `theory/{주제}/01_*.md` - 세부 이론 필요시
4. `tutorials/` - 코드 예시 확인

### 연구자/학생
1. `theory/{주제}/00_overview.md` - 종합 이론
2. `theory/{주제}/01_*.md` - 세부 이론 문서 깊이 있게 학습
3. `theory/{주제}/study_*.md` - 학습 가이드 참고
4. `tutorials/` - 구현 확인

---

## 📖 주제별 문서 읽기 순서

### 임베딩
1. `theory/01_cs_foundations_for_ai.md` - CS 기초 (선택)
2. `theory/embeddings/study_01_embeddings_learning.md` - 학습 가이드
3. `theory/embeddings/00_overview.md` - 종합 이론
4. `theory/embeddings/01_vector_space_foundations.md` - 벡터 공간 이론
5. `theory/embeddings/02_cosine_similarity_deep_dive.md` - 코사인 유사도
6. `theory/embeddings/practice_01_embeddings_usage.md` - 실무 활용
7. `tutorials/01_embeddings_tutorial.py` - 실습

### RAG
1. `theory/rag/study_01_rag_learning.md` - 학습 가이드
2. `theory/rag/00_overview.md` - 종합 이론
3. `theory/rag/01_rag_probabilistic_model.md` - RAG 확률 모델
4. `theory/rag/practice_01_rag_usage.md` - 실무 가이드
5. `tutorials/02_rag_tutorial.py` - 실습

---

## 🔍 빠른 검색

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

---

## 📝 문서 기여

문서를 개선하거나 추가하고 싶으시면:
1. 해당 주제 폴더에 문서 작성
2. 이 README 업데이트
3. Pull Request 제출

---

**최종 업데이트**: 2025-01-XX
