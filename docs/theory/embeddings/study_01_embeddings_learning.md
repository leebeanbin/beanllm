# Embeddings 학습 가이드: 체계적인 임베딩 학습 경로

**학습 가이드 문서**  
**대상**: AI 엔지니어 지망생, 임베딩을 처음 배우는 개발자

---

## 목차

1. [학습 목표와 범위](#1-학습-목표와-범위)
2. [단계별 학습 경로](#2-단계별-학습-경로)
3. [필수 개념 이해](#3-필수-개념-이해)
4. [실습 프로젝트](#4-실습-프로젝트)
5. [심화 학습](#5-심화-학습)
6. [학습 자료](#6-학습-자료)

---

## 1. 학습 목표와 범위

### 1.1 학습 목표

**초급 목표:**
- 임베딩의 기본 개념 이해
- 간단한 검색 시스템 구축
- llmkit으로 임베딩 사용

**중급 목표:**
- 벡터 공간과 수학적 기초 이해
- 다양한 유사도 측정 방법
- 성능 최적화

**고급 목표:**
- 임베딩 모델 Fine-tuning
- 대규모 시스템 설계
- 프로덕션 배포

### 1.2 학습 범위

**포함:**
- 벡터 공간 이론
- 코사인 유사도
- 임베딩 모델
- 검색 알고리즘
- 실무 활용

**제외:**
- 임베딩 모델 학습 (Pre-trained 모델 사용)
- 수학적 증명 (이론 문서 참고)

---

## 2. 단계별 학습 경로

### 2.1 1주차: 기본 개념

**학습 내용:**
1. **임베딩이란?**
   - 텍스트를 숫자 벡터로 변환
   - 의미를 보존하는 표현

2. **기본 사용법**
   ```python
   from llmkit import Embedding
   
   emb = Embedding()
   text = "고양이는 귀여워"
   vector = emb.embed_sync([text])[0]
   print(f"벡터 차원: {len(vector)}")  # 1536
   ```

3. **간단한 검색**
   ```python
   from llmkit import VectorStore
   
   docs = ["고양이는 귀여워", "강아지는 충실해"]
   store = VectorStore.from_documents(docs)
   results = store.similarity_search("귀여운 동물", k=1)
   ```

**실습:**
- 튜토리얼 코드 실행
- 간단한 검색 시스템 만들기

**참고 문서:**
- [이론: 벡터 공간 기초](./01_vector_space_foundations.md)
- [실무: 기본 사용법](./practice_01_embeddings_usage.md)

---

### 2.2 2주차: 유사도와 거리

**학습 내용:**
1. **코사인 유사도**
   - 방향(의미) 측정
   - 범위: -1 ~ 1

2. **유클리드 거리**
   - 절대 거리 측정
   - 범위: 0 ~ ∞

3. **실제 계산**
   ```python
   from llmkit.embeddings import cosine_similarity, euclidean_distance
   
   vec1 = emb.embed_sync(["고양이"])[0]
   vec2 = emb.embed_sync(["강아지"])[0]
   
   sim = cosine_similarity(vec1, vec2)  # 0.7 정도
   dist = euclidean_distance(vec1, vec2)  # 거리
   ```

**실습:**
- 다양한 텍스트 쌍의 유사도 계산
- 유사도 분포 시각화

**참고 문서:**
- [이론: 코사인 유사도](./02_cosine_similarity_deep_dive.md)
- [이론: 유클리드 거리](./03_euclidean_distance_and_norms.md)

---

### 2.3 3주차: 고급 기법

**학습 내용:**
1. **MMR (Maximal Marginal Relevance)**
   - 다양성 고려 검색
   - 중복 제거

2. **Hard Negative Mining**
   - 학습에 유용한 샘플 선택
   - Contrastive Learning

3. **Query Expansion**
   - 쿼리 확장
   - 검색 품질 향상

**실습:**
```python
# MMR 검색
results = store.mmr_search(query, k=5, lambda_param=0.6)

# Hard Negative Mining
from llmkit.embeddings import find_hard_negatives
hard_negatives = find_hard_negatives(query_vec, candidate_vecs)
```

**참고 문서:**
- [이론: MMR](./05_mmr_maximal_marginal_relevance.md)
- [이론: Contrastive Learning](./04_contrastive_learning_and_hard_negatives.md)

---

### 2.4 4주차: 실무 프로젝트

**프로젝트: 문서 검색 시스템**

**요구사항:**
1. 문서 로딩 및 전처리
2. 임베딩 생성 및 저장
3. 검색 기능 구현
4. 성능 최적화

**구현:**
```python
from llmkit import DocumentLoader, TextSplitter, Embedding, VectorStore

# 1. 문서 로딩
docs = DocumentLoader.load("documents/")

# 2. 텍스트 분할
chunks = TextSplitter.split(docs, chunk_size=500)

# 3. 임베딩 생성
emb = Embedding()
embeddings = emb.embed_sync([chunk.content for chunk in chunks])

# 4. 벡터 저장소 구축
store = VectorStore.from_embeddings(
    [chunk.content for chunk in chunks],
    embeddings
)

# 5. 검색
results = store.similarity_search("질문", k=5)
```

**참고 문서:**
- [실무: 전체 가이드](./practice_01_embeddings_usage.md)

---

## 3. 필수 개념 이해

### 3.1 벡터 공간

**핵심 개념:**
- 벡터: 숫자의 배열
- 차원: 벡터의 크기
- 공간: 벡터들의 집합

**시각화:**
```
2D 공간:
        y
        ↑
        |     v (3, 4)
        |    /
        |   /
        |  /
        | /
        |/________→ x
       u (1, 2)
```

**학습 자료:**
- [이론: 벡터 공간 기초](./01_vector_space_foundations.md)

### 3.2 유사도 측정

**코사인 유사도:**
- 방향(의미) 측정
- 크기 무관
- 텍스트 임베딩에 적합

**유클리드 거리:**
- 절대 거리 측정
- 크기 중요
- 특징 벡터에 적합

**학습 자료:**
- [이론: 코사인 유사도](./02_cosine_similarity_deep_dive.md)
- [이론: 유클리드 거리](./03_euclidean_distance_and_norms.md)

### 3.3 임베딩 모델

**Transformer 기반:**
- BERT, GPT 등
- 문맥 이해
- 다국어 지원

**학습 자료:**
- [실무: 모델 선택](./practice_01_embeddings_usage.md#1-임베딩-모델-선택-가이드)

---

## 4. 실습 프로젝트

### 4.1 프로젝트 1: 간단한 검색 시스템

**목표:** 기본 검색 기능 구현

**단계:**
1. 문서 준비
2. 임베딩 생성
3. 검색 구현
4. 결과 평가

**코드:**
```python
from llmkit import Embedding, VectorStore

# 문서 준비
documents = [
    "고양이는 포유동물이다",
    "강아지는 충실한 동물이다",
    "새는 날아다닌다"
]

# 임베딩 생성
emb = Embedding()
store = VectorStore.from_documents(documents)

# 검색
query = "동물"
results = store.similarity_search(query, k=2)

for i, result in enumerate(results, 1):
    print(f"{i}. {result.document.content} (유사도: {result.score:.3f})")
```

### 4.2 프로젝트 2: 추천 시스템

**목표:** 사용자 기반 추천

**단계:**
1. 사용자 프로필 임베딩
2. 콘텐츠 임베딩
3. 유사도 계산
4. 상위 N개 추천

### 4.3 프로젝트 3: 클러스터링

**목표:** 문서 자동 분류

**단계:**
1. 문서 임베딩
2. K-means 클러스터링
3. 클러스터 분석
4. 시각화

---

## 5. 심화 학습

### 5.1 고급 주제

**1. Contrastive Learning**
- Hard Negative Mining
- InfoNCE Loss
- 학습 과정

**2. MMR 최적화**
- 다양성 측정
- Lambda 파라미터
- Greedy 알고리즘

**3. 벡터 DB 최적화**
- ANN 알고리즘
- 인덱싱 전략
- 성능 튜닝

**참고 문서:**
- [이론: Contrastive Learning](./04_contrastive_learning_and_hard_negatives.md)
- [이론: MMR](./05_mmr_maximal_marginal_relevance.md)

### 5.2 프로덕션 고려사항

**1. 성능 최적화**
- 배치 처리
- 캐싱
- 비동기 처리

**2. 비용 관리**
- 모델 선택
- 캐싱 전략
- 사용량 모니터링

**3. 모니터링**
- 품질 메트릭
- 성능 추적
- 에러 처리

**참고 문서:**
- [실무: 성능 최적화](./practice_01_embeddings_usage.md#3-성능-최적화-전략)
- [실무: 비용 관리](./practice_01_embeddings_usage.md#4-비용-관리)

---

## 6. 학습 자료

### 6.1 추천 도서

**기초:**
- "Hands-On Machine Learning" - 임베딩 기초
- "Natural Language Processing with Python" - NLP 기초

**고급:**
- "Deep Learning" (Ian Goodfellow) - 딥러닝 기초
- "Neural Network Methods for Natural Language Processing" - NLP 심화

### 6.2 온라인 강의

**무료:**
- Coursera: Machine Learning (Andrew Ng)
- Fast.ai: Practical Deep Learning
- Hugging Face: NLP Course

**유료:**
- DeepLearning.AI: NLP Specialization
- Udacity: Natural Language Processing

### 6.3 실습 플랫폼

- **Kaggle**: 임베딩 경진대회
- **Google Colab**: 무료 GPU
- **Hugging Face Spaces**: 데모 배포

---

## 학습 체크리스트

### 초급 (1-2주)
- [ ] 임베딩 기본 개념 이해
- [ ] llmkit으로 간단한 검색 구현
- [ ] 코사인 유사도 이해

### 중급 (3-4주)
- [ ] 벡터 공간 이론 이해
- [ ] 다양한 유사도 측정 방법
- [ ] 실무 프로젝트 완료

### 고급 (5-8주)
- [ ] 고급 기법 이해 (MMR, Hard Negative)
- [ ] 성능 최적화
- [ ] 프로덕션 배포 경험

---

## 다음 단계

임베딩을 마스터한 후:
1. **RAG 학습**: [RAG 학습 가이드](../rag/study_01_rag_learning.md)
2. **벡터 DB 심화**: Pinecone, Qdrant 등
3. **Fine-tuning**: 도메인 특화 모델 학습

---

**작성일**: 2025-01-XX  
**버전**: 1.0

