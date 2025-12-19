# Embeddings 실무 가이드: 프로덕션에서의 임베딩 활용

**실무 적용 문서**  
**대상**: AI 엔지니어, ML 엔지니어, 백엔드 개발자

---

## 목차

1. [임베딩 모델 선택 가이드](#1-임베딩-모델-선택-가이드)
2. [실무에서의 임베딩 사용 패턴](#2-실무에서의-임베딩-사용-패턴)
3. [성능 최적화 전략](#3-성능-최적화-전략)
4. [비용 관리](#4-비용-관리)
5. [모니터링과 디버깅](#5-모니터링과-디버깅)
6. [베스트 프랙티스](#6-베스트-프랙티스)
7. [트러블슈팅](#7-트러블슈팅)

---

## 1. 임베딩 모델 선택 가이드

### 1.1 모델 비교표

| 모델 | 차원 | 비용 | 속도 | 다국어 | 권장 용도 |
|------|------|------|------|--------|----------|
| text-embedding-3-small | 1536 | 낮음 | 빠름 | ✓ | 일반 검색, 대규모 데이터 |
| text-embedding-3-large | 3072 | 중간 | 중간 | ✓ | 고품질 검색, 소규모 데이터 |
| text-embedding-ada-002 | 1536 | 낮음 | 빠름 | ✓ | 레거시, 호환성 |
| embed-multilingual-v3.0 | 1024 | 중간 | 중간 | ✓✓ | 다국어 서비스 |
| voyage-large-2 | 1024 | 중간 | 빠름 | ✓ | 긴 텍스트, 문서 검색 |

### 1.2 선택 기준

**1. 데이터 규모**
- 소규모 (< 10만 문서): `text-embedding-3-large`
- 중규모 (10만 ~ 100만): `text-embedding-3-small`
- 대규모 (> 100만): `text-embedding-3-small` + 벡터 DB 최적화

**2. 언어 요구사항**
- 한국어만: `text-embedding-3-small`
- 다국어: `embed-multilingual-v3.0`
- 영어 중심: `voyage-large-2`

**3. 예산**
- 제한적: `text-embedding-3-small`
- 여유: `text-embedding-3-large`

### 1.3 llmkit 사용 예시

```python
from llmkit import Embedding

# 자동 선택 (권장)
emb = Embedding()  # 기본: text-embedding-3-small

# 명시적 선택
emb = Embedding(model="text-embedding-3-large")

# 다국어 모델
emb = Embedding(model="embed-multilingual-v3.0")
```

---

## 2. 실무에서의 임베딩 사용 패턴

### 2.1 검색 시스템

**패턴 1: 단순 유사도 검색**

```python
from llmkit import Embedding, VectorStore

# 1. 문서 임베딩 생성
emb = Embedding()
docs = ["고양이는 귀여워", "강아지는 충실해", "새는 날아다녀"]
embeddings = emb.embed_sync(docs)

# 2. 벡터 저장소에 저장
store = VectorStore.from_embeddings(docs, embeddings)

# 3. 검색
query = "귀여운 동물"
results = store.similarity_search(query, k=3)
```

**패턴 2: MMR로 다양성 확보**

```python
# 중복 제거된 검색 결과
results = store.mmr_search(query, k=5, lambda_param=0.6)
```

**패턴 3: 하이브리드 검색**

```python
# 벡터 + 키워드 검색
results = store.hybrid_search(query, k=5, alpha=0.7)
```

### 2.2 추천 시스템

```python
# 사용자 프로필 임베딩
user_profile = "AI, 머신러닝, 딥러닝에 관심"
user_embedding = emb.embed_sync([user_profile])[0]

# 콘텐츠 임베딩과 비교
content_embeddings = emb.embed_sync(contents)
similarities = [cosine_similarity(user_embedding, ce) for ce in content_embeddings]

# 상위 N개 추천
top_n = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:10]
```

### 2.3 클러스터링

```python
from sklearn.cluster import KMeans
import numpy as np

# 문서 임베딩
embeddings = emb.embed_sync(documents)
X = np.array(embeddings)

# K-means 클러스터링
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X)

# 클러스터별 문서 그룹화
for i in range(5):
    cluster_docs = [documents[j] for j in range(len(documents)) if clusters[j] == i]
    print(f"Cluster {i}: {len(cluster_docs)} documents")
```

---

## 3. 성능 최적화 전략

### 3.1 배치 처리

**❌ 비효율적:**
```python
# 하나씩 처리 (느림)
for doc in documents:
    embedding = emb.embed_sync([doc])[0]
```

**✅ 효율적:**
```python
# 배치 처리 (빠름)
embeddings = emb.embed_sync(documents)  # 한 번에 처리
```

**성능 비교:**
- 단일 처리: 1000개 문서 → ~100초
- 배치 처리: 1000개 문서 → ~10초 (10배 빠름)

### 3.2 캐싱 전략

```python
from llmkit.embeddings import EmbeddingCache

# 캐시 사용
cache = EmbeddingCache(ttl=3600, max_size=10000)
emb = Embedding()

def get_embedding_cached(text: str):
    # 캐시 확인
    cached = cache.get(text)
    if cached:
        return cached
    
    # 캐시 미스: 새로 생성
    embedding = emb.embed_sync([text])[0]
    cache.set(text, embedding)
    return embedding
```

**캐시 효과:**
- 캐시 히트율 80% → 비용 80% 절감
- 응답 시간 90% 단축

### 3.3 비동기 처리

```python
import asyncio
from llmkit import Embedding

async def process_documents_async(documents):
    emb = Embedding()
    
    # 비동기 배치 처리
    tasks = []
    batch_size = 100
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        tasks.append(emb.embed_async(batch))
    
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]
```

---

## 4. 비용 관리

### 4.1 비용 계산

**OpenAI 임베딩 비용 (2024 기준):**
- text-embedding-3-small: $0.02 / 1M tokens
- text-embedding-3-large: $0.13 / 1M tokens

**예시 계산:**
```python
# 1000개 문서, 평균 500 토큰
total_tokens = 1000 * 500 = 500,000 tokens
cost = 500,000 / 1,000,000 * 0.02 = $0.01
```

### 4.2 비용 절감 전략

**1. 캐싱**
- 중복 문서 제거
- 캐시 TTL 설정

**2. 배치 처리**
- API 호출 횟수 최소화
- 배치 크기 최적화

**3. 모델 선택**
- 필요 이상의 고차원 모델 사용 지양
- text-embedding-3-small로 시작

**4. 인덱싱 전략**
- 자주 변경되지 않는 문서는 사전 인덱싱
- 증분 업데이트

---

## 5. 모니터링과 디버깅

### 5.1 주요 메트릭

```python
# 임베딩 품질 모니터링
def monitor_embedding_quality(query, results):
    metrics = {
        "avg_similarity": sum(r.score for r in results) / len(results),
        "min_similarity": min(r.score for r in results),
        "max_similarity": max(r.score for r in results),
        "result_count": len(results)
    }
    return metrics
```

### 5.2 디버깅 도구

```python
# 유사도 분포 확인
import matplotlib.pyplot as plt

similarities = [r.score for r in results]
plt.hist(similarities, bins=20)
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.title("Similarity Distribution")
plt.show()
```

### 5.3 로깅

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_with_logging(query, k=5):
    logger.info(f"Search query: {query}, k={k}")
    results = store.similarity_search(query, k=k)
    logger.info(f"Found {len(results)} results")
    for i, r in enumerate(results, 1):
        logger.debug(f"Result {i}: score={r.score:.3f}, content={r.document.content[:50]}...")
    return results
```

---

## 6. 베스트 프랙티스

### 6.1 문서 전처리

**✅ 권장:**
- 텍스트 정규화 (공백, 특수문자)
- 불필요한 정보 제거
- 의미 단위로 분할

**❌ 지양:**
- 너무 짧은 청크 (< 50자)
- 너무 긴 청크 (> 2000자)
- 중복 문서

### 6.2 임베딩 저장

**✅ 권장:**
- 벡터 DB 사용 (Pinecone, Qdrant, Chroma)
- 메타데이터 함께 저장
- 버전 관리

**❌ 지양:**
- 파일 시스템 직접 저장
- 메모리에만 저장
- 버전 관리 없음

### 6.3 검색 품질 개선

**1. 쿼리 전처리**
```python
def preprocess_query(query: str) -> str:
    # 소문자 변환
    query = query.lower()
    # 불필요한 단어 제거
    stop_words = ["은", "는", "이", "가", "을", "를"]
    words = query.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
```

**2. 쿼리 확장**
```python
from llmkit.embeddings import query_expansion

# 관련 쿼리 생성
expanded_queries = query_expansion(
    query="고양이",
    embedding=emb,
    top_k=3
)
# ["고양이", "고양이 사료", "고양이 건강"]
```

---

## 7. 트러블슈팅

### 7.1 일반적인 문제

**문제 1: 검색 결과가 관련 없음**

**원인:**
- 임베딩 모델이 도메인에 맞지 않음
- 쿼리와 문서의 언어 불일치

**해결:**
- 도메인 특화 모델 사용
- Fine-tuning 고려
- 쿼리 전처리 개선

**문제 2: 검색 속도가 느림**

**원인:**
- 대규모 데이터셋
- 비효율적인 검색 알고리즘

**해결:**
- ANN 사용 (FAISS, HNSW)
- 벡터 DB 최적화
- 인덱싱 전략 개선

**문제 3: 비용이 너무 높음**

**원인:**
- 중복 임베딩 생성
- 불필요한 재계산

**해결:**
- 캐싱 도입
- 배치 처리
- 모델 다운그레이드

### 7.2 성능 튜닝 체크리스트

- [ ] 배치 처리 사용
- [ ] 캐싱 구현
- [ ] 적절한 모델 선택
- [ ] 벡터 DB 최적화
- [ ] 인덱싱 전략 수립
- [ ] 모니터링 설정
- [ ] 비용 추적

---

## 참고 자료

- [이론 문서](./01_vector_space_foundations.md)
- [코사인 유사도 이론](./02_cosine_similarity_deep_dive.md)
- [llmkit Embeddings 문서](../../README.md#embeddings)

---

**작성일**: 2025-01-XX  
**버전**: 1.0

