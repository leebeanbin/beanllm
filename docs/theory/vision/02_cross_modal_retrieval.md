# Cross-modal Retrieval: 교차 모달 검색

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit VisionRAG 실제 구현 분석

---

## 목차

1. [교차 모달 검색의 정의](#1-교차-모달-검색의-정의)
2. [이미지-텍스트 검색](#2-이미지-텍스트-검색)
3. [텍스트-이미지 검색](#3-텍스트-이미지-검색)
4. [검색 알고리즘](#4-검색-알고리즘)
5. [CS 관점: 구현과 성능](#5-cs-관점-구현과-성능)

---

## 1. 교차 모달 검색의 정의

### 1.1 교차 모달 검색 문제

#### 정의 1.1.1: Cross-modal Retrieval

**교차 모달 검색**은 한 모달리티로 다른 모달리티를 검색합니다:

$$
\text{Image2Text}: I \rightarrow \{T_1, T_2, \ldots, T_k\}
$$

$$
\text{Text2Image}: T \rightarrow \{I_1, I_2, \ldots, I_k\}
$$

### 1.2 검색 과정

#### 시각적 표현: 교차 모달 검색

```
텍스트 → 이미지 검색:

쿼리: "고양이"
    │
    ▼
┌─────────────┐
│ Text        │
│ Encoder     │
└──────┬──────┘
       │
       │ E_T: [512]
       │
       ▼
┌─────────────────┐
│  공통 임베딩 공간 │
│    ℝ^512        │
└──────┬──────────┘
       │
       │ 유사도 계산
       ▼
┌─────────────────┐
│  이미지 컬렉션    │
│  I₁, I₂, ..., Iₙ│
└──────┬──────────┘
       │
       ▼
상위 k개 이미지 반환
```

---

## 2. 이미지-텍스트 검색

### 2.1 검색 알고리즘

#### 알고리즘 2.1.1: Image2Text Search

```
Algorithm: Image2TextSearch(image, text_candidates, k)
Input:
  - image: 이미지 I
  - text_candidates: 텍스트 리스트 {T₁, ..., Tₙ}
  - k: 반환할 개수
Output: 상위 k개 텍스트

1. E_I ← CLIP.embed_image(image)  // O(d)
2. E_Ts ← CLIP.embed_texts(text_candidates)  // O(n·d)
3. similarities ← [cos(E_I, E_Ti) for E_Ti in E_Ts]  // O(n·d)
4. top_k ← ArgMaxK(similarities, k)  // O(n log k)
5. return [text_candidates[i] for i in top_k]
```

**시간 복잡도:** $O(n \cdot d + n \log k)$

---

## 3. 텍스트-이미지 검색

### 3.1 검색 알고리즘

#### 알고리즘 3.1.1: Text2Image Search

```
Algorithm: Text2ImageSearch(text, image_candidates, k)
1. E_T ← CLIP.embed_text(text)
2. E_Is ← CLIP.embed_images(image_candidates)
3. similarities ← [cos(E_T, E_Ii) for E_Ii in E_Is]
4. top_k ← ArgMaxK(similarities, k)
5. return [image_candidates[i] for i in top_k]
```

---

## 4. 검색 알고리즘

### 4.1 효율적인 검색

#### 최적화 4.1.1: 인덱싱

**이미지 임베딩 사전 계산:**

```python
# 인덱싱 단계
image_embeddings = {}
for image_id, image in image_collection.items():
    embedding = clip.embed_image(image)
    image_embeddings[image_id] = embedding

# 검색 단계 (빠름)
query_embedding = clip.embed_text(query)
similarities = compute_similarities(query_embedding, image_embeddings)
```

**효과:**
- 인덱싱: $O(n \cdot d)$ (한 번)
- 검색: $O(n \cdot d)$ (매번, 하지만 임베딩 계산 제외)

---

## 5. CS 관점: 구현과 성능

### 5.1 llmkit 구현

#### 구현 5.1.1: VisionRAG

```python
# vision_rag.py
class VisionRAG:
    def search(self, query: str, k: int = 5):
        """
        텍스트로 이미지 검색
        """
        # 1. 쿼리 임베딩
        query_emb = self.clip.embed_text(query)
        
        # 2. 모든 이미지와 유사도 계산
        similarities = []
        for image_id, image_emb in self.image_embeddings.items():
            sim = cosine_similarity(query_emb, image_emb)
            similarities.append((image_id, sim))
        
        # 3. 상위 k개
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.images[i] for i, _ in similarities[:k]]
```

---

## 질문과 답변 (Q&A)

### Q1: 교차 모달 검색의 정확도는?

**A:** CLIP 성능:

- **Image2Text:** Top-1 정확도 ~60%
- **Text2Image:** Top-1 정확도 ~55%
- **Top-5:** ~85%

**해석:**
- 완벽하지 않지만 실용적
- Fine-tuning으로 개선 가능

---

## 참고 문헌

1. **Radford et al. (2021)**: "Learning Transferable Visual Models From Natural Language Supervision"
2. **Li et al. (2021)**: "BLIP: Bootstrapping Language-Image Pre-training"

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

