# Vision RAG Theory: 멀티모달 임베딩과 교차 모달 검색

**석사 수준 이론 문서**  
**기반**: llmkit VisionRAG, CLIPEmbedding 실제 구현 분석

---

## 목차

### Part I: 멀티모달 임베딩 이론
1. [CLIP 아키텍처의 수학적 모델](#part-i-멀티모달-임베딩-이론)
2. [Contrastive Learning의 정보 이론](#12-contrastive-learning의-정보-이론)
3. [공통 임베딩 공간의 기하학](#13-공통-임베딩-공간의-기하학)

### Part II: 교차 모달 검색
4. [이미지-텍스트 검색의 수학적 모델](#part-ii-교차-모달-검색)
5. [Cross-modal Retrieval 알고리즘](#42-cross-modal-retrieval-알고리즘)
6. [Multimodal RAG 파이프라인](#43-multimodal-rag-파이프라인)

### Part III: 이미지 캡셔닝
7. [Image Captioning 모델](#part-iii-이미지-캡셔닝)
8. [BLIP 아키텍처](#72-blip-아키텍처)
9. [캡션 기반 검색](#73-캡션-기반-검색)

---

## Part I: 멀티모달 임베딩 이론

### 1.1 CLIP 아키텍처의 수학적 모델

#### 정의 1.1.1: CLIP (Contrastive Language-Image Pre-training)

**CLIP**은 이미지와 텍스트를 같은 벡터 공간에 매핑합니다:

$$
E_I = f_{\text{image}}(I) \in \mathbb{R}^d
$$

$$
E_T = f_{\text{text}}(T) \in \mathbb{R}^d
$$

여기서 $d$는 임베딩 차원 (CLIP: 512)입니다.

#### 시각적 표현: CLIP 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                  CLIP 아키텍처                           │
└─────────────────────────────────────────────────────────┘

이미지 입력                    텍스트 입력
    │                            │
    │ I: [224×224×3]             │ T: "a cat"
    │                            │
    ▼                            ▼
┌─────────────┐              ┌─────────────┐
│ Vision      │              │ Text        │
│ Encoder     │              │ Encoder     │
│ (ViT/ResNet)│              │ (Transformer)│
└──────┬──────┘              └──────┬──────┘
       │                            │
       │ E_I: [512]                 │ E_T: [512]
       │                            │
       └──────────┬─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  공통 임베딩 공간 │
         │    ℝ^512        │
         └─────────────────┘
                  │
                  ▼
         sim(E_I, E_T) = cos(E_I, E_T)
```

#### 구체적 수치 예시

**예시 1.1.1: CLIP 임베딩 계산**

**입력:**
- 이미지: $I \in \mathbb{R}^{224 \times 224 \times 3}$ (RGB 이미지)
- 텍스트: $T$ = "a cat"

**처리 과정:**

1. **이미지 인코더:**
   $$
   I \rightarrow \text{Vision Transformer} \rightarrow E_I \in \mathbb{R}^{512}
   $$
   - Vision Transformer (ViT): 이미지를 패치로 분할 → Transformer 처리
   - 예: $E_I = [0.12, 0.45, -0.23, \ldots, 0.78]$ (512차원)
   - L2 정규화: $E_I = \frac{E_I}{\|E_I\|}$ (단위 벡터)

2. **텍스트 인코더:**
   $$
   T \rightarrow \text{Tokenize} \rightarrow \text{Transformer} \rightarrow E_T \in \mathbb{R}^{512}
   $$
   - 토큰화: "a cat" → ["a", "cat"] (또는 BPE 토큰)
   - Transformer: 텍스트 임베딩 생성
   - 예: $E_T = [0.15, 0.42, -0.18, \ldots, 0.81]$ (512차원)
   - L2 정규화: $E_T = \frac{E_T}{\|E_T\|}$ (단위 벡터)

3. **유사도 계산:**
   $$
   \text{sim}(E_I, E_T) = \cos(E_I, E_T) = \frac{E_I \cdot E_T}{\|E_I\| \|E_T\|} = E_I \cdot E_T
   $$
   (정규화되어 있으므로 내적 = 코사인 유사도)
   
   예: $\text{sim} = 0.12 \times 0.15 + 0.45 \times 0.42 + \ldots \approx 0.87$

**llmkit 구현:**
```python
# domain/vision/embeddings.py: CLIPEmbedding
# facade/vision_rag_facade.py: VisionRAG
class CLIPEmbedding(BaseEmbedding):
    """
    CLIP 임베딩: E_I = f_image(I), E_T = f_text(T)
    
    수학적 표현:
    - 이미지: I ∈ ℝ^(224×224×3) → E_I ∈ ℝ^512
    - 텍스트: T (문자열) → E_T ∈ ℝ^512
    - 유사도: sim(E_I, E_T) = cos(E_I, E_T)
    
    실제 구현:
    - domain/vision/embeddings.py: CLIPEmbedding
    - transformers 라이브러리 사용 (CLIPModel, CLIPProcessor)
    - L2 정규화 자동 적용
    """
    def __init__(self, model: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            model: CLIP 모델 이름
            - "openai/clip-vit-base-patch32": 기본 (512차원)
            - "openai/clip-vit-large-patch14": 대형 (768차원)
        """
        super().__init__(model=model)
        self._model = None
        self._processor = None
    
    def embed_images(self, images: List[Union[str, Path]]) -> List[List[float]]:
        """
        이미지 임베딩: E_I = f_image(I)
        
        Process:
        1. 이미지 로드 및 전처리 (224×224 리사이즈)
        2. Vision Transformer (ViT) 처리
        3. L2 정규화
        
        실제 구현:
        - domain/vision/embeddings.py: CLIPEmbedding.embed_images()
        """
        self._load_model()
        
        # 이미지 로드 및 전처리
        pil_images = [Image.open(img) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        
        # Vision Encoder 실행
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
        
        # L2 정규화
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().tolist()
    
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 임베딩: E_T = f_text(T)
        
        실제 구현:
        - domain/vision/embeddings.py: CLIPEmbedding.embed_sync()
        """
        self._load_model()
        
        # 텍스트 전처리 및 토큰화
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        
        # Text Encoder 실행
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
        
        # L2 정규화
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().tolist()
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        유사도 계산: sim(E_I, E_T) = cos(E_I, E_T) = E_I · E_T
        
        (정규화되어 있으므로 내적 = 코사인 유사도)
        """
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2))
```
   $$
   \text{sim}(E_I, E_T) = \cos(E_I, E_T) = \frac{E_I \cdot E_T}{\|E_I\| \|E_T\|}
   $$
   $$
   = \frac{0.12 \times 0.15 + 0.45 \times 0.42 + \cdots}{1.0 \times 1.0} \approx 0.89
   $$

**해석:** 유사도 0.89는 이미지와 텍스트가 매우 관련이 있음을 의미합니다.

#### 아키텍처 1.1.1: Dual Encoder

**이미지 인코더:**

$$
f_{\text{image}}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^d
$$

**텍스트 인코더:**

$$
f_{\text{text}}: \text{Token Sequence} \rightarrow \mathbb{R}^d
$$

#### 시각적 비교: 이미지 vs 텍스트 임베딩

```
임베딩 공간 ℝ^512:

        E_T("cat") ★
           /
          /
         /  θ ≈ 15°
        /
       ★ E_I(cat_image)
       
cos(θ) ≈ 0.89 (높은 유사도)

다른 예시:
        E_T("dog") ★
           |
           |  θ ≈ 60°
           |
       ★ E_I(cat_image)
       
cos(θ) ≈ 0.50 (중간 유사도)
```

**llmkit 구현:**
```python
# domain/vision/embeddings.py: CLIPEmbedding
class CLIPEmbedding(BaseEmbedding):
    """
    CLIP 임베딩: E_I = f_image(I), E_T = f_text(T)
    
    실제 구현:
    - domain/vision/embeddings.py: CLIPEmbedding
    - transformers 라이브러리 사용 (CLIPModel, CLIPProcessor)
    """
    def __init__(self, model: str = "openai/clip-vit-base-patch32"):
        """
        실제 구현:
        - domain/vision/embeddings.py: CLIPEmbedding.__init__()
        """
        # CLIP 모델 로드 (lazy loading)
        self.model_name = model
        self._model = None
        self._processor = None
    
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 임베딩: E_T = f_text(T)
        
        실제 구현:
        - domain/vision/embeddings.py: CLIPEmbedding.embed_sync()
        """
        self._load_model()
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
        # L2 정규화
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().tolist()
```

---

### 1.2 Contrastive Learning의 정보 이론

#### 정의 1.2.1: Contrastive Loss

**Contrastive Loss**는 유사한 쌍을 가깝게, 다른 쌍을 멀게 배치합니다:

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j) / \tau)}
$$

여기서:
- $(I_i, T_i)$: Positive 쌍 (매칭되는 이미지-텍스트)
- $(I_i, T_j)$: Negative 쌍 (매칭되지 않는 쌍)
- $\tau$: Temperature parameter

#### 정리 1.2.1: 정보 이론적 해석

**Contrastive Loss는 Mutual Information을 최대화합니다:**

$$
\max I(I; T) = H(T) - H(T | I)
$$

**증명 스케치:**
Contrastive Learning의 목적 함수는 InfoNCE로, Mutual Information의 하한입니다.

---

### 1.3 공통 임베딩 공간의 기하학

#### 정의 1.3.1: 공통 임베딩 공간

**이미지와 텍스트가 같은 공간에 매핑:**

$$
\mathcal{E} = \{E_I, E_T | I \in \mathcal{I}, T \in \mathcal{T}\} \subset \mathbb{R}^d
$$

#### 정리 1.3.1: 교차 모달 유사도

**이미지-텍스트 유사도:**

$$
\text{sim}(I, T) = \cos(E_I, E_T) = \frac{E_I \cdot E_T}{\|E_I\| \|E_T\|}
$$

**llmkit 구현:**
```python
# domain/vision/embeddings.py: CLIPEmbedding.similarity()
def similarity(self, image_vec: List[float], text_vec: List[float]) -> float:
    """
    교차 모달 유사도: sim(I, T) = cos(E_I, E_T) = E_I · E_T (정규화됨)
    
    실제 구현:
    - domain/vision/embeddings.py: CLIPEmbedding.similarity()
    - 정규화되어 있으므로 내적만 계산
    """
    a = np.array(image_vec)
    b = np.array(text_vec)
    return float(np.dot(a, b))  # 이미 normalized됨
```

---

## Part II: 교차 모달 검색

### 2.1 이미지-텍스트 검색의 수학적 모델

#### 정의 2.1.1: Cross-modal Retrieval

**텍스트로 이미지 검색:**

$$
\text{retrieve}(T) = \arg\max_{I \in \mathcal{I}} \text{sim}(E_T, E_I)
$$

**이미지로 텍스트 검색:**

$$
\text{retrieve}(I) = \arg\max_{T \in \mathcal{T}} \text{sim}(E_I, E_T)
$$

**llmkit 구현:**
```python
# facade/vision_rag_facade.py: VisionRAG
class VisionRAG:
    def query(self, query: str, k: int = 5) -> str:
        """
        텍스트 쿼리로 이미지 검색:
        retrieve(T) = argmax_I sim(E_T, E_I)
        """
        # 1. 쿼리 임베딩
        query_vec = self.vision_embedding.embed_sync([query])[0]
        
        # 2. 벡터 검색
        results = self.vector_store.similarity_search_by_vector(
            query_vec, k=k
        )
        
        # 3. LLM으로 답변 생성
        context = self._build_context(results)
        answer = await self.llm.chat([{
            "role": "user",
            "content": f"{context}\n\nQuestion: {query}"
        }])
        return answer.content
```

---

### 2.2 Cross-modal Retrieval 알고리즘

#### 알고리즘 2.2.1: k-NN Cross-modal Search

**시간 복잡도:**
- Naive: $O(|\mathcal{I}| \cdot d)$
- Indexed: $O(\log |\mathcal{I}| \cdot d)$ (HNSW 등)

**llmkit 구현:**
```python
# facade/vision_rag_facade.py: VisionRAG
def _search_images(self, query_vec: List[float], k: int) -> List[VectorSearchResult]:
    """
    k-NN Cross-modal Search
    시간 복잡도: O(log |I| · d) (인덱싱 사용)
    """
    return self.vector_store.similarity_search_by_vector(query_vec, k=k)
```

---

### 2.3 Multimodal RAG 파이프라인

#### 정의 2.3.1: Multimodal RAG

**Multimodal RAG 파이프라인:**

$$
\text{MultimodalRAG}(Q) = \text{LLM}(Q, \text{Retrieve}(Q, \mathcal{I} \cup \mathcal{T}))
$$

**단계별 분해:**

1. **이미지 로딩**: $\mathcal{I} = \text{load\_images}(\text{source})$
2. **캡션 생성**: $C = \text{caption}(\mathcal{I})$
3. **임베딩**: $E = \text{embed}(\mathcal{I}, C)$
4. **저장**: $V = \text{store}(E)$
5. **검색**: $R = \text{retrieve}(Q, V, k)$
6. **생성**: $A = \text{LLM}(Q, R)$

**llmkit 구현:**
```python
# facade/vision_rag_facade.py: VisionRAG.from_images()
# service/impl/vision_rag_service_impl.py: VisionRAGServiceImpl.build_chain()
@classmethod
def from_images(
    cls,
    source: Union[str, Path],
    generate_captions: bool = True,
    **kwargs
) -> 'VisionRAG':
    """
    전체 Multimodal RAG 파이프라인:
    1. I = load_images(source)
    2. C = caption(I) if generate_captions
    3. E = embed(I, C)
    4. V = store(E)
    5. VisionRAG 생성
    
    실제 구현:
    - facade/vision_rag_facade.py: VisionRAG.from_images()
    - service/impl/vision_rag_service_impl.py: VisionRAGServiceImpl.build_chain()
    """
    # 1. 이미지 로딩
    images = load_images(source, generate_captions=generate_captions)
    
    # 2. 임베딩
    vision_embed = CLIPEmbedding()
    
    # 3. 벡터 저장
    vector_store = from_documents(images, embed_func=vision_embed.embed_sync)
    
    return cls(vector_store=vector_store, **kwargs)
```

---

## Part III: 이미지 캡셔닝

### 3.1 Image Captioning 모델

#### 정의 3.1.1: Image Captioning

**Image Captioning**은 이미지를 텍스트로 변환합니다:

$$
\text{caption} = \text{argmax}_{C} P(C | I)
$$

**생성 모델:**

$$
P(C | I) = \prod_{i=1}^{|C|} P(c_i | c_{<i}, I)
$$

---

### 3.2 BLIP 아키텍처

#### 정의 3.2.1: BLIP (Bootstrapping Language-Image Pre-training)

**BLIP**은 Vision-Language 모델로 이미지 캡션을 생성합니다:

$$
\text{caption} = \text{BLIP}(I)
$$

**아키텍처:**
- **Vision Encoder**: 이미지 → 특징 벡터
- **Text Decoder**: 특징 벡터 → 캡션

**llmkit 구현:**
```python
# vision_loaders.py: ImageLoader
class ImageLoader:
    def load_with_captions(
        self,
        images: List[Union[str, Path]],
        model: str = "Salesforce/blip-image-captioning-base"
    ) -> List[ImageDocument]:
        """
        BLIP으로 캡션 생성: caption = BLIP(I)
        """
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        processor = BlipProcessor.from_pretrained(model)
        model = BlipForConditionalGeneration.from_pretrained(model)
        
        captions = []
        for image in images:
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        
        return [ImageDocument(path=img, caption=caption) 
                for img, caption in zip(images, captions)]
```

---

### 3.3 캡션 기반 검색

#### 정의 3.3.1: Hybrid Search (이미지 + 캡션)

**이미지와 캡션을 모두 사용한 검색:**

$$
\text{score}(Q, I) = \alpha \cdot \text{sim}(E_Q, E_I) + (1-\alpha) \cdot \text{sim}(E_Q, E_C)
$$

여기서 $C$는 이미지 $I$의 캡션입니다.

**llmkit 구현:**
```python
# facade/vision_rag_facade.py: MultimodalRAG
class MultimodalRAG(VisionRAG):
    """
    이미지 + 캡션 통합 검색
    """
    def query(self, query: str, k: int = 5, alpha: float = 0.7) -> str:
        """
        Hybrid Search:
        score = α · sim(E_Q, E_I) + (1-α) · sim(E_Q, E_C)
        """
        # 이미지 임베딩 검색
        image_results = self._search_by_image(query, k=k*2)
        
        # 캡션 임베딩 검색
        caption_results = self._search_by_caption(query, k=k*2)
        
        # 점수 결합
        combined = self._combine_results(
            image_results, caption_results, alpha=alpha
        )
        
        return combined[:k]
```

---

## 참고 문헌

1. **Radford et al. (2021)**: "Learning Transferable Visual Models From Natural Language Supervision" - CLIP
2. **Li et al. (2022)**: "BLIP: Bootstrapping Language-Image Pre-training" - BLIP
3. **Wei & Zou (2019)**: "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
