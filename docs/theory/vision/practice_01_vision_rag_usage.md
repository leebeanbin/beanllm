# Vision RAG 실무 가이드: 이미지 검색 시스템

**실무 적용 문서**

---

## 목차

1. [Vision RAG 기본](#1-vision-rag-기본)
2. [이미지 임베딩](#2-이미지-임베딩)
3. [멀티모달 검색](#3-멀티모달-검색)
4. [실무 패턴](#4-실무-패턴)

---

## 1. Vision RAG 기본

```python
from llmkit.vision_rag import VisionRAG

rag = VisionRAG.from_images("images/")

results = rag.query("고양이 사진", k=5)
```

---

## 2. 이미지 임베딩

```python
from llmkit.vision_embeddings import CLIPEmbedding

clip = CLIPEmbedding()

# 이미지 임베딩
image_emb = clip.embed_image("cat.jpg")

# 텍스트 임베딩
text_emb = clip.embed_text("a cat")
```

---

## 3. 멀티모달 검색

```python
from llmkit.vision_rag import MultimodalRAG

rag = MultimodalRAG()

# 이미지 추가
rag.add_image("cat1.jpg")
rag.add_image("cat2.jpg")

# 텍스트로 검색
results = rag.search("귀여운 고양이", k=2)
```

---

**작성일**: 2025-01-XX  
**버전**: 1.0

