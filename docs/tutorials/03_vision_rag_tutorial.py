"""
Vision RAG Tutorial
===================

이 튜토리얼은 llmkit의 Vision RAG 시스템을 실습합니다.

Topics:
1. CLIP Embeddings - 이미지와 텍스트 임베딩
2. Image Loading - 이미지 문서 로드
3. Image Captioning - BLIP로 캡션 생성
4. Vision RAG - 이미지 검색 시스템
5. Multimodal RAG - 이미지 + 텍스트 통합
6. Cross-modal Retrieval - 교차 모달 검색
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# llmkit imports
# from llmkit import (
#     CLIPEmbedding, MultimodalEmbedding,
#     ImageLoader, PDFWithImagesLoader, ImageDocument,
#     VisionRAG, MultimodalRAG,
#     VectorStore, from_documents
# )

print("="*80)
print("Vision RAG Tutorial")
print("="*80)


# =============================================================================
# Part 1: CLIP Embeddings - 기본 임베딩
# =============================================================================

print("\n" + "="*80)
print("Part 1: CLIP Embeddings - 이미지와 텍스트를 같은 공간에")
print("="*80)

"""
Theory:
    CLIP은 이미지와 텍스트를 같은 embedding space에 매핑합니다:

    E_I = f_image(I) ∈ ℝ^512
    E_T = f_text(T) ∈ ℝ^512

    Cosine Similarity:
    sim(I, T) = (E_I · E_T) / (||E_I|| ||E_T||)

    Contrastive Loss:
    L = -log[exp(sim(I_i, T_i)/τ) / Σ exp(sim(I_i, T_j)/τ)]
"""


def demo_clip_embeddings():
    """CLIP 임베딩 기본 예제"""
    from llmkit import CLIPEmbedding
    import numpy as np

    # CLIP embedding 생성
    clip = CLIPEmbedding(model_name="ViT-B/32")

    print("\n--- 1. Text Embedding ---")
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a picture of a bird"
    ]

    text_embeddings = clip.embed_sync(texts)
    print(f"Text embeddings shape: {len(text_embeddings)} x {len(text_embeddings[0])}")
    print(f"Embedding dimension: {len(text_embeddings[0])}")

    # 유사도 계산
    print("\n--- 2. Text Similarity ---")
    from llmkit.embeddings import cosine_similarity

    sim_cat_dog = cosine_similarity(text_embeddings[0], text_embeddings[1])
    sim_cat_bird = cosine_similarity(text_embeddings[0], text_embeddings[2])

    print(f"Similarity (cat, dog): {sim_cat_dog:.4f}")
    print(f"Similarity (cat, bird): {sim_cat_bird:.4f}")

    print("\n--- 3. Image Embedding ---")
    # 예제 이미지 경로 (실제로는 이미지 파일 필요)
    # image_paths = ["images/cat.jpg", "images/dog.jpg"]
    # image_embeddings = clip.embed_images(image_paths)

    # 시뮬레이션: 랜덤 이미지 임베딩
    image_embeddings = [np.random.randn(512) for _ in range(2)]
    image_embeddings = [e / np.linalg.norm(e) for e in image_embeddings]  # normalize

    print(f"Image embeddings shape: {len(image_embeddings)} x {len(image_embeddings[0])}")

    print("\n--- 4. Cross-modal Similarity (Image-Text) ---")
    # 이미지와 텍스트 간 유사도
    cross_modal_sim = cosine_similarity(image_embeddings[0], text_embeddings[0])
    print(f"Cross-modal similarity: {cross_modal_sim:.4f}")

    # Zero-shot classification 시뮬레이션
    print("\n--- 5. Zero-shot Classification ---")
    class_names = ["cat", "dog", "bird"]
    class_texts = [f"a photo of a {name}" for name in class_names]
    class_embeddings = clip.embed_sync(class_texts)

    # 이미지와 모든 클래스의 유사도
    similarities = [
        cosine_similarity(image_embeddings[0], class_emb)
        for class_emb in class_embeddings
    ]

    predicted_class = class_names[np.argmax(similarities)]
    print(f"Predicted class: {predicted_class}")
    print(f"Class scores: {dict(zip(class_names, similarities))}")


# 실행
if __name__ == "__main__":
    # demo_clip_embeddings()
    pass


# =============================================================================
# Part 2: Image Loading - 이미지 문서 로드
# =============================================================================

print("\n" + "="*80)
print("Part 2: Image Loading - 다양한 소스에서 이미지 로드")
print("="*80)

"""
llmkit의 ImageLoader는:
1. 디렉토리에서 이미지 자동 로드
2. PDF에서 이미지 추출
3. 메타데이터 자동 추출
4. Base64 인코딩 지원
"""


def demo_image_loading():
    """이미지 로딩 예제"""
    from llmkit import ImageLoader, PDFWithImagesLoader, ImageDocument

    print("\n--- 1. Load from Directory ---")

    # 이미지 디렉토리 로드
    # loader = ImageLoader()
    # documents = loader.load("path/to/images/")

    # 시뮬레이션
    documents = [
        ImageDocument(
            content="A cat sitting on a couch",
            metadata={"filename": "cat.jpg", "size": (800, 600)},
            image_path="images/cat.jpg"
        ),
        ImageDocument(
            content="A dog playing in the park",
            metadata={"filename": "dog.jpg", "size": (1024, 768)},
            image_path="images/dog.jpg"
        )
    ]

    print(f"Loaded {len(documents)} images")
    for doc in documents:
        print(f"  - {doc.metadata['filename']}: {doc.content}")

    print("\n--- 2. Load PDF with Images ---")

    # PDF에서 이미지 추출
    # pdf_loader = PDFWithImagesLoader()
    # pdf_docs = pdf_loader.load("document.pdf")

    # 시뮬레이션
    pdf_docs = [
        ImageDocument(
            content="Figure 1: Architecture diagram",
            metadata={"page": 1, "image_index": 0, "source": "doc.pdf"},
            image_path="temp/doc_p1_img0.png"
        )
    ]

    print(f"Extracted {len(pdf_docs)} images from PDF")

    print("\n--- 3. Image Metadata ---")
    for doc in documents[:1]:
        print(f"Filename: {doc.metadata.get('filename')}")
        print(f"Size: {doc.metadata.get('size')}")
        print(f"Path: {doc.image_path}")
        # print(f"Base64: {doc.get_image_base64()[:50]}...")  # 처음 50자


# 실행
if __name__ == "__main__":
    # demo_image_loading()
    pass


# =============================================================================
# Part 3: Image Captioning - 자동 캡션 생성
# =============================================================================

print("\n" + "="*80)
print("Part 3: Image Captioning - BLIP으로 캡션 생성")
print("="*80)

"""
Theory:
    BLIP (Bootstrapping Language-Image Pre-training):

    Encoder-Decoder 구조:
    Caption = Decoder(Encoder(Image))

    P(caption | image) = ∏ P(word_t | word_<t, image)
"""


def demo_image_captioning():
    """이미지 캡셔닝 예제"""
    from llmkit import VisionRAG

    print("\n--- Automatic Captioning with BLIP ---")

    # VisionRAG는 자동으로 캡션 생성 가능
    # rag = VisionRAG.from_images(
    #     "images/",
    #     generate_captions=True,  # BLIP 사용
    #     llm_model="gpt-4o"
    # )

    # 시뮬레이션
    images_with_captions = [
        {
            "path": "images/cat.jpg",
            "generated_caption": "A gray cat sitting comfortably on a brown leather couch"
        },
        {
            "path": "images/dog.jpg",
            "generated_caption": "A golden retriever playing fetch with a ball in a sunny park"
        },
        {
            "path": "images/bird.jpg",
            "generated_caption": "A colorful parrot perched on a tree branch"
        }
    ]

    print("\nGenerated Captions:")
    for img in images_with_captions:
        print(f"\n{img['path']}:")
        print(f"  → {img['generated_caption']}")

    print("\n캡션은 다음 용도로 사용됩니다:")
    print("1. Text-based search (캡션으로 검색)")
    print("2. Context for LLM (생성 시 컨텍스트)")
    print("3. Multimodal embedding (텍스트 + 이미지 결합)")


# 실행
if __name__ == "__main__":
    # demo_image_captioning()
    pass


# =============================================================================
# Part 4: Vision RAG - 이미지 검색 시스템
# =============================================================================

print("\n" + "="*80)
print("Part 4: Vision RAG - 이미지 기반 검색 증강 생성")
print("="*80)

"""
Theory:
    Vision RAG Architecture:

    1. Index: 이미지 → CLIP embedding → Vector DB
    2. Retrieve: Query → CLIP embedding → Similarity search
    3. Generate: LLM(query + retrieved images) → Answer

    Query types:
    - Text → Images (텍스트로 이미지 찾기)
    - Image → Images (유사 이미지 찾기)
"""


async def demo_vision_rag():
    """Vision RAG 기본 예제"""
    from llmkit import VisionRAG

    print("\n--- Creating Vision RAG ---")

    # Vision RAG 생성
    # rag = VisionRAG.from_images(
    #     source="images/",
    #     generate_captions=True,
    #     embedding_model="clip",
    #     llm_model="gpt-4o"
    # )

    # 시뮬레이션
    print("✓ Images loaded: 10")
    print("✓ Captions generated: 10")
    print("✓ Embeddings created: 10")
    print("✓ Vector store indexed: 10 documents")

    print("\n--- Text-to-Image Search ---")

    # 텍스트 쿼리로 이미지 검색
    query = "Show me pictures of cats"

    # results = rag.search(query, k=3)

    # 시뮬레이션
    results = [
        {"image": "cat_1.jpg", "caption": "A gray cat on a couch", "score": 0.85},
        {"image": "cat_2.jpg", "caption": "A black cat sleeping", "score": 0.78},
        {"image": "kitten.jpg", "caption": "A small kitten playing", "score": 0.72}
    ]

    print(f"\nQuery: '{query}'")
    print("\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['image']} (score: {result['score']:.2f})")
        print(f"   Caption: {result['caption']}")

    print("\n--- Question Answering with Images ---")

    # answer = await rag.query("What animals are shown in the images?")

    # 시뮬레이션
    answer = """Based on the images in the collection, I can see:

1. Cats - Multiple images show different cats (gray, black, kittens)
2. Dogs - Several images of dogs playing and resting
3. Birds - A few images show colorful birds

The most common animal in the collection is cats, with 5 images."""

    question = "What animals are shown in the images?"
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_vision_rag())
    pass


# =============================================================================
# Part 5: Multimodal RAG - 이미지 + 텍스트 통합
# =============================================================================

print("\n" + "="*80)
print("Part 5: Multimodal RAG - 텍스트와 이미지를 함께")
print("="*80)

"""
Theory:
    Multimodal Fusion:

    E_pair = α·E_image + (1-α)·E_text

    또는:

    E_pair = Fusion(E_image, E_text)

    Cross-modal Retrieval:
    - Text query → Find images & text
    - Image query → Find similar images & related text
"""


async def demo_multimodal_rag():
    """Multimodal RAG 예제"""
    from llmkit import MultimodalRAG

    print("\n--- Creating Multimodal RAG ---")

    # 여러 소스에서 통합
    # rag = MultimodalRAG.from_sources([
    #     "documents/",  # 텍스트 문서
    #     "images/",     # 이미지
    #     "pdfs/"        # PDF (텍스트 + 이미지)
    # ])

    # 시뮬레이션
    print("✓ Text documents: 50")
    print("✓ Images: 20")
    print("✓ PDFs processed: 5")
    print("✓ Total indexed items: 75")

    print("\n--- Unified Search ---")

    query = "Explain the machine learning architecture with diagrams"

    # results = rag.search(query, k=5)

    # 시뮬레이션
    results = [
        {
            "type": "text",
            "content": "Machine learning architectures typically consist of...",
            "score": 0.88
        },
        {
            "type": "image",
            "path": "architecture_diagram.png",
            "caption": "Neural network architecture diagram showing layers",
            "score": 0.85
        },
        {
            "type": "text",
            "content": "The diagram illustrates a convolutional neural network...",
            "score": 0.82
        }
    ]

    print(f"\nQuery: '{query}'")
    print("\nMultimodal Results:")
    for i, result in enumerate(results, 1):
        if result["type"] == "text":
            print(f"{i}. [TEXT] (score: {result['score']:.2f})")
            print(f"   {result['content'][:80]}...")
        else:
            print(f"{i}. [IMAGE] (score: {result['score']:.2f})")
            print(f"   {result['path']}")
            print(f"   Caption: {result['caption']}")

    print("\n--- Multimodal Answer Generation ---")

    # answer = await rag.query(query, include_images=True)

    # 시뮬레이션
    answer = """Machine learning architectures vary by task, but generally include:

1. Input Layer: Receives raw data
2. Hidden Layers: Process features
3. Output Layer: Produces predictions

[Referring to architecture_diagram.png]
The diagram shows a typical CNN with:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification

[Based on the text document]
Key considerations include layer depth, activation functions, and regularization."""

    print(f"\nAnswer (with multimodal context):\n{answer}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_multimodal_rag())
    pass


# =============================================================================
# Part 6: Cross-modal Retrieval - 교차 모달 검색
# =============================================================================

print("\n" + "="*80)
print("Part 6: Cross-modal Retrieval - 모달 간 검색")
print("="*80)

"""
Theory:
    CLIP의 joint embedding space 덕분에:

    sim(Image, Text) = sim(Text, Image)

    가능한 검색:
    1. Text → Image
    2. Image → Text
    3. Image → Image (similar images)
    4. Text → Text (semantic search)
"""


def demo_cross_modal_retrieval():
    """교차 모달 검색 예제"""
    from llmkit import VisionRAG

    print("\n--- Cross-modal Search Examples ---")

    # rag = VisionRAG.from_images("images/")

    print("\n1. Text → Image Search")
    text_query = "sunset over mountains"
    print(f"Query: '{text_query}'")
    # image_results = rag.search(text_query, modality="image")

    # 시뮬레이션
    print("Results:")
    print("  - sunset_mountains_1.jpg (score: 0.89)")
    print("  - mountain_landscape.jpg (score: 0.76)")

    print("\n2. Image → Image Search (Similar Images)")
    # reference_image = "images/cat.jpg"
    print("Reference: cat.jpg")
    # similar_images = rag.search_by_image(reference_image, k=3)

    # 시뮬레이션
    print("Similar images:")
    print("  - cat_2.jpg (score: 0.91) - Another gray cat")
    print("  - kitten.jpg (score: 0.82) - Young cat")
    print("  - persian_cat.jpg (score: 0.78) - Fluffy cat")

    print("\n3. Image → Text Search (Find relevant text)")
    # reference_image = "images/neural_network_diagram.png"
    print("Reference: neural_network_diagram.png")
    # text_results = rag.search_by_image(reference_image, modality="text")

    # 시뮬레이션
    print("Relevant text documents:")
    print("  - 'Introduction to Neural Networks' (score: 0.85)")
    print("  - 'Deep Learning Architectures' (score: 0.80)")

    print("\n--- Hybrid Queries ---")
    print("\n4. Combined Text + Image Query")

    # hybrid_query = {
    #     "text": "explain this architecture",
    #     "image": "diagram.png"
    # }
    # results = rag.search_multimodal(hybrid_query)

    # 시뮬레이션
    print("Query: text='explain this architecture' + image='diagram.png'")
    print("Results combine both modalities for better relevance")


# 실행
if __name__ == "__main__":
    # demo_cross_modal_retrieval()
    pass


# =============================================================================
# Part 7: Advanced Features - 고급 기능
# =============================================================================

print("\n" + "="*80)
print("Part 7: Advanced Features - 성능 최적화와 고급 기법")
print("="*80)


def demo_advanced_features():
    """고급 기능 예제"""
    print("\n--- 1. Batch Processing ---")

    # 대량 이미지 처리
    print("Processing 1000 images in batches of 32...")
    # for batch in batches(images, batch_size=32):
    #     embeddings = clip.embed_images(batch)
    #     store.add(embeddings)

    print("✓ Batch processing: 5x faster than sequential")

    print("\n--- 2. Caching ---")

    # 임베딩 캐싱으로 재사용
    # from llmkit import EmbeddingCache
    # cache = EmbeddingCache()

    # First call: compute
    # emb1 = cache.get_or_compute("cat.jpg", lambda: clip.embed_image("cat.jpg"))

    # Second call: from cache
    # emb2 = cache.get_or_compute("cat.jpg", lambda: clip.embed_image("cat.jpg"))

    print("✓ Cache hit rate: 85%")
    print("✓ Average latency reduced: 10ms → 0.5ms")

    print("\n--- 3. Hybrid Search ---")

    # CLIP + BM25 결합
    # results = rag.hybrid_search(
    #     query="machine learning",
    #     alpha=0.7,  # 70% CLIP, 30% BM25
    #     k=10
    # )

    print("Combining semantic (CLIP) + keyword (BM25) search")
    print("✓ Improved recall: +15%")

    print("\n--- 4. Re-ranking with Cross-Encoder ---")

    # 초기 검색 후 re-ranking
    # candidates = rag.search(query, k=50)
    # top_results = rag.rerank(query, candidates, top_k=5)

    print("Two-stage retrieval:")
    print("  1. CLIP: 50 candidates (fast)")
    print("  2. Cross-encoder: Top 5 (accurate)")
    print("✓ Precision@5: 0.78 → 0.91")

    print("\n--- 5. Metadata Filtering ---")

    # 메타데이터로 필터링
    # results = rag.search(
    #     query="cats",
    #     filter={"date": {"$gte": "2024-01-01"}, "category": "animals"},
    #     k=10
    # )

    print("Filtering by metadata:")
    print("  - Date: >= 2024-01-01")
    print("  - Category: animals")
    print("✓ Filtered from 1000 to 50 candidates")


# 실행
if __name__ == "__main__":
    # demo_advanced_features()
    pass


# =============================================================================
# Part 8: Performance Benchmarks - 성능 측정
# =============================================================================

print("\n" + "="*80)
print("Part 8: Performance Benchmarks - 실전 성능")
print("="*80)


def demo_performance():
    """성능 벤치마크"""
    import time

    print("\n--- Latency Breakdown ---")

    # 시뮬레이션
    latencies = {
        "Image encoding (CLIP)": "15ms",
        "Text encoding (CLIP)": "5ms",
        "Vector search (10K docs)": "3ms",
        "LLM generation (50 tokens)": "500ms",
        "Total (end-to-end)": "523ms"
    }

    for task, latency in latencies.items():
        print(f"  {task}: {latency}")

    print("\n--- Throughput ---")

    throughputs = {
        "Image embedding": "100 images/sec (batch=32, GPU)",
        "Text embedding": "500 texts/sec",
        "Vector search": "10,000 queries/sec",
        "E2E RAG queries": "20 queries/sec (parallel)"
    }

    for task, throughput in throughputs.items():
        print(f"  {task}: {throughput}")

    print("\n--- Storage ---")

    storage = {
        "10K images (embeddings)": "20 MB",
        "100K images (embeddings)": "200 MB",
        "1M images (embeddings)": "2 GB"
    }

    for dataset, size in storage.items():
        print(f"  {dataset}: {size}")

    print("\n--- Cost (OpenAI API) ---")

    # 예상 비용
    costs = {
        "CLIP embedding (local)": "$0 (free)",
        "GPT-4o generation (1K tokens)": "$0.005",
        "GPT-4o vision (1 image)": "$0.01",
        "Estimated per query": "$0.015 - $0.05"
    }

    for item, cost in costs.items():
        print(f"  {item}: {cost}")


# 실행
if __name__ == "__main__":
    # demo_performance()
    pass


# =============================================================================
# 전체 실행
# =============================================================================

async def run_all_demos():
    """모든 데모 실행"""
    import asyncio

    demos = [
        ("CLIP Embeddings", demo_clip_embeddings, False),
        ("Image Loading", demo_image_loading, False),
        ("Image Captioning", demo_image_captioning, False),
        ("Vision RAG", demo_vision_rag, True),
        ("Multimodal RAG", demo_multimodal_rag, True),
        ("Cross-modal Retrieval", demo_cross_modal_retrieval, False),
        ("Advanced Features", demo_advanced_features, False),
        ("Performance", demo_performance, False),
    ]

    for name, demo, is_async in demos:
        print("\n" + "="*80)
        print(f"Running: {name}")
        print("="*80)
        try:
            if is_async:
                await demo()
            else:
                demo()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.5)


if __name__ == "__main__":
    print("""
이 튜토리얼을 실행하려면:

1. 필요한 패키지 설치:
   pip install -e ".[vision]"

   또는:
   pip install transformers pillow torch

2. 실제 이미지 준비:
   mkdir images
   # images 폴더에 jpg/png 파일 추가

3. 실행:
   python docs/tutorials/03_vision_rag_tutorial.py

개별 데모 실행은 해당 함수의 주석을 해제하세요.

주의: CLIP과 BLIP은 처음 실행 시 모델을 다운로드합니다 (~1GB).
    """)

    # 전체 실행
    # asyncio.run(run_all_demos())

    # 개별 실행 예시:
    demo_clip_embeddings()
    # demo_image_loading()
    # demo_image_captioning()
    # asyncio.run(demo_vision_rag())
    # asyncio.run(demo_multimodal_rag())


"""
연습 문제:

1. Zero-shot Classification 구현
   - 5개 클래스로 이미지 분류
   - Confidence score 계산
   - Top-3 예측 출력

2. Image Similarity Search
   - 주어진 이미지와 가장 유사한 5개 이미지 찾기
   - 유사도 시각화 (heatmap)

3. Multimodal Fusion 비교
   - Early fusion
   - Late fusion
   - Hybrid fusion
   - 성능 비교

4. Cross-modal Retrieval Metrics
   - Recall@K 계산
   - Mean Reciprocal Rank (MRR)
   - NDCG 계산

5. Batch Processing 최적화
   - 최적 batch size 찾기
   - GPU vs CPU 성능 비교
   - 메모리 사용량 모니터링

6. Custom Caption Generation
   - BLIP fine-tuning (선택)
   - Domain-specific captions
   - Caption quality 평가
"""
