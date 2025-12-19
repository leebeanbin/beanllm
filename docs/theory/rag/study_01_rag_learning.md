# RAG 학습 가이드: 체계적인 RAG 학습 경로

**학습 가이드 문서**  
**대상**: AI 엔지니어 지망생, RAG를 처음 배우는 개발자

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
- RAG의 기본 개념 이해
- 간단한 RAG 시스템 구축
- llmkit으로 RAG 사용

**중급 목표:**
- RAG 확률 모델 이해
- 검색 전략 선택
- 프롬프트 엔지니어링

**고급 목표:**
- 고급 검색 기법 (MMR, Hybrid)
- 성능 최적화
- 프로덕션 배포

### 1.2 학습 범위

**포함:**
- RAG 파이프라인
- 문서 처리
- 검색 알고리즘
- LLM 생성
- 실무 활용

**사전 요구사항:**
- 임베딩 기초 (이전 학습)
- Python 기초
- LLM 기본 이해

---

## 2. 단계별 학습 경로

### 2.1 1주차: RAG 기초

**학습 내용:**
1. **RAG란?**
   - 검색 증강 생성
   - LLM의 한계 극복
   - 외부 지식 활용

2. **기본 사용법**
   ```python
   from llmkit import RAGChain
   
   # 문서에서 RAG 생성
   rag = RAGChain.from_documents("documents.pdf")
   
   # 질문하기
   answer = rag.query("이 문서의 주요 내용은?")
   print(answer)
   ```

3. **간단한 파이프라인 이해**
   - 문서 로딩
   - 임베딩 생성
   - 검색
   - 생성

**실습:**
- 튜토리얼 코드 실행
- 간단한 RAG 시스템 만들기

**참고 문서:**
- [이론: RAG 확률 모델](./01_rag_probabilistic_model.md)
- [실무: 기본 사용법](./practice_01_rag_usage.md)

---

### 2.2 2주차: 문서 처리

**학습 내용:**
1. **문서 로딩**
   ```python
   from llmkit import DocumentLoader
   
   # 다양한 형식 지원
   docs = DocumentLoader.load("file.pdf")
   docs = DocumentLoader.load("file.txt")
   ```

2. **텍스트 분할 (Chunking)**
   ```python
   from llmkit import TextSplitter
   
   chunks = TextSplitter.split(
       documents=docs,
       chunk_size=500,
       chunk_overlap=50
   )
   ```

3. **임베딩 생성**
   ```python
   from llmkit import Embedding
   
   emb = Embedding()
   embeddings = emb.embed_sync([chunk.content for chunk in chunks])
   ```

**실습:**
- 다양한 문서 형식 로딩
- 청킹 전략 실험
- 임베딩 생성 및 저장

**참고 문서:**
- [임베딩 학습 가이드](../embeddings/study_01_embeddings_learning.md)

---

### 2.3 3주차: 검색 전략

**학습 내용:**
1. **기본 검색**
   ```python
   results = rag.retrieve("질문", k=5)
   ```

2. **MMR 검색**
   ```python
   results = rag.retrieve("질문", k=5, mmr=True)
   ```

3. **하이브리드 검색**
   ```python
   results = rag.retrieve("질문", k=5, hybrid=True)
   ```

4. **재순위화**
   ```python
   results = rag.retrieve("질문", k=5, rerank=True)
   ```

**실습:**
- 다양한 검색 전략 비교
- 검색 품질 평가
- 최적 전략 선택

**참고 문서:**
- [실무: 검색 전략](./practice_01_rag_usage.md#3-검색-전략-선택)

---

### 2.4 4주차: 프롬프트 엔지니어링

**학습 내용:**
1. **기본 프롬프트**
   - 컨텍스트 주입
   - 질문 포함
   - 답변 형식

2. **개선된 프롬프트**
   - 지시사항 추가
   - 소스 인용
   - 불확실성 처리

3. **도메인 특화 프롬프트**
   - 의료, 법률 등
   - 전문 용어
   - 형식 요구사항

**실습:**
```python
CUSTOM_PROMPT = """You are a helpful assistant.

Context:
{context}

Question: {question}

Answer based only on the context. Cite sources."""

rag = RAGChain(
    vector_store=store,
    llm=llm,
    prompt_template=CUSTOM_PROMPT
)
```

**참고 문서:**
- [실무: 프롬프트 엔지니어링](./practice_01_rag_usage.md#4-프롬프트-엔지니어링)

---

### 2.5 5주차: 실무 프로젝트

**프로젝트: 문서 Q&A 시스템**

**요구사항:**
1. 문서 로딩 및 전처리
2. 벡터 저장소 구축
3. 검색 기능 구현
4. 답변 생성
5. 성능 최적화

**구현:**
```python
from llmkit import RAGChain

# 1. RAG 시스템 구축
rag = RAGChain.from_documents(
    "documents/",
    chunk_size=500,
    embedding_model="text-embedding-3-small"
)

# 2. 질문하기
answer, sources = rag.query(
    "질문",
    k=5,
    include_sources=True
)

# 3. 결과 출력
print(f"답변: {answer}")
print(f"\n소스:")
for i, source in enumerate(sources, 1):
    print(f"{i}. {source.document.content[:100]}...")
```

---

## 3. 필수 개념 이해

### 3.1 RAG 파이프라인

**단계:**
1. **검색 (Retrieval)**: 관련 문서 찾기
2. **컨텍스트 구성**: 문서 결합
3. **생성 (Generation)**: LLM으로 답변 생성

**학습 자료:**
- [이론: RAG 확률 모델](./01_rag_probabilistic_model.md)

### 3.2 검색 알고리즘

**벡터 검색:**
- 코사인 유사도
- 임베딩 기반

**하이브리드 검색:**
- 벡터 + 키워드
- RRF 결합

**MMR:**
- 다양성 고려
- 중복 제거

**학습 자료:**
- [실무: 검색 전략](./practice_01_rag_usage.md#3-검색-전략-선택)

### 3.3 프롬프트 엔지니어링

**기본 원칙:**
- 명확한 지시사항
- 컨텍스트 활용
- 형식 지정

**학습 자료:**
- [실무: 프롬프트 엔지니어링](./practice_01_rag_usage.md#4-프롬프트-엔지니어링)

---

## 4. 실습 프로젝트

### 4.1 프로젝트 1: 간단한 Q&A 시스템

**목표:** 기본 RAG 기능 구현

**단계:**
1. 문서 준비
2. RAG 시스템 구축
3. 질문하기
4. 결과 평가

### 4.2 프로젝트 2: 도메인 특화 RAG

**목표:** 특정 도메인에 맞춘 RAG

**단계:**
1. 도메인 문서 수집
2. 전처리 및 청킹
3. 도메인 특화 프롬프트
4. 평가 및 개선

### 4.3 프로젝트 3: 프로덕션 RAG

**목표:** 배포 가능한 RAG 시스템

**단계:**
1. API 서버 구축
2. 성능 최적화
3. 모니터링 설정
4. 배포

---

## 5. 심화 학습

### 5.1 고급 주제

**1. 고급 검색 기법**
- MMR 최적화
- 하이브리드 검색
- 재순위화

**2. 성능 최적화**
- 캐싱
- 배치 처리
- 비동기 처리

**3. 평가 방법**
- Recall@K
- Precision@K
- MRR

**참고 문서:**
- [실무: 성능 최적화](./practice_01_rag_usage.md#5-성능-최적화)

### 5.2 프로덕션 고려사항

**1. 확장성**
- 수평 확장
- 로드 밸런싱
- 캐싱 전략

**2. 모니터링**
- 품질 메트릭
- 성능 추적
- 에러 처리

**3. 비용 관리**
- API 비용
- 인프라 비용
- 최적화

**참고 문서:**
- [실무: 프로덕션 배포](./practice_01_rag_usage.md#6-프로덕션-배포)

---

## 6. 학습 자료

### 6.1 추천 도서

**기초:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (논문)
- "Building LLM Applications" - RAG 실무

**고급:**
- "Natural Language Processing with Transformers" - Transformer 기초
- "Designing Machine Learning Systems" - 시스템 설계

### 6.2 온라인 강의

**무료:**
- Hugging Face: NLP Course
- LangChain: RAG Tutorial

**유료:**
- DeepLearning.AI: NLP Specialization
- Udacity: Natural Language Processing

### 6.3 실습 플랫폼

- **LangChain**: RAG 예제
- **LlamaIndex**: RAG 프레임워크
- **Hugging Face Spaces**: 데모 배포

---

## 학습 체크리스트

### 초급 (1-2주)
- [ ] RAG 기본 개념 이해
- [ ] llmkit으로 간단한 RAG 구현
- [ ] 문서 처리 이해

### 중급 (3-4주)
- [ ] 검색 전략 이해
- [ ] 프롬프트 엔지니어링
- [ ] 실무 프로젝트 완료

### 고급 (5-8주)
- [ ] 고급 검색 기법
- [ ] 성능 최적화
- [ ] 프로덕션 배포 경험

---

## 다음 단계

RAG를 마스터한 후:
1. **멀티 에이전트**: 여러 에이전트 협업
2. **Vision RAG**: 이미지 + 텍스트
3. **Fine-tuning**: 도메인 특화 모델

---

**작성일**: 2025-01-XX  
**버전**: 1.0

