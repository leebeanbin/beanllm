# AI 엔지니어링 로드맵: 체계적인 학습 경로

**학습 가이드 문서**  
**대상**: AI 엔지니어 지망생, 전환 개발자

---

## 목차

1. [학습 단계별 로드맵](#1-학습-단계별-로드맵)
2. [필수 지식 영역](#2-필수-지식-영역)
3. [실무 프로젝트 추천](#3-실무-프로젝트-추천)
4. [학습 자료](#4-학습-자료)
5. [인증과 포트폴리오](#5-인증과-포트폴리오)

---

## 1. 학습 단계별 로드맵

### 1.1 초급 (0-6개월)

**목표**: AI/ML 기초 이해

**학습 내용:**
1. **Python 기초**
   - 기본 문법
   - 데이터 구조
   - 객체지향 프로그래밍

2. **수학 기초**
   - 선형 대수
   - 확률과 통계
   - 미적분

3. **ML 기초**
   - 지도 학습
   - 비지도 학습
   - 평가 메트릭

**프로젝트:**
- Iris 분류
- 주가 예측
- 이미지 분류 (MNIST)

### 1.2 중급 (6-12개월)

**목표**: 딥러닝과 NLP 이해

**학습 내용:**
1. **딥러닝**
   - 신경망 기초
   - CNN, RNN
   - Transformer

2. **NLP**
   - 텍스트 전처리
   - 임베딩
   - 언어 모델

3. **프레임워크**
   - PyTorch / TensorFlow
   - Hugging Face
   - LangChain

**프로젝트:**
- 감정 분석
- 챗봇 구축
- 문서 요약

### 1.3 고급 (12-24개월)

**목표**: 프로덕션 시스템 구축

**학습 내용:**
1. **LLM**
   - GPT, BERT
   - Fine-tuning
   - Prompt Engineering

2. **RAG**
   - 벡터 검색
   - 문서 처리
   - 검색 최적화

3. **프로덕션**
   - MLOps
   - 모델 배포
   - 모니터링

**프로젝트:**
- RAG 시스템
- 멀티 에이전트
- 프로덕션 배포

---

## 2. 필수 지식 영역

### 2.1 수학

**선형 대수:**
- 벡터, 행렬
- 내적, 외적
- 고유값, 고유벡터

**확률과 통계:**
- 확률 분포
- 베이즈 정리
- 가설 검정

**미적분:**
- 미분
- 적분
- 최적화

### 2.2 프로그래밍

**Python:**
- 기본 문법
- NumPy, Pandas
- 비동기 프로그래밍

**시스템:**
- Linux
- Docker
- Kubernetes

### 2.3 ML/DL

**기계 학습:**
- 회귀, 분류
- 클러스터링
- 차원 축소

**딥러닝:**
- 신경망
- 역전파
- 정규화

**LLM:**
- Transformer
- Attention
- Fine-tuning

---

## 3. 실무 프로젝트 추천

### 3.1 초급 프로젝트

**1. 문서 검색 시스템**
```python
from llmkit import RAGChain

rag = RAGChain.from_documents("documents/")
answer = rag.query("질문")
```

**2. 감정 분석 API**
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
```

**3. 챗봇**
```python
from llmkit import Agent

agent = Agent(model="gpt-4o-mini")
response = agent.run("안녕하세요!")
```

### 3.2 중급 프로젝트

**1. RAG 시스템**
- 문서 처리
- 벡터 검색
- 답변 생성

**2. 멀티 에이전트 시스템**
- 에이전트 협업
- 작업 분배
- 결과 통합

**3. 문서 요약 시스템**
- 문서 분석
- 요약 생성
- 품질 평가

### 3.3 고급 프로젝트

**1. 프로덕션 RAG**
- 확장 가능한 아키텍처
- 모니터링
- A/B 테스트

**2. 도메인 특화 시스템**
- 의료, 법률 등
- Fine-tuning
- 평가

**3. 멀티모달 시스템**
- 텍스트 + 이미지
- Vision RAG
- 통합 검색

---

## 4. 학습 자료

### 4.1 온라인 강의

**무료:**
- Coursera: Machine Learning (Andrew Ng)
- Fast.ai: Practical Deep Learning
- Hugging Face: NLP Course

**유료:**
- DeepLearning.AI
- Udacity
- Pluralsight

### 4.2 도서

**기초:**
- "Hands-On Machine Learning"
- "Deep Learning" (Ian Goodfellow)

**고급:**
- "Natural Language Processing with Transformers"
- "Building LLM Applications"

### 4.3 실습 플랫폼

- Kaggle
- Google Colab
- Hugging Face Spaces

---

## 5. 인증과 포트폴리오

### 5.1 인증

**추천 인증:**
- AWS Certified Machine Learning
- Google Cloud Professional ML Engineer
- Microsoft Azure AI Engineer

### 5.2 포트폴리오

**필수 요소:**
1. GitHub 프로젝트
2. 블로그 포스트
3. 데모 애플리케이션

**프로젝트 예시:**
- RAG 시스템
- 챗봇
- 문서 분석 도구

---

## 학습 체크리스트

### 초급
- [ ] Python 기초 완료
- [ ] 수학 기초 복습
- [ ] 첫 ML 프로젝트 완료

### 중급
- [ ] 딥러닝 이해
- [ ] NLP 프로젝트 완료
- [ ] 프레임워크 숙련

### 고급
- [ ] LLM 이해
- [ ] RAG 시스템 구축
- [ ] 프로덕션 배포 경험

---

## 추천 학습 순서

1. **1-2주**: Python 기초
2. **3-4주**: 수학 기초
3. **5-8주**: ML 기초
4. **9-12주**: 딥러닝
5. **13-16주**: NLP
6. **17-20주**: LLM
7. **21-24주**: 프로덕션

---

**작성일**: 2025-01-XX  
**버전**: 1.0

