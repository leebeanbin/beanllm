# 데이터 흐름 및 병목 분석

**목적**: llmkit의 주요 데이터 흐름을 정리하고, 네트워크 I/O와 CPU 연산 지점을 식별하여 병목을 파악합니다.

---

## 목차

1. [RAG 파이프라인](#1-rag-파이프라인)
2. [Agent/Tool 실행](#2-agenttool-실행)
3. [State Graph 실행](#3-state-graph-실행)
4. [Multi-Agent 시스템](#4-multi-agent-시스템)
5. [Evaluation 시스템](#5-evaluation-시스템)
6. [병목 지점 요약](#6-병목-지점-요약)

---

## 1. RAG 파이프라인

### 1.1 데이터 흐름도

```
사용자 쿼리
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 쿼리 임베딩 생성                 │ ← 🔴 네트워크 I/O (API 호출)
│    - OpenAI/Anthropic Embedding API │    또는 CPU 연산 (로컬 모델)
│    - embed_sync([query])            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. 벡터 검색                        │ ← 🟡 CPU 연산 (유사도 계산)
│    - similarity_search(query, k)    │    또는 네트워크 I/O (Pinecone 등)
│    - 코사인 유사도 계산              │
│    - Top-k 선택                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 재순위화 (선택적)                │ ← 🟡 CPU 연산
│    - rerank(query, results)         │    또는 네트워크 I/O (Cross-encoder)
│    - Cross-encoder 점수 계산         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. 컨텍스트 생성                    │ ← 🟢 CPU 연산 (문자열 조작)
│    - _build_context(results)        │
│    - 검색 결과를 문자열로 결합      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. 프롬프트 생성                    │ ← 🟢 CPU 연산 (문자열 조작)
│    - _build_prompt(query, context)  │
│    - 템플릿 포맷팅                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 6. LLM 호출                         │ ← 🔴 네트워크 I/O (가장 큰 병목)
│    - ChatService.chat(request)      │    지연 시간: 1-10초
│    - OpenAI/Anthropic/Google API    │
└──────────────┬──────────────────────┘
               │
               ▼
응답 반환
```

### 1.2 네트워크 I/O 지점

| 단계 | 위치 | 설명 | 예상 지연 시간 |
|------|------|------|---------------|
| 쿼리 임베딩 | `domain/embeddings/providers.py:OpenAIEmbedding.embed_sync()` | OpenAI Embedding API 호출 | 100-500ms |
| 벡터 검색 | `infrastructure/vector_stores/pinecone.py` (Pinecone 사용 시) | Pinecone API 호출 | 50-200ms |
| LLM 호출 | `_source_providers/openai_provider.py:chat()` | OpenAI Chat API 호출 | **1-10초** ⚠️ |

**총 네트워크 지연 시간**: 약 1.2-10.7초

### 1.3 CPU 연산 지점

| 단계 | 위치 | 설명 | 시간 복잡도 | 최적화 여부 |
|------|------|------|-------------|------------|
| 벡터 유사도 계산 | `domain/embeddings/utils.py:cosine_similarity()` | 코사인 유사도 계산 | O(d) | ✅ NumPy 사용 |
| 배치 유사도 계산 | `domain/embeddings/utils.py:batch_cosine_similarity()` | 여러 벡터와의 유사도 | O(n·d) | ✅ NumPy 벡터화 |
| 벡터 검색 (FAISS) | `infrastructure/vector_stores/faiss.py` | HNSW 인덱스 검색 | O(log n·d) | ✅ FAISS 최적화 |
| 재순위화 | `domain/vector_stores/search.py:rerank()` | Cross-encoder 점수 계산 | O(k·d) | ⚠️ 단일 쿼리 |
| 컨텍스트 생성 | `service/impl/rag_service_impl.py:_build_context()` | 문자열 결합 | O(k·L) | ✅ 단순 연산 |
| 프롬프트 생성 | `service/impl/rag_service_impl.py:_build_prompt()` | 템플릿 포맷팅 | O(L) | ✅ 단순 연산 |

**주요 CPU 병목**: 벡터 검색 (대규모 데이터셋), 재순위화 (Cross-encoder)

### 1.4 메모리 사용

| 데이터 | 위치 | 크기 | 최적화 여부 |
|--------|------|------|------------|
| 임베딩 벡터 | `domain/embeddings/` | d × 4 bytes (float32) | ✅ float32 사용 |
| 벡터 스토어 | `infrastructure/vector_stores/` | n × d × 4 bytes | ⚠️ 전체 로드 시 |
| 검색 결과 | `domain/vector_stores/base.py:VectorSearchResult` | k × (문서 크기) | ✅ 제한적 |

---

## 2. Agent/Tool 실행

### 2.1 데이터 흐름도

```
사용자 태스크
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 프롬프트 생성                    │ ← 🟢 CPU 연산
│    - REACT_PROMPT.format()         │
│    - 도구 설명 포함                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. LLM 호출 (Thought)              │ ← 🔴 네트워크 I/O
│    - ChatService.chat(request)     │    지연 시간: 1-5초
│    - ReAct 패턴 응답 생성           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 응답 파싱                        │ ← 🟡 CPU 연산
│    - _parse_response(content)      │
│    - 정규표현식으로 Action 추출     │
│    - JSON 파싱                      │
└──────────────┬──────────────────────┘
               │
               ├─ Final Answer? ──┐
               │                  │
               ▼                  │
┌─────────────────────────────────────┐
│ 4. Tool 실행                        │ ← 🔴 네트워크 I/O (외부 API)
│    - _execute_tool(name, args)     │    또는 🟡 CPU 연산 (로컬 계산)
│    - ToolRegistry.execute()         │    지연 시간: 0.1-5초
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. 히스토리 업데이트                 │ ← 🟢 CPU 연산
│    - conversation_history += ...    │
│    - messages 배열 재구성           │
└──────────────┬──────────────────────┘
               │
               └─── 반복 (최대 max_steps) ───┘
```

### 2.2 네트워크 I/O 지점

| 단계 | 위치 | 설명 | 예상 지연 시간 |
|------|------|------|---------------|
| LLM 호출 (각 스텝) | `service/impl/agent_service_impl.py:run()` | OpenAI Chat API | **1-5초** ⚠️ |
| Tool 실행 (외부 API) | `domain/tools/advanced/api.py:ExternalAPITool.call()` | HTTP 요청 | 0.1-5초 |
| Tool 실행 (웹 검색) | `facade/web_search_facade.py` | 검색 엔진 API | 0.5-2초 |

**총 네트워크 지연 시간**: 
- 최소 (1스텝): 1-5초
- 평균 (3-5스텝): **3-25초** ⚠️
- 최대 (max_steps): 10-50초

### 2.3 CPU 연산 지점

| 단계 | 위치 | 설명 | 시간 복잡도 | 최적화 여부 |
|------|------|------|-------------|------------|
| 프롬프트 생성 | `service/impl/agent_service_impl.py:_format_tools()` | 도구 설명 문자열 생성 | O(t) | ✅ 단순 연산 |
| 응답 파싱 | `service/impl/agent_service_impl.py:_parse_response()` | 정규표현식 매칭 | O(L) | ⚠️ 정규표현식 |
| JSON 파싱 | `service/impl/agent_service_impl.py:_parse_response()` | Action Input 파싱 | O(L) | ✅ 내장 json |
| 히스토리 업데이트 | `service/impl/agent_service_impl.py:run()` | 문자열 결합 | O(L) | ✅ 단순 연산 |

**주요 CPU 병목**: 응답 파싱 (정규표현식), 히스토리 누적 (긴 대화)

### 2.4 메모리 사용

| 데이터 | 위치 | 크기 | 최적화 여부 |
|--------|------|------|------------|
| 대화 히스토리 | `service/impl/agent_service_impl.py:conversation_history` | 누적 증가 | ⚠️ 제한 없음 |
| 스텝 기록 | `service/impl/agent_service_impl.py:steps` | O(max_steps) | ✅ 제한적 |

---

## 3. State Graph 실행

### 3.1 데이터 흐름도

```
초기 상태
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 상태 복사                        │ ← 🟡 CPU 연산
│    - GraphState.copy()              │    최적화: 얕은 복사
│    - 또는 copy.deepcopy()           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. 노드 실행                        │ ← 🔴 네트워크 I/O (LLM 노드)
│    - node_func(state)               │    또는 🟡 CPU 연산 (일반 노드)
│    - LLMNode.execute()               │    지연 시간: 0.1-10초
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 상태 업데이트                    │ ← 🟢 CPU 연산
│    - state.update(update)           │
│    - GraphState 업데이트             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. 다음 노드 결정                   │ ← 🟢 CPU 연산
│    - _get_next_node()               │
│    - 조건부 엣지 평가                │
└──────────────┬──────────────────────┘
               │
               └─── 반복 (최대 max_iterations) ───┘
```

### 3.2 네트워크 I/O 지점

| 단계 | 위치 | 설명 | 예상 지연 시간 |
|------|------|------|---------------|
| LLM 노드 실행 | `domain/graph/nodes.py:LLMNode.execute()` | LLM API 호출 | **1-10초** ⚠️ |
| Agent 노드 실행 | `domain/graph/nodes.py:AgentNode.execute()` | Agent 실행 (여러 LLM 호출) | **3-50초** ⚠️ |

**총 네트워크 지연 시간**: 
- 단순 그래프 (3-5 노드): 3-50초
- 복잡한 그래프 (10+ 노드): **10-500초** ⚠️

### 3.3 CPU 연산 지점

| 단계 | 위치 | 설명 | 시간 복잡도 | 최적화 여부 |
|------|------|------|-------------|------------|
| 상태 복사 | `domain/graph/graph_state.py:copy()` | 얕은 복사 | O(n) | ✅ 최적화됨 |
| 상태 복사 (깊은) | `service/impl/state_graph_service_impl.py:invoke()` | deepcopy | O(n·m) | ⚠️ 최소화 |
| 조건 평가 | `domain/graph/nodes.py:ConditionalNode.execute()` | 조건 함수 실행 | O(1) | ✅ 단순 연산 |
| 다음 노드 결정 | `service/impl/state_graph_service_impl.py:_get_next_node()` | 엣지/조건부 엣지 확인 | O(1) | ✅ 단순 연산 |

**주요 CPU 병목**: 상태 복사 (깊은 복사), 체크포인트 저장 (디스크 I/O)

### 3.4 메모리 사용

| 데이터 | 위치 | 크기 | 최적화 여부 |
|--------|------|------|------------|
| 그래프 상태 | `domain/graph/graph_state.py:GraphState` | 상태 크기에 비례 | ⚠️ 제한 없음 |
| 실행 기록 | `domain/state_graph.py:GraphExecution` | O(노드 수) | ✅ 제한적 |
| 체크포인트 | `domain/state_graph.py:Checkpoint` | 디스크 저장 | ✅ 선택적 |

---

## 4. Multi-Agent 시스템

### 4.1 데이터 흐름도

```
초기 메시지
    │
    ▼
┌─────────────────────────────────────┐
│ 1. CommunicationBus 초기화         │ ← 🟢 CPU 연산
│    - 메시지 큐 생성                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. 에이전트별 병렬 실행             │ ← 🔴 네트워크 I/O (병렬)
│    - Agent 1: LLM 호출             │    지연 시간: max(각 에이전트)
│    - Agent 2: LLM 호출             │
│    - Agent 3: LLM 호출             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 메시지 전달                      │ ← 🟢 CPU 연산
│    - CommunicationBus.send()        │
│    - 메시지 큐에 추가               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. 에이전트 간 통신                 │ ← 🟢 CPU 연산
│    - 메시지 수신                    │
│    - 상태 업데이트                  │
└──────────────┬──────────────────────┘
               │
               └─── 반복 (최대 라운드) ───┘
```

### 4.2 네트워크 I/O 지점

| 단계 | 위치 | 설명 | 예상 지연 시간 |
|------|------|------|---------------|
| 각 에이전트 LLM 호출 | `service/impl/multi_agent_service_impl.py:run()` | 병렬 LLM 호출 | **1-10초** ⚠️ |
| Tool 실행 (에이전트별) | `domain/tools/` | 각 에이전트의 Tool 실행 | 0.1-5초 |

**총 네트워크 지연 시간**: 
- 병렬 실행: max(각 에이전트) = **1-10초** (병렬화 효과)
- 순차 실행: sum(각 에이전트) = **3-30초** (비효율적)

### 4.3 CPU 연산 지점

| 단계 | 위치 | 설명 | 시간 복잡도 | 최적화 여부 |
|------|------|------|-------------|------------|
| 메시지 전달 | `domain/multi_agent/communication.py:CommunicationBus.send()` | 큐에 추가 | O(1) | ✅ 효율적 |
| 메시지 수신 | `domain/multi_agent/communication.py:CommunicationBus.receive()` | 큐에서 제거 | O(1) | ✅ 효율적 |
| 상태 동기화 | `service/impl/multi_agent_service_impl.py` | 상태 업데이트 | O(n) | ✅ 단순 연산 |

**주요 CPU 병목**: 메시지 큐 관리 (대량 메시지)

### 4.4 메모리 사용

| 데이터 | 위치 | 크기 | 최적화 여부 |
|--------|------|------|------------|
| 메시지 큐 | `domain/multi_agent/communication.py:CommunicationBus` | O(메시지 수) | ⚠️ 제한 없음 |
| 에이전트 상태 | `service/impl/multi_agent_service_impl.py` | O(에이전트 수) | ✅ 제한적 |

---

## 5. Evaluation 시스템

### 5.1 데이터 흐름도

```
평가 데이터셋
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 데이터셋 로드                    │ ← 🟡 디스크 I/O
│    - EvaluationDataset.load()      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. 각 샘플 평가 (순차/병렬)         │ ← 🔴 네트워크 I/O
│    - LLM 호출 (예측 생성)           │    지연 시간: 1-10초/샘플
│    - 또는 RAG/Agent 실행            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 메트릭 계산                      │ ← 🟡 CPU 연산
│    - ExactMatchMetric               │
│    - F1ScoreMetric                  │
│    - BLEUMetric                     │
│    - SemanticSimilarityMetric      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. 결과 집계                        │ ← 🟢 CPU 연산
│    - 평균 점수 계산                  │
│    - 통계 분석                       │
└──────────────┬──────────────────────┘
               │
               ▼
평가 결과
```

### 5.2 네트워크 I/O 지점

| 단계 | 위치 | 설명 | 예상 지연 시간 |
|------|------|------|---------------|
| LLM 호출 (각 샘플) | `service/impl/evaluation_service_impl.py` | 예측 생성 | **1-10초/샘플** ⚠️ |
| RAG 실행 (각 샘플) | `service/impl/evaluation_service_impl.py` | RAG 파이프라인 | **2-15초/샘플** ⚠️ |
| Agent 실행 (각 샘플) | `service/impl/evaluation_service_impl.py` | Agent 실행 | **3-50초/샘플** ⚠️ |

**총 네트워크 지연 시간**: 
- 100개 샘플 (순차): **100-5000초** (16분-83분) ⚠️
- 100개 샘플 (병렬 10): **10-500초** (병렬화 효과)

### 5.3 CPU 연산 지점

| 단계 | 위치 | 설명 | 시간 복잡도 | 최적화 여부 |
|------|------|------|-------------|------------|
| Exact Match | `domain/evaluation/metrics.py:ExactMatchMetric` | 문자열 비교 | O(L) | ✅ 단순 연산 |
| F1 Score | `domain/evaluation/metrics.py:F1ScoreMetric` | 토큰화 + F1 계산 | O(L) | ✅ 효율적 |
| BLEU | `domain/evaluation/metrics.py:BLEUMetric` | n-gram 계산 | O(L²) | ⚠️ 중간 |
| Semantic Similarity | `domain/evaluation/metrics.py:SemanticSimilarityMetric` | 임베딩 + 유사도 | O(d) | ✅ NumPy 사용 |
| LLM Judge | `domain/evaluation/metrics.py:LLMJudgeMetric` | LLM 호출 | 네트워크 I/O | ⚠️ 추가 비용 |

**주요 CPU 병목**: BLEU 계산 (n-gram), 대규모 데이터셋 집계

### 5.4 메모리 사용

| 데이터 | 위치 | 크기 | 최적화 여부 |
|--------|------|------|------------|
| 평가 데이터셋 | `domain/evaluation/dataset.py` | O(샘플 수 × 샘플 크기) | ⚠️ 전체 로드 |
| 예측 결과 | `service/impl/evaluation_service_impl.py` | O(샘플 수 × 결과 크기) | ⚠️ 누적 저장 |
| 메트릭 결과 | `domain/evaluation/metrics.py` | O(샘플 수) | ✅ 제한적 |

---

## 6. 병목 지점 요약

### 6.1 네트워크 I/O 병목 (🔴)

| 우선순위 | 위치 | 설명 | 개선 방안 |
|---------|------|------|----------|
| **1** | LLM API 호출 (모든 기능) | 가장 큰 지연 시간 (1-10초) | - 배치 처리<br>- 스트리밍 활용<br>- 캐싱 |
| **2** | Agent 실행 (반복 LLM 호출) | 여러 번의 LLM 호출 (3-50초) | - 스텝 수 최소화<br>- 프롬프트 최적화 |
| **3** | Evaluation (대량 샘플) | 순차 실행 시 매우 느림 (100-5000초) | - **병렬 처리 필수**<br>- 배치 평가 |
| **4** | 임베딩 API 호출 | RAG에서 매번 호출 (100-500ms) | - 캐싱<br>- 로컬 모델 사용 |
| **5** | Tool 실행 (외부 API) | Agent에서 사용 (0.1-5초) | - 타임아웃 설정<br>- 재시도 로직 |

### 6.2 CPU 연산 병목 (🟡)

| 우선순위 | 위치 | 설명 | 개선 방안 |
|---------|------|------|----------|
| **1** | 벡터 검색 (대규모) | O(n·d) 또는 O(log n·d) | - ANN 인덱스 (HNSW)<br>- 배치 검색 |
| **2** | 재순위화 (Cross-encoder) | O(k·d) | - 배치 처리<br>- GPU 활용 |
| **3** | 상태 복사 (깊은 복사) | O(n·m) | - 얕은 복사 우선<br>- GraphState.copy() 사용 |
| **4** | BLEU 계산 | O(L²) | - 최적화된 라이브러리<br>- 배치 처리 |
| **5** | 응답 파싱 (정규표현식) | O(L) | - 구조화된 출력 활용<br>- JSON Schema |

### 6.3 메모리 병목 (🟠)

| 우선순위 | 위치 | 설명 | 개선 방안 |
|---------|------|------|----------|
| **1** | 벡터 스토어 (전체 로드) | n × d × 4 bytes | - 지연 로딩<br>- 청크 단위 처리 |
| **2** | 대화 히스토리 (누적) | 무제한 증가 | - 최대 길이 제한<br>- 요약/압축 |
| **3** | 평가 데이터셋 (전체 로드) | O(샘플 수 × 크기) | - 스트리밍 로드<br>- 배치 처리 |

### 6.4 개선 우선순위

#### 즉시 개선 가능 (High Impact, Low Effort)

1. **Evaluation 병렬 처리**
   - 현재: 순차 실행
   - 개선: `asyncio.gather()`로 병렬 실행
   - 예상 효과: **10-100배 속도 향상**

2. **임베딩 캐싱**
   - 현재: 매번 API 호출
   - 개선: `EmbeddingCache` 활용
   - 예상 효과: **반복 쿼리 100% 속도 향상**

3. **상태 복사 최적화**
   - 현재: `copy.deepcopy()` 과다 사용
   - 개선: `GraphState.copy()` 사용 (완료)
   - 예상 효과: **10-50% 속도 향상**

#### 중기 개선 (High Impact, Medium Effort)

4. **벡터 검색 배치 처리**
   - 현재: 단일 쿼리만 처리
   - 개선: 배치 검색 API 추가
   - 예상 효과: **5-10배 속도 향상**

5. **Agent 스텝 수 최소화**
   - 현재: 최대 스텝까지 반복
   - 개선: 조기 종료, 프롬프트 최적화
   - 예상 효과: **30-50% 속도 향상**

6. **대화 히스토리 관리**
   - 현재: 무제한 누적
   - 개선: 최대 길이 제한, 요약
   - 예상 효과: **메모리 사용량 감소**

#### 장기 개선 (High Impact, High Effort)

7. **스트리밍 최적화**
   - 현재: 일부만 지원
   - 개선: 전체 파이프라인 스트리밍
   - 예상 효과: **사용자 경험 향상**

8. **GPU 활용**
   - 현재: CPU만 사용
   - 개선: 로컬 모델 GPU 가속
   - 예상 효과: **10-100배 속도 향상** (로컬 모델)

---

## 7. 측정 및 모니터링

### 7.1 성능 측정 포인트

```python
# 예시: RAG 파이프라인 성능 측정
import time
from llmkit import RAGChain

rag = RAGChain.from_documents("doc.pdf")

# 각 단계별 시간 측정
start = time.time()
results = rag.retrieve("query", k=4)  # 검색 시간
search_time = time.time() - start

start = time.time()
answer = rag.query("query")  # 전체 시간
total_time = time.time() - start

llm_time = total_time - search_time  # LLM 호출 시간
```

### 7.2 병목 식별 방법

1. **프로파일링**
   ```bash
   python -m cProfile -o profile.stats your_script.py
   python -m pstats profile.stats
   ```

2. **타이밍 측정**
   - 각 단계별 `time.time()` 측정
   - 네트워크 I/O vs CPU 연산 구분

3. **메모리 프로파일링**
   ```bash
   python -m memory_profiler your_script.py
   ```

4. **비동기 작업 모니터링**
   - `asyncio` 작업 추적
   - 병렬 실행 효율성 확인

---

## 8. 결론

### 주요 병목 지점

1. **네트워크 I/O**: LLM API 호출이 가장 큰 병목 (1-10초)
2. **순차 실행**: Evaluation, Agent 반복에서 비효율적
3. **메모리 사용**: 벡터 스토어, 대화 히스토리 무제한 증가

### 개선 효과 예상

- **Evaluation 병렬 처리**: 10-100배 속도 향상
- **임베딩 캐싱**: 반복 쿼리 100% 속도 향상
- **상태 복사 최적화**: 10-50% 속도 향상 (완료)
- **벡터 검색 배치**: 5-10배 속도 향상

### 다음 단계

1. Evaluation 병렬 처리 구현
2. 임베딩 캐싱 강화
3. 벡터 검색 배치 처리 추가
4. 성능 벤치마크 수립
