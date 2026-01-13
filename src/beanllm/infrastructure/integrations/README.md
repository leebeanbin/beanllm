# Integrations - 외부 프레임워크 통합

## 개요

이 모듈은 beanLLM과 외부 LLM 프레임워크 간의 통합을 제공합니다. Infrastructure 레이어의 일부로, 외부 시스템과의 상호 운용성을 담당합니다.

## 지원하는 통합

### 1. LangGraph 통합

**목적**: 복잡한 에이전트 워크플로우를 위한 그래프 기반 프레임워크 통합

**주요 기능**:
- beanLLM State Graph → LangGraph StateGraph 변환
- beanLLM Agent → LangGraph Agent 통합
- Workflow Builder (beanLLM 스타일)

**설치**:
```bash
pip install langgraph
```

**사용 예제**:
```python
from beanllm.infrastructure.integrations import LangGraphBridge, create_workflow
from beanllm.domain.state_graph import GraphState

# State 정의
class AgentState(GraphState):
    query: str
    documents: list
    answer: str

# Workflow 생성
workflow = create_workflow(AgentState)
result = workflow.run({"query": "What is AI?"})
print(result["answer"])
```

**참고 자료**:
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)

### 2. LlamaIndex 통합

**목적**: LLM 애플리케이션을 위한 데이터 프레임워크 통합

**주요 기능**:
- beanLLM Document → LlamaIndex Document 변환
- beanLLM Embeddings → LlamaIndex Embeddings 래핑
- Query Engine을 beanLLM 스타일로 제공

**설치**:
```bash
pip install llama-index
```

**사용 예제**:
```python
from beanllm.infrastructure.integrations import (
    LlamaIndexBridge,
    create_llamaindex_query_engine
)
from beanllm.domain.loaders import TextLoader
from beanllm.domain.embeddings import OpenAIEmbedding

# 문서 로드
loader = TextLoader("document.txt")
docs = loader.load()

# 임베딩 모델
embedding = OpenAIEmbedding()

# Query Engine 생성
query_engine = create_llamaindex_query_engine(
    documents=docs,
    embedding_function=embedding.embed,
    similarity_top_k=5
)

# 쿼리 실행
response = query_engine.query("What is this document about?")
print(response)
```

**참고 자료**:
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [LlamaIndex 문서](https://docs.llamaindex.ai/)

## 아키텍처

### 브릿지 패턴

이 모듈은 브릿지 패턴을 사용하여 beanLLM과 외부 프레임워크 간의 변환을 처리합니다:

```
beanLLM 타입 → Bridge → 외부 프레임워크 타입
```

### 선택적 의존성

모든 통합은 선택적 의존성으로 설계되어 있습니다:
- 의존성이 없어도 beanLLM의 다른 기능은 정상 작동
- 필요한 경우에만 해당 프레임워크를 설치
- try-except로 안전하게 처리

## 모듈 구조

```
integrations/
├── __init__.py          # 통합 모듈 export
├── README.md            # 이 문서
├── langgraph/
│   ├── __init__.py
│   ├── bridge.py       # beanLLM ↔ LangGraph 변환
│   └── workflow.py     # LangGraph 워크플로우 빌더
└── llamaindex/
    ├── __init__.py
    ├── bridge.py       # beanLLM ↔ LlamaIndex 변환
    └── query_engine.py # LlamaIndex Query Engine 래퍼
```

## 사용 시나리오

### 시나리오 1: LangGraph로 복잡한 워크플로우 구현

기존 beanLLM State Graph를 LangGraph의 고급 기능(조건부 분기, Human-in-the-loop 등)과 함께 사용하고 싶을 때:

```python
from beanllm.infrastructure.integrations import LangGraphBridge
from beanllm.domain.state_graph import GraphState

class MyState(GraphState):
    query: str
    result: str

bridge = LangGraphBridge()
langgraph_state = bridge.create_state_schema(MyState)
# LangGraph의 고급 기능 사용
```

### 시나리오 2: LlamaIndex의 고급 RAG 기능 활용

beanLLM의 문서와 임베딩을 LlamaIndex의 고급 RAG 기능(Multi-step retrieval, Query transformation 등)과 함께 사용하고 싶을 때:

```python
from beanllm.infrastructure.integrations import create_llamaindex_query_engine
from beanllm.domain.loaders import DocumentLoader
from beanllm.domain.embeddings import Embedding

docs = DocumentLoader.load("documents/")
embedding = Embedding(model="text-embedding-3-small")

query_engine = create_llamaindex_query_engine(
    documents=docs,
    embedding_function=embedding.embed
)
# LlamaIndex의 고급 RAG 기능 사용
```

## 주의사항

1. **의존성 관리**: 필요한 통합만 설치하여 불필요한 의존성을 피하세요.
2. **타입 변환**: 브릿지를 통해 변환된 타입은 원본과 동일하지 않을 수 있습니다.
3. **성능**: 브릿지 변환 과정에서 약간의 오버헤드가 발생할 수 있습니다.

## 확장

새로운 통합을 추가하려면:

1. `integrations/` 폴더에 새 서브 폴더 생성
2. `bridge.py` 파일에 브릿지 클래스 구현
3. `__init__.py`에서 export
4. 메인 `__init__.py`에 추가

## 문제 해결

### ImportError 발생 시

의존성이 설치되지 않았을 수 있습니다:
```bash
# LangGraph
pip install langgraph

# LlamaIndex
pip install llama-index
```

### 변환 오류 발생 시

타입이 호환되지 않을 수 있습니다. 브릿지 클래스의 문서를 확인하세요.

