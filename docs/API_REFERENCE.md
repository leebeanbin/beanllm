# 📚 beanllm API Reference

Complete API reference for all beanllm components.

## Table of Contents

### Core Components
- [ClientFacade](#clientfacade) - Basic LLM client
- [RAGFacade](#ragfacade) - RAG (Retrieval-Augmented Generation) system
- [AgentFacade](#agentfacade) - AI agent with tools
- [ChainFacade](#chainfacade) - Chain execution

### Advanced Features
- [MultiAgentFacade](#multiagentfacade) - Multi-agent collaboration
- [GraphFacade](#graphfacade) - Graph-based workflows
- [StateGraphFacade](#stategraphfacade) - State-based graph execution
- [AudioFacade](#audiofacade) - Audio processing (speech-to-text, text-to-speech)

### Specialized Features
- [VisionRAGFacade](#visionragfacade) - Vision + RAG with image understanding
- [WebSearchFacade](#websearchfacade) - Web search integration
- [EvaluationFacade](#evaluationfacade) - LLM evaluation and metrics
- [FinetuningFacade](#finetuningfacade) - Model fine-tuning

---

## Installation

```bash
# Basic installation
pip install beanllm

# With all providers
pip install beanllm[all]

# Specific providers
pip install beanllm[openai,anthropic]
```

---

## Quick Start

```python
from beanllm import ClientFacade

# Initialize client
client = ClientFacade(model="gpt-4")

# Simple chat
response = client.chat("Hello, how are you?")
print(response.content)
```

---

# API Documentation

## Core Components

### ClientFacade

기본 LLM 클라이언트. 가장 간단한 채팅 인터페이스를 제공합니다.

#### `__init__(model, provider=None, api_key=None, **kwargs)`

**파라미터:**
- `model` (str): 모델 이름 (예: "gpt-4", "claude-3-opus", "gemini-pro")
- `provider` (str, optional): Provider 이름. 생략 시 모델명에서 자동 감지
- `api_key` (str, optional): API 키. 생략 시 환경변수에서 로드
- `**kwargs`: Provider별 추가 설정

**예제:**
```python
from beanllm import ClientFacade

# OpenAI
client = ClientFacade(model="gpt-4")

# Anthropic (provider 자동 감지)
client = ClientFacade(model="claude-3-opus-20240229")

# 명시적 provider 지정
client = ClientFacade(model="gpt-4", provider="openai")
```

#### `chat(messages, system=None, temperature=None, max_tokens=None, **kwargs)` (async)

채팅 완료를 수행합니다.

**파라미터:**
- `messages` (List[Dict[str, str]]): 메시지 리스트 `[{"role": "user", "content": "..."}]`
- `system` (str, optional): 시스템 프롬프트
- `temperature` (float, optional): 샘플링 온도 (0.0-2.0)
- `max_tokens` (int, optional): 최대 생성 토큰 수
- `**kwargs`: 추가 파라미터

**반환:** `ChatResponse`

**예제:**
```python
response = await client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1000
)
print(response.content)
```

#### `stream(messages, **kwargs)` (async)

스트리밍 방식으로 채팅 완료를 수행합니다.

**파라미터:** `chat()`와 동일

**반환:** `AsyncIterator[str]`

**예제:**
```python
async for chunk in client.stream(messages=[{"role": "user", "content": "Tell me a story"}]):
    print(chunk, end="", flush=True)
```

---

### RAGFacade

RAG (Retrieval-Augmented Generation) 시스템. 문서 기반 질의응답을 제공합니다.

#### `__init__(model, vector_store=None, embedding_model=None, **kwargs)`

**파라미터:**
- `model` (str): LLM 모델 이름
- `vector_store` (str, optional): 벡터 저장소 ("chroma", "faiss", "pinecone" 등)
- `embedding_model` (str, optional): 임베딩 모델 이름
- `**kwargs`: 추가 설정

**예제:**
```python
from beanllm import RAGFacade

rag = RAGFacade(
    model="gpt-4",
    vector_store="chroma",
    embedding_model="text-embedding-3-small"
)
```

#### `add_documents(documents)` (async)

문서를 벡터 저장소에 추가합니다.

**파라미터:**
- `documents` (List[str] | List[Document]): 문서 리스트

**예제:**
```python
documents = [
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
]
await rag.add_documents(documents)
```

#### `query(question, top_k=3, **kwargs)` (async)

질문에 대한 답변을 생성합니다.

**파라미터:**
- `question` (str): 질문
- `top_k` (int): 검색할 문서 수
- `**kwargs`: 추가 파라미터

**반환:** `RAGResponse`

**예제:**
```python
response = await rag.query(
    question="What is Python?",
    top_k=3
)
print(response.answer)
print(response.sources)  # 사용된 문서들
```

---

### AgentFacade

도구를 사용할 수 있는 AI 에이전트.

#### `__init__(model, tools=None, **kwargs)`

**파라미터:**
- `model` (str): LLM 모델 이름
- `tools` (List[Tool], optional): 사용할 도구 리스트
- `**kwargs`: 추가 설정

**예제:**
```python
from beanllm import AgentFacade
from beanllm.domain.tools import Calculator, WebSearch

agent = AgentFacade(
    model="gpt-4",
    tools=[Calculator(), WebSearch()]
)
```

#### `run(task, max_iterations=10, **kwargs)` (async)

에이전트를 실행하여 작업을 수행합니다.

**파라미터:**
- `task` (str): 수행할 작업
- `max_iterations` (int): 최대 반복 횟수
- `**kwargs`: 추가 파라미터

**반환:** `AgentResponse`

**예제:**
```python
response = await agent.run(
    task="Calculate 123 * 456 and search for the result online",
    max_iterations=5
)
print(response.final_answer)
print(response.steps)  # 실행 단계
```

---

### ChainFacade

여러 단계를 순차적으로 실행하는 체인.

#### `__init__(steps=None, **kwargs)`

**파라미터:**
- `steps` (List[Callable], optional): 실행할 단계들
- `**kwargs`: 추가 설정

#### `add_step(step, name=None)`

체인에 단계를 추가합니다.

**파라미터:**
- `step` (Callable): 실행할 함수
- `name` (str, optional): 단계 이름

#### `run(input_data, **kwargs)` (async)

체인을 실행합니다.

**파라미터:**
- `input_data` (Any): 입력 데이터
- `**kwargs`: 추가 파라미터

**반환:** `ChainResponse`

**예제:**
```python
from beanllm import ChainFacade

chain = ChainFacade()
chain.add_step(lambda x: x.upper(), name="uppercase")
chain.add_step(lambda x: x + "!", name="add_exclamation")

response = await chain.run("hello")
print(response.result)  # "HELLO!"
```

---

## Advanced Features

### MultiAgentFacade

여러 에이전트가 협업하는 시스템.

#### `__init__(agents=None, strategy="sequential", **kwargs)`

**파라미터:**
- `agents` (List[Agent], optional): 에이전트 리스트
- `strategy` (str): 협업 전략 ("sequential", "parallel", "debate")
- `**kwargs`: 추가 설정

#### `run(task, **kwargs)` (async)

멀티 에이전트 시스템을 실행합니다.

**반환:** `MultiAgentResponse`

**예제:**
```python
from beanllm import MultiAgentFacade, AgentFacade

researcher = AgentFacade(model="gpt-4", name="Researcher")
writer = AgentFacade(model="gpt-4", name="Writer")

multi_agent = MultiAgentFacade(
    agents=[researcher, writer],
    strategy="sequential"
)

response = await multi_agent.run("Research AI trends and write a summary")
```

---

### GraphFacade

그래프 기반 워크플로우.

#### `add_node(node, name)`

그래프에 노드를 추가합니다.

#### `add_edge(from_node, to_node, condition=None)`

노드 간 엣지를 추가합니다.

#### `run(initial_state, **kwargs)` (async)

그래프를 실행합니다.

---

### StateGraphFacade

상태 기반 그래프 실행 시스템.

#### `__init__(state_schema=None, **kwargs)`

**파라미터:**
- `state_schema` (Dict, optional): 상태 스키마 정의

#### `add_node(name, function)`

상태 그래프에 노드를 추가합니다.

#### `set_entry_point(node_name)`

진입점을 설정합니다.

#### `add_conditional_edges(source, condition_fn, mapping)`

조건부 엣지를 추가합니다.

#### `run(initial_state, **kwargs)` (async)

상태 그래프를 실행합니다.

**예제:**
```python
from beanllm import StateGraphFacade

graph = StateGraphFacade(state_schema={"count": 0, "message": ""})

def increment(state):
    state["count"] += 1
    return state

graph.add_node("increment", increment)
graph.set_entry_point("increment")

result = await graph.run({"count": 0, "message": "start"})
```

---

### AudioFacade

음성 처리 (STT, TTS).

#### `transcribe(audio_file, **kwargs)` (async)

음성을 텍스트로 변환합니다 (Speech-to-Text).

**파라미터:**
- `audio_file` (str | bytes): 오디오 파일 경로 또는 바이트
- `**kwargs`: 추가 파라미터

**반환:** `AudioResponse`

**예제:**
```python
from beanllm import AudioFacade

audio = AudioFacade(model="whisper-1")
response = await audio.transcribe("speech.mp3")
print(response.text)
```

#### `synthesize(text, voice="alloy", **kwargs)` (async)

텍스트를 음성으로 변환합니다 (Text-to-Speech).

**파라미터:**
- `text` (str): 변환할 텍스트
- `voice` (str): 음성 종류
- `**kwargs`: 추가 파라미터

**반환:** 오디오 바이트

---

## Specialized Features

### VisionRAGFacade

이미지 + 텍스트 기반 RAG.

#### `add_images(image_paths)` (async)

이미지를 벡터 저장소에 추가합니다.

#### `query(question, image_context=True, **kwargs)` (async)

이미지 컨텍스트를 포함하여 질의합니다.

---

### WebSearchFacade

웹 검색 통합.

#### `search(query, num_results=5, **kwargs)` (async)

웹 검색을 수행합니다.

**파라미터:**
- `query` (str): 검색 쿼리
- `num_results` (int): 결과 수

**예제:**
```python
from beanllm import WebSearchFacade

search = WebSearchFacade(engine="google")
results = await search.search("latest AI news", num_results=5)
```

---

### EvaluationFacade

LLM 평가 및 메트릭.

#### `evaluate(predictions, references, metrics=None, **kwargs)` (async)

모델 출력을 평가합니다.

**파라미터:**
- `predictions` (List[str]): 예측 결과
- `references` (List[str]): 정답 참조
- `metrics` (List[str], optional): 사용할 메트릭 ("bleu", "rouge", etc.)

**반환:** `EvaluationResponse`

**예제:**
```python
from beanllm import EvaluationFacade

evaluator = EvaluationFacade()
results = await evaluator.evaluate(
    predictions=["The cat sat on the mat"],
    references=["A cat was sitting on the mat"],
    metrics=["bleu", "rouge"]
)
print(results.scores)
```

---

### FinetuningFacade

모델 파인튜닝.

#### `create_job(training_data, model, **kwargs)` (async)

파인튜닝 작업을 생성합니다.

**파라미터:**
- `training_data` (str | List): 훈련 데이터 파일 경로 또는 데이터
- `model` (str): 기본 모델
- `**kwargs`: 추가 파라미터

#### `check_status(job_id)` (async)

파인튜닝 작업 상태를 확인합니다.

**예제:**
```python
from beanllm import FinetuningFacade

finetuner = FinetuningFacade(provider="openai")
job = await finetuner.create_job(
    training_data="training.jsonl",
    model="gpt-3.5-turbo"
)
status = await finetuner.check_status(job.id)
```

---

## Common Types

### Response Objects

All facade methods return specific response objects:

- `ChatResponse` - Chat completion response
- `RAGResponse` - RAG query response
- `AgentResponse` - Agent execution response
- `AudioResponse` - Audio processing response
- `EvaluationResponse` - Evaluation results
- etc.

### Common Parameters

Most facades support these common parameters:

- `model` (str): Model name (e.g., "gpt-4", "claude-3-opus")
- `temperature` (float): Sampling temperature (0.0 - 2.0)
- `max_tokens` (int): Maximum tokens to generate
- `stream` (bool): Enable streaming responses

---

## Error Handling

```python
from beanllm import ClientFacade
from beanllm.utils.exceptions import BeanLLMError

try:
    client = ClientFacade(model="gpt-4")
    response = client.chat("Hello")
except BeanLLMError as e:
    print(f"Error: {e}")
```

---

## Environment Variables

beanllm uses environment variables for API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google
export GOOGLE_API_KEY="your-key"

# Or use .env file
```

---

## Additional Resources

- [GitHub Repository](https://github.com/leebeanbin/beanllm)
- [PyPI Package](https://pypi.org/project/beanllm/)
- [Examples](../examples/)
- [Architecture Guide](../ARCHITECTURE.md)

---

**Last Updated:** 2025-12-28
**Version:** 0.1.1
