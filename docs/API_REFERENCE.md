# 📚 beanllm API Reference

Complete API reference for all beanllm components.

## Table of Contents

### Core Components
- [Client](#client) - Basic LLM client
- [RAGChain](#ragchain) - RAG (Retrieval-Augmented Generation) system
- [Agent](#agent) - AI agent with tools
- [Chain](#chain) - Chain execution

### Advanced Features
- [MultiAgentCoordinator](#multiagentcoordinator) - Multi-agent collaboration
- [Graph](#graph) - Graph-based workflows
- [StateGraph](#stategraph) - State-based graph execution
- [Audio](#audio) - Audio processing (speech-to-text, text-to-speech)

### Specialized Features
- [VisionRAG](#visionrag) - Vision + RAG with image understanding
- [WebSearch](#websearch) - Web search integration
- [Evaluator](#evaluator) - LLM evaluation and metrics
- [FineTuningManager](#finetuningmanager) - Model fine-tuning

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
import asyncio
from beanllm import Client

async def main():
    # Initialize client
    client = Client(model="gpt-4")

    # Simple chat
    response = await client.chat(
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print(response.content)

asyncio.run(main())
```

---

# API Documentation

## Core Components

### Client

기본 LLM 클라이언트. 가장 간단한 채팅 인터페이스를 제공합니다.

#### `__init__(model, provider=None, api_key=None, **kwargs)`

**파라미터:**
- `model` (str): 모델 이름 (예: "gpt-4", "claude-3-opus", "gemini-pro")
- `provider` (str, optional): Provider 이름. 생략 시 모델명에서 자동 감지
- `api_key` (str, optional): API 키. 생략 시 환경변수에서 로드
- `**kwargs`: Provider별 추가 설정

**예제:**
```python
from beanllm import Client

# OpenAI
client = Client(model="gpt-4")

# Anthropic (provider 자동 감지)
client = Client(model="claude-3-opus-20240229")

# 명시적 provider 지정
client = Client(model="gpt-4", provider="openai")
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

#### `stream_chat(messages, **kwargs)` (async)

스트리밍 방식으로 채팅 완료를 수행합니다.

**파라미터:** `chat()`와 동일

**반환:** `AsyncIterator[str]`

**예제:**
```python
async for chunk in client.stream_chat(messages=[{"role": "user", "content": "Tell me a story"}]):
    print(chunk, end="", flush=True)
```

---

### RAGChain

RAG (Retrieval-Augmented Generation) 시스템. 문서 기반 질의응답을 제공합니다.

#### `from_documents(source, chunk_size=500, chunk_overlap=50, embedding_model="text-embedding-3-small", llm_model="gpt-4o-mini", **kwargs)`

팩토리 메서드로 RAG 시스템을 생성합니다.

**파라미터:**
- `source` (str | List): 문서 경로 또는 문서 리스트
- `chunk_size` (int): 청크 크기
- `chunk_overlap` (int): 청크 겹침
- `embedding_model` (str): 임베딩 모델 이름
- `llm_model` (str): LLM 모델 이름
- `**kwargs`: 추가 설정

**예제:**
```python
from beanllm import RAGChain

rag = RAGChain.from_documents(
    source="documents.txt",
    chunk_size=500,
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4"
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

### Agent

도구를 사용할 수 있는 AI 에이전트.

#### `__init__(model, tools=None, max_iterations=10, **kwargs)`

**파라미터:**
- `model` (str): LLM 모델 이름
- `tools` (List[Tool], optional): 사용할 도구 리스트
- `max_iterations` (int): 최대 반복 횟수
- `**kwargs`: 추가 설정

**예제:**
```python
from beanllm import Agent
from beanllm import search_web, calculator

agent = Agent(
    model="gpt-4",
    tools=[search_web, calculator]
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

### Chain

여러 단계를 순차적으로 실행하는 체인.

#### `__init__(client, memory=None, verbose=False)`

**파라미터:**
- `client` (Client): LLM 클라이언트
- `memory` (Memory, optional): 메모리 객체
- `verbose` (bool): 디버그 출력 여부

#### `run(user_input, **kwargs)` (async)

체인을 실행합니다.

**파라미터:**
- `user_input` (str): 사용자 입력
- `**kwargs`: 추가 파라미터

**반환:** `ChainResult`

**예제:**
```python
from beanllm import Chain, Client

client = Client(model="gpt-4")
chain = Chain(client=client)

response = await chain.run("Translate 'hello' to French")
print(response.output)
```

---

## Advanced Features

### MultiAgentCoordinator

여러 에이전트가 협업하는 시스템.

#### `__init__(agents, communication_bus=None)`

**파라미터:**
- `agents` (Dict[str, Agent]): 에이전트 딕셔너리 (id: agent)
- `communication_bus` (CommunicationBus, optional): 통신 버스

#### `execute_sequential(task, agent_order, **kwargs)` (async)

순차적으로 에이전트를 실행합니다.

**파라미터:**
- `task` (str): 작업
- `agent_order` (List[str]): 에이전트 실행 순서

#### `execute_debate(task, agent_ids=None, rounds=3, **kwargs)` (async)

토론 방식으로 에이전트를 실행합니다.

**예제:**
```python
from beanllm import MultiAgentCoordinator, Agent

researcher = Agent(model="gpt-4")
writer = Agent(model="gpt-4")

coordinator = MultiAgentCoordinator(
    agents={"researcher": researcher, "writer": writer}
)

result = await coordinator.execute_sequential(
    task="Research AI trends and write a summary",
    agent_order=["researcher", "writer"]
)
```

---

### Graph

그래프 기반 워크플로우.

#### `__init__(enable_cache=True)`

**파라미터:**
- `enable_cache` (bool): 캐싱 활성화 여부

#### `add_node(node)`

그래프에 노드를 추가합니다.

#### `add_edge(from_node, to_node)`

노드 간 엣지를 추가합니다.

#### `run(initial_state, verbose=False)` (async)

그래프를 실행합니다.

---

### StateGraph

상태 기반 그래프 실행 시스템.

#### `__init__(state_schema=None, config=None)`

**파라미터:**
- `state_schema` (Dict, optional): 상태 스키마 정의
- `config` (GraphConfig, optional): 그래프 설정

#### `add_node(name, func)`

상태 그래프에 노드를 추가합니다.

#### `set_entry_point(node_name)`

진입점을 설정합니다.

#### `add_conditional_edge(from_node, condition_func, edge_mapping=None)`

조건부 엣지를 추가합니다.

#### `invoke(initial_state, execution_id=None)` (async)

상태 그래프를 실행합니다.

**예제:**
```python
from beanllm import StateGraph

graph = StateGraph(state_schema={"count": 0, "message": ""})

def increment(state):
    state["count"] += 1
    return state

graph.add_node("increment", increment)
graph.set_entry_point("increment")

result = await graph.invoke({"count": 0, "message": "start"})
```

---

### Audio

음성 처리 (STT, TTS).

#### WhisperSTT - Speech-to-Text

```python
from beanllm import WhisperSTT

stt = WhisperSTT(model="base")
text = stt.transcribe("speech.mp3")
print(text)
```

#### TextToSpeech - Text-to-Speech

```python
from beanllm import TextToSpeech

tts = TextToSpeech(provider="openai", voice="alloy")
audio_bytes = tts.synthesize("Hello, world!")
```

#### AudioRAG - 오디오 검색 및 QA

```python
from beanllm import AudioRAG

audio_rag = AudioRAG()
audio_rag.add_audio("interview.mp3", audio_id="interview_1")
results = audio_rag.search("What did they say about AI?", top_k=3)
```

---

## Specialized Features

### VisionRAG

이미지 + 텍스트 기반 RAG.

#### `from_images(source, generate_captions=True, llm_model="gpt-4o", **kwargs)`

이미지로부터 VisionRAG를 생성합니다.

**파라미터:**
- `source` (str | List): 이미지 경로 또는 리스트
- `generate_captions` (bool): 자동 캡션 생성 여부
- `llm_model` (str): LLM 모델

**예제:**
```python
from beanllm import VisionRAG

vision_rag = VisionRAG.from_images(
    source="images/",
    generate_captions=True,
    llm_model="gpt-4o"
)

# 이미지 검색 및 질의
response = vision_rag.query(
    question="What objects are in the images?",
    k=3,
    include_images=True
)
```

---

### WebSearch

웹 검색 통합.

#### `search(query, engine=None, **kwargs)`

웹 검색을 수행합니다.

**파라미터:**
- `query` (str): 검색 쿼리
- `engine` (str, optional): 검색 엔진 ("google", "bing", "duckduckgo")

**예제:**
```python
from beanllm import WebSearch

search = WebSearch(default_engine="duckduckgo")
results = search.search("latest AI news")

for result in results:
    print(result.title, result.url)
```

---

### Evaluator

LLM 평가 및 메트릭.

#### `evaluate(prediction, reference, **kwargs)`

모델 출력을 평가합니다.

**파라미터:**
- `prediction` (str): 예측 결과
- `reference` (str): 정답 참조

**반환:** `EvaluationResult`

**예제:**
```python
from beanllm import Evaluator

evaluator = Evaluator(metrics=["bleu", "rouge", "f1"])
result = evaluator.evaluate(
    prediction="The cat sat on the mat",
    reference="A cat was sitting on the mat"
)
print(result.scores)
```

---

### FineTuningManager

모델 파인튜닝.

#### `prepare_and_upload(examples, output_path, validate=True)`

훈련 데이터를 준비하고 업로드합니다.

#### `start_training(model, training_file, validation_file=None, **kwargs)`

파인튜닝 작업을 시작합니다.

**예제:**
```python
from beanllm import FineTuningManager

manager = FineTuningManager(provider="openai")

# 데이터 준비
file_id = manager.prepare_and_upload(
    examples=[...],
    output_path="training.jsonl"
)

# 훈련 시작
job = manager.start_training(
    model="gpt-3.5-turbo",
    training_file=file_id
)

# 진행 상황 확인
progress = manager.get_training_progress(job.id)
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
import asyncio
from beanllm import Client
from beanllm.utils.exceptions import LLMKitError

async def main():
    try:
        client = Client(model="gpt-4")
        response = await client.chat(
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.content)
    except LLMKitError as e:
        print(f"Error: {e}")

asyncio.run(main())
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
