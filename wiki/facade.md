# Facade API

beanllm의 공개 API는 4개의 핵심 Facade 클래스로 구성됩니다.

모든 Facade는 `FacadeBase`를 상속하며 DI 컨테이너를 통해 Handler와 Service를 사용합니다.

---

## Client — 채팅·스트리밍·임베딩

가장 기본적인 LLM 인터페이스입니다.

### 기본 사용

```python
import asyncio
from beanllm import Client

async def main():
    client = Client(model="gpt-4o-mini")

    # 채팅
    response = await client.chat([
        {"role": "user", "content": "Explain beanllm in one sentence."}
    ])
    print(response.content)

    # 시스템 프롬프트
    response = await client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=512,
    )

asyncio.run(main())
```

### 스트리밍

```python
async def stream_example():
    client = Client(model="claude-sonnet-4-6", provider="claude")

    async for chunk in client.stream_chat(
        messages=[{"role": "user", "content": "Write a haiku."}],
        temperature=0.9,
    ):
        print(chunk, end="", flush=True)
```

### 프로바이더 명시 vs 자동 감지

```python
# 모델명으로 자동 감지
client = Client(model="gpt-4o")          # → openai
client = Client(model="claude-sonnet-4-6")  # → claude

# 명시적 지정
client = Client(model="llama3", provider="ollama")

# API 키 직접 전달 (환경변수 우선)
client = Client(model="gpt-4o-mini", api_key="sk-...")
```

자세한 파라미터는 [docs/api/client.md](../docs/api/client.md)를 참고하세요.

---

## RAGChain — 문서 기반 QA

문서를 청킹하고 벡터 검색으로 관련 컨텍스트를 찾아 LLM에 전달하는 파이프라인입니다.

### 기본 사용

```python
from beanllm import RAGChain

# 단순 사용 — 파일에서 바로 RAG 구성
rag = RAGChain.from_documents("docs/spec.pdf")
answer = rag.query("What are the main features?")
print(answer)

# 복수 파일
rag = RAGChain.from_documents(["doc1.pdf", "doc2.md", "data/notes.txt"])
```

### 세밀한 제어

```python
from beanllm import Client, RAGChain

llm = Client(model="gpt-4o-mini")

rag = RAGChain(
    llm=llm,
    chunk_size=512,
    chunk_overlap=64,
    k=5,                          # 검색 상위 k개
    rerank=True,                   # 크로스인코더 리랭킹
    prompt_template="Context:\n{context}\n\nQuestion: {question}\nAnswer:",
)

rag.add_documents(["report.pdf"])
answer = rag.query("주요 위험 요소는?", k=8, rerank=True)
```

### 벡터 스토어 연동

```python
import chromadb
from beanllm import RAGChain

# Chroma 영구 저장
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection("my_docs")

rag = RAGChain(vector_store=collection)
rag.add_documents(["doc.pdf"])
answer = rag.query("...")
```

지원 벡터 스토어: ChromaDB, FAISS (선택 설치), Pinecone (선택 설치).

자세한 파라미터는 [docs/api/rag.md](../docs/api/rag.md)를 참고하세요.

---

## Agent — ReAct 도구 호출

ReAct (Reasoning + Acting) 패턴으로 도구를 사용하여 복잡한 작업을 수행합니다.

### 기본 사용

```python
from beanllm import Agent, Tool

# 도구 정의
def web_search(query: str) -> str:
    """Search the web for information."""
    # 실제 구현
    return f"Search results for: {query}"

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# 에이전트 생성
agent = Agent(
    model="gpt-4o-mini",
    tools=[
        Tool.from_function(web_search),
        Tool.from_function(calculator),
    ],
    max_steps=10,
)

# 실행
import asyncio
result = asyncio.run(agent.run("What is 15% of the population of South Korea?"))
print(result.answer)
print(f"Steps taken: {result.total_steps}")
```

### 도구 데코레이터

```python
from beanllm.domain.tools import tool

@tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    # 구현
    return f"Weather in {city}: sunny, 22°C"

agent = Agent(model="claude-sonnet-4-6", tools=[get_weather])
```

### AgentResult

```python
result = await agent.run("Find the GDP of Japan and divide it by 10")

result.answer        # 최종 답변 (str)
result.steps         # List[AgentStep]
result.total_steps   # 실행된 단계 수
result.success       # 성공 여부 (bool)
result.error         # 실패 시 오류 메시지

# 각 단계 확인
for step in result.steps:
    print(f"Step {step.step_number}: {step.thought}")
    if step.action:
        print(f"  Action: {step.action}({step.action_input})")
        print(f"  Observation: {step.observation}")
```

자세한 파라미터는 [docs/api/agent.md](../docs/api/agent.md)를 참고하세요.

---

## StateGraph — DAG 워크플로우

방향성 비순환 그래프(DAG)로 복잡한 멀티스텝 워크플로우를 구성합니다.

### 기본 사용

```python
from typing import TypedDict
from beanllm import StateGraph
from beanllm.domain.state_graph import END

# 상태 스키마 정의
class PipelineState(TypedDict):
    input_text: str
    summary: str
    translated: str
    score: float

# 노드 함수 정의
async def summarize(state: PipelineState) -> PipelineState:
    client = Client(model="gpt-4o-mini")
    resp = await client.chat([{"role": "user", "content": f"Summarize: {state['input_text']}"}])
    state["summary"] = resp.content
    return state

async def translate(state: PipelineState) -> PipelineState:
    client = Client(model="gpt-4o-mini")
    resp = await client.chat([{"role": "user", "content": f"Translate to Korean: {state['summary']}"}])
    state["translated"] = resp.content
    return state

def score(state: PipelineState) -> PipelineState:
    state["score"] = len(state["summary"]) / 1000
    return state

# 그래프 구성
graph = StateGraph(PipelineState)
graph.add_node("summarize", summarize)
graph.add_node("translate", translate)
graph.add_node("score", score)

graph.set_entry_point("summarize")
graph.add_edge("summarize", "translate")
graph.add_edge("translate", "score")
graph.add_edge("score", END)

# 실행
result = graph.invoke({"input_text": "Long document text...", "summary": "", "translated": "", "score": 0.0})
print(result["translated"])
```

### 조건부 엣지

```python
def router(state: PipelineState) -> str:
    if state["score"] > 0.5:
        return "detailed_node"
    return "simple_node"

graph.add_conditional_edges("score_node", router, {
    "detailed_node": "detailed_node",
    "simple_node": "simple_node",
})
```

### 체크포인트

```python
from beanllm.domain.state_graph import Checkpoint

# 체크포인트로 실행 재개
checkpoint = Checkpoint(thread_id="workflow-001")
result = graph.invoke(initial_state, config={"checkpoint": checkpoint})
```

---

## 고급 Facade

핵심 4개 외에 다음 Facade도 제공됩니다.

| Facade | 설명 |
|--------|------|
| `MultiAgent` | 복수 에이전트 협업 오케스트레이션 |
| `KnowledgeGraph` | 그래프 기반 지식 검색 (networkx) |
| `RAGDebug` | RAG 파이프라인 디버깅·시각화 |
| `Orchestrator` | 복수 모델 병렬 실행 |
| `Optimizer` | 베이지안 최적화로 프롬프트 파라미터 탐색 |
| `WhisperSTT` | Whisper 기반 음성→텍스트 |
| `VisionRAG` | 이미지·PDF 비전 기반 RAG |
| `WebSearch` | 웹 검색 결합 QA |

---

## 관련 문서

- [docs/api/client.md](../docs/api/client.md) — Client 전체 파라미터
- [docs/api/rag.md](../docs/api/rag.md) — RAGChain 전체 파라미터
- [docs/api/agent.md](../docs/api/agent.md) — Agent 전체 파라미터
- [Architecture](architecture.md) — 레이어 설계
