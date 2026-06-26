# API Reference

beanllm 공개 API 레퍼런스 인덱스.

## 클래스 레퍼런스

| 문서 | 클래스 | 설명 |
|------|--------|------|
| [client.md](client.md) | `Client` | 채팅·스트리밍·임베딩 |
| [rag.md](rag.md) | `RAGChain` | 문서 기반 QA 파이프라인 |
| [agent.md](agent.md) | `Agent` | ReAct 패턴 도구 호출 |
| [models.md](models.md) | — | 지원 모델 전체 목록 |

## 빠른 시작

```python
from beanllm import Client, RAGChain, Agent, StateGraph

# 채팅
client = Client(model="gpt-4o-mini")
response = await client.chat([{"role": "user", "content": "Hello"}])

# RAG
rag = RAGChain.from_documents("doc.pdf")
answer = rag.query("What is this about?")

# Agent
from beanllm.domain.tools import tool

@tool
def search(query: str) -> str:
    return f"Results for {query}"

agent = Agent(model="gpt-4o-mini", tools=[search])
result = await agent.run("Find information about Python")
```

## 설치

```bash
pip install beanllm              # 핵심 (~5MB, Ollama 포함)
pip install beanllm[openai]      # + OpenAI SDK
pip install beanllm[anthropic]   # + Anthropic SDK
pip install beanllm[gemini]      # + Google Generative AI SDK
pip install beanllm[ml]          # + torch, marker-pdf (OCR·음성)
pip install beanllm[all]         # 전체
```
