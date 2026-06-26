# RAGChain

`beanllm.RAGChain` — 문서 기반 QA 파이프라인 (Facade 패턴)

RAGChain은 문서를 청킹하고 벡터 검색으로 관련 컨텍스트를 찾아 LLM에 전달하는 완전한 RAG 파이프라인을 제공합니다.

## Import

```python
from beanllm import RAGChain
```

---

## `from_documents` (클래스 메서드)

```python
@classmethod
def from_documents(
    cls,
    documents: Union[str, Path, List[Union[str, Path]]],
    model: str = "gpt-4o-mini",
    provider: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs: Any,
) -> "RAGChain"
```

파일에서 바로 RAGChain을 구성하는 편의 메서드.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `documents` | `str \| Path \| List[...]` | — (필수) | 파일 경로. PDF, MD, TXT 지원 |
| `model` | `str` | `"gpt-4o-mini"` | 사용할 LLM 모델 |
| `provider` | `str \| None` | `None` | 프로바이더. `None`이면 자동 감지 |
| `chunk_size` | `int` | `512` | 청크당 최대 토큰 수 |
| `chunk_overlap` | `int` | `64` | 인접 청크 겹침 토큰 수 |

**예시:**

```python
# 단일 파일
rag = RAGChain.from_documents("report.pdf")

# 복수 파일
rag = RAGChain.from_documents(["doc1.pdf", "doc2.md", "notes.txt"])

# 모델 지정
rag = RAGChain.from_documents("spec.pdf", model="claude-sonnet-4-6", provider="claude")
```

---

## `__init__`

```python
RAGChain(
    llm: Optional[Client] = None,
    vector_store: Optional[Any] = None,
    chunk_size: int = DEFAULT_RAG_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_RAG_CHUNK_OVERLAP,
    k: int = 4,
    rerank: bool = False,
    prompt_template: Optional[str] = None,
    **kwargs: Any,
)
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `llm` | `Client \| None` | `None` | 사용할 LLM 클라이언트. `None`이면 내부 생성 |
| `vector_store` | `Any \| None` | `None` | 벡터 스토어 인스턴스 (ChromaDB collection 등) |
| `chunk_size` | `int` | `512` | 청크당 최대 토큰 수 |
| `chunk_overlap` | `int` | `64` | 청크 간 겹침 토큰 수 |
| `k` | `int` | `4` | 검색 시 반환할 상위 청크 수 |
| `rerank` | `bool` | `False` | 크로스인코더 리랭킹 사용 여부 |
| `prompt_template` | `str \| None` | `None` | 커스텀 프롬프트 템플릿. `{context}`, `{question}` 플레이스홀더 사용 |

---

## `add_documents`

```python
def add_documents(
    self,
    documents: Union[str, Path, List[Union[str, Path]]],
) -> None
```

문서를 청킹하여 벡터 스토어에 추가합니다.

```python
rag = RAGChain()
rag.add_documents(["doc1.pdf", "doc2.md"])
rag.add_documents("additional.pdf")  # 추가 문서 계속 삽입 가능
```

---

## `query`

```python
def query(
    self,
    question: str,
    k: Optional[int] = None,
    rerank: Optional[bool] = None,
    **kwargs: Any,
) -> str
```

질문에 대해 RAG 파이프라인을 실행하고 답변을 반환합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `question` | `str` | — (필수) | 질문 |
| `k` | `int \| None` | 생성자 값 | 검색 청크 수 (이 호출에만 적용) |
| `rerank` | `bool \| None` | 생성자 값 | 리랭킹 여부 (이 호출에만 적용) |

**반환:** `str` — 생성된 답변

**예시:**

```python
rag = RAGChain.from_documents("whitepaper.pdf")

# 기본 질의
answer = rag.query("What is the main contribution?")
print(answer)

# 호출별 파라미터 오버라이드
answer = rag.query("List all limitations.", k=8, rerank=True)
```

---

## 벡터 스토어 연동

### ChromaDB (기본 권장)

```python
import chromadb
from beanllm import RAGChain

# 인메모리
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("my_docs")

# 영구 저장
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("project_docs")

rag = RAGChain(vector_store=collection)
rag.add_documents(["doc.pdf"])
```

### FAISS

```python
# pip install faiss-cpu
import faiss
from beanllm import RAGChain

# FAISS 인덱스 직접 사용
rag = RAGChain(vector_store_type="faiss")
```

### Pinecone

```python
# pip install pinecone-client
import pinecone

pc = pinecone.Pinecone(api_key="...")
index = pc.Index("my-index")

rag = RAGChain(vector_store=index)
```

---

## 커스텀 프롬프트

```python
template = """You are a precise technical assistant.

Context:
{context}

Question: {question}

Answer (be concise and cite the source if possible):"""

rag = RAGChain(
    prompt_template=template,
    k=6,
    rerank=True,
)
rag.add_documents("technical_spec.pdf")
answer = rag.query("What is the API rate limit?")
```

---

## 비동기 사용

`query()`는 동기 메서드입니다. 비동기 환경에서는 executor를 사용하거나 `asyncio.to_thread()`를 활용하세요.

```python
import asyncio

async def async_query():
    rag = RAGChain.from_documents("doc.pdf")
    answer = await asyncio.to_thread(rag.query, "What is the main topic?")
    return answer
```

---

## 관련 문서

- [wiki/facade.md](../../wiki/facade.md#ragchain--문서-기반-qa) — RAGChain 고수준 가이드
- [wiki/providers.md](../../wiki/providers.md) — 지원 벡터 스토어
- [client.md](client.md) — Client 레퍼런스
