# 🚀 llmkit 빠른 시작 가이드

## 📦 설치

### Poetry 사용 (권장)

```bash
# 프로젝트 클론
git clone https://github.com/yourusername/llmkit.git
cd llmkit

# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install --extras all  # 모든 Provider 포함
# 또는
poetry install --extras openai  # OpenAI만

# 가상 환경 활성화
poetry shell
```

### pip 사용

```bash
# 기본 설치
pip install llmkit

# 특정 Provider 추가
pip install llmkit[openai]
pip install llmkit[anthropic]
pip install llmkit[gemini]
pip install llmkit[ollama]

# 모든 Provider
pip install llmkit[all]

# 개발 도구 포함
pip install llmkit[dev,all]
```

---

## ⚙️ 환경 설정

### 1. .env 파일 생성

```bash
# 프로젝트 루트에 .env 파일 생성
touch .env
```

### 2. API 키 설정

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GEMINI_API_KEY=...

# Ollama (로컬, API 키 불필요)
OLLAMA_HOST=http://localhost:11434
```

### 3. 환경 변수 로드

```python
# 자동으로 .env 파일 로드됨
from llmkit import Client
# 또는
from dotenv import load_dotenv
load_dotenv()
```

---

## 🎯 기본 사용법

### 1. 간단한 채팅

```python
from llmkit import Client

# Client 생성 (자동으로 사용 가능한 Provider 선택)
client = Client(model="gpt-4o")

# 채팅
response = client.chat("안녕하세요!")
print(response.content)

# 스트리밍
for chunk in client.stream("긴 이야기를 들려주세요"):
    print(chunk.content, end="", flush=True)
```

### 2. Provider 선택

```python
# OpenAI 사용
client = Client(model="gpt-4o")

# Claude 사용
client = Client(model="claude-3-5-sonnet-20241022")

# Gemini 사용
client = Client(model="gemini-2.0-flash-exp")

# Ollama 사용 (로컬)
client = Client(model="qwen2.5:7b")
```

### 3. 파라미터 설정

```python
response = client.chat(
    "창의적인 이야기를 써주세요",
    temperature=0.9,      # 창의성
    max_tokens=1000,      # 최대 토큰
    system="당신은 창의적인 작가입니다"  # 시스템 메시지
)
```

---

## 📄 RAG (Retrieval-Augmented Generation)

### 1. 문서에서 RAG 생성

```python
from llmkit import RAGChain

# 문서 폴더에서 RAG 생성
rag = RAGChain.from_documents("docs/")

# 질문하기
answer = rag.query("이 문서의 주요 내용은?")
print(answer)

# 소스 포함
answer, sources = rag.query(
    "구체적인 예시를 들어 설명해주세요",
    include_sources=True
)

for source in sources:
    print(f"출처: {source.document.metadata.get('source')}")
    print(f"유사도: {source.similarity:.4f}")
```

### 2. 커스텀 RAG 구성

```python
from llmkit import (
    DocumentLoader,
    RecursiveCharacterTextSplitter,
    OpenAIEmbedding,
    ChromaVectorStore,
    RAGChain
)

# 1. 문서 로드
docs = DocumentLoader.load("my_documents/")

# 2. 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 3. 임베딩 생성
embedding = OpenAIEmbedding(model="text-embedding-3-small")

# 4. 벡터 스토어 생성
vector_store = ChromaVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./my_vector_db"
)

# 5. RAG 생성
rag = RAGChain(
    vector_store=vector_store,
    llm=Client(model="gpt-4o")
)

# 사용
answer = rag.query("질문")
```

---

## 🤖 Agent (도구 사용)

### 1. 기본 Agent

```python
from llmkit import Agent, Tool

# 도구 정의
@Tool.from_function
def calculator(expression: str) -> str:
    """수학 표현식을 계산합니다"""
    return str(eval(expression))

@Tool.from_function
def get_weather(city: str) -> str:
    """도시의 날씨를 가져옵니다"""
    # 실제 API 호출
    return f"{city}의 날씨는 맑음입니다"

# Agent 생성
agent = Agent(
    llm=Client(model="gpt-4o"),
    tools=[calculator, get_weather],
    max_iterations=10
)

# 실행
result = agent.run("25 * 17를 계산하고, 서울의 날씨를 알려주세요")
print(result.output)
```

### 2. 내장 도구 사용

```python
from llmkit import Agent, search_web, get_current_time

# 내장 도구 사용
agent = Agent(
    llm=Client(model="gpt-4o"),
    tools=[search_web, get_current_time]
)

result = agent.run("현재 시간을 알려주고, 오늘의 뉴스를 검색해주세요")
```

---

## 🕸️ Graph Workflows

### 1. 간단한 Graph

```python
from llmkit import StateGraph, END

# Graph 생성
graph = StateGraph()

# 노드 정의
def analyze(state):
    state["analysis"] = client.chat(f"분석: {state['input']}")
    return state

def improve(state):
    state["output"] = client.chat(f"개선: {state['input']}")
    return state

# 노드 추가
graph.add_node("analyze", analyze)
graph.add_node("improve", improve)

# 조건부 엣지
def should_improve(state):
    score = float(state["analysis"].content.split("점수:")[1])
    return "improve" if score < 0.8 else "end"

graph.add_conditional_edges(
    "analyze",
    should_improve,
    {"improve": "improve", "end": END}
)

# 실행
result = graph.compile().invoke({"input": "초안 텍스트"})
print(result["output"])
```

### 2. LangGraph 스타일

```python
from llmkit import Graph, create_simple_graph

# 간단한 Graph 생성
graph = create_simple_graph(
    nodes={
        "research": lambda s: {"info": "연구 결과"},
        "write": lambda s: {"draft": "초안"},
        "review": lambda s: {"final": "최종"}
    },
    edges=[
        ("research", "write"),
        ("write", "review")
    ]
)

result = graph.run({"topic": "AI"})
```

---

## 👥 Multi-Agent Systems

### 1. Debate 패턴

```python
from llmkit import MultiAgentCoordinator, DebateStrategy, Agent

# 여러 Agent 생성
researcher = Agent(
    llm=Client(model="gpt-4o"),
    role="연구자",
    tools=[search_web]
)

writer = Agent(
    llm=Client(model="gpt-4o"),
    role="작가"
)

critic = Agent(
    llm=Client(model="gpt-4o"),
    role="비평가"
)

# Coordinator 생성
coordinator = MultiAgentCoordinator(
    agents=[researcher, writer, critic],
    strategy=DebateStrategy(rounds=3)
)

# 실행
result = coordinator.coordinate("양자 컴퓨팅에 대한 기사를 작성해주세요")
print(result.final_output)
```

### 2. Sequential 패턴

```python
from llmkit import SequentialStrategy

coordinator = MultiAgentCoordinator(
    agents=[researcher, writer, critic],
    strategy=SequentialStrategy()
)

result = coordinator.coordinate("작업을 순차적으로 수행")
```

---

## 🖼️ Vision RAG

### 1. 이미지 기반 질의응답

```python
from llmkit import VisionRAG, CLIPEmbedding, ImageLoader

# 이미지 로드
images = ImageLoader.load("images/")

# Vision RAG 생성
vision_rag = VisionRAG.from_images(
    images=images,
    embedding=CLIPEmbedding(),
    llm=Client(model="gpt-4o")  # Vision 지원 모델
)

# 텍스트 질의
answer = vision_rag.query("이 이미지들에 어떤 객체들이 있나요?")

# 이미지 + 텍스트 질의
answer = vision_rag.query_with_image(
    "reference.jpg",
    "이 이미지와 유사한 이미지를 찾아 설명해주세요"
)
```

---

## 🎙️ Audio Processing

### 1. Speech-to-Text

```python
from llmkit import WhisperSTT

stt = WhisperSTT()
result = stt.transcribe("audio.mp3", language="ko")
print(result.text)

# 세그먼트별 결과
for segment in result.segments:
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
```

### 2. Text-to-Speech

```python
from llmkit import TextToSpeech

tts = TextToSpeech(provider="openai")
audio = tts.synthesize(
    "안녕하세요, 반갑습니다",
    voice="alloy",
    speed=1.0
)

# 파일 저장
audio.save("output.mp3")
```

### 3. Audio RAG

```python
from llmkit import AudioRAG

# 오디오 파일에서 RAG 생성
audio_rag = AudioRAG.from_audio_files([
    "podcast1.mp3",
    "podcast2.mp3"
])

# 질문
answer = audio_rag.query("AI에 대해 무엇이 논의되었나요?")
```

---

## 🌐 Web Search

### 1. 웹 검색

```python
from llmkit import DuckDuckGoSearch, WebScraper

# 검색 (API 키 불필요!)
search = DuckDuckGoSearch()
results = search.search("최신 AI 뉴스", max_results=5)

for result in results:
    print(f"{result.title}: {result.url}")
    print(f"요약: {result.snippet}")

# 콘텐츠 스크래핑
scraper = WebScraper()
content = scraper.scrape(results[0].url)
print(content)
```

---

## 📊 Evaluation

### 1. 텍스트 평가

```python
from llmkit import evaluate_text

prediction = "고양이가 매트 위에 앉아있다"
reference = "고양이가 매트 위에 앉아 있습니다"

result = evaluate_text(
    prediction=prediction,
    reference=reference,
    metrics=["bleu", "rouge-1", "rouge-l", "f1"]
)

print(f"BLEU: {result.bleu:.4f}")
print(f"ROUGE-1: {result.rouge_1:.4f}")
print(f"평균 점수: {result.average_score:.4f}")
```

### 2. RAG 평가

```python
from llmkit import evaluate_rag

rag_result = evaluate_rag(
    question="AI란 무엇인가요?",
    answer="AI는 인공지능입니다...",
    contexts=["컨텍스트 1", "컨텍스트 2"],
    ground_truth="AI는..."
)

print(f"Faithfulness: {rag_result.faithfulness:.4f}")
print(f"Answer Relevance: {rag_result.answer_relevance:.4f}")
```

---

## 🛠️ 고급 기능

### 1. Memory 사용

```python
from llmkit import BufferMemory

memory = BufferMemory(max_messages=10)

# 대화 추가
memory.add_message("user", "내 이름은 홍길동이야")
memory.add_message("assistant", "안녕하세요, 홍길동님!")

# 대화 기록 가져오기
history = memory.get_messages()
print(history)
```

### 2. Output Parsers

```python
from llmkit import PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)

response = client.chat(
    "홍길동, 30세에 대한 정보를 JSON 형식으로 반환해주세요",
    output_parser=parser
)

person = response.parsed  # Person 객체
print(person.name, person.age)
```

### 3. Prompt Templates

```python
from llmkit import PromptTemplate, FewShotPromptTemplate

# 기본 템플릿
template = PromptTemplate(
    template="{source}에서 {target}로 번역: {text}",
    input_variables=["source", "target", "text"]
)

prompt = template.format(
    source="영어",
    target="한국어",
    text="Hello"
)

# Few-shot 템플릿
few_shot = FewShotPromptTemplate(
    examples=[
        {"input": "2+2", "output": "4"},
        {"input": "3*5", "output": "15"}
    ],
    example_template=PromptTemplate(
        template="Q: {input}\nA: {output}",
        input_variables=["input", "output"]
    ),
    prefix="수학 문제를 풀어주세요:",
    suffix="Q: {input}\nA:"
)
```

---

## 🔧 개발 도구

### Makefile 사용

```bash
# 개발 도구 설치
make install-dev

# 빠른 자동 수정
make quick-fix

# 타입 체크
make type-check

# 린트 체크
make lint

# 전체 검사 및 수정
make all
```

### Poetry 사용

```bash
# 의존성 추가
poetry add openai
poetry add --group dev pytest

# 의존성 업데이트
poetry update

# 가상 환경 정보
poetry env info
```

---

## 📚 다음 단계

1. **문서 읽기**: [`docs/`](docs/) 폴더의 상세 문서
2. **예제 실행**: [`examples/`](examples/) 폴더의 예제 코드
3. **튜토리얼**: [`docs/tutorials/`](docs/tutorials/) 폴더의 튜토리얼
4. **아키텍처 이해**: [`ARCHITECTURE.md`](ARCHITECTURE.md) 참고

---

## ❓ 문제 해결

### Provider를 찾을 수 없음

```bash
# Provider 설치 확인
poetry install --extras all
# 또는
pip install llmkit[all]
```

### API 키 오류

```bash
# .env 파일 확인
cat .env

# 환경 변수 확인
echo $OPENAI_API_KEY
```

### Import 오류

```python
# 올바른 import 방법
from llmkit import Client  # ✅
# from llmkit.client import Client  # ❌ (구버전)
```

---

**더 자세한 내용은 [README.md](README.md)와 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요!**
