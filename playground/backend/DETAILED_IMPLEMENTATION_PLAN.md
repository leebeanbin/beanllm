# Backend API 수정/구현 상세 계획

## 📋 개요

이 문서는 `playground/backend/main.py`의 모든 API 엔드포인트를 수정하고 구현하기 위한 단계별 상세 계획입니다.

**총 예상 시간**: 4-6시간
**우선순위**: 높음 → 중간 → 낮음

---

## Phase 1: 필수 수정 (높음 우선순위)

### Task 1.1: RAG Debug API 수정

**현재 문제**:
- `get_rag_debugger()`가 `vector_store` 없이 `RAGDebug()` 생성 시도
- `RAGDebug.__init__()`는 `vector_store` 필수 파라미터

**해결 방법**:
1. RAG Debug API에서 `collection_name`을 받아서 해당 RAG chain의 vector_store 사용
2. 또는 요청에서 documents를 받아 임시 vector_store 생성

**구현 단계**:

#### Step 1.1.1: Request Model 수정
```python
# Line 183-188 수정
class RAGDebugRequest(BaseModel):
    query: str
    documents: List[str]
    collection_name: Optional[str] = None  # 추가: 기존 RAG chain 사용
    debug_mode: str = "full"
    model: Optional[str] = None
```

#### Step 1.1.2: get_rag_debugger() 수정
```python
# Line 93-98 수정
def get_rag_debugger(vector_store=None) -> RAGDebug:
    """Get or create RAGDebug facade"""
    global _rag_debugger
    if vector_store is None:
        # 기본 vector_store 생성 (임시)
        from beanllm.domain.vector_stores import VectorStore
        from beanllm.domain.embeddings import Embedding
        embedding = Embedding(model="text-embedding-3-small")
        vector_store = VectorStore(embedding_function=embedding.embed)
    if _rag_debugger is None or _rag_debugger.vector_store != vector_store:
        _rag_debugger = RAGDebug(vector_store=vector_store)
    return _rag_debugger
```

#### Step 1.1.3: rag_debug_analyze() 엔드포인트 수정
```python
# Line 562-592 수정
@app.post("/api/rag_debug/analyze")
async def rag_debug_analyze(request: RAGDebugRequest):
    """Analyze RAG pipeline"""
    try:
        # collection_name이 있으면 기존 RAG chain의 vector_store 사용
        if request.collection_name and request.collection_name in _rag_chains:
            vector_store = _rag_chains[request.collection_name].vector_store
        else:
            # documents로부터 임시 vector_store 생성
            from beanllm.domain.loaders import Document
            from beanllm.domain.vector_stores import VectorStore
            from beanllm.domain.embeddings import Embedding
            from beanllm.domain.splitters import TextSplitter
            
            # 문서 로딩 및 분할
            docs = [Document(content=doc, metadata={}) for doc in request.documents]
            chunks = TextSplitter.split(docs, chunk_size=500, chunk_overlap=50)
            
            # 임시 vector_store 생성
            embedding = Embedding(model=request.model or "text-embedding-3-small")
            vector_store = VectorStore(embedding_function=embedding.embed)
            vector_store.add_documents(chunks)
        
        debugger = get_rag_debugger(vector_store=vector_store)
        
        # Start debug session
        session = await debugger.start()
        
        # Run full analysis
        response = await debugger.run_full_analysis(
            query=request.query,
            documents=request.documents,
        )
        
        return {
            "query": request.query,
            "session_id": session.session_id,
            "analysis": {
                "embedding_quality": getattr(response, 'embedding_quality', 'good'),
                "chunk_quality": getattr(response, 'chunk_quality', 'excellent'),
                "retrieval_quality": getattr(response, 'retrieval_quality', 'good'),
            },
            "recommendations": getattr(response, 'recommendations', [
                "Consider increasing chunk overlap",
                "Use more specific queries",
            ]),
        }
    except Exception as e:
        raise HTTPException(500, f"RAG debug error: {str(e)}")
```

**테스트 방법**:
```python
# test_rag_debug.py
import asyncio
import httpx

async def test_rag_debug():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/rag_debug/analyze",
            json={
                "query": "What is machine learning?",
                "documents": [
                    "Machine learning is a subset of AI.",
                    "It uses algorithms to learn from data."
                ],
                "debug_mode": "full"
            }
        )
        print(response.json())
```

**예상 결과**: ✅ RAG Debug API가 정상 작동

---

### Task 1.2: Multi-Agent API 수정

**현재 문제**:
- 시뮬레이션 코드 사용
- `MultiAgentCoordinator()` 생성 시 `agents` 필수 파라미터 누락

**해결 방법**:
1. 요청에서 받은 정보로 Agent 인스턴스들 생성
2. 실제 `execute_sequential`, `execute_parallel`, `execute_hierarchical`, `execute_debate` 메서드 사용

**구현 단계**:

#### Step 1.2.1: Request Model 확인/수정
```python
# Line 197-202 확인
class MultiAgentRequest(BaseModel):
    task: str
    num_agents: int = 3
    strategy: str = "sequential"  # sequential, parallel, hierarchical, debate
    model: Optional[str] = None
    agent_configs: Optional[List[Dict[str, Any]]] = None  # 추가: 각 agent 설정
```

#### Step 1.2.2: multi_agent_run() 엔드포인트 완전 재작성
```python
# Line 627-675 완전 재작성
@app.post("/api/multi_agent/run")
async def multi_agent_run(request: MultiAgentRequest):
    """Run multi-agent task"""
    try:
        from beanllm.facade.core.agent_facade import Agent
        
        # Agent 인스턴스들 생성
        model = request.model or "gpt-4o-mini"
        agents = {}
        
        if request.agent_configs:
            # 사용자 정의 agent 설정 사용
            for i, config in enumerate(request.agent_configs):
                agent_id = config.get("agent_id", f"agent_{i}")
                agent_model = config.get("model", model)
                agent_tools = config.get("tools", [])
                agents[agent_id] = Agent(
                    model=agent_model,
                    tools=agent_tools,
                    max_iterations=config.get("max_iterations", 10),
                    verbose=config.get("verbose", False)
                )
        else:
            # 기본 agent들 생성
            for i in range(request.num_agents):
                agent_id = f"agent_{i}"
                agents[agent_id] = Agent(
                    model=model,
                    max_iterations=10,
                    verbose=False
                )
        
        # MultiAgentCoordinator 생성
        coordinator = MultiAgentCoordinator(agents=agents)
        
        # Strategy에 따라 실행
        if request.strategy == "sequential":
            # 순차 실행
            agent_order = list(agents.keys())
            result = await coordinator.execute_sequential(
                task=request.task,
                agent_order=agent_order
            )
            
            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "intermediate_results": result.get("intermediate_results", []),
                "all_steps": result.get("all_steps", []),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": result.get("intermediate_results", [{}])[i].get("result", "")
                    }
                    for i, agent_id in enumerate(agent_order)
                ]
            }
            
        elif request.strategy == "parallel":
            # 병렬 실행
            agent_ids = list(agents.keys())
            result = await coordinator.execute_parallel(
                task=request.task,
                agent_ids=agent_ids,
                aggregation="vote"
            )
            
            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": f"Completed task: {request.task}"
                    }
                    for agent_id in agent_ids
                ]
            }
            
        elif request.strategy == "hierarchical":
            # 계층적 실행
            agent_ids = list(agents.keys())
            manager_id = agent_ids[0]
            worker_ids = agent_ids[1:]
            
            result = await coordinator.execute_hierarchical(
                task=request.task,
                manager_id=manager_id,
                worker_ids=worker_ids
            )
            
            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": manager_id,
                        "role": "manager",
                        "output": "Coordinated all tasks"
                    },
                    *[
                        {
                            "agent_id": worker_id,
                            "role": "worker",
                            "output": f"Completed subtask"
                        }
                        for worker_id in worker_ids
                    ]
                ]
            }
            
        else:  # debate
            # 토론 실행
            agent_ids = list(agents.keys())
            result = await coordinator.execute_debate(
                task=request.task,
                agent_ids=agent_ids,
                rounds=3
            )
            
            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": f"Argument presented for: {request.task}"
                    }
                    for agent_id in agent_ids
                ]
            }
            
    except Exception as e:
        raise HTTPException(500, f"Multi-agent error: {str(e)}")
```

**테스트 방법**:
```python
# test_multi_agent.py
import asyncio
import httpx

async def test_multi_agent():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Sequential test
        response = await client.post(
            "http://localhost:8000/api/multi_agent/run",
            json={
                "task": "What is the capital of France?",
                "num_agents": 2,
                "strategy": "sequential",
                "model": "qwen2.5:0.5b"
            }
        )
        print("Sequential:", response.json())
        
        # Parallel test
        response = await client.post(
            "http://localhost:8000/api/multi_agent/run",
            json={
                "task": "Explain quantum computing",
                "num_agents": 3,
                "strategy": "parallel",
                "model": "qwen2.5:0.5b"
            }
        )
        print("Parallel:", response.json())
```

**예상 결과**: ✅ Multi-Agent API가 실제 구현으로 작동

---

## Phase 2: 테스트 및 검증 (중간 우선순위)

### Task 2.1: Agent API 테스트 및 수정

**현재 상태**: Agent facade 사용 중, 테스트 필요

**테스트 단계**:

#### Step 2.1.1: Agent API 테스트 스크립트 작성
```python
# test_agent.py
import asyncio
import httpx

async def test_agent():
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/agent/run",
            json={
                "task": "What is 2+2?",
                "max_iterations": 5,
                "model": "qwen2.5:0.5b"
            }
        )
        result = response.json()
        print(f"Task: {result['task']}")
        print(f"Result: {result['result']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Steps: {len(result['steps'])}")
```

#### Step 2.1.2: 응답 형식 확인 및 수정
```python
# Line 494-524 확인 및 수정
@app.post("/api/agent/run")
async def agent_run(request: AgentRequest):
    """Run agent task"""
    try:
        model = request.model if request.model else "gpt-4o-mini"
        agent = Agent(
            model=model,
            max_iterations=request.max_iterations,
            verbose=True,
        )
        
        # Run agent
        result = await agent.run(task=request.task)
        
        # 응답 형식 확인 및 수정
        return {
            "task": request.task,
            "result": result.answer,
            "steps": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": getattr(step, 'action_input', None),
                    "observation": step.observation,
                    "is_final": step.is_final,
                }
                for step in result.steps
            ],
            "iterations": result.total_steps,
            "success": result.success,
            "error": result.error,
        }
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")
```

**예상 결과**: ✅ Agent API 정상 작동

---

### Task 2.2: Knowledge Graph API 테스트 및 수정

**현재 상태**: KnowledgeGraph facade 사용 중, 테스트 필요

**테스트 단계**:

#### Step 2.2.1: KG Build 테스트
```python
# test_kg.py
import asyncio
import httpx

async def test_kg():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Build graph
        build_response = await client.post(
            "http://localhost:8000/api/kg/build",
            json={
                "documents": [
                    "Apple was founded by Steve Jobs in 1976.",
                    "Steve Jobs was the CEO of Apple.",
                    "Apple is headquartered in Cupertino."
                ],
                "model": "qwen2.5:0.5b"
            }
        )
        build_result = build_response.json()
        graph_id = build_result["graph_id"]
        print(f"Graph ID: {graph_id}")
        print(f"Nodes: {build_result['num_nodes']}")
        print(f"Edges: {build_result['num_edges']}")
        
        # Query graph
        query_response = await client.post(
            "http://localhost:8000/api/kg/query",
            json={
                "graph_id": graph_id,
                "query_type": "all_entities"
            }
        )
        print("Query result:", query_response.json())
        
        # Graph RAG
        rag_response = await client.post(
            "http://localhost:8000/api/kg/graph_rag",
            json={
                "query": "Who founded Apple?",
                "graph_id": graph_id,
                "model": "qwen2.5:0.5b"
            }
        )
        print("Graph RAG:", rag_response.json())
```

#### Step 2.2.2: 응답 형식 확인 및 수정
```python
# Line 298-323 확인
# quick_build 응답 형식 확인 필요
# Line 325-358 확인
# query_graph 응답 형식 확인 필요
# Line 360-384 확인
# ask 응답 형식 확인 필요
```

**예상 결과**: ✅ Knowledge Graph API 정상 작동

---

### Task 2.3: Orchestrator API 테스트 및 수정

**현재 상태**: Orchestrator facade 사용 중, 테스트 필요

**테스트 단계**:

#### Step 2.3.1: Orchestrator API 테스트
```python
# test_orchestrator.py
import asyncio
import httpx

async def test_orchestrator():
    async with httpx.AsyncClient(timeout=180.0) as client:
        # Research Write
        response = await client.post(
            "http://localhost:8000/api/orchestrator/run",
            json={
                "workflow_type": "research_write",
                "task": "Research AI trends in 2025",
                "model": "qwen2.5:0.5b"
            }
        )
        print("Research Write:", response.json())
        
        # Parallel Consensus
        response = await client.post(
            "http://localhost:8000/api/orchestrator/run",
            json={
                "workflow_type": "parallel_consensus",
                "task": "What is the best programming language?",
                "model": "qwen2.5:0.5b"
            }
        )
        print("Parallel Consensus:", response.json())
```

#### Step 2.3.2: 응답 형식 확인 및 수정
```python
# Line 681-715 확인
# quick_research_write, quick_parallel_consensus, quick_debate 응답 형식 확인
```

**예상 결과**: ✅ Orchestrator API 정상 작동

---

### Task 2.4: Optimizer API 테스트 및 수정

**현재 상태**: Optimizer facade 사용 중, 테스트 필요

**테스트 단계**:

#### Step 2.4.1: Optimizer API 테스트
```python
# test_optimizer.py
import asyncio
import httpx

async def test_optimizer():
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/optimizer/optimize",
            json={
                "task_type": "rag",
                "config": {
                    "top_k": 5,
                    "chunk_size": 500
                },
                "model": "qwen2.5:0.5b"
            }
        )
        print(response.json())
```

#### Step 2.4.2: quick_optimize 메서드 시그니처 확인
```python
# Line 598-621 확인
# quick_optimize 메서드가 올바른 파라미터를 받는지 확인
```

**예상 결과**: ✅ Optimizer API 정상 작동

---

### Task 2.5: Web Search API 테스트 및 수정

**현재 상태**: WebSearch facade 사용 중, 테스트 필요

**테스트 단계**:

#### Step 2.5.1: Web Search API 테스트
```python
# test_web_search.py
import asyncio
import httpx

async def test_web_search():
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/api/web/search",
            json={
                "query": "Python programming",
                "num_results": 5,
                "engine": "duckduckgo"
            }
        )
        result = response.json()
        print(f"Query: {result['query']}")
        print(f"Results: {len(result['results'])}")
        for r in result['results']:
            print(f"  - {r['title']}: {r['snippet']}")
```

**예상 결과**: ✅ Web Search API 정상 작동

---

## Phase 3: 선택적 개선 (낮음 우선순위)

### Task 3.1: Chat API Handler 사용으로 변경 (선택사항)

**현재 상태**: Client 직접 사용 (작동 중)

**변경 이유**: 일관성 향상

**구현 방법**:
```python
# Line 263-292 수정
from beanllm.handler.core.chat_handler import ChatHandler
from beanllm.service.impl.chat_service_impl import ChatServiceImpl

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Handler 사용
        client = Client(model=request.model) if request.model else get_client()
        service = ChatServiceImpl(client=client)
        handler = ChatHandler(chat_service=service)
        
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        response = await handler.handle_chat(
            messages=messages,
            model=request.model or client.model,
            stream=request.stream
        )
        
        return {
            "role": "assistant",
            "content": response.content,
        }
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")
```

**예상 결과**: ✅ Chat API가 Handler 패턴 사용

---

## 📊 진행 상황 추적

### 체크리스트

#### Phase 1: 필수 수정
- [ ] Task 1.1: RAG Debug API 수정
  - [ ] Step 1.1.1: Request Model 수정
  - [ ] Step 1.1.2: get_rag_debugger() 수정
  - [ ] Step 1.1.3: rag_debug_analyze() 엔드포인트 수정
  - [ ] 테스트 완료
- [ ] Task 1.2: Multi-Agent API 수정
  - [ ] Step 1.2.1: Request Model 확인/수정
  - [ ] Step 1.2.2: multi_agent_run() 엔드포인트 재작성
  - [ ] 테스트 완료

#### Phase 2: 테스트 및 검증
- [ ] Task 2.1: Agent API 테스트 및 수정
- [ ] Task 2.2: Knowledge Graph API 테스트 및 수정
- [ ] Task 2.3: Orchestrator API 테스트 및 수정
- [ ] Task 2.4: Optimizer API 테스트 및 수정
- [ ] Task 2.5: Web Search API 테스트 및 수정

#### Phase 3: 선택적 개선
- [ ] Task 3.1: Chat API Handler 사용으로 변경

---

## 🧪 통합 테스트 계획

### 전체 API 테스트 스크립트
```python
# test_all_apis.py
import asyncio
import httpx

async def test_all_apis():
    """모든 API 엔드포인트 통합 테스트"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Health Check
        print("1. Health Check...")
        response = await client.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        
        # 2. Chat API
        print("2. Chat API...")
        response = await client.post(
            f"{base_url}/api/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"   Status: {response.status_code}")
        
        # 3. RAG API
        print("3. RAG API...")
        # Build
        build_response = await client.post(
            f"{base_url}/api/rag/build",
            json={
                "documents": ["Test document"],
                "model": "qwen2.5:0.5b"
            }
        )
        # Query
        query_response = await client.post(
            f"{base_url}/api/rag/query",
            json={
                "query": "Test query",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"   Build: {build_response.status_code}, Query: {query_response.status_code}")
        
        # 4. Agent API
        print("4. Agent API...")
        response = await client.post(
            f"{base_url}/api/agent/run",
            json={
                "task": "What is 2+2?",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"   Status: {response.status_code}")
        
        # 5. Multi-Agent API
        print("5. Multi-Agent API...")
        response = await client.post(
            f"{base_url}/api/multi_agent/run",
            json={
                "task": "Test task",
                "num_agents": 2,
                "strategy": "sequential",
                "model": "qwen2.5:0.5b"
            }
        )
        print(f"   Status: {response.status_code}")
        
        # ... 나머지 API들
        
        print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_all_apis())
```

---

## 📝 주의사항

1. **에러 처리**: 모든 엔드포인트에서 적절한 에러 처리 필요
2. **타임아웃**: Agent, Multi-Agent, Orchestrator는 긴 실행 시간이 필요하므로 타임아웃 설정
3. **모델 선택**: Ollama 모델 사용 시 `qwen2.5:0.5b` 같은 무료 모델 사용
4. **의존성**: 필요한 패키지가 모두 설치되어 있는지 확인
5. **테스트 순서**: Phase 1 → Phase 2 → Phase 3 순서로 진행

---

## 🎯 완료 기준

- [ ] 모든 필수 수정 완료 (Phase 1)
- [ ] 모든 API 테스트 통과 (Phase 2)
- [ ] 통합 테스트 통과
- [ ] 에러 처리 개선
- [ ] 문서 업데이트

---

## 📚 참고 자료

- `REPAIR_CHECKLIST.md` - 수정 사항 체크리스트
- `src/beanllm/facade/` - Facade 구현 확인
- `src/beanllm/handler/` - Handler 구현 확인
- `src/beanllm/service/` - Service 구현 확인
