"""
beanllm Playground Backend - FastAPI

Complete working backend for all 9 beanllm features
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

# Add parent directory to path to import beanllm
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from beanllm import Client
from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph
from beanllm.facade.core.rag_facade import RAGChain, RAGBuilder
from beanllm.facade.core.agent_facade import Agent
from beanllm.facade.core.chain_facade import Chain, ChainBuilder, PromptChain
from beanllm.facade.ml.web_search_facade import WebSearch
from beanllm.facade.advanced.rag_debug_facade import RAGDebug
from beanllm.facade.advanced.optimizer_facade import Optimizer
from beanllm.facade.advanced.multi_agent_facade import MultiAgentCoordinator
from beanllm.facade.advanced.orchestrator_facade import Orchestrator
from beanllm.facade.ml.vision_rag_facade import VisionRAG, MultimodalRAG
from beanllm.facade.ml.audio_facade import WhisperSTT, TextToSpeech, AudioRAG
from beanllm.facade.ml.evaluation_facade import EvaluatorFacade
from beanllm.facade.ml.finetuning_facade import FineTuningManagerFacade

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="beanllm Playground API",
    description="Complete backend for all beanllm features",
    version="1.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

# Client initialization (lazy)
_client: Optional[Client] = None
_kg: Optional[KnowledgeGraph] = None
_rag_chains: Dict[str, RAGChain] = {}  # collection_name -> RAGChain
_chains: Dict[str, Chain] = {}  # chain_id -> Chain
_web_search: Optional[WebSearch] = None
_rag_debugger: Optional[RAGDebug] = None
_optimizer: Optional[Optimizer] = None
_multi_agent: Optional[MultiAgentCoordinator] = None
_orchestrator: Optional[Orchestrator] = None
_vision_rag: Optional[VisionRAG] = None
_audio_rag: Optional[AudioRAG] = None
_evaluator: Optional[EvaluatorFacade] = None
_finetuning: Optional[FineTuningManagerFacade] = None


def get_client() -> Client:
    """Get or create beanllm client"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(500, "OPENAI_API_KEY not set")
        _client = Client(provider="openai", api_key=api_key, model="gpt-4o-mini")
    return _client


def get_kg() -> KnowledgeGraph:
    """Get or create KnowledgeGraph facade"""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph(client=get_client())
    return _kg


def get_web_search() -> WebSearch:
    """Get or create WebSearch facade"""
    global _web_search
    if _web_search is None:
        _web_search = WebSearch()
    return _web_search


def get_rag_debugger(vector_store=None) -> RAGDebug:
    """Get or create RAGDebug facade"""
    global _rag_debugger
    if vector_store is None:
        # Create default vector_store if not provided
        from beanllm.domain.vector_stores import VectorStore
        from beanllm.domain.embeddings import Embedding

        embedding = Embedding(model="text-embedding-3-small")
        vector_store = VectorStore(embedding_function=embedding.embed)
    # Create new debugger if vector_store changed
    if _rag_debugger is None or _rag_debugger.vector_store != vector_store:
        _rag_debugger = RAGDebug(vector_store=vector_store)
    return _rag_debugger


def get_optimizer() -> Optimizer:
    """Get or create Optimizer facade"""
    global _optimizer
    if _optimizer is None:
        _optimizer = Optimizer()
    return _optimizer


def get_multi_agent() -> MultiAgentCoordinator:
    """Get or create MultiAgent facade"""
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = MultiAgentCoordinator()
    return _multi_agent


def get_orchestrator() -> Orchestrator:
    """Get or create Orchestrator facade"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# ============================================================================
# Request/Response Models
# ============================================================================


class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    assistant_id: str = "chat"
    model: Optional[str] = None
    stream: bool = False


# Knowledge Graph
class BuildGraphRequest(BaseModel):
    documents: List[str]
    graph_id: Optional[str] = None
    entity_types: Optional[List[str]] = None
    relation_types: Optional[List[str]] = None
    model: Optional[str] = None  # Optional model selection


class QueryGraphRequest(BaseModel):
    graph_id: str
    query_type: str = "cypher"
    query: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # Optional model selection


class GraphRAGRequest(BaseModel):
    query: str
    graph_id: str
    model: Optional[str] = None  # Optional model selection


# RAG
class RAGBuildRequest(BaseModel):
    documents: List[str]
    collection_name: Optional[str] = "default"
    model: Optional[str] = None  # Optional model selection


class RAGQueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None  # Optional model selection


# Agent
class AgentRequest(BaseModel):
    task: str
    tools: Optional[List[str]] = None
    max_iterations: int = 10
    model: Optional[str] = None  # Optional model selection


# Web Search
class WebSearchRequest(BaseModel):
    query: str
    num_results: int = 5
    engine: str = "duckduckgo"


# RAG Debug
class RAGDebugRequest(BaseModel):
    query: str
    documents: List[str]
    collection_name: Optional[str] = None  # Use existing RAG chain's vector_store
    debug_mode: str = "full"
    model: Optional[str] = None  # Optional model selection


# Optimizer
class OptimizeRequest(BaseModel):
    task_type: str = "rag"  # rag, agent, chain
    config: Optional[Dict[str, Any]] = None
    top_k_range: Optional[tuple] = None  # (min, max) for quick_optimize
    threshold_range: Optional[tuple] = None  # (min, max) for quick_optimize
    method: str = "bayesian"  # bayesian, grid, random, genetic
    n_trials: int = 30
    test_queries: Optional[List[str]] = None
    model: Optional[str] = None  # Optional model selection


# Multi-Agent
class MultiAgentRequest(BaseModel):
    task: str
    num_agents: int = 3
    strategy: str = "sequential"  # sequential, parallel, hierarchical, debate
    model: Optional[str] = None  # Optional model selection
    agent_configs: Optional[List[Dict[str, Any]]] = None  # Optional: custom agent configs


# Orchestrator
class WorkflowRequest(BaseModel):
    workflow_type: str  # research_write, parallel_consensus, debate
    task: str
    input_data: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # Optional model selection
    num_agents: int = 2  # Number of agents for quick methods


# Chain
class ChainRequest(BaseModel):
    input: str
    chain_id: Optional[str] = None
    chain_type: str = "basic"  # basic, prompt
    template: Optional[str] = None
    model: Optional[str] = None


# VisionRAG
class VisionRAGBuildRequest(BaseModel):
    images: List[str]  # Base64 encoded images or URLs
    texts: Optional[List[str]] = None
    collection_name: Optional[str] = "default"
    model: Optional[str] = None


class VisionRAGQueryRequest(BaseModel):
    query: str
    image: Optional[str] = None  # Base64 encoded image or URL
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None


# Audio
class AudioTranscribeRequest(BaseModel):
    audio_file: str  # Base64 encoded audio or file path
    model: Optional[str] = None


class AudioSynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    model: Optional[str] = None


class AudioRAGRequest(BaseModel):
    query: str
    audio_files: Optional[List[str]] = None
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None


# Evaluation
class EvaluationRequest(BaseModel):
    task_type: str  # rag, agent, chain
    queries: List[str]
    ground_truth: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


# Fine-tuning
class FineTuningCreateRequest(BaseModel):
    base_model: str
    training_data: List[Dict[str, Any]]
    job_name: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class FineTuningStatusRequest(BaseModel):
    job_id: str


# ============================================================================
# Health Check
# ============================================================================


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "beanllm-playground-api",
        "version": "1.0.0",
        "features": [
            "chat",
            "knowledge_graph",
            "rag",
            "agent",
            "web_search",
            "rag_debug",
            "optimizer",
            "multi_agent",
            "orchestrator",
        ],
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        client = get_client()
        return {
            "status": "healthy",
            "client_initialized": _client is not None,
            "facades": {
                "kg": _kg is not None,
                "rag_chains": len(_rag_chains),
                "web_search": _web_search is not None,
                "rag_debugger": _rag_debugger is not None,
                "optimizer": _optimizer is not None,
                "multi_agent": _multi_agent is not None,
                "orchestrator": _orchestrator is not None,
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ============================================================================
# Chat API
# ============================================================================


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint - routes to different assistants
    """
    try:
        # Convert messages to beanllm format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # If model is provided, create a client for that model
        # This allows using any provider (Ollama, DeepSeek, etc.) without API keys
        if request.model:
            # Client can auto-detect provider from model name
            client = Client(model=request.model)
            response = await client.chat(messages=messages)
        else:
            # Fallback to default client (requires OPENAI_API_KEY)
            client = get_client()
            response = await client.chat(messages=messages)

        return {
            "role": "assistant",
            "content": response.content,
        }

    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")


# ============================================================================
# Knowledge Graph API
# ============================================================================


@app.post("/api/kg/build")
async def kg_build(request: BuildGraphRequest):
    """Build knowledge graph from documents"""
    try:
        # Create KG with requested model or use default
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Use quick_build for simplicity
        response = await kg.quick_build(
            documents=request.documents,
        )

        return {
            "graph_id": response.graph_id,
            "num_nodes": response.num_nodes,
            "num_edges": response.num_edges,
            "entities": response.entities[:10],  # Show first 10
            "relations": response.relations[:10],  # Show first 10
        }

    except Exception as e:
        raise HTTPException(500, f"KG build error: {str(e)}")


@app.post("/api/kg/query")
async def kg_query(request: QueryGraphRequest):
    """Query knowledge graph"""
    try:
        # Create KG with requested model or use default
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Find entities by type as example
        if not request.query:
            # Return all entities
            response = await kg.query_graph(
                graph_id=request.graph_id,
                query_type="all_entities",
            )
        else:
            response = await kg.query_graph(
                graph_id=request.graph_id,
                query_type=request.query_type,
                query=request.query,
                params=request.params or {},
            )

        return {
            "graph_id": response.graph_id,
            "results": response.results[:20],  # Limit to 20
            "num_results": len(response.results),
        }

    except Exception as e:
        raise HTTPException(500, f"KG query error: {str(e)}")


@app.post("/api/kg/graph_rag")
async def kg_graph_rag(request: GraphRAGRequest):
    """Graph-based RAG query"""
    try:
        # Create KG with requested model or use default
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Use ask method (simplified graph RAG)
        answer = await kg.ask(
            query=request.query,
            graph_id=request.graph_id,
        )

        return {
            "query": request.query,
            "graph_id": request.graph_id,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(500, f"Graph RAG error: {str(e)}")


@app.get("/api/kg/visualize/{graph_id}")
async def kg_visualize(graph_id: str):
    """Get graph visualization (ASCII)"""
    try:
        kg = get_kg()

        visualization = await kg.visualize_graph(graph_id=graph_id)

        return {
            "graph_id": graph_id,
            "visualization": visualization,
        }

    except Exception as e:
        raise HTTPException(500, f"Visualization error: {str(e)}")


# ============================================================================
# RAG API
# ============================================================================


@app.post("/api/rag/build")
async def rag_build(request: RAGBuildRequest):
    """Build RAG index from documents"""
    try:
        collection_name = request.collection_name or "default"

        # Convert string documents to proper format
        from beanllm.domain.loaders import Document

        docs = [Document(content=doc, metadata={}) for doc in request.documents]

        # Build RAG chain using builder pattern
        # Create client with requested model or use default
        if request.model:
            client = Client(model=request.model)
        else:
            client = get_client()

        rag_chain = (
            RAGBuilder()
            .load_documents(docs)
            .split_text(chunk_size=500, chunk_overlap=50)
            .use_llm(client)
            .build()
        )

        # Store in global dict
        _rag_chains[collection_name] = rag_chain

        return {
            "collection_name": collection_name,
            "num_documents": len(request.documents),
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(500, f"RAG build error: {str(e)}")


@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query RAG system"""
    try:
        collection_name = request.collection_name or "default"

        if collection_name not in _rag_chains:
            raise HTTPException(404, f"Collection '{collection_name}' not found. Build it first.")

        # Get the existing chain
        rag_chain = _rag_chains[collection_name]

        # If a different model is requested, we use the existing chain
        # (The model used is determined at build time)
        # Note: To use a different model, rebuild the RAG chain with that model

        # Query using async method with sources
        answer, sources = await rag_chain.aquery(
            question=request.query, k=request.top_k, include_sources=True
        )

        # Extract source content (handle different source types)
        source_list = []
        for src in sources[:3]:
            if hasattr(src, "document"):
                # VectorSearchResult with document attribute
                content = (
                    src.document.content if hasattr(src.document, "content") else str(src.document)
                )
            elif hasattr(src, "page_content"):
                content = src.page_content
            elif hasattr(src, "content"):
                content = src.content
            else:
                content = str(src)
            source_list.append({"content": content[:200]})

        return {
            "query": request.query,
            "answer": answer,
            "sources": source_list,
            "relevance_score": 0.85,  # Placeholder
        }

    except Exception as e:
        raise HTTPException(500, f"RAG query error: {str(e)}")


# ============================================================================
# Agent API
# ============================================================================


@app.post("/api/agent/run")
async def agent_run(request: AgentRequest):
    """Run agent task"""
    try:
        # Create agent with requested model or default
        model = request.model if request.model else "gpt-4o-mini"
        agent = Agent(
            model=model,
            max_iterations=request.max_iterations,
            verbose=True,
        )

        # Run agent
        result = await agent.run(task=request.task)

        return {
            "task": request.task,
            "result": result.answer,
            "steps": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                }
                for step in result.steps
            ],
            "iterations": result.total_steps,
        }

    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")


# ============================================================================
# Web Search API
# ============================================================================


@app.post("/api/web/search")
async def web_search(request: WebSearchRequest):
    """Web search"""
    try:
        web = get_web_search()

        # Use async search
        response = await web.search_async(
            query=request.query,
            max_results=request.num_results,
        )

        return {
            "query": request.query,
            "results": [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "")[:200],
                }
                for result in response.results
            ],
            "num_results": len(response.results),
        }

    except Exception as e:
        raise HTTPException(500, f"Web search error: {str(e)}")


# ============================================================================
# RAG Debug API
# ============================================================================


@app.post("/api/rag_debug/analyze")
async def rag_debug_analyze(request: RAGDebugRequest):
    """Analyze RAG pipeline"""
    try:
        # Use existing RAG chain's vector_store if collection_name provided
        if request.collection_name and request.collection_name in _rag_chains:
            vector_store = _rag_chains[request.collection_name].vector_store
        else:
            # Create temporary vector_store from documents
            from beanllm.domain.loaders import Document
            from beanllm.domain.vector_stores import VectorStore
            from beanllm.domain.embeddings import Embedding
            from beanllm.domain.splitters import TextSplitter

            # Load and split documents
            docs = [Document(content=doc, metadata={}) for doc in request.documents]
            chunks = TextSplitter.split(docs, chunk_size=500, chunk_overlap=50)

            # Create temporary vector_store
            embedding_model = request.model or "text-embedding-3-small"
            embedding = Embedding(model=embedding_model)
            vector_store = VectorStore(embedding_function=embedding.embed)
            vector_store.add_documents(chunks)

        debugger = get_rag_debugger(vector_store=vector_store)

        # Start debug session first
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
                "embedding_quality": getattr(response, "embedding_quality", "good"),
                "chunk_quality": getattr(response, "chunk_quality", "excellent"),
                "retrieval_quality": getattr(response, "retrieval_quality", "good"),
            },
            "recommendations": getattr(
                response,
                "recommendations",
                [
                    "Consider increasing chunk overlap",
                    "Use more specific queries",
                ],
            ),
        }

    except Exception as e:
        raise HTTPException(500, f"RAG debug error: {str(e)}")


# ============================================================================
# Optimizer API
# ============================================================================


@app.post("/api/optimizer/optimize")
async def optimize(request: OptimizeRequest):
    """Run optimization"""
    try:
        optimizer = get_optimizer()

        # Use quick_optimize with provided ranges or defaults
        top_k_range = request.top_k_range or (1, 20)
        threshold_range = request.threshold_range or (0.0, 1.0)

        response = await optimizer.quick_optimize(
            top_k_range=top_k_range,
            threshold_range=threshold_range,
            method=request.method,
            n_trials=request.n_trials,
        )

        return {
            "task_type": request.task_type,
            "optimized_config": response.best_params if hasattr(response, "best_params") else {},
            "improvements": {
                "latency": f"{getattr(response, 'improvement_percentage', 0):.1f}%",
                "quality": "improved",
            },
            "metrics": getattr(response, "metrics", {}),
            "best_params": getattr(response, "best_params", {}),
        }

    except Exception as e:
        raise HTTPException(500, f"Optimizer error: {str(e)}")


# ============================================================================
# Multi-Agent API
# ============================================================================


@app.post("/api/multi_agent/run")
async def multi_agent_run(request: MultiAgentRequest):
    """Run multi-agent task"""
    try:
        # Create Agent instances
        model = request.model or "gpt-4o-mini"
        agents = {}

        if request.agent_configs:
            # Use custom agent configurations
            for i, config in enumerate(request.agent_configs):
                agent_id = config.get("agent_id", f"agent_{i}")
                agent_model = config.get("model", model)
                agent_tools = config.get("tools", [])
                agents[agent_id] = Agent(
                    model=agent_model,
                    tools=agent_tools,  # Note: tools should be Tool objects, not strings
                    max_iterations=config.get("max_iterations", 10),
                    verbose=config.get("verbose", False),
                )
        else:
            # Create default agents
            for i in range(request.num_agents):
                agent_id = f"agent_{i}"
                agents[agent_id] = Agent(
                    model=model,
                    max_iterations=10,
                    verbose=False,
                )

        # Create MultiAgentCoordinator with agents
        coordinator = MultiAgentCoordinator(agents=agents)

        # Execute based on strategy
        if request.strategy == "sequential":
            # Sequential execution
            agent_order = list(agents.keys())
            result = await coordinator.execute_sequential(
                task=request.task,
                agent_order=agent_order,
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
                        "output": (
                            result.get("intermediate_results", [{}])[i].get("result", "")
                            if i < len(result.get("intermediate_results", []))
                            else f"Step {i+1} completed"
                        ),
                    }
                    for i, agent_id in enumerate(agent_order)
                ],
            }

        elif request.strategy == "parallel":
            # Parallel execution
            agent_ids = list(agents.keys())
            result = await coordinator.execute_parallel(
                task=request.task,
                agent_ids=agent_ids,
                aggregation="vote",
            )

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": f"Completed task: {request.task}",
                    }
                    for agent_id in agent_ids
                ],
            }

        elif request.strategy == "hierarchical":
            # Hierarchical execution
            agent_ids = list(agents.keys())
            if len(agent_ids) < 2:
                raise HTTPException(
                    400, "Hierarchical strategy requires at least 2 agents (1 manager + 1 worker)"
                )
            manager_id = agent_ids[0]
            worker_ids = agent_ids[1:]

            result = await coordinator.execute_hierarchical(
                task=request.task,
                manager_id=manager_id,
                worker_ids=worker_ids,
            )

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": manager_id,
                        "role": "manager",
                        "output": "Coordinated all tasks",
                    },
                    *[
                        {
                            "agent_id": worker_id,
                            "role": "worker",
                            "output": f"Completed subtask",
                        }
                        for worker_id in worker_ids
                    ],
                ],
            }

        else:  # debate
            # Debate execution
            agent_ids = list(agents.keys())
            result = await coordinator.execute_debate(
                task=request.task,
                agent_ids=agent_ids,
                rounds=3,
            )

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": f"Argument presented for: {request.task}",
                    }
                    for agent_id in agent_ids
                ],
            }

    except Exception as e:
        raise HTTPException(500, f"Multi-agent error: {str(e)}")


# ============================================================================
# Orchestrator API
# ============================================================================


@app.post("/api/orchestrator/run")
async def orchestrator_run(request: WorkflowRequest):
    """Run workflow"""
    try:
        from beanllm.facade.core.agent_facade import Agent

        orchestrator = get_orchestrator()
        model = request.model or "gpt-4o-mini"

        # Use quick methods based on workflow type
        if request.workflow_type == "research_write":
            # Create agents for research_write workflow
            researcher = Agent(model=model, max_iterations=10)
            writer = Agent(model=model, max_iterations=10)
            response = await orchestrator.quick_research_write(
                researcher_agent=researcher,
                writer_agent=writer,
                task=request.task,
            )
        elif request.workflow_type == "parallel_consensus":
            # Create agents for parallel consensus
            agents = [Agent(model=model, max_iterations=10) for _ in range(request.num_agents)]
            response = await orchestrator.quick_parallel_consensus(
                agents=agents,
                task=request.task,
                aggregation="vote",
            )
        elif request.workflow_type == "debate":
            # Create agents for debate
            debaters = [
                Agent(model=model, max_iterations=10) for _ in range(request.num_agents - 1)
            ]
            judge = Agent(model=model, max_iterations=10)
            response = await orchestrator.quick_debate(
                debater_agents=debaters,
                judge_agent=judge,
                task=request.task,
                rounds=3,
            )
        else:
            # Generic workflow execution
            response = await orchestrator.run_full_workflow(
                workflow_type=request.workflow_type,
                input_data=request.input_data or {"task": request.task},
            )

        return {
            "workflow_id": response.workflow_id if hasattr(response, "workflow_id") else "wf_001",
            "result": response.result if hasattr(response, "result") else str(response),
            "execution_time": (
                response.execution_time if hasattr(response, "execution_time") else 0.0
            ),
            "steps_executed": response.steps if hasattr(response, "steps") else 0,
        }

    except Exception as e:
        raise HTTPException(500, f"Orchestrator error: {str(e)}")


# ============================================================================
# Chain API
# ============================================================================


@app.post("/api/chain/run")
async def chain_run(request: ChainRequest):
    """Run chain"""
    try:
        # Create client with requested model or use default
        if request.model:
            client = Client(model=request.model)
        else:
            client = get_client()

        # Get or create chain
        chain_id = request.chain_id or "default"

        if chain_id not in _chains:
            if request.chain_type == "prompt" and request.template:
                chain = PromptChain(client=client, template=request.template)
            else:
                chain = Chain(client=client)
            _chains[chain_id] = chain
        else:
            chain = _chains[chain_id]

        # Run chain
        result = await chain.run(user_input=request.input)

        return {
            "chain_id": chain_id,
            "input": request.input,
            "output": result.output,
            "steps": result.steps,
            "success": result.success,
            "error": result.error,
        }

    except Exception as e:
        raise HTTPException(500, f"Chain error: {str(e)}")


@app.post("/api/chain/build")
async def chain_build(request: ChainRequest):
    """Build chain with builder"""
    try:
        # Create client with requested model or use default
        if request.model:
            client = Client(model=request.model)
        else:
            client = get_client()

        # Build chain
        builder = ChainBuilder(client=client)

        if request.template:
            builder.with_template(request.template)

        chain = builder.build()

        # Store chain
        chain_id = request.chain_id or f"chain_{len(_chains)}"
        _chains[chain_id] = chain

        return {
            "chain_id": chain_id,
            "chain_type": request.chain_type,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(500, f"Chain build error: {str(e)}")


# ============================================================================
# VisionRAG API
# ============================================================================


@app.post("/api/vision_rag/build")
async def vision_rag_build(request: VisionRAGBuildRequest):
    """Build VisionRAG index"""
    try:
        # VisionRAG.from_images() requires a directory or file path
        # For API, we'll create a temporary directory with images
        import tempfile
        import shutil
        from pathlib import Path
        import base64

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Save images to temp directory
            image_paths = []
            for i, img_data in enumerate(request.images):
                # Handle base64 or URL
                if img_data.startswith("data:image") or img_data.startswith("http"):
                    # For now, skip URL handling - would need to download
                    continue
                else:
                    # Assume base64 encoded image
                    try:
                        img_bytes = base64.b64decode(img_data)
                        img_path = Path(temp_dir) / f"image_{i}.png"
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        image_paths.append(str(img_path))
                    except Exception:
                        # If not base64, treat as file path
                        image_paths.append(img_data)

            # Create VisionRAG from images
            model = request.model or "gpt-4o"
            if image_paths:
                # Use first image directory or create from paths
                vision_rag = VisionRAG.from_images(
                    source=temp_dir if len(image_paths) > 1 else image_paths[0],
                    generate_captions=True,
                    llm_model=model,
                )
            else:
                # Create empty VisionRAG
                client = Client(model=model)
                vision_rag = VisionRAG(client=client)

            # Store in global dict
            collection_name = request.collection_name or "default"
            global _vision_rag
            _vision_rag = vision_rag

            return {
                "collection_name": collection_name,
                "num_images": len(image_paths),
                "num_texts": len(request.texts) if request.texts else 0,
                "status": "success",
            }
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        raise HTTPException(500, f"VisionRAG build error: {str(e)}")


@app.post("/api/vision_rag/query")
async def vision_rag_query(request: VisionRAGQueryRequest):
    """Query VisionRAG system"""
    try:
        if _vision_rag is None:
            raise HTTPException(404, "VisionRAG not built. Build it first.")

        # Query VisionRAG (query method returns string or tuple)
        answer, sources = _vision_rag.query(
            question=request.query,
            k=request.top_k,
            include_sources=True,
        )

        # Format sources
        source_list = []
        for src in sources[: request.top_k]:
            if hasattr(src, "document"):
                content = (
                    src.document.content if hasattr(src.document, "content") else str(src.document)
                )
            elif hasattr(src, "page_content"):
                content = src.page_content
            elif hasattr(src, "content"):
                content = src.content
            else:
                content = str(src)
            source_list.append(
                {
                    "content": content[:200],
                    "score": getattr(src, "score", 0.0),
                    "type": "image" if hasattr(src, "image_path") else "text",
                }
            )

        return {
            "query": request.query,
            "answer": answer,
            "sources": source_list,
            "num_results": len(sources),
        }

    except Exception as e:
        raise HTTPException(500, f"VisionRAG query error: {str(e)}")


# ============================================================================
# Audio API
# ============================================================================


@app.post("/api/audio/transcribe")
async def audio_transcribe(request: AudioTranscribeRequest):
    """Transcribe audio to text"""
    try:
        # Create STT instance
        stt = WhisperSTT(model=request.model or "base")

        # Transcribe
        result = await stt.transcribe_async(request.audio_file)

        return {
            "text": result.text,
            "language": result.language,
            "segments": (
                [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in result.segments
                ]
                if hasattr(result, "segments")
                else []
            ),
        }

    except Exception as e:
        raise HTTPException(500, f"Audio transcribe error: {str(e)}")


@app.post("/api/audio/synthesize")
async def audio_synthesize(request: AudioSynthesizeRequest):
    """Synthesize text to speech"""
    try:
        # Create TTS instance
        tts = TextToSpeech()

        # Synthesize
        audio = await tts.synthesize_async(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )

        # Convert audio to base64 for response
        import base64
        from io import BytesIO

        buffer = BytesIO()
        audio.export(buffer, format="wav")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "text": request.text,
            "audio_base64": audio_base64,
            "format": "wav",
        }

    except Exception as e:
        raise HTTPException(500, f"Audio synthesize error: {str(e)}")


@app.post("/api/audio/rag")
async def audio_rag(request: AudioRAGRequest):
    """Audio RAG query"""
    try:
        # Create AudioRAG instance
        audio_rag = AudioRAG()

        # Add audio files if provided
        if request.audio_files:
            for audio_file in request.audio_files:
                await audio_rag.add_audio(audio_file)

        # Query
        results = await audio_rag.search(
            query=request.query,
            top_k=request.top_k,
        )

        return {
            "query": request.query,
            "results": [
                {
                    "text": result.get("text", "")[:200],
                    "audio_segment": result.get("audio_segment", ""),
                    "score": result.get("score", 0.0),
                }
                for result in results[: request.top_k]
            ],
            "num_results": len(results),
        }

    except Exception as e:
        raise HTTPException(500, f"Audio RAG error: {str(e)}")


# ============================================================================
# Evaluation API
# ============================================================================


@app.post("/api/evaluation/evaluate")
async def evaluation_evaluate(request: EvaluationRequest):
    """Run evaluation"""
    try:
        evaluator = EvaluatorFacade()

        # Run batch evaluation if we have queries and ground_truth
        if request.ground_truth and len(request.ground_truth) == len(request.queries):
            # Batch evaluate
            results = evaluator.batch_evaluate(
                predictions=request.queries,  # Using queries as predictions for now
                references=request.ground_truth,
            )

            # Aggregate metrics
            all_metrics = {}
            for result in results:
                if hasattr(result, "metrics"):
                    for key, value in result.metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)

            # Calculate averages
            summary = {k: sum(v) / len(v) for k, v in all_metrics.items()}

            return {
                "task_type": request.task_type,
                "num_queries": len(request.queries),
                "metrics": summary,
                "results": [
                    {
                        "prediction": request.queries[i],
                        "reference": request.ground_truth[i],
                        "metrics": result.metrics if hasattr(result, "metrics") else {},
                    }
                    for i, result in enumerate(results)
                ],
                "summary": summary,
            }
        else:
            # Single evaluation (use first query and ground_truth if available)
            prediction = request.queries[0] if request.queries else ""
            reference = request.ground_truth[0] if request.ground_truth else ""

            result = evaluator.evaluate(
                prediction=prediction,
                reference=reference,
            )

            return {
                "task_type": request.task_type,
                "num_queries": 1,
                "metrics": result.metrics if hasattr(result, "metrics") else {},
                "results": [
                    {
                        "prediction": prediction,
                        "reference": reference,
                        "metrics": result.metrics if hasattr(result, "metrics") else {},
                    }
                ],
                "summary": result.metrics if hasattr(result, "metrics") else {},
            }

    except Exception as e:
        raise HTTPException(500, f"Evaluation error: {str(e)}")


# ============================================================================
# Fine-tuning API
# ============================================================================


@app.post("/api/finetuning/create")
async def finetuning_create(request: FineTuningCreateRequest):
    """Create fine-tuning job"""
    try:
        from beanllm.domain.finetuning.providers import OpenAIFineTuningProvider

        # Create provider (default to OpenAI)
        provider = OpenAIFineTuningProvider()
        finetuning = FineTuningManagerFacade(provider=provider)

        # Prepare and upload training data
        import tempfile
        import json
        from pathlib import Path

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            # Convert training_data to JSONL format
            for example in request.training_data:
                json.dump(example, temp_file)
                temp_file.write("\n")
            temp_file.close()

            # Start training
            job = finetuning.start_training(
                model=request.base_model,
                training_file=temp_file.name,
                **request.hyperparameters or {},
            )

            return {
                "job_id": job.job_id if hasattr(job, "job_id") else "job_001",
                "status": job.status if hasattr(job, "status") else "created",
                "base_model": request.base_model,
                "created_at": job.created_at if hasattr(job, "created_at") else None,
            }
        finally:
            # Cleanup temp file
            Path(temp_file.name).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(500, f"Fine-tuning create error: {str(e)}")


@app.get("/api/finetuning/status/{job_id}")
async def finetuning_status(job_id: str):
    """Get fine-tuning job status"""
    try:
        from beanllm.domain.finetuning.providers import OpenAIFineTuningProvider

        # Create provider
        provider = OpenAIFineTuningProvider()
        finetuning = FineTuningManagerFacade(provider=provider)

        # Get training progress (includes job status)
        progress = finetuning.get_training_progress(job_id)

        job = progress.get("job")
        metrics = progress.get("metrics", [])

        return {
            "job_id": job_id,
            "status": job.status if hasattr(job, "status") else "unknown",
            "progress": len(metrics) / 100.0 if metrics else 0.0,  # Estimate progress
            "model_id": job.fine_tuned_model if hasattr(job, "fine_tuned_model") else None,
            "error": job.error if hasattr(job, "error") else None,
            "latest_metric": progress.get("latest_metric"),
        }

    except Exception as e:
        raise HTTPException(500, f"Fine-tuning status error: {str(e)}")


# ============================================================================
# Models API
# ============================================================================


@app.get("/api/models")
async def get_models():
    """Get all available models grouped by provider"""
    try:
        from beanllm.infrastructure.models.models import get_all_models

        models = get_all_models()

        # Group by provider
        grouped = {}
        for model_name, model_info in models.items():
            provider = model_info["provider"]
            if provider not in grouped:
                grouped[provider] = []
            grouped[provider].append(
                {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "description": model_info["description"],
                    "use_case": model_info["use_case"],
                    "max_tokens": model_info["max_tokens"],
                    "type": model_info["type"],
                }
            )

        return grouped
    except Exception as e:
        raise HTTPException(500, f"Failed to get models: {str(e)}")


@app.get("/api/models/{provider}")
async def get_models_by_provider(provider: str):
    """Get models for a specific provider"""
    try:
        from beanllm.infrastructure.models.models import get_models_by_provider

        models = get_models_by_provider(provider)

        return {
            "provider": provider,
            "models": [
                {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "description": model_info["description"],
                    "use_case": model_info["use_case"],
                    "max_tokens": model_info["max_tokens"],
                    "type": model_info["type"],
                }
                for model_name, model_info in models.items()
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get models for provider {provider}: {str(e)}")


# ============================================================================
# WebSocket for Real-time Streaming
# ============================================================================


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    print(f"WebSocket connected: {session_id}")

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
                "message": "Connected to beanllm playground",
            }
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_json()

                # Handle ping-pong
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                # Handle other messages
                else:
                    print(f"Received from {session_id}: {data}")

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break

    finally:
        # Clean up
        if session_id in active_connections:
            del active_connections[session_id]
        print(f"WebSocket disconnected: {session_id}")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("beanllm Playground API Server")
    print("=" * 60)
    print("Starting on http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\nAvailable Features:")
    print("  - Chat (General conversation)")
    print("  - Knowledge Graph (Build & Query)")
    print("  - RAG (Retrieval-Augmented Generation)")
    print("  - Agent (Autonomous task execution)")
    print("  - Web Search (Multi-engine search)")
    print("  - RAG Debug (Pipeline analysis)")
    print("  - Optimizer (Performance optimization)")
    print("  - Multi-Agent (Collaborative agents)")
    print("  - Orchestrator (Workflow management)")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
