# beanllm Playground Backend

FastAPI-based backend server for beanllm playground.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
OPENAI_API_KEY=sk-...
```

### 3. Start Server

```bash
python main.py
```

Server will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## API Endpoints

### Health Check

- `GET /` - Service status
- `GET /health` - Detailed health check with client initialization status

### Chat

```python
POST /api/chat
{
  "messages": [{"role": "user", "content": "Hello"}],
  "assistant_id": "chat",  # or "kg", "rag", "optimizer", "orchestrator"
  "stream": false
}
```

### Knowledge Graph

#### Build Graph

```python
POST /api/kg/build
{
  "documents": ["Document content..."],
  "graph_id": "my-graph",  # optional
  "entity_types": ["Person", "Organization"],  # optional
  "relation_types": ["WORKS_FOR", "KNOWS"]  # optional
}
```

Response:

```json
{
  "graph_id": "my-graph",
  "num_nodes": 10,
  "num_edges": 15,
  "density": 0.33,
  "num_connected_components": 1
}
```

#### Query Graph

```python
POST /api/kg/query
{
  "graph_id": "my-graph",
  "query_type": "cypher",
  "query": "MATCH (n) RETURN n LIMIT 10",
  "params": {}
}
```

#### Graph RAG

```python
POST /api/kg/graph_rag
{
  "query": "What do you know about John?",
  "graph_id": "my-graph"
}
```

#### Visualize Graph

```python
GET /api/kg/visualize/{graph_id}
```

Returns ASCII visualization of the graph.

### WebSocket

```python
WS /ws/{session_id}
```

Connect for real-time updates.

## Development

### Auto-Reload

The server runs with auto-reload enabled in development mode:

```python
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,  # Auto-reload on file changes
    log_level="info",
)
```

### Adding New Endpoints

Add new endpoints in `main.py`:

```python
@app.post("/api/new-feature")
async def new_feature(request: NewFeatureRequest):
    """New feature endpoint"""
    # Implementation
    return {"status": "ok"}
```

### Error Handling

All endpoints use FastAPI's `HTTPException` for error handling:

```python
from fastapi import HTTPException

raise HTTPException(500, f"Error message: {str(e)}")
```

## Tech Stack

- **FastAPI**: Modern async Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Request/response validation
- **WebSockets**: Real-time communication
- **beanllm**: Core LLM framework

## Notes

- All-in-one `main.py` for rapid development
- Global state for lazy initialization of Client and KnowledgeGraph
- CORS enabled for Next.js frontend (localhost:3000)
- Modular structure can be implemented later for production

## License

MIT
