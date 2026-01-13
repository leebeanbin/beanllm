"""Direct RAG test without FastAPI"""
import sys
sys.path.insert(0, "/Users/leejungbin/Downloads/llmkit/src")

from beanllm import Client
from beanllm.facade.core.rag_facade import RAGBuilder
from beanllm.domain.loaders import Document

# Create client
client = Client(model="qwen2.5:0.5b")

# Create documents
docs = [
    Document(content="beanllm은 통합 LLM 관리 도구입니다.", metadata={}),
    Document(content="주요 기능: RAG, Agent, Knowledge Graph", metadata={}),
]

print("Building RAG chain...")
try:
    rag_chain = (
        RAGBuilder()
        .load_documents(docs)
        .split_text(chunk_size=100, chunk_overlap=20)
        .use_llm(client)
        .build()
    )
    print("SUCCESS! RAG chain built")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
