"""
MCP Server Configuration

환경 변수 및 기본 설정을 관리합니다.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MCPServerConfig:
    """MCP Server 설정"""

    # Server info
    SERVER_NAME: str = "beanllm-mcp-server"
    SERVER_VERSION: str = "0.1.0"
    SERVER_DESCRIPTION: str = "Model Context Protocol server wrapping beanllm functionality"

    # Host and port
    HOST: str = os.getenv("MCP_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("MCP_PORT", "8765"))

    # LLM Provider API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Database URLs (세션 관리)
    MONGODB_URI: Optional[str] = os.getenv("MONGODB_URI")
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")

    # Google Workspace
    GOOGLE_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")

    # Default models
    DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "qwen2.5:0.5b")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", "nomic-embed-text:latest"
    )

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_stores"

    # Chunk settings (RAG)
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Retrieval settings
    DEFAULT_TOP_K: int = 5

    # Session settings
    SESSION_TTL_SECONDS: int = 3600  # 1 hour

    @classmethod
    def ensure_directories(cls):
        """필수 디렉토리 생성"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(exist_ok=True)


# Initialize
MCPServerConfig.ensure_directories()
