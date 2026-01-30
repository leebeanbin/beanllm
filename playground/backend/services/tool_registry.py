"""
Tool Registry Service

beanllm의 기능들을 도구로 래핑하고 관리하는 레지스트리.
Intent Classifier가 분류한 의도에 따라 적절한 도구를 선택하여 실행.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from services.intent_classifier import IntentType

logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    """도구 상태"""
    AVAILABLE = "available"      # 사용 가능
    UNAVAILABLE = "unavailable"  # 의존성 없음
    REQUIRES_KEY = "requires_key"  # API 키 필요
    ERROR = "error"              # 오류 상태


@dataclass
class ToolRequirement:
    """도구 실행 요구사항"""
    api_keys: List[str] = field(default_factory=list)  # 필요한 API 키
    packages: List[str] = field(default_factory=list)  # 필요한 패키지
    services: List[str] = field(default_factory=list)  # 필요한 서비스 (mongo, redis 등)


@dataclass
class Tool:
    """도구 정의"""
    name: str                           # 도구 이름 (고유)
    description: str                    # 도구 설명
    description_ko: str                 # 한국어 설명
    intent_types: List[IntentType]      # 지원하는 의도 타입
    requirements: ToolRequirement       # 실행 요구사항
    facade_class: Optional[str] = None  # beanllm Facade 클래스 경로
    handler: Optional[Callable] = None  # 커스텀 핸들러 (없으면 Facade 사용)
    is_streaming: bool = False          # 스트리밍 지원 여부
    priority: int = 0                   # 우선순위 (높을수록 우선)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "description_ko": self.description_ko,
            "intent_types": [it.value for it in self.intent_types],
            "requirements": {
                "api_keys": self.requirements.api_keys,
                "packages": self.requirements.packages,
                "services": self.requirements.services,
            },
            "is_streaming": self.is_streaming,
            "priority": self.priority,
        }


@dataclass
class ToolCheckResult:
    """도구 사용 가능성 검사 결과"""
    tool: Tool
    status: ToolStatus
    missing_keys: List[str] = field(default_factory=list)
    missing_packages: List[str] = field(default_factory=list)
    missing_services: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.status == ToolStatus.AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool.to_dict(),
            "status": self.status.value,
            "is_available": self.is_available,
            "missing_keys": self.missing_keys,
            "missing_packages": self.missing_packages,
            "missing_services": self.missing_services,
            "error_message": self.error_message,
        }


# ===========================================
# Tool Definitions (beanllm Facade 연동)
# ===========================================

REGISTERED_TOOLS: List[Tool] = [
    # ===== Core Tools =====
    Tool(
        name="chat",
        description="Basic LLM chat conversation",
        description_ko="기본 LLM 대화",
        intent_types=[IntentType.CHAT],
        requirements=ToolRequirement(
            api_keys=[],  # Ollama는 키 불필요
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.core.Client",
        is_streaming=True,
        priority=100,
    ),

    Tool(
        name="rag",
        description="Document retrieval and Q&A with RAG",
        description_ko="RAG 기반 문서 검색 및 질의응답",
        intent_types=[IntentType.RAG],
        requirements=ToolRequirement(
            api_keys=[],  # Ollama 임베딩 사용 가능
            packages=["beanllm", "chromadb"],
        ),
        facade_class="beanllm.facade.core.RAGChain",
        is_streaming=True,
        priority=90,
    ),

    Tool(
        name="agent",
        description="Tool-using agent for automated tasks",
        description_ko="도구 사용 에이전트",
        intent_types=[IntentType.AGENT],
        requirements=ToolRequirement(
            api_keys=[],
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.core.Agent",
        is_streaming=True,
        priority=85,
    ),

    # ===== Advanced Tools =====
    Tool(
        name="multi_agent",
        description="Multi-agent debate and collaboration",
        description_ko="멀티 에이전트 토론/협업",
        intent_types=[IntentType.MULTI_AGENT],
        requirements=ToolRequirement(
            api_keys=[],
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.advanced.MultiAgent",
        is_streaming=True,
        priority=80,
    ),

    Tool(
        name="knowledge_graph",
        description="Knowledge graph operations with Neo4j",
        description_ko="지식 그래프 (Neo4j)",
        intent_types=[IntentType.KNOWLEDGE_GRAPH],
        requirements=ToolRequirement(
            api_keys=["neo4j"],
            packages=["beanllm", "neo4j"],
            services=["neo4j"],
        ),
        facade_class="beanllm.facade.advanced.KnowledgeGraph",
        is_streaming=False,
        priority=70,
    ),

    # ===== Web Search =====
    Tool(
        name="web_search",
        description="Web search with Tavily or SerpAPI",
        description_ko="웹 검색 (Tavily/SerpAPI)",
        intent_types=[IntentType.WEB_SEARCH],
        requirements=ToolRequirement(
            api_keys=["tavily"],  # 또는 serpapi
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.ml.WebSearch",
        is_streaming=False,
        priority=75,
    ),

    # ===== Google Services =====
    Tool(
        name="google_drive",
        description="Google Drive file operations",
        description_ko="Google Drive 파일 관리",
        intent_types=[IntentType.GOOGLE_DRIVE],
        requirements=ToolRequirement(
            api_keys=["google_oauth"],
            packages=["beanllm", "google-api-python-client"],
        ),
        is_streaming=False,
        priority=60,
    ),

    Tool(
        name="google_docs",
        description="Google Docs document operations",
        description_ko="Google Docs 문서 관리",
        intent_types=[IntentType.GOOGLE_DOCS],
        requirements=ToolRequirement(
            api_keys=["google_oauth"],
            packages=["beanllm", "google-api-python-client"],
        ),
        is_streaming=False,
        priority=60,
    ),

    Tool(
        name="google_gmail",
        description="Gmail email operations",
        description_ko="Gmail 이메일 관리",
        intent_types=[IntentType.GOOGLE_GMAIL],
        requirements=ToolRequirement(
            api_keys=["google_oauth"],
            packages=["beanllm", "google-api-python-client"],
        ),
        is_streaming=False,
        priority=60,
    ),

    Tool(
        name="google_calendar",
        description="Google Calendar event management",
        description_ko="Google Calendar 일정 관리",
        intent_types=[IntentType.GOOGLE_CALENDAR],
        requirements=ToolRequirement(
            api_keys=["google_oauth"],
            packages=["beanllm", "google-api-python-client"],
        ),
        is_streaming=False,
        priority=60,
    ),

    Tool(
        name="google_sheets",
        description="Google Sheets spreadsheet operations",
        description_ko="Google Sheets 스프레드시트 관리",
        intent_types=[IntentType.GOOGLE_SHEETS],
        requirements=ToolRequirement(
            api_keys=["google_oauth"],
            packages=["beanllm", "google-api-python-client"],
        ),
        is_streaming=False,
        priority=60,
    ),

    # ===== Media Tools =====
    Tool(
        name="audio_transcribe",
        description="Audio transcription with Whisper",
        description_ko="음성 전사 (Whisper)",
        intent_types=[IntentType.AUDIO],
        requirements=ToolRequirement(
            api_keys=[],  # 로컬 Whisper 사용 가능
            packages=["beanllm", "openai-whisper"],
        ),
        facade_class="beanllm.facade.ml.Audio",
        is_streaming=False,
        priority=50,
    ),

    Tool(
        name="vision",
        description="Image analysis and understanding",
        description_ko="이미지 분석",
        intent_types=[IntentType.VISION],
        requirements=ToolRequirement(
            api_keys=["openai"],  # GPT-4V 필요
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.ml.VisionRAG",
        is_streaming=False,
        priority=50,
    ),

    Tool(
        name="ocr",
        description="Text extraction from images",
        description_ko="이미지 텍스트 추출 (OCR)",
        intent_types=[IntentType.OCR],
        requirements=ToolRequirement(
            api_keys=[],  # Tesseract 로컬
            packages=["beanllm", "pytesseract"],
        ),
        is_streaming=False,
        priority=50,
    ),

    # ===== Code & Analysis =====
    Tool(
        name="code",
        description="Code generation and analysis",
        description_ko="코드 생성 및 분석",
        intent_types=[IntentType.CODE],
        requirements=ToolRequirement(
            api_keys=[],
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.core.Client",  # 코드 모드 Chat
        is_streaming=True,
        priority=70,
    ),

    Tool(
        name="evaluation",
        description="Model and RAG evaluation",
        description_ko="모델/RAG 평가",
        intent_types=[IntentType.EVALUATION],
        requirements=ToolRequirement(
            api_keys=[],
            packages=["beanllm"],
        ),
        facade_class="beanllm.facade.ml.Evaluation",
        is_streaming=False,
        priority=40,
    ),
]


class ToolRegistry:
    """
    도구 레지스트리 - 사용 가능한 도구 관리

    Usage:
        registry = ToolRegistry()
        tools = registry.get_tools_for_intent(IntentType.RAG)
        check = registry.check_requirements(tools[0])
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in REGISTERED_TOOLS}
        self._intent_map: Dict[IntentType, List[Tool]] = {}
        self._build_intent_map()

    def _build_intent_map(self):
        """Intent → Tool 매핑 구축"""
        for tool in self._tools.values():
            for intent_type in tool.intent_types:
                if intent_type not in self._intent_map:
                    self._intent_map[intent_type] = []
                self._intent_map[intent_type].append(tool)

        # 우선순위로 정렬
        for intent_type in self._intent_map:
            self._intent_map[intent_type].sort(key=lambda t: t.priority, reverse=True)

    def get_tool(self, name: str) -> Optional[Tool]:
        """이름으로 도구 조회"""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """모든 도구 조회"""
        return list(self._tools.values())

    def get_tools_for_intent(self, intent_type: IntentType) -> List[Tool]:
        """
        Intent에 맞는 도구 조회 (우선순위 순)

        Args:
            intent_type: Intent 타입

        Returns:
            해당 Intent를 지원하는 도구 목록
        """
        return self._intent_map.get(intent_type, [])

    def check_requirements(self, tool: Tool) -> ToolCheckResult:
        """
        도구 사용 가능 여부 검사

        Args:
            tool: 검사할 도구

        Returns:
            ToolCheckResult 객체
        """
        missing_keys = []
        missing_packages = []
        missing_services = []

        # 1. API 키 검사
        for key in tool.requirements.api_keys:
            if not self._check_api_key(key):
                missing_keys.append(key)

        # 2. 패키지 검사
        for package in tool.requirements.packages:
            if not self._check_package(package):
                missing_packages.append(package)

        # 3. 서비스 검사
        for service in tool.requirements.services:
            if not self._check_service(service):
                missing_services.append(service)

        # 상태 결정
        if missing_packages:
            status = ToolStatus.UNAVAILABLE
        elif missing_keys:
            status = ToolStatus.REQUIRES_KEY
        elif missing_services:
            status = ToolStatus.UNAVAILABLE
        else:
            status = ToolStatus.AVAILABLE

        return ToolCheckResult(
            tool=tool,
            status=status,
            missing_keys=missing_keys,
            missing_packages=missing_packages,
            missing_services=missing_services,
        )

    def _check_api_key(self, key_name: str) -> bool:
        """API 키 존재 여부 확인"""
        import os

        key_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "tavily": "TAVILY_API_KEY",
            "serpapi": "SERPAPI_API_KEY",
            "neo4j": "NEO4J_PASSWORD",
            "google_oauth": "GOOGLE_OAUTH_CLIENT_ID",
            "pinecone": "PINECONE_API_KEY",
            "qdrant": "QDRANT_API_KEY",
            "weaviate": "WEAVIATE_API_KEY",
        }

        env_var = key_env_map.get(key_name, f"{key_name.upper()}_API_KEY")
        return bool(os.getenv(env_var))

    def _check_package(self, package: str) -> bool:
        """패키지 설치 여부 확인"""
        import importlib.util

        # 특수 패키지 매핑
        package_map = {
            "openai-whisper": "whisper",
            "google-api-python-client": "googleapiclient",
        }

        import_name = package_map.get(package, package.replace("-", "_"))

        try:
            spec = importlib.util.find_spec(import_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False

    def _check_service(self, service: str) -> bool:
        """외부 서비스 연결 가능 여부 확인"""
        import os

        service_checks = {
            "neo4j": lambda: bool(os.getenv("NEO4J_URI")),
            "mongodb": lambda: bool(os.getenv("MONGODB_URI")),
            "redis": lambda: bool(os.getenv("REDIS_URL")),
            "kafka": lambda: bool(os.getenv("KAFKA_BOOTSTRAP_SERVERS")),
        }

        check_fn = service_checks.get(service)
        if check_fn:
            return check_fn()
        return False

    def get_available_tools(self) -> List[ToolCheckResult]:
        """
        사용 가능한 모든 도구 조회 (상태 포함)

        Returns:
            각 도구의 상태 목록
        """
        results = []
        for tool in self._tools.values():
            check = self.check_requirements(tool)
            results.append(check)

        # 상태 순 정렬: AVAILABLE > REQUIRES_KEY > UNAVAILABLE
        status_order = {
            ToolStatus.AVAILABLE: 0,
            ToolStatus.REQUIRES_KEY: 1,
            ToolStatus.UNAVAILABLE: 2,
            ToolStatus.ERROR: 3,
        }
        results.sort(key=lambda r: (status_order[r.status], -r.tool.priority))

        return results

    def get_best_tool_for_intent(
        self,
        intent_type: IntentType,
        only_available: bool = True
    ) -> Optional[ToolCheckResult]:
        """
        Intent에 가장 적합한 도구 선택

        Args:
            intent_type: Intent 타입
            only_available: True면 사용 가능한 도구만 반환

        Returns:
            가장 적합한 도구 또는 None
        """
        tools = self.get_tools_for_intent(intent_type)

        for tool in tools:
            check = self.check_requirements(tool)
            if only_available and not check.is_available:
                continue
            return check

        # 사용 가능한 도구가 없으면 첫 번째 도구라도 반환 (상태 표시용)
        if not only_available and tools:
            return self.check_requirements(tools[0])

        return None

    def register_tool(self, tool: Tool):
        """
        새 도구 등록

        Args:
            tool: 등록할 도구
        """
        self._tools[tool.name] = tool
        for intent_type in tool.intent_types:
            if intent_type not in self._intent_map:
                self._intent_map[intent_type] = []
            self._intent_map[intent_type].append(tool)
            self._intent_map[intent_type].sort(key=lambda t: t.priority, reverse=True)

        logger.info(f"✅ Registered tool: {tool.name}")


# Singleton instance
tool_registry = ToolRegistry()
