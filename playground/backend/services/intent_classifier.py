"""
Intent Classifier Service

Analyzes user input and classifies intent to route to appropriate tools/features.
Uses a hybrid approach: rule-based for speed, LLM-based for complex cases.
"""

import logging
import re
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Supported intent types for agentic routing."""

    # Basic
    CHAT = "chat"                        # General conversation

    # RAG & Search
    RAG = "rag"                          # Document retrieval & QA
    WEB_SEARCH = "web_search"            # Web search

    # Agents
    AGENT = "agent"                      # Tool-using agent
    MULTI_AGENT = "multi_agent"          # Multi-agent coordination

    # Knowledge
    KNOWLEDGE_GRAPH = "kg"               # Knowledge graph operations

    # Google Services
    GOOGLE_DRIVE = "google_drive"        # Google Drive operations
    GOOGLE_DOCS = "google_docs"          # Google Docs operations
    GOOGLE_GMAIL = "google_gmail"        # Gmail operations
    GOOGLE_CALENDAR = "google_calendar"  # Calendar operations
    GOOGLE_SHEETS = "google_sheets"      # Sheets operations

    # Media
    AUDIO = "audio"                      # Audio processing (Whisper, TTS)
    VISION = "vision"                    # Image analysis
    OCR = "ocr"                          # Text extraction from images

    # Code & Analysis
    CODE = "code"                        # Code generation/analysis
    EVALUATION = "evaluation"            # Model/RAG evaluation


@dataclass
class IntentResult:
    """Result of intent classification."""

    primary_intent: IntentType
    confidence: float  # 0.0 - 1.0
    secondary_intents: List[IntentType] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "secondary_intents": [i.value for i in self.secondary_intents],
            "extracted_entities": self.extracted_entities,
            "requires_confirmation": self.requires_confirmation,
            "reasoning": self.reasoning,
        }


# ===========================================
# Keyword Rules for Rule-Based Classification
# ===========================================

INTENT_KEYWORDS: Dict[IntentType, List[str]] = {
    IntentType.RAG: [
        # Korean
        "문서", "pdf", "파일", "검색", "찾아", "조회", "참고", "읽어",
        "자료", "데이터", "분석", "요약",
        # English
        "document", "file", "search", "find", "lookup", "reference",
        "retrieve", "data", "analyze", "summarize", "pdf",
    ],

    IntentType.WEB_SEARCH: [
        # Korean
        "웹", "인터넷", "검색", "최신", "뉴스", "온라인", "사이트",
        # English
        "web", "internet", "online", "news", "latest", "current",
        "search online", "google", "look up online",
    ],

    IntentType.AGENT: [
        # Korean
        "에이전트", "도구", "실행", "자동", "작업", "수행",
        # English
        "agent", "tool", "execute", "automate", "perform", "run",
        "use tool", "call function",
    ],

    IntentType.MULTI_AGENT: [
        # Korean
        "토론", "협업", "여러", "다중", "팀", "논쟁", "debating",
        # English
        "debate", "collaborate", "multiple", "team", "discuss",
        "agents", "multi-agent", "coordination",
    ],

    IntentType.KNOWLEDGE_GRAPH: [
        # Korean
        "지식 그래프", "그래프", "관계", "노드", "엔티티", "연결",
        # English
        "knowledge graph", "graph", "relationship", "entity",
        "node", "connection", "neo4j", "kg",
    ],

    IntentType.GOOGLE_DRIVE: [
        # Korean
        "드라이브", "구글 드라이브", "google drive", "내 드라이브",
        "파일 목록", "드라이브에서",
        # English
        "drive", "my drive", "google drive", "gdrive",
    ],

    IntentType.GOOGLE_DOCS: [
        # Korean
        "문서 작성", "구글 문서", "google docs", "독스",
        "문서 편집", "문서 만들기",
        # English
        "docs", "google docs", "document create", "write document",
    ],

    IntentType.GOOGLE_GMAIL: [
        # Korean
        "이메일", "메일", "gmail", "지메일", "편지", "수신함",
        "발송", "보내기",
        # English
        "email", "mail", "gmail", "inbox", "send mail", "compose",
    ],

    IntentType.GOOGLE_CALENDAR: [
        # Korean
        "일정", "캘린더", "calendar", "스케줄", "예약", "약속",
        "미팅", "회의",
        # English
        "calendar", "schedule", "event", "meeting", "appointment",
        "book", "reserve",
    ],

    IntentType.GOOGLE_SHEETS: [
        # Korean
        "스프레드시트", "시트", "sheets", "엑셀", "표", "데이터",
        # English
        "spreadsheet", "sheets", "excel", "table", "csv",
    ],

    IntentType.AUDIO: [
        # Korean
        "음성", "오디오", "듣기", "녹음", "전사", "음악", "소리",
        "말하기", "읽어줘", "tts",
        # English
        "audio", "voice", "listen", "transcribe", "whisper",
        "speech", "tts", "text to speech", "speak",
    ],

    IntentType.VISION: [
        # Korean
        "이미지", "사진", "그림", "비전", "보기", "시각",
        # English
        "image", "picture", "photo", "vision", "visual", "see",
        "look at", "analyze image",
    ],

    IntentType.OCR: [
        # Korean
        "ocr", "텍스트 추출", "글자 인식", "스캔", "이미지에서 텍스트",
        # English
        "ocr", "extract text", "text recognition", "scan",
        "read text from image",
    ],

    IntentType.CODE: [
        # Korean
        "코드", "프로그래밍", "개발", "함수", "클래스", "버그",
        "디버그", "리팩토링",
        # English
        "code", "programming", "function", "class", "debug",
        "refactor", "implement", "develop", "script",
    ],

    IntentType.EVALUATION: [
        # Korean
        "평가", "테스트", "검증", "성능", "정확도", "벤치마크",
        # English
        "evaluate", "test", "benchmark", "accuracy", "performance",
        "metrics", "score",
    ],
}

# Keywords that strongly indicate specific intents (higher weight)
STRONG_INDICATORS: Dict[IntentType, List[str]] = {
    IntentType.RAG: ["pdf", "문서에서", "파일에서", "from document", "in the file"],
    IntentType.WEB_SEARCH: ["최신", "뉴스", "latest news", "current", "today"],
    IntentType.GOOGLE_DRIVE: ["내 드라이브", "my drive", "google drive"],
    IntentType.GOOGLE_GMAIL: ["이메일 보내", "send email", "gmail"],
    IntentType.AUDIO: ["음성으로", "읽어줘", "speak", "transcribe"],
    IntentType.OCR: ["이미지에서 텍스트", "extract text from"],
}


class IntentClassifier:
    """
    Hybrid intent classifier using rules and optional LLM.

    Usage:
        classifier = IntentClassifier()
        result = await classifier.classify("내 드라이브에서 회의록 찾아줘")
        print(result.primary_intent)  # IntentType.GOOGLE_DRIVE
    """

    def __init__(self, use_llm_fallback: bool = True):
        """
        Initialize the classifier.

        Args:
            use_llm_fallback: Whether to use LLM for uncertain cases
        """
        self.use_llm_fallback = use_llm_fallback
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[IntentType, List[re.Pattern]]:
        """Pre-compile regex patterns for performance."""
        patterns = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            patterns[intent] = [
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                for kw in keywords
            ]
        return patterns

    async def classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_intents: Optional[List[IntentType]] = None,
    ) -> IntentResult:
        """
        Classify user query to determine intent.

        Args:
            query: User's natural language query
            context: Optional context (previous messages, current state, etc.)
            available_intents: Optional list of allowed intents (for filtering)

        Returns:
            IntentResult with classified intent and confidence
        """
        # Step 1: Rule-based classification
        rule_result = self._classify_by_rules(query)

        # Step 2: Filter by available intents if specified
        if available_intents:
            if rule_result.primary_intent not in available_intents:
                # Find best available intent
                for intent in rule_result.secondary_intents:
                    if intent in available_intents:
                        rule_result.primary_intent = intent
                        rule_result.confidence *= 0.8  # Reduce confidence
                        break
                else:
                    # Default to CHAT if no match
                    rule_result.primary_intent = IntentType.CHAT
                    rule_result.confidence = 0.5

        # Step 3: LLM fallback for low confidence
        if self.use_llm_fallback and rule_result.confidence < 0.6:
            try:
                llm_result = await self._classify_by_llm(query, context, available_intents)
                if llm_result.confidence > rule_result.confidence:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed, using rule result: {e}")

        return rule_result

    def _classify_by_rules(self, query: str) -> IntentResult:
        """
        Rule-based classification using keyword matching.

        Returns:
            IntentResult with scores based on keyword matches
        """
        query_lower = query.lower()
        scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

        # Score based on keyword matches
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    scores[intent] += 1.0

        # Bonus for strong indicators
        for intent, indicators in STRONG_INDICATORS.items():
            for indicator in indicators:
                if indicator.lower() in query_lower:
                    scores[intent] += 2.0

        # Extract entities
        entities = self._extract_entities(query)

        # Sort by score
        sorted_intents = sorted(
            [(intent, score) for intent, score in scores.items() if score > 0],
            key=lambda x: x[1],
            reverse=True
        )

        if not sorted_intents:
            # Default to CHAT with medium confidence
            return IntentResult(
                primary_intent=IntentType.CHAT,
                confidence=0.7,
                secondary_intents=[],
                extracted_entities=entities,
                reasoning="No specific keywords matched, defaulting to chat"
            )

        # Calculate confidence based on score distribution
        max_score = sorted_intents[0][1]
        total_score = sum(s for _, s in sorted_intents)
        confidence = min(max_score / max(total_score, 1) * 0.8 + 0.2, 0.95)

        primary = sorted_intents[0][0]
        secondary = [intent for intent, _ in sorted_intents[1:3]]

        return IntentResult(
            primary_intent=primary,
            confidence=confidence,
            secondary_intents=secondary,
            extracted_entities=entities,
            reasoning=f"Matched keywords for {primary.value} with score {max_score}"
        )

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract relevant entities from the query."""
        entities = {}

        # Extract file extensions
        file_ext_pattern = r"\.(pdf|docx?|xlsx?|pptx?|txt|md|csv|json)\b"
        file_matches = re.findall(file_ext_pattern, query, re.IGNORECASE)
        if file_matches:
            entities["file_types"] = list(set(file_matches))

        # Extract URLs
        url_pattern = r"https?://[^\s]+"
        url_matches = re.findall(url_pattern, query)
        if url_matches:
            entities["urls"] = url_matches

        # Extract email addresses
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        email_matches = re.findall(email_pattern, query)
        if email_matches:
            entities["emails"] = email_matches

        # Extract dates (simple patterns)
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # 2024-01-23
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # 1/23/2024
            r"(오늘|내일|어제|today|tomorrow|yesterday)",
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities["dates"] = entities.get("dates", []) + list(matches)

        return entities

    async def _classify_by_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_intents: Optional[List[IntentType]] = None,
    ) -> IntentResult:
        """
        LLM-based classification for complex queries.

        Uses beanllm's Client for classification.
        """
        try:
            from beanllm import Client
        except ImportError:
            logger.warning("beanllm not installed, cannot use LLM classification")
            raise

        # Build intent options
        if available_intents:
            intent_options = [i.value for i in available_intents]
        else:
            intent_options = [i.value for i in IntentType]

        # Create classification prompt
        system_prompt = f"""You are an intent classifier. Classify the user's query into one of these intents:
{', '.join(intent_options)}

Respond with JSON only:
{{"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Add context if available
        if context and context.get("previous_messages"):
            context_str = "\n".join([
                f"{m['role']}: {m['content'][:100]}"
                for m in context["previous_messages"][-3:]
            ])
            messages[0]["content"] += f"\n\nRecent conversation:\n{context_str}"

        # Call LLM
        client = Client(model="gpt-4o-mini")  # Use fast model for classification
        response = await client.chat(messages, temperature=0.1)

        # Parse response
        import json
        try:
            result = json.loads(response.content)
            intent_str = result.get("intent", "chat")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            # Map to IntentType
            try:
                primary_intent = IntentType(intent_str)
            except ValueError:
                primary_intent = IntentType.CHAT
                confidence = 0.5

            return IntentResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=[],
                extracted_entities=self._extract_entities(query),
                reasoning=f"LLM: {reasoning}"
            )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response.content}")
            raise

    def get_required_keys_for_intent(self, intent: IntentType) -> List[str]:
        """
        Get required API keys for an intent.

        Returns:
            List of provider names that need API keys
        """
        requirements = {
            IntentType.CHAT: [],  # Ollama always available
            IntentType.RAG: [],  # Uses embeddings, might need OpenAI
            IntentType.WEB_SEARCH: ["tavily"],  # Or serpapi
            IntentType.AGENT: [],
            IntentType.MULTI_AGENT: [],
            IntentType.KNOWLEDGE_GRAPH: ["neo4j"],
            IntentType.GOOGLE_DRIVE: ["google_oauth"],
            IntentType.GOOGLE_DOCS: ["google_oauth"],
            IntentType.GOOGLE_GMAIL: ["google_oauth"],
            IntentType.GOOGLE_CALENDAR: ["google_oauth"],
            IntentType.GOOGLE_SHEETS: ["google_oauth"],
            IntentType.AUDIO: [],  # Whisper might need OpenAI
            IntentType.VISION: [],
            IntentType.OCR: [],
            IntentType.CODE: [],
            IntentType.EVALUATION: [],
        }
        return requirements.get(intent, [])


# Singleton instance
intent_classifier = IntentClassifier()
