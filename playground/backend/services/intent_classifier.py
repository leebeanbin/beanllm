"""
Intent Classifier Service

Analyzes user input and classifies intent to route to appropriate tools/features.
Uses a hybrid approach: rule-based for speed, LLM-based for complex cases.

Features:
- Rule-based classification (fast, keyword matching)
- LLM-based classification (complex queries)
- QueryRefiner integration (query preprocessing)
- PromptBuilder integration (dynamic prompt generation)
- Context-aware classification
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Supported intent types for agentic routing."""

    # Basic
    CHAT = "chat"  # General conversation

    # RAG & Search
    RAG = "rag"  # Document retrieval & QA
    WEB_SEARCH = "web_search"  # Web search

    # Agents
    AGENT = "agent"  # Tool-using agent
    MULTI_AGENT = "multi_agent"  # Multi-agent coordination

    # Knowledge
    KNOWLEDGE_GRAPH = "kg"  # Knowledge graph operations

    # Google Services
    GOOGLE_DRIVE = "google_drive"  # Google Drive operations
    GOOGLE_DOCS = "google_docs"  # Google Docs operations
    GOOGLE_GMAIL = "google_gmail"  # Gmail operations
    GOOGLE_CALENDAR = "google_calendar"  # Calendar operations
    GOOGLE_SHEETS = "google_sheets"  # Sheets operations

    # Media
    AUDIO = "audio"  # Audio processing (Whisper, TTS)
    VISION = "vision"  # Image analysis
    OCR = "ocr"  # Text extraction from images

    # Code & Analysis
    CODE = "code"  # Code generation/analysis
    EVALUATION = "evaluation"  # Model/RAG evaluation


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
        "문서",
        "pdf",
        "파일",
        "검색",
        "찾아",
        "조회",
        "참고",
        "읽어",
        "자료",
        "데이터",
        "분석",
        "요약",
        # English
        "document",
        "file",
        "search",
        "find",
        "lookup",
        "reference",
        "retrieve",
        "data",
        "analyze",
        "summarize",
        "pdf",
    ],
    IntentType.WEB_SEARCH: [
        # Korean
        "웹",
        "인터넷",
        "검색",
        "최신",
        "뉴스",
        "온라인",
        "사이트",
        # English
        "web",
        "internet",
        "online",
        "news",
        "latest",
        "current",
        "search online",
        "google",
        "look up online",
    ],
    IntentType.AGENT: [
        # Korean
        "에이전트",
        "도구",
        "실행",
        "자동",
        "작업",
        "수행",
        # English
        "agent",
        "tool",
        "execute",
        "automate",
        "perform",
        "run",
        "use tool",
        "call function",
    ],
    IntentType.MULTI_AGENT: [
        # Korean
        "토론",
        "협업",
        "여러",
        "다중",
        "팀",
        "논쟁",
        "debating",
        # English
        "debate",
        "collaborate",
        "multiple",
        "team",
        "discuss",
        "agents",
        "multi-agent",
        "coordination",
    ],
    IntentType.KNOWLEDGE_GRAPH: [
        # Korean
        "지식 그래프",
        "그래프",
        "관계",
        "노드",
        "엔티티",
        "연결",
        # English
        "knowledge graph",
        "graph",
        "relationship",
        "entity",
        "node",
        "connection",
        "neo4j",
        "kg",
    ],
    IntentType.GOOGLE_DRIVE: [
        # Korean
        "드라이브",
        "구글 드라이브",
        "google drive",
        "내 드라이브",
        "파일 목록",
        "드라이브에서",
        # English
        "drive",
        "my drive",
        "google drive",
        "gdrive",
    ],
    IntentType.GOOGLE_DOCS: [
        # Korean
        "문서 작성",
        "구글 문서",
        "google docs",
        "독스",
        "문서 편집",
        "문서 만들기",
        # English
        "docs",
        "google docs",
        "document create",
        "write document",
    ],
    IntentType.GOOGLE_GMAIL: [
        # Korean
        "이메일",
        "메일",
        "gmail",
        "지메일",
        "편지",
        "수신함",
        "발송",
        "보내기",
        # English
        "email",
        "mail",
        "gmail",
        "inbox",
        "send mail",
        "compose",
    ],
    IntentType.GOOGLE_CALENDAR: [
        # Korean
        "일정",
        "캘린더",
        "calendar",
        "스케줄",
        "예약",
        "약속",
        "미팅",
        "회의",
        # English
        "calendar",
        "schedule",
        "event",
        "meeting",
        "appointment",
        "book",
        "reserve",
    ],
    IntentType.GOOGLE_SHEETS: [
        # Korean
        "스프레드시트",
        "시트",
        "sheets",
        "엑셀",
        "표",
        "데이터",
        # English
        "spreadsheet",
        "sheets",
        "excel",
        "table",
        "csv",
    ],
    IntentType.AUDIO: [
        # Korean
        "음성",
        "오디오",
        "듣기",
        "녹음",
        "전사",
        "음악",
        "소리",
        "말하기",
        "읽어줘",
        "tts",
        # English
        "audio",
        "voice",
        "listen",
        "transcribe",
        "whisper",
        "speech",
        "tts",
        "text to speech",
        "speak",
    ],
    IntentType.VISION: [
        # Korean
        "이미지",
        "사진",
        "그림",
        "비전",
        "보기",
        "시각",
        # English
        "image",
        "picture",
        "photo",
        "vision",
        "visual",
        "see",
        "look at",
        "analyze image",
    ],
    IntentType.OCR: [
        # Korean
        "ocr",
        "텍스트 추출",
        "글자 인식",
        "스캔",
        "이미지에서 텍스트",
        # English
        "ocr",
        "extract text",
        "text recognition",
        "scan",
        "read text from image",
    ],
    IntentType.CODE: [
        # Korean
        "코드",
        "프로그래밍",
        "개발",
        "함수",
        "클래스",
        "버그",
        "디버그",
        "리팩토링",
        # English
        "code",
        "programming",
        "function",
        "class",
        "debug",
        "refactor",
        "implement",
        "develop",
        "script",
    ],
    IntentType.EVALUATION: [
        # Korean
        "평가",
        "테스트",
        "검증",
        "성능",
        "정확도",
        "벤치마크",
        # English
        "evaluate",
        "test",
        "benchmark",
        "accuracy",
        "performance",
        "metrics",
        "score",
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

    Features:
    - Rule-based classification (keyword matching)
    - LLM-based classification (complex queries)
    - QueryRefiner integration (query preprocessing)
    - Context-aware classification (previous intents)

    Usage:
        classifier = IntentClassifier()
        result = await classifier.classify("내 드라이브에서 회의록 찾아줘")
        print(result.primary_intent)  # IntentType.GOOGLE_DRIVE
    """

    # 신뢰도 임계값
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_THRESHOLD = 0.4

    def __init__(
        self,
        use_llm_fallback: bool = True,
        use_query_refiner: bool = True,
    ):
        """
        Initialize the classifier.

        Args:
            use_llm_fallback: Whether to use LLM for uncertain cases
            use_query_refiner: Whether to use QueryRefiner for preprocessing
        """
        self.use_llm_fallback = use_llm_fallback
        self.use_query_refiner = use_query_refiner
        self._compiled_patterns = self._compile_patterns()
        self._intent_history: List[IntentType] = []  # 최근 의도 이력

    def _compile_patterns(self) -> Dict[IntentType, List[re.Pattern]]:
        """Pre-compile regex patterns for performance."""
        patterns = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            patterns[intent] = [
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords
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
        # Step 0: Query preprocessing with QueryRefiner
        refined_query = query
        keywords = []

        if self.use_query_refiner:
            try:
                from services.query_refiner import query_refiner

                refine_context = None
                if context and context.get("previous_queries"):
                    refine_context = {"previous_queries": context["previous_queries"]}

                refined = query_refiner.refine(query, refine_context)
                refined_query = refined.refined
                keywords = refined.keywords

                logger.debug(f"Query refined: '{query}' → '{refined_query}'")
            except Exception as e:
                logger.debug(f"QueryRefiner not available: {e}")

        # Step 1: Rule-based classification
        rule_result = self._classify_by_rules(refined_query)

        # Add extracted keywords to entities
        if keywords:
            rule_result.extracted_entities["keywords"] = keywords

        # Step 2: Context-based adjustment
        if context:
            rule_result = self._adjust_by_context(rule_result, context)

        # Step 3: Filter by available intents if specified
        if available_intents:
            rule_result = self._filter_by_available_intents(rule_result, available_intents)

        # Step 4: LLM fallback for low confidence
        if self.use_llm_fallback and rule_result.confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            try:
                llm_result = await self._classify_by_llm(query, context, available_intents)
                if llm_result.confidence > rule_result.confidence:
                    rule_result = llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed, using rule result: {e}")

        # Step 5: Update intent history
        self._intent_history.append(rule_result.primary_intent)
        if len(self._intent_history) > 10:
            self._intent_history = self._intent_history[-10:]

        return rule_result

    def _adjust_by_context(
        self,
        result: IntentResult,
        context: Dict[str, Any],
    ) -> IntentResult:
        """Adjust classification based on context."""
        # 이전 의도가 있으면 연속성 고려
        previous_intents = context.get("previous_intents", [])
        if previous_intents:
            last_intent_str = previous_intents[-1]
            try:
                last_intent = IntentType(last_intent_str)

                # 같은 의도가 연속되면 신뢰도 증가
                if result.primary_intent == last_intent:
                    result.confidence = min(result.confidence * 1.1, 0.95)
                    result.reasoning = f"{result.reasoning} (context boost: same intent)"

                # Google 서비스 연속 사용 시 연속성 유지
                google_intents = {
                    IntentType.GOOGLE_DRIVE,
                    IntentType.GOOGLE_DOCS,
                    IntentType.GOOGLE_GMAIL,
                    IntentType.GOOGLE_CALENDAR,
                    IntentType.GOOGLE_SHEETS,
                }
                if last_intent in google_intents and result.primary_intent in google_intents:
                    result.confidence = min(result.confidence * 1.05, 0.95)

            except ValueError:
                pass

        # 세션에 RAG 데이터가 있으면 RAG 의도 우선
        if context.get("has_rag_data") and result.confidence < 0.7:
            if "문서" in context.get("query", "") or "검색" in context.get("query", ""):
                result.secondary_intents.insert(0, IntentType.RAG)

        return result

    def _filter_by_available_intents(
        self,
        result: IntentResult,
        available_intents: List[IntentType],
    ) -> IntentResult:
        """Filter result by available intents."""
        if result.primary_intent not in available_intents:
            # Find best available intent from secondary
            for intent in result.secondary_intents:
                if intent in available_intents:
                    result.primary_intent = intent
                    result.confidence *= 0.8  # Reduce confidence
                    result.reasoning = f"{result.reasoning} (filtered to {intent.value})"
                    break
            else:
                # Default to CHAT if no match
                if IntentType.CHAT in available_intents:
                    result.primary_intent = IntentType.CHAT
                    result.confidence = 0.5
                    result.reasoning = "No matching intent, defaulting to chat"
                elif available_intents:
                    result.primary_intent = available_intents[0]
                    result.confidence = 0.4

        return result

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
            reverse=True,
        )

        if not sorted_intents:
            # Default to CHAT with medium confidence
            return IntentResult(
                primary_intent=IntentType.CHAT,
                confidence=0.7,
                secondary_intents=[],
                extracted_entities=entities,
                reasoning="No specific keywords matched, defaulting to chat",
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
            reasoning=f"Matched keywords for {primary.value} with score {max_score}",
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

        Uses beanllm's Client and PromptBuilder for classification.
        """
        try:
            from beanllm.facade.core import Client
        except ImportError:
            logger.warning("beanllm not installed, cannot use LLM classification")
            raise

        # Build intent options
        if available_intents:
            intent_options = [i.value for i in available_intents]
        else:
            intent_options = [i.value for i in IntentType]

        # Use PromptBuilder if available
        try:
            from services.prompt_builder import prompt_builder

            recent_intents = None
            if self._intent_history:
                recent_intents = [i.value for i in self._intent_history[-3:]]

            prompt = prompt_builder.build_intent_classification_prompt(
                query=query,
                available_intents=intent_options,
                recent_intents=recent_intents,
            )
            messages = [{"role": "user", "content": prompt}]

        except ImportError:
            # Fallback to basic prompt
            system_prompt = f"""You are an intent classifier. Classify the user's query into one of these intents:
{', '.join(intent_options)}

Respond with JSON only:
{{"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            # Add context if available
            if context and context.get("previous_messages"):
                context_str = "\n".join(
                    [
                        f"{m['role']}: {m['content'][:100]}"
                        for m in context["previous_messages"][-3:]
                    ]
                )
                messages[0]["content"] += f"\n\nRecent conversation:\n{context_str}"

        # Call LLM (use lightweight model for classification)
        try:
            client = Client(model="qwen2.5:0.5b")  # Ollama model for speed
            response = await client.chat(messages, temperature=0.1)
        except Exception:
            # Fallback to gpt-4o-mini if Ollama not available
            client = Client(model="gpt-4o-mini")
            response = await client.chat(messages, temperature=0.1)

        # Parse response
        try:
            # Try to extract JSON from response
            content = response.content
            json_match = re.search(r"\{[^{}]*\}", content)
            if json_match:
                content = json_match.group()

            result = json.loads(content)
            intent_str = result.get("intent", "chat")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            # Map to IntentType
            try:
                primary_intent = IntentType(intent_str)
            except ValueError:
                # Try partial match
                for intent in IntentType:
                    if intent.value in intent_str.lower():
                        primary_intent = intent
                        break
                else:
                    primary_intent = IntentType.CHAT
                    confidence = 0.5

            return IntentResult(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=[],
                extracted_entities=self._extract_entities(query),
                reasoning=f"LLM: {reasoning}",
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
