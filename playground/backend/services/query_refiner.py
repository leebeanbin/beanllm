"""
QueryRefiner Service - 쿼리 재구성 및 확장

검색 품질 향상을 위한 쿼리 재구성, 확장, 동의어 생성
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RefinedQuery:
    """재구성된 쿼리 결과"""

    original: str
    refined: str
    expanded_queries: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    confidence: float = 1.0


class QueryRefiner:
    """
    쿼리 재구성 서비스

    기능:
    - 쿼리 정제 (불필요한 단어 제거)
    - 쿼리 확장 (동의어, 관련 용어)
    - 키워드 추출
    - 의도 파악
    - 컨텍스트 기반 재구성
    """

    # 불용어 (검색에 불필요한 단어)
    STOP_WORDS_KO = {
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "에서",
        "로",
        "으로",
        "와",
        "과",
        "도",
        "만",
        "까지",
        "부터",
        "에게",
        "한테",
        "께",
        "좀",
        "그냥",
        "제발",
        "혹시",
        "아마",
        "뭔가",
    }

    STOP_WORDS_EN = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "shall",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "been",
        "being",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "and",
        "any",
        "as",
        "at",
        "because",
        "before",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "here",
        "how",
        "if",
        "in",
        "into",
        "just",
        "more",
        "most",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "out",
        "over",
        "own",
        "same",
        "so",
        "some",
        "such",
        "than",
        "then",
        "there",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "when",
        "where",
        "while",
        "why",
        "with",
        "please",
        "just",
        "maybe",
        "perhaps",
    }

    # 질문 패턴
    QUESTION_PATTERNS = {
        "what": "정의/설명",
        "how": "방법/과정",
        "why": "이유/원인",
        "when": "시간/시기",
        "where": "장소/위치",
        "who": "인물/주체",
        "which": "선택/비교",
        "뭐": "정의/설명",
        "어떻게": "방법/과정",
        "왜": "이유/원인",
        "언제": "시간/시기",
        "어디": "장소/위치",
        "누가": "인물/주체",
    }

    # 도메인별 동의어/관련어
    SYNONYMS = {
        # 프로그래밍
        "코드": ["code", "소스코드", "프로그램", "스크립트"],
        "함수": ["function", "메서드", "method", "procedure"],
        "에러": ["error", "오류", "버그", "bug", "exception"],
        "설치": ["install", "설정", "setup", "configuration"],
        "api": ["API", "엔드포인트", "endpoint", "interface"],
        # AI/ML
        "모델": ["model", "LLM", "AI", "인공지능"],
        "학습": ["training", "학습", "fine-tuning", "튜닝"],
        "임베딩": ["embedding", "벡터", "vector", "representation"],
        "rag": ["RAG", "검색증강생성", "retrieval augmented generation"],
        # 일반
        "문서": ["document", "문서", "파일", "file", "docs"],
        "데이터": ["data", "데이터", "정보", "information"],
    }

    def __init__(self, enable_llm_expansion: bool = False):
        """
        Args:
            enable_llm_expansion: LLM을 사용한 쿼리 확장 활성화
        """
        self._enable_llm = enable_llm_expansion

    def refine(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RefinedQuery:
        """
        쿼리 재구성

        Args:
            query: 원본 쿼리
            context: 컨텍스트 정보 (이전 대화, 세션 정보 등)

        Returns:
            재구성된 쿼리 결과
        """
        # 1. 기본 정제
        refined = self._clean_query(query)

        # 2. 키워드 추출
        keywords = self._extract_keywords(refined)

        # 3. 의도 파악
        intent = self._detect_intent(query)

        # 4. 쿼리 확장
        expanded = self._expand_query(refined, keywords, context)

        # 5. 컨텍스트 반영
        if context:
            refined = self._apply_context(refined, context)

        return RefinedQuery(
            original=query,
            refined=refined,
            expanded_queries=expanded,
            keywords=keywords,
            intent=intent,
            confidence=self._calculate_confidence(query, refined),
        )

    def _clean_query(self, query: str) -> str:
        """쿼리 정제 (불필요한 문자/단어 제거)"""
        # 1. 앞뒤 공백 제거
        cleaned = query.strip()

        # 2. 중복 공백 제거
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 3. 특수문자 정리 (유지: 알파벳, 숫자, 한글, 공백, 일부 기호)
        cleaned = re.sub(r"[^\w\s가-힣\-_.,?!]", "", cleaned)

        # 4. 불용어 제거 (옵션)
        words = cleaned.split()
        words = [
            w for w in words if w.lower() not in self.STOP_WORDS_EN and w not in self.STOP_WORDS_KO
        ]

        return " ".join(words) if words else cleaned

    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        words = query.split()

        # 불용어 제외
        keywords = [
            w
            for w in words
            if len(w) > 1 and w.lower() not in self.STOP_WORDS_EN and w not in self.STOP_WORDS_KO
        ]

        # 중복 제거 (순서 유지)
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)

        return unique_keywords

    def _detect_intent(self, query: str) -> Optional[str]:
        """의도 파악"""
        query_lower = query.lower()

        for pattern, intent in self.QUESTION_PATTERNS.items():
            if pattern in query_lower:
                return intent

        # 기본 의도 추측
        if "?" in query or "뭐" in query or "어떻게" in query:
            return "질문"
        if any(kw in query_lower for kw in ["해줘", "해", "하세요", "please", "can you"]):
            return "요청"
        if any(kw in query_lower for kw in ["찾아", "검색", "search", "find"]):
            return "검색"

        return None

    def _expand_query(
        self,
        query: str,
        keywords: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """쿼리 확장 (동의어, 관련어 추가)"""
        expanded = []

        # 1. 동의어 기반 확장
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.SYNONYMS:
                for synonym in self.SYNONYMS[keyword_lower]:
                    expanded_query = query.replace(keyword, synonym)
                    if expanded_query != query and expanded_query not in expanded:
                        expanded.append(expanded_query)

        # 2. 키워드 조합
        if len(keywords) >= 2:
            # 키워드만 조합한 쿼리
            keyword_query = " ".join(keywords)
            if keyword_query not in expanded:
                expanded.append(keyword_query)

        # 3. 컨텍스트 기반 확장
        if context:
            previous_queries = context.get("previous_queries", [])
            for prev_query in previous_queries[-2:]:  # 최근 2개
                # 이전 쿼리 키워드와 현재 키워드 조합
                prev_keywords = self._extract_keywords(prev_query)
                combined = list(set(keywords + prev_keywords[:2]))
                combined_query = " ".join(combined)
                if combined_query not in expanded:
                    expanded.append(combined_query)

        return expanded[:5]  # 최대 5개

    def _apply_context(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """컨텍스트 반영"""
        # 이전 대화에서 언급된 주제 반영
        previous_topics = context.get("topics", [])
        if previous_topics and len(query.split()) <= 3:
            # 짧은 쿼리면 주제 추가
            topic = previous_topics[-1]
            if topic.lower() not in query.lower():
                return f"{topic} {query}"

        return query

    def _calculate_confidence(self, original: str, refined: str) -> float:
        """재구성 신뢰도 계산"""
        if original == refined:
            return 1.0

        # 변경 비율 계산
        original_words = set(original.lower().split())
        refined_words = set(refined.lower().split())

        if not original_words:
            return 0.5

        overlap = len(original_words & refined_words)
        ratio = overlap / len(original_words)

        return round(ratio, 2)

    async def refine_with_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        model: str = "qwen2.5:0.5b",
    ) -> RefinedQuery:
        """
        LLM을 사용한 고급 쿼리 재구성

        Args:
            query: 원본 쿼리
            context: 컨텍스트 정보
            model: 사용할 모델

        Returns:
            재구성된 쿼리 결과
        """
        # 기본 재구성 먼저 수행
        result = self.refine(query, context)

        if not self._enable_llm:
            return result

        try:
            from beanllm.facade.core import Client

            client = Client(model=model)

            # 프롬프트 구성
            context_info = ""
            if context:
                previous = context.get("previous_queries", [])
                if previous:
                    context_info = f"\n이전 질문들: {', '.join(previous[-3:])}"

            prompt = f"""다음 사용자 질문을 검색에 최적화된 형태로 재구성해주세요.
{context_info}

원본 질문: {query}

응답 형식:
재구성된 질문: [더 명확하고 검색에 적합한 질문]
키워드: [핵심 키워드들, 쉼표로 구분]
확장 질문: [관련된 추가 질문 1-2개, 쉼표로 구분]"""

            response = await client.chat([{"role": "user", "content": prompt}])

            # 응답 파싱
            content = response.content

            # 재구성된 질문 추출
            if "재구성된 질문:" in content:
                refined_match = re.search(
                    r"재구성된 질문:\s*(.+?)(?:\n|키워드:|$)",
                    content,
                    re.DOTALL,
                )
                if refined_match:
                    result.refined = refined_match.group(1).strip()

            # 키워드 추출
            if "키워드:" in content:
                keywords_match = re.search(
                    r"키워드:\s*(.+?)(?:\n|확장|$)",
                    content,
                    re.DOTALL,
                )
                if keywords_match:
                    keywords = [k.strip() for k in keywords_match.group(1).split(",") if k.strip()]
                    result.keywords = keywords

            # 확장 질문 추출
            if "확장 질문:" in content:
                expanded_match = re.search(
                    r"확장 질문:\s*(.+?)$",
                    content,
                    re.DOTALL,
                )
                if expanded_match:
                    expanded = [q.strip() for q in expanded_match.group(1).split(",") if q.strip()]
                    result.expanded_queries = expanded

            logger.info(f"LLM query refinement: '{query}' → '{result.refined}'")

        except Exception as e:
            logger.warning(f"LLM query refinement failed, using basic: {e}")

        return result

    def generate_multi_query(
        self,
        query: str,
        num_queries: int = 3,
    ) -> List[str]:
        """
        하이브리드 검색을 위한 다중 쿼리 생성

        Args:
            query: 원본 쿼리
            num_queries: 생성할 쿼리 수

        Returns:
            다중 쿼리 리스트
        """
        queries = [query]  # 원본 포함

        # 재구성
        refined = self.refine(query)

        # 재구성된 쿼리 추가
        if refined.refined != query:
            queries.append(refined.refined)

        # 확장 쿼리 추가
        queries.extend(refined.expanded_queries)

        # 키워드만으로 쿼리
        if refined.keywords:
            keyword_query = " ".join(refined.keywords[:3])
            if keyword_query not in queries:
                queries.append(keyword_query)

        # 중복 제거 및 개수 제한
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries[:num_queries]


# Singleton instance
query_refiner = QueryRefiner()
