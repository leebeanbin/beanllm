"""
PromptBuilder Service - 프롬프트 동적 구성

컨텍스트에 따라 요약, 검색, 채팅 프롬프트를 동적으로 구성합니다.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """프롬프트 템플릿"""

    template: str
    input_variables: List[str]

    def format(self, **kwargs) -> str:
        """템플릿에 변수 대입"""
        result = self.template
        for var in self.input_variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result


class PromptComposer:
    """프롬프트 조합기"""

    def __init__(self):
        self._parts: List[str] = []
        self._templates: List[PromptTemplate] = []

    def add_text(self, text: str) -> "PromptComposer":
        """텍스트 추가"""
        self._parts.append(text.strip())
        return self

    def add_template(self, template: PromptTemplate) -> "PromptComposer":
        """템플릿 추가"""
        self._templates.append(template)
        return self

    def compose(self, **kwargs) -> str:
        """모든 부분을 조합하여 최종 프롬프트 생성"""
        result_parts = self._parts.copy()

        for template in self._templates:
            result_parts.append(template.format(**kwargs))

        return "\n\n".join(result_parts)


class PromptOptimizer:
    """프롬프트 최적화기"""

    def add_instructions(
        self,
        base_prompt: str,
        instructions: List[str],
    ) -> str:
        """프롬프트에 추가 지시사항 추가"""
        if not instructions:
            return base_prompt

        instruction_text = "\n".join(f"- {inst}" for inst in instructions)
        return f"{base_prompt}\n\n추가 지시사항:\n{instruction_text}"

    def add_constraints(
        self,
        base_prompt: str,
        constraints: List[str],
    ) -> str:
        """프롬프트에 제약 조건 추가"""
        if not constraints:
            return base_prompt

        constraint_text = "\n".join(f"- {const}" for const in constraints)
        return f"{base_prompt}\n\n제약 조건:\n{constraint_text}"


class PromptBuilder:
    """
    프롬프트 동적 구성 서비스

    컨텍스트에 따라 요약, 검색, 채팅 프롬프트를 동적으로 구성합니다.
    """

    def __init__(self):
        self.optimizer = PromptOptimizer()
        self._summarization_template = PromptTemplate(
            template="""다음 대화 내용을 요약해주세요. 다음 정보는 반드시 포함해야 합니다:

1. 주요 주제 및 목적
2. 중요한 결정 사항
3. 사용자가 언급한 특별한 요구사항
4. 해결된 문제와 해결 방법
5. 미완료된 작업이나 다음 단계

대화 내용:
{conversation_history}

요약 (200-300자):""",
            input_variables=["conversation_history"],
        )

    def build_summarization_prompt(
        self,
        conversation_history: str,
        session_context: Optional[Dict[str, Any]] = None,
        previous_summaries: Optional[List[str]] = None,
    ) -> str:
        """
        요약 프롬프트 동적 구성

        Args:
            conversation_history: 대화 내용
            session_context: 세션 컨텍스트 (uploaded_files, intent_history 등)
            previous_summaries: 이전 요약들

        Returns:
            동적으로 구성된 요약 프롬프트
        """
        base_prompt = self._summarization_template.format(conversation_history=conversation_history)

        instructions = []

        # 세션 컨텍스트 반영
        if session_context:
            uploaded_files = session_context.get("uploaded_files", [])
            if uploaded_files:
                instructions.append(f"세션에 업로드된 파일: {', '.join(uploaded_files)}")

            tool_usage = session_context.get("tool_usage", [])
            if tool_usage:
                instructions.append(f"사용된 도구: {', '.join(set(tool_usage))}")

            intent_history = session_context.get("intent_history", [])
            if len(intent_history) > 1:
                instructions.append(f"의도 변화: {' → '.join(intent_history[-3:])}")

        # 이전 요약이 있으면 연속성 유지
        if previous_summaries and previous_summaries[-1]:
            instructions.append(f"이전 요약: {previous_summaries[-1][:200]}...")
            instructions.append("연속성을 유지하며 요약하세요.")

        if instructions:
            base_prompt = self.optimizer.add_instructions(base_prompt, instructions)

        return base_prompt

    def build_structured_summarization_prompt(
        self,
        conversation_history: str,
        intent_history: Optional[List[str]] = None,
        tool_usage_history: Optional[List[str]] = None,
    ) -> str:
        """
        구조화된 요약 프롬프트 (컨텍스트 기반)

        Args:
            conversation_history: 대화 내용
            intent_history: 의도 변화 이력
            tool_usage_history: 도구 사용 이력

        Returns:
            구조화된 요약 프롬프트
        """
        composer = PromptComposer()

        # 기본 구조
        composer.add_text("""다음 대화를 구조화된 형식으로 요약해주세요:

주제: [대화의 주요 주제]
목적: [사용자의 목적]
주요 내용:
- [핵심 포인트 1]
- [핵심 포인트 2]
- [핵심 포인트 3]
중요 정보: [보존해야 할 특별한 정보]
다음 단계: [미완료 작업이나 다음 단계]""")

        # 사용된 도구 정보 추가
        if tool_usage_history:
            unique_tools = list(set(tool_usage_history))
            composer.add_text(f"사용된 도구: {', '.join(unique_tools)}")

        # 의도 변화 추적
        if intent_history and len(intent_history) > 1:
            recent_intents = intent_history[-3:]
            composer.add_text(f"의도 변화: {' → '.join(recent_intents)}")

        # 대화 내용 템플릿
        composer.add_template(
            PromptTemplate(
                template="대화 내용:\n{conversation_history}",
                input_variables=["conversation_history"],
            )
        )

        return composer.compose(conversation_history=conversation_history)

    def build_context_prompt(
        self,
        summary: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        LLM에 전달할 컨텍스트 프롬프트 구성

        Args:
            summary: 이전 대화 요약
            session_context: 세션 컨텍스트

        Returns:
            컨텍스트 프롬프트
        """
        parts = [f"이전 대화 요약:\n{summary}"]

        if session_context:
            # 업로드된 파일 정보
            uploaded_files = session_context.get("uploaded_files", [])
            if uploaded_files:
                parts.append(f"세션 파일: {', '.join(uploaded_files)}")

            # RAG 컬렉션 정보
            rag_collection = session_context.get("rag_collection")
            if rag_collection:
                parts.append(f"RAG 컬렉션: {rag_collection}")

            # Google 연동 정보
            google_connected = session_context.get("google_connected", False)
            if google_connected:
                parts.append("Google 서비스 연결됨")

        parts.append("\n최근 대화:")

        return "\n".join(parts)

    def build_rag_query_prompt(
        self,
        query: str,
        context_documents: List[str],
        session_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        RAG 쿼리 프롬프트 구성

        Args:
            query: 사용자 쿼리
            context_documents: 검색된 문서들
            session_context: 세션 컨텍스트

        Returns:
            RAG 쿼리 프롬프트
        """
        context_text = "\n\n---\n\n".join(context_documents)

        prompt = f"""다음 문서를 참고하여 질문에 답변해주세요.

참고 문서:
{context_text}

질문: {query}

답변:"""

        # 세션 컨텍스트 반영
        if session_context:
            instructions = []

            previous_queries = session_context.get("previous_queries", [])
            if previous_queries:
                instructions.append(f"이전 질문들: {', '.join(previous_queries[-3:])}")

            if instructions:
                prompt = self.optimizer.add_instructions(prompt, instructions)

        return prompt

    def build_intent_classification_prompt(
        self,
        query: str,
        available_intents: List[str],
        recent_intents: Optional[List[str]] = None,
    ) -> str:
        """
        의도 분류 프롬프트 구성

        Args:
            query: 사용자 쿼리
            available_intents: 사용 가능한 의도 목록
            recent_intents: 최근 의도 이력

        Returns:
            의도 분류 프롬프트
        """
        intent_list = "\n".join(f"- {intent}" for intent in available_intents)

        prompt = f"""사용자 쿼리의 의도를 분류해주세요.

사용 가능한 의도:
{intent_list}

사용자 쿼리: {query}

가장 적합한 의도를 선택하고, 신뢰도(0.0-1.0)와 함께 응답해주세요.
형식: {{"intent": "...", "confidence": 0.X, "reasoning": "..."}}"""

        # 최근 의도 이력 반영
        if recent_intents:
            prompt = self.optimizer.add_instructions(
                prompt,
                [f"최근 의도 이력: {' → '.join(recent_intents[-3:])}"],
            )

        return prompt


# Singleton instance
prompt_builder = PromptBuilder()
