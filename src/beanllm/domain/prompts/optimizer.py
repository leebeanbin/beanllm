"""
Prompts Optimizer - 프롬프트 최적화 도구
"""

import json
from typing import Any, Dict, List, Optional


class PromptOptimizer:
    """
    프롬프트 최적화 도구

    프롬프트를 자동으로 개선합니다.
    """

    @staticmethod
    def add_instructions(prompt: str, instructions: List[str]) -> str:
        """명령어 추가"""
        instruction_text = "\n".join(f"- {inst}" for inst in instructions)
        return f"{prompt}\n\nInstructions:\n{instruction_text}"

    @staticmethod
    def add_constraints(prompt: str, constraints: List[str]) -> str:
        """제약조건 추가"""
        constraint_text = "\n".join(f"- {const}" for const in constraints)
        return f"{prompt}\n\nConstraints:\n{constraint_text}"

    @staticmethod
    def add_output_format(
        prompt: str, format_description: str, example: Optional[str] = None
    ) -> str:
        """출력 포맷 명시"""
        result = f"{prompt}\n\nOutput Format:\n{format_description}"
        if example:
            result += f"\n\nExample Output:\n{example}"
        return result

    @staticmethod
    def add_json_output(prompt: str, schema: Dict[str, Any]) -> str:
        """JSON 출력 형식 추가"""
        schema_str = json.dumps(schema, indent=2)
        return f"{prompt}\n\nPlease respond in JSON format:\n{schema_str}"

    @staticmethod
    def add_thinking_process(prompt: str) -> str:
        """사고 과정 요청 추가"""
        return (
            f"{prompt}\n\n"
            "Please think step-by-step:\n"
            "1. Analyze the problem\n"
            "2. Consider possible solutions\n"
            "3. Choose the best approach\n"
            "4. Provide your answer"
        )

    @staticmethod
    def add_role_context(prompt: str, role: str, expertise: List[str]) -> str:
        """역할 컨텍스트 추가"""
        expertise_text = ", ".join(expertise)
        role_prompt = f"You are a {role} with expertise in {expertise_text}.\n\n{prompt}"
        return role_prompt
