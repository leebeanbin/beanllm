"""
Prompts 테스트 - 프롬프트 시스템 테스트
"""

import pytest
from unittest.mock import Mock


class TestPromptComposer:
    """PromptComposer 테스트"""

    def test_prompt_composer_compose(self):
        """프롬프트 작성 테스트"""
        try:
            from llmkit.domain.prompts.composer import PromptComposer
            from llmkit.domain.prompts.templates import PromptTemplate

            composer = PromptComposer()
            template = PromptTemplate(template="Hello {name}", input_variables=["name"])
            composer.add_template(template)
            prompt = composer.compose(name="World")

            assert isinstance(prompt, str)
            assert "World" in prompt
        except ImportError:
            pytest.skip("PromptComposer not available")


class TestPromptFactory:
    """PromptFactory 테스트"""

    def test_prompt_factory_create(self):
        """프롬프트 생성 테스트"""
        try:
            from llmkit.domain.prompts.factory import create_prompt_template

            template = create_prompt_template(
                template="Test {variable}",
                input_variables=["variable"],
            )
            prompt = template.format(variable="value")

            assert isinstance(prompt, str)
            assert "value" in prompt
        except ImportError:
            pytest.skip("PromptFactory not available")


class TestPredefinedPrompts:
    """Predefined Prompts 테스트"""

    def test_predefined_prompts_rag(self):
        """RAG 프롬프트 테스트 (question_answering 사용)"""
        try:
            from llmkit.domain.prompts.predefined import PredefinedTemplates

            template = PredefinedTemplates.question_answering()
            prompt = template.format(context="Test context", question="Test question")

            assert isinstance(prompt, str)  # ChatPromptTemplate.format()은 문자열 반환
            assert len(prompt) > 0
            assert "Test context" in prompt
            assert "Test question" in prompt
        except ImportError:
            pytest.skip("Predefined prompts not available")

            assert "Test context" in prompt
            assert "Test question" in prompt
        except ImportError:
            pytest.skip("Predefined prompts not available")

            assert "Test context" in prompt
            assert "Test question" in prompt
        except ImportError:
            pytest.skip("Predefined prompts not available")
