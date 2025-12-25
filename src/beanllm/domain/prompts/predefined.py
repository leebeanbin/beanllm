"""
Prompts Predefined - 사전 정의된 템플릿
"""

from .templates import ChatPromptTemplate, PromptTemplate


class PredefinedTemplates:
    """자주 사용되는 템플릿 모음"""

    @staticmethod
    def translation() -> PromptTemplate:
        """번역 템플릿"""
        return PromptTemplate(
            template="Translate the following text from {source_lang} to {target_lang}:\n\n{text}",
            input_variables=["source_lang", "target_lang", "text"],
        )

    @staticmethod
    def summarization() -> PromptTemplate:
        """요약 템플릿"""
        return PromptTemplate(
            template="Summarize the following text in {max_sentences} sentences:\n\n{text}",
            input_variables=["text", "max_sentences"],
        )

    @staticmethod
    def question_answering() -> ChatPromptTemplate:
        """QA 템플릿"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions based on the given context.",
                ),
                ("user", "Context: {context}\n\nQuestion: {question}\n\nAnswer:"),
            ]
        )

    @staticmethod
    def code_generation() -> ChatPromptTemplate:
        """코드 생성 템플릿"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert {language} programmer."),
                ("user", "Write {language} code to {task}.\n\nRequirements:\n{requirements}"),
            ]
        )

    @staticmethod
    def chain_of_thought() -> PromptTemplate:
        """Chain-of-Thought 템플릿"""
        return PromptTemplate(
            template=(
                "{question}\n\n"
                "Let's think step by step:\n"
                "1. First, let's identify what we know\n"
                "2. Next, let's determine what we need to find\n"
                "3. Then, let's work through the solution\n"
                "4. Finally, let's verify our answer"
            ),
            input_variables=["question"],
        )

    @staticmethod
    def react_agent() -> ChatPromptTemplate:
        """ReAct Agent 템플릿"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful assistant that uses tools to answer questions.\n"
                        "Use the following format:\n\n"
                        "Thought: Consider what to do\n"
                        "Action: The action to take\n"
                        "Observation: The result of the action\n"
                        "... (repeat as needed)\n"
                        "Final Answer: The final answer"
                    ),
                ),
                ("user", "{input}\n\nAvailable tools: {tools}"),
            ]
        )
