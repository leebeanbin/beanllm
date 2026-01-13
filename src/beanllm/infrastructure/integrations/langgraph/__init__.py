"""
LangGraph Integration - LangGraph 통합 (2024-2025)

LangGraph는 복잡한 에이전트 워크플로우를 위한 그래프 기반 프레임워크입니다.

LangGraph 특징:
- State Machine 기반 워크플로우
- Conditional Edges (조건부 분기)
- Human-in-the-loop
- Persistence & Checkpointing
- Streaming

beanLLM 통합:
- beanLLM State Graph → LangGraph StateGraph 변환
- beanLLM Agent → LangGraph Agent 통합
- Workflow Builder (beanLLM 스타일)

Requirements:
    pip install langgraph

References:
    - https://github.com/langchain-ai/langgraph
    - https://langchain-ai.github.io/langgraph/
"""

from .bridge import LangGraphBridge
from .workflow import (
    LangGraphWorkflow,
    WorkflowBuilder,
    create_workflow,
)

__all__ = [
    "LangGraphBridge",
    "LangGraphWorkflow",
    "WorkflowBuilder",
    "create_workflow",
]
