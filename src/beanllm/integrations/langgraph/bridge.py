"""
LangGraph Bridge - beanLLM ↔ LangGraph 브릿지

beanLLM의 State Graph를 LangGraph 형식으로 변환합니다.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class LangGraphBridge:
    """
    beanLLM ↔ LangGraph 브릿지

    beanLLM의 State Graph를 LangGraph 형식으로 변환합니다.

    Features:
    - beanLLM GraphState → LangGraph State
    - beanLLM Node → LangGraph Node
    - Edge & Conditional Edge 변환

    Example:
        ```python
        from beanllm.integrations.langgraph import LangGraphBridge
        from beanllm.domain.state_graph import GraphState

        # beanLLM State
        class MyState(GraphState):
            query: str
            documents: list
            answer: str

        # LangGraph State로 변환
        bridge = LangGraphBridge()
        langgraph_state = bridge.create_state_schema(MyState)
        ```
    """

    @staticmethod
    def create_state_schema(bean_state_class: type) -> type:
        """
        beanLLM GraphState → LangGraph State Schema 변환

        Args:
            bean_state_class: beanLLM GraphState 클래스

        Returns:
            LangGraph State 클래스
        """
        try:
            import operator
            from typing import Annotated, TypedDict

            from langgraph.graph import MessagesState
        except ImportError:
            raise ImportError(
                "langgraph is required for LangGraphBridge. "
                "Install it with: pip install langgraph"
            )

        # State 필드 추출
        state_fields = {}

        # beanLLM GraphState의 필드를 LangGraph State로 매핑
        if hasattr(bean_state_class, "__annotations__"):
            for field_name, field_type in bean_state_class.__annotations__.items():
                # 리스트는 operator.add로 합침
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    state_fields[field_name] = Annotated[field_type, operator.add]
                else:
                    state_fields[field_name] = field_type

        # TypedDict 생성
        LangGraphState = type(
            "LangGraphState", (TypedDict,), state_fields
        )

        logger.info(f"Created LangGraph State schema with fields: {list(state_fields.keys())}")

        return LangGraphState

    @staticmethod
    def wrap_node_function(
        node_fn: Callable[[Dict], Dict],
    ) -> Callable[[Dict], Dict]:
        """
        beanLLM Node Function → LangGraph Node Function 래핑

        Args:
            node_fn: beanLLM 노드 함수 (state -> state)

        Returns:
            LangGraph 노드 함수
        """

        def wrapped_node(state: Dict) -> Dict:
            """LangGraph Node Wrapper"""
            # beanLLM 노드 함수 호출
            result = node_fn(state)

            # 결과 반환 (LangGraph는 diff만 반환해도 됨)
            return result

        return wrapped_node

    @staticmethod
    def wrap_conditional_edge(
        condition_fn: Callable[[Dict], str],
    ) -> Callable[[Dict], str]:
        """
        beanLLM Conditional Edge → LangGraph Conditional Edge 래핑

        Args:
            condition_fn: beanLLM 조건 함수 (state -> next_node_name)

        Returns:
            LangGraph 조건 함수
        """

        def wrapped_condition(state: Dict) -> str:
            """LangGraph Conditional Edge Wrapper"""
            # beanLLM 조건 함수 호출
            next_node = condition_fn(state)
            return next_node

        return wrapped_condition
