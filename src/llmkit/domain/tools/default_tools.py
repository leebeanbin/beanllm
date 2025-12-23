"""
기본 도구들
"""

from datetime import datetime

from .tool_registry import register_tool


@register_tool
def echo(text: str) -> str:
    """입력을 그대로 반환"""
    return text


@register_tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    간단한 계산기

    Args:
        operation: 연산 (add, subtract, multiply, divide)
        a: 첫 번째 숫자
        b: 두 번째 숫자
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }

    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"

    return operations[operation](a, b)


@register_tool
def get_current_time() -> str:
    """현재 시간 가져오기"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@register_tool
def search_web(query: str) -> str:
    """
    웹 검색 (시뮬레이션)

    실제 구현 시 Google Search API 등을 사용
    """
    return f"[검색 결과 시뮬레이션] '{query}'에 대한 검색 결과:\n- 결과 1\n- 결과 2\n- 결과 3"
