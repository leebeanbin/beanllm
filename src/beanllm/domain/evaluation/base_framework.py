"""
Base Evaluation Framework - 평가 프레임워크 추상 클래스

beanLLM의 모든 외부 평가 프레임워크 래퍼는 이 추상 클래스를 상속해야 합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class BaseEvaluationFramework(ABC):
    """
    평가 프레임워크 베이스 클래스

    DeepEval, LM Eval Harness 등 외부 평가 프레임워크를 통합하기 위한
    공통 인터페이스를 정의합니다.

    BaseMetric과의 차이:
    - BaseMetric: beanLLM 자체 메트릭 (Accuracy, F1, BLEU 등)
    - BaseEvaluationFramework: 외부 평가 프레임워크 래퍼

    Example:
        ```python
        from beanllm.domain.evaluation import BaseEvaluationFramework

        class MyFrameworkWrapper(BaseEvaluationFramework):
            def evaluate(self, **kwargs) -> Dict[str, Any]:
                # 평가 로직
                return {"score": 0.95}

            def list_tasks(self) -> List[str]:
                return ["task1", "task2"]
        ```
    """

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        평가 실행

        각 프레임워크마다 필요한 파라미터가 다르므로 **kwargs로 받습니다.

        Returns:
            평가 결과 딕셔너리
            최소한 다음 형식을 포함해야 합니다:
            - 점수/메트릭 정보
            - 태스크/메트릭 이름
            - 기타 프레임워크별 정보

        Example:
            ```python
            # DeepEval
            result = evaluator.evaluate(
                metric="answer_relevancy",
                question="What is AI?",
                answer="AI is..."
            )

            # LM Eval Harness
            result = evaluator.evaluate(
                tasks=["mmlu", "hellaswag"],
                num_fewshot=5
            )
            ```
        """
        pass

    @abstractmethod
    def list_tasks(self) -> Union[List[str], Dict[str, str]]:
        """
        사용 가능한 태스크/메트릭 목록 조회

        Returns:
            태스크 이름 리스트 또는 {이름: 설명} 딕셔너리

        Example:
            ```python
            # 리스트 형식
            tasks = evaluator.list_tasks()
            # ["mmlu", "hellaswag", "arc_easy"]

            # 딕셔너리 형식 (설명 포함)
            tasks = evaluator.list_tasks()
            # {"mmlu": "Multitask Language Understanding", ...}
            ```
        """
        pass

    def __repr__(self) -> str:
        """래퍼 정보 출력"""
        return f"{self.__class__.__name__}()"
