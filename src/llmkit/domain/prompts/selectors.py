"""
Prompts Selectors - 예제 선택기
"""

import random
from typing import Any, Callable, Dict, List, Optional

from .types import PromptExample


class ExampleSelector:
    """Few-shot 예제 선택 전략"""

    @staticmethod
    def similarity_based(
        examples: List[PromptExample],
        input_data: Dict[str, Any],
        top_k: int = 3,
        similarity_fn: Optional[Callable] = None,
    ) -> List[PromptExample]:
        """유사도 기반 예제 선택"""
        if similarity_fn is None:
            # 기본: 간단한 문자열 유사도
            def default_similarity(ex1: str, ex2: str) -> float:
                # Jaccard similarity
                set1 = set(ex1.lower().split())
                set2 = set(ex2.lower().split())
                if not set1 or not set2:
                    return 0.0
                intersection = set1 & set2
                union = set1 | set2
                return len(intersection) / len(union)

            similarity_fn = default_similarity

        # 입력과 각 예제의 유사도 계산
        input_text = str(input_data.get("input", ""))
        scored_examples = []

        for example in examples:
            score = similarity_fn(input_text, example.input)
            scored_examples.append((score, example))

        # 점수 기준 정렬
        scored_examples.sort(reverse=True, key=lambda x: x[0])

        # top_k 반환
        return [ex for _, ex in scored_examples[:top_k]]

    @staticmethod
    def length_based(examples: List[PromptExample], max_length: int) -> List[PromptExample]:
        """길이 제한 기반 예제 선택"""
        selected = []
        current_length = 0

        for example in examples:
            example_length = len(example.input) + len(example.output)
            if current_length + example_length <= max_length:
                selected.append(example)
                current_length += example_length
            else:
                break

        return selected

    @staticmethod
    def random(examples: List[PromptExample], k: int) -> List[PromptExample]:
        """랜덤 선택"""
        return random.sample(examples, min(k, len(examples)))
