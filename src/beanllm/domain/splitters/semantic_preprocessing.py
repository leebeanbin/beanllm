"""
Semantic Preprocessing - 텍스트 전처리 및 문장 분할

의미 기반 분할을 위한 텍스트 정규화 및 문장 단위 분할 유틸리티.
"""

import re
from typing import Callable, List, Optional


def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분할합니다.

    마침표, 느낌표, 물음표, 줄바꿈을 문장 경계로 처리합니다.

    Args:
        text: 분할할 텍스트

    Returns:
        문장 리스트 (빈 문장 제외)

    Example:
        >>> split_into_sentences("Hello. How are you? I'm fine!")
        ['Hello.', 'How are you?', "I'm fine!"]
    """
    sentence_endings = r"(?<=[.!?])\s+|(?<=\n)\s*"
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]


def normalize_text(text: str) -> str:
    """
    텍스트 정규화 (연속 공백, 불필요한 줄바꿈 제거).

    Args:
        text: 정규화할 텍스트

    Returns:
        정규화된 텍스트
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()
