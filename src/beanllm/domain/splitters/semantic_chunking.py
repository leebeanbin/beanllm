"""
Semantic Chunking - 청크 병합 및 강제 분할 유틸리티

최소/최대 크기 제약을 적용한 청크 병합 및 강제 분할.
"""

from typing import List


def merge_small_chunks(
    chunks: List[str],
    min_size: int,
    max_size: int,
) -> List[str]:
    """
    최소 크기 미만의 청크를 인접 청크와 병합합니다.

    Args:
        chunks: 청크 리스트
        min_size: 최소 청크 크기 (문자 수)
        max_size: 최대 청크 크기 (문자 수)

    Returns:
        병합된 청크 리스트

    Example:
        >>> merge_small_chunks(["Hi", "Hello world", "Bye"], 5, 20)
        ['Hi Hello world', 'Bye']
    """
    merged: List[str] = []
    current = ""

    for chunk in chunks:
        if len(current) + len(chunk) <= max_size:
            current += (" " if current else "") + chunk
        else:
            if current and len(current) >= min_size:
                merged.append(current)
            current = chunk

    if current:
        if len(current) >= min_size:
            merged.append(current)
        elif merged:
            if len(merged[-1]) + len(current) <= max_size:
                merged[-1] += " " + current
            else:
                merged.append(current)
        else:
            merged.append(current)

    return merged


def force_split_by_size(text: str, max_size: int) -> List[str]:
    """
    최대 크기 초과 시 단어 경계에서 강제 분할합니다.

    Args:
        text: 분할할 텍스트
        max_size: 최대 청크 크기 (문자 수)

    Returns:
        분할된 청크 리스트
    """
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + max_size

        if end < len(text):
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx

        chunks.append(text[start:end].strip())
        start = end

    return chunks
