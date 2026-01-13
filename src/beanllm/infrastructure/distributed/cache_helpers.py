"""
분산 캐시 헬퍼 함수

RAG 검색 결과, LLM 응답, Agent/Chain 실행 결과 캐싱을 위한 헬퍼 함수
"""

import hashlib
import json
from typing import Any, Dict, Optional

from .factory import get_cache


def get_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    캐시 키 생성

    Args:
        prefix: 캐시 키 접두사 (예: "rag:query", "llm:response")
        *args: 위치 인자
        **kwargs: 키워드 인자

    Returns:
        캐시 키 문자열
    """
    # 인자를 JSON으로 직렬화하여 해시 생성
    key_data = {"args": args, "kwargs": kwargs}
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    return f"{prefix}:{key_hash}"


async def get_cached_result(cache_key: str, ttl: Optional[int] = None) -> Optional[Any]:
    """
    캐시에서 결과 가져오기

    Args:
        cache_key: 캐시 키
        ttl: TTL (초, None이면 기본값 사용)

    Returns:
        캐시된 결과 또는 None
    """
    cache = get_cache(ttl=ttl)
    return await cache.get(cache_key)


async def set_cached_result(cache_key: str, result: Any, ttl: Optional[int] = None) -> None:
    """
    캐시에 결과 저장

    Args:
        cache_key: 캐시 키
        result: 저장할 결과
        ttl: TTL (초, None이면 기본값 사용)
    """
    cache = get_cache(ttl=ttl)
    await cache.set(cache_key, result, ttl=ttl)


# RAG 검색 결과 캐시
async def get_rag_search_cache(query: str, vector_store_id: str, k: int, **kwargs) -> Optional[Any]:
    """
    RAG 검색 결과 캐시에서 가져오기

    Args:
        query: 검색 쿼리
        vector_store_id: 벡터 스토어 ID
        k: 검색 결과 수
        **kwargs: 추가 파라미터

    Returns:
        캐시된 검색 결과 또는 None
    """
    cache_key = get_cache_key("rag:search", query, vector_store_id, k, **kwargs)
    return await get_cached_result(cache_key, ttl=3600)  # 1시간 TTL


async def set_rag_search_cache(query: str, vector_store_id: str, k: int, results: Any, **kwargs) -> None:
    """
    RAG 검색 결과 캐시에 저장

    Args:
        query: 검색 쿼리
        vector_store_id: 벡터 스토어 ID
        k: 검색 결과 수
        results: 검색 결과
        **kwargs: 추가 파라미터
    """
    cache_key = get_cache_key("rag:search", query, vector_store_id, k, **kwargs)
    await set_cached_result(cache_key, results, ttl=3600)  # 1시간 TTL


# LLM 응답 캐시
async def get_llm_response_cache(messages: list, model: str, **kwargs) -> Optional[Any]:
    """
    LLM 응답 캐시에서 가져오기

    Args:
        messages: 메시지 리스트
        model: 모델 이름
        **kwargs: 추가 파라미터

    Returns:
        캐시된 응답 또는 None
    """
    cache_key = get_cache_key("llm:response", messages, model, **kwargs)
    return await get_cached_result(cache_key, ttl=7200)  # 2시간 TTL


async def set_llm_response_cache(messages: list, model: str, response: Any, **kwargs) -> None:
    """
    LLM 응답 캐시에 저장

    Args:
        messages: 메시지 리스트
        model: 모델 이름
        response: LLM 응답
        **kwargs: 추가 파라미터
    """
    cache_key = get_cache_key("llm:response", messages, model, **kwargs)
    await set_cached_result(cache_key, response, ttl=7200)  # 2시간 TTL


# Agent 실행 결과 캐시
async def get_agent_result_cache(agent_id: str, input_data: Any, **kwargs) -> Optional[Any]:
    """
    Agent 실행 결과 캐시에서 가져오기

    Args:
        agent_id: Agent ID
        input_data: 입력 데이터
        **kwargs: 추가 파라미터

    Returns:
        캐시된 결과 또는 None
    """
    cache_key = get_cache_key("agent:result", agent_id, input_data, **kwargs)
    return await get_cached_result(cache_key, ttl=1800)  # 30분 TTL


async def set_agent_result_cache(agent_id: str, input_data: Any, result: Any, **kwargs) -> None:
    """
    Agent 실행 결과 캐시에 저장

    Args:
        agent_id: Agent ID
        input_data: 입력 데이터
        result: 실행 결과
        **kwargs: 추가 파라미터
    """
    cache_key = get_cache_key("agent:result", agent_id, input_data, **kwargs)
    await set_cached_result(cache_key, result, ttl=1800)  # 30분 TTL


# Chain 실행 결과 캐시
async def get_chain_result_cache(chain_id: str, input_data: Any, **kwargs) -> Optional[Any]:
    """
    Chain 실행 결과 캐시에서 가져오기

    Args:
        chain_id: Chain ID
        input_data: 입력 데이터
        **kwargs: 추가 파라미터

    Returns:
        캐시된 결과 또는 None
    """
    cache_key = get_cache_key("chain:result", chain_id, input_data, **kwargs)
    return await get_cached_result(cache_key, ttl=1800)  # 30분 TTL


async def set_chain_result_cache(chain_id: str, input_data: Any, result: Any, **kwargs) -> None:
    """
    Chain 실행 결과 캐시에 저장

    Args:
        chain_id: Chain ID
        input_data: 입력 데이터
        result: 실행 결과
        **kwargs: 추가 파라미터
    """
    cache_key = get_cache_key("chain:result", chain_id, input_data, **kwargs)
    await set_cached_result(cache_key, result, ttl=1800)  # 30분 TTL

