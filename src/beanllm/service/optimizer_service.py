"""
IOptimizerService - Auto-Optimizer 서비스 인터페이스
SOLID 원칙:
- ISP: 최적화 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from beanllm.dto.request.advanced.optimizer_request import (
    ABTestRequest,
    BenchmarkRequest,
    OptimizeRequest,
    ProfileRequest,
)
from beanllm.dto.response.advanced.optimizer_response import (
    ABTestResponse,
    BenchmarkResponse,
    OptimizeResponse,
    ProfileResponse,
    RecommendationResponse,
)


class IOptimizerService(ABC):
    """
    Auto-Optimizer 서비스 인터페이스

    책임:
    - RAG/Agent 시스템 자동 최적화 비즈니스 로직 정의
    - 벤치마킹, 프로파일링, 파라미터 최적화, A/B 테스팅

    SOLID:
    - ISP: 최적화 관련 메서드만
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """
        시스템 벤치마킹 (synthetic queries, baseline 측정)

        Args:
            request: 벤치마크 요청 DTO

        Returns:
            BenchmarkResponse: 벤치마크 결과
        """
        pass

    @abstractmethod
    async def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """
        파라미터 자동 최적화 (Bayesian/Grid search)

        Args:
            request: 최적화 요청 DTO

        Returns:
            OptimizeResponse: 최적화 결과 (최적 파라미터)
        """
        pass

    @abstractmethod
    async def profile(self, request: ProfileRequest) -> ProfileResponse:
        """
        컴포넌트별 프로파일링 (latency, cost 분석)

        Args:
            request: 프로파일링 요청 DTO

        Returns:
            ProfileResponse: 프로파일링 결과
        """
        pass

    @abstractmethod
    async def ab_test(self, request: ABTestRequest) -> ABTestResponse:
        """
        A/B 테스팅 (side-by-side comparison)

        Args:
            request: A/B 테스트 요청 DTO

        Returns:
            ABTestResponse: A/B 테스트 결과
        """
        pass

    @abstractmethod
    async def get_recommendations(self, profile_id: str) -> RecommendationResponse:
        """
        최적화 권장사항 생성

        Args:
            profile_id: 프로파일 ID

        Returns:
            RecommendationResponse: 권장사항 목록
        """
        pass

    @abstractmethod
    async def compare_configs(
        self, config_ids: List[str]
    ) -> Dict[str, Any]:
        """
        여러 설정 비교

        Args:
            config_ids: 비교할 설정 ID 목록

        Returns:
            Dict: 비교 결과
        """
        pass
