"""
Circuit Breaker 테스트 - 에러 처리 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, patch
import asyncio

try:
    from llmkit.utils.error_handling import CircuitBreaker, CircuitBreakerConfig
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False


@pytest.mark.skipif(not CIRCUIT_BREAKER_AVAILABLE, reason="CircuitBreaker not available")
class TestCircuitBreaker:
    """CircuitBreaker 테스트"""

    @pytest.fixture
    def circuit_breaker(self):
        """CircuitBreaker 인스턴스"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=1.0,
        )
        return CircuitBreaker(config=config)

    def test_circuit_breaker_success(self, circuit_breaker):
        """정상 실행 테스트"""
        def test_func():
            return "success"

        result = circuit_breaker.call(test_func)
        assert result == "success"
        state = circuit_breaker.get_state()
        assert state["state"] == "closed"

    def test_circuit_breaker_failure(self, circuit_breaker):
        """실패 누적 테스트"""
        def test_func():
            raise Exception("Test error")

        # 실패 누적
        for _ in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(test_func)

        # Circuit이 열림
        state = circuit_breaker.get_state()
        assert state["state"] == "open"

    def test_circuit_breaker_open_state(self, circuit_breaker):
        """열린 상태에서 호출 테스트"""
        from llmkit.utils.error_handling import CircuitState
        import time
        
        # Circuit을 열림 상태로 만듦
        circuit_breaker.failure_count = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = time.time()

        def test_func():
            return "should not execute"

        # Circuit이 열려있으면 즉시 실패
        from llmkit.utils.error_handling import CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(test_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self, circuit_breaker):
        """비동기 함수 테스트"""
        # CircuitBreaker는 동기 함수만 지원하므로 동기 함수로 테스트
        def sync_func():
            return "async success"

        result = circuit_breaker.call(sync_func)
        assert result == "async success"

    def test_circuit_breaker_recovery(self, circuit_breaker):
        """복구 테스트"""
        import time
        from llmkit.utils.error_handling import CircuitState
        
        # Circuit을 열림 상태로 만듦
        circuit_breaker.failure_count = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = time.time() - 2.0  # 과거 시간으로 설정

        def test_func():
            return "recovered"

        # 복구 시간이 지나면 half-open 상태로 전환되어 호출 가능
        result = circuit_breaker.call(test_func)
        assert result == "recovered"
        state = circuit_breaker.get_state()
        assert state["state"] in ["closed", "half_open"]


