"""
A/B Testing for Prompts - 프롬프트 A/B 테스트
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

from .versioning import PromptVersionManager


@dataclass
class ABTestConfig:
    """A/B 테스트 설정"""

    prompt_a: str
    prompt_b: str
    prompt_a_version: str = "v1"
    prompt_b_version: str = "v2"
    traffic_split: float = 0.5  # A:B 비율 (0.5 = 50:50)
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "latency"])
    min_samples: int = 100  # 최소 샘플 수


@dataclass
class ABTestResult:
    """A/B 테스트 결과"""

    config: ABTestConfig
    results_a: List[Dict[str, Any]] = field(default_factory=list)
    results_b: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def get_summary(self) -> Dict[str, Any]:
        """결과 요약"""
        summary = {}
        for metric in self.config.metrics:
            values_a = [r.get(metric, 0) for r in self.results_a if metric in r]
            values_b = [r.get(metric, 0) for r in self.results_b if metric in r]

            summary[metric] = {
                "a_mean": sum(values_a) / len(values_a) if values_a else 0,
                "b_mean": sum(values_b) / len(values_b) if values_b else 0,
                "a_std": self._std(values_a),
                "b_std": self._std(values_b),
                "improvement": (
                    (sum(values_b) / len(values_b) - sum(values_a) / len(values_a))
                    if values_a and values_b
                    else 0
                ),
            }

        return summary

    def _std(self, values: List[float]) -> float:
        """표준편차 계산"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5


class ABTestRunner:
    """A/B 테스트 실행기"""

    def __init__(self, version_manager: PromptVersionManager):
        self.version_manager = version_manager

    async def run_test(
        self,
        config: ABTestConfig,
        test_cases: List[Dict[str, Any]],
        llm_client: Any,  # Client 타입
        evaluation_function: Optional[Callable] = None,
    ) -> ABTestResult:
        """
        A/B 테스트 실행

        Args:
            config: A/B 테스트 설정
            test_cases: 테스트 케이스 리스트 [{"input": "...", "expected": "..."}]
            llm_client: LLM 클라이언트
            evaluation_function: 평가 함수 (input, expected, output) -> metrics

        Returns:
            ABTestResult: 테스트 결과
        """
        result = ABTestResult(config=config)

        # 트래픽 분할
        for test_case in test_cases:
            # 랜덤하게 A 또는 B 선택
            use_b = random.random() < config.traffic_split

            prompt = config.prompt_b if use_b else config.prompt_a

            # 프롬프트 포맷팅 (변수 치환)
            try:
                formatted_prompt = prompt.format(**test_case)
            except KeyError:
                # 변수가 없으면 그대로 사용
                formatted_prompt = prompt

            # LLM 호출
            try:
                # llm_client.chat() 메서드 호출 (비동기)
                if hasattr(llm_client, "chat"):
                    response = await llm_client.chat(
                        messages=[{"role": "user", "content": formatted_prompt}]
                    )
                    output = response.content if hasattr(response, "content") else str(response)
                elif hasattr(llm_client, "handle_chat"):
                    # Handler를 통한 호출
                    response = await llm_client.handle_chat(
                        messages=[{"role": "user", "content": formatted_prompt}]
                    )
                    output = response.content if hasattr(response, "content") else str(response)
                else:
                    # 직접 호출 불가능한 경우
                    raise ValueError("llm_client must have 'chat' or 'handle_chat' method")

                # 평가
                if evaluation_function:
                    metrics = evaluation_function(
                        test_case.get("input", ""),
                        test_case.get("expected"),
                        output,
                    )
                else:
                    # 기본 평가 (정확도만)
                    expected = test_case.get("expected", "")
                    accuracy = 1.0 if output.strip() == expected.strip() else 0.0
                    metrics = {
                        "accuracy": accuracy,
                        "latency": 0.0,  # 기본값
                    }

                # 결과 저장
                test_result = {
                    "input": test_case.get("input", ""),
                    "output": output,
                    "expected": test_case.get("expected"),
                    **metrics,
                }

                if use_b:
                    result.results_b.append(test_result)
                else:
                    result.results_a.append(test_result)

                # 버전 사용 기록 (프롬프트 이름이 필요한 경우)
                # 여기서는 버전만 기록

            except Exception as e:
                # 에러 처리
                error_result = {
                    "input": test_case.get("input", ""),
                    "error": str(e),
                    **{metric: 0.0 for metric in config.metrics},
                }
                if use_b:
                    result.results_b.append(error_result)
                else:
                    result.results_a.append(error_result)

        result.end_time = datetime.now()
        return result

    def analyze_results(self, result: ABTestResult) -> Dict[str, Any]:
        """
        결과 분석 (통계적 유의성 검증)

        Args:
            result: A/B 테스트 결과

        Returns:
            {
                "summary": {...},  # 요약 통계
                "statistical_significance": {...},  # 통계적 유의성
                "recommendation": str  # 추천 프롬프트
            }
        """
        summary = result.get_summary()

        # 통계적 유의성 검증 (t-test)
        significance = {}
        if SCIPY_AVAILABLE:
            for metric in result.config.metrics:
                values_a = [r.get(metric, 0) for r in result.results_a if metric in r]
                values_b = [r.get(metric, 0) for r in result.results_b if metric in r]

                if len(values_a) < 2 or len(values_b) < 2:
                    significance[metric] = {
                        "p_value": None,
                        "significant": False,
                        "reason": "Insufficient samples",
                    }
                    continue

                # t-test 수행
                try:
                    t_stat, p_value = stats.ttest_ind(values_a, values_b)

                    significance[metric] = {
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,  # 95% 신뢰도
                        "t_statistic": float(t_stat),
                    }
                except Exception as e:
                    significance[metric] = {
                        "p_value": None,
                        "significant": False,
                        "reason": str(e),
                    }
        else:
            # scipy 없으면 유의성 검증 불가
            for metric in result.config.metrics:
                significance[metric] = {
                    "p_value": None,
                    "significant": False,
                    "reason": "scipy not available",
                }

        # 추천 프롬프트 (accuracy 기준, 통계적으로 유의한 경우만)
        recommendation = "A"
        if "accuracy" in significance and significance["accuracy"].get("significant", False):
            if summary["accuracy"]["improvement"] > 0:
                recommendation = "B"

        return {
            "summary": summary,
            "statistical_significance": significance,
            "recommendation": recommendation,
            "sample_sizes": {
                "a": len(result.results_a),
                "b": len(result.results_b),
            },
        }
