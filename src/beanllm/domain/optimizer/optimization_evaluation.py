"""
Optimization Evaluation - 수렴 분석 및 벤치마킹 유틸리티

최적화 히스토리 기반 수렴 그래프 데이터 생성.
"""

from typing import List, Tuple


def get_convergence_plot_data(
    history: List[dict],
) -> Tuple[List[int], List[float]]:
    """
    수렴 그래프 데이터 반환.

    각 시행까지의 최고 점수를 누적하여 반환합니다.

    Args:
        history: 최적화 히스토리 [{"trial_num", "params", "score"}, ...]

    Returns:
        (trial_nums, best_scores_so_far) 튜플
    """
    if not history:
        return [], []

    trial_nums: List[int] = []
    best_scores: List[float] = []
    current_best = float("-inf")

    for entry in history:
        trial_nums.append(entry["trial_num"])
        if entry["score"] > current_best:
            current_best = entry["score"]
        best_scores.append(current_best)

    return trial_nums, best_scores
