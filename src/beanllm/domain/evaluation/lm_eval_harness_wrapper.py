"""
LM Evaluation Harness Wrapper - 표준 벤치마크 평가 (2024-2025)

EleutherAI의 LM Evaluation Harness는 LLM 벤치마크의 사실상 표준입니다.

LM Eval Harness 특징:
- 60+ 표준 벤치마크 지원
- MMLU, MMLU-Pro, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval 등
- HuggingFace Transformers 통합
- 멀티모달 지원 (Vision, Speech)
- Few-shot learning
- 재현 가능한 평가

Requirements:
    pip install lm-eval

References:
    - https://github.com/EleutherAI/lm-evaluation-harness
    - https://www.eleuther.ai/projects/large-language-model-evaluation/
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_framework import BaseEvaluationFramework

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

# LM Eval Harness 설치 여부 체크
try:
    HAS_LM_EVAL = True
    # 실제 import는 사용 시점에 수행
except ImportError:
    HAS_LM_EVAL = False


class LMEvalHarnessWrapper(BaseEvaluationFramework):
    """
    LM Evaluation Harness 통합 래퍼

    EleutherAI의 표준 벤치마크 프레임워크를 beanLLM에서 사용합니다.

    지원 벤치마크:
    - MMLU: 57개 주제의 멀티태스크 이해도 (57 tasks)
    - MMLU-Pro: 향상된 MMLU (14K questions)
    - HellaSwag: 상식 추론 (10K questions)
    - ARC (Easy/Challenge): 과학 문제 풀이
    - TruthfulQA: 진실성 평가 (817 questions)
    - GSM8K: 수학 문제 풀이 (8.5K questions)
    - HumanEval: 코딩 능력 (164 problems)
    - MATH: 고급 수학 (12.5K problems)
    - BBH: BIG-Bench Hard (23 tasks)
    - DROP: 읽기 이해 + 수학
    - WinoGrande: 상식 추론
    - PIQA: 물리 상식
    - SIQA: 사회 상식

    Example:
        ```python
        from beanllm.domain.evaluation import LMEvalHarnessWrapper

        # HuggingFace 모델 평가
        evaluator = LMEvalHarnessWrapper(
            model="hf",
            model_args="pretrained=meta-llama/Llama-3.2-1B"
        )

        # MMLU 평가
        results = evaluator.evaluate(
            tasks=["mmlu"],
            num_fewshot=5
        )
        print(results)  # {"mmlu": {"acc": 0.45, "acc_norm": 0.47}}

        # 여러 벤치마크 동시 평가
        results = evaluator.evaluate(
            tasks=["mmlu", "hellaswag", "arc_easy", "truthfulqa_mc2"],
            num_fewshot=5
        )

        # 로컬 모델 평가 (Ollama)
        evaluator = LMEvalHarnessWrapper(
            model="local-completions",
            model_args="base_url=http://localhost:11434,model=llama3.2:1b"
        )
        results = evaluator.evaluate(tasks=["gsm8k"], num_fewshot=8)
        ```
    """

    # 인기 있는 벤치마크 태스크
    POPULAR_TASKS = {
        # 종합 평가
        "mmlu": "Massive Multitask Language Understanding (57 tasks)",
        "mmlu_pro": "Enhanced MMLU with harder questions (14K)",
        "bbh": "BIG-Bench Hard (23 challenging tasks)",

        # 추론
        "hellaswag": "Commonsense reasoning (10K questions)",
        "arc_easy": "ARC Easy (science Q&A)",
        "arc_challenge": "ARC Challenge (harder science Q&A)",
        "winogrande": "Commonsense reasoning (1.3K questions)",
        "piqa": "Physical commonsense reasoning",
        "siqa": "Social commonsense reasoning",

        # 진실성
        "truthfulqa_mc1": "TruthfulQA (single-choice)",
        "truthfulqa_mc2": "TruthfulQA (multi-choice)",

        # 수학
        "gsm8k": "Grade School Math (8.5K questions)",
        "math": "Advanced Math (12.5K problems)",

        # 코딩
        "humaneval": "Code generation (164 Python problems)",
        "mbpp": "Mostly Basic Python Programming (1K problems)",

        # 읽기 이해
        "drop": "Reading comprehension + reasoning",
        "race": "Reading comprehension (high/middle school)",

        # 한국어
        "kobest": "Korean language understanding",
        "klue": "Korean Language Understanding Evaluation",
    }

    def __init__(
        self,
        model: str = "hf",
        model_args: str = "",
        batch_size: Union[int, str] = "auto",
        device: Optional[str] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Args:
            model: 모델 타입
                - "hf": HuggingFace Transformers
                - "local-completions": 로컬 API (Ollama, vLLM 등)
                - "openai-completions": OpenAI API
                - "anthropic": Anthropic API
            model_args: 모델 인자 (쉼표로 구분)
                예: "pretrained=meta-llama/Llama-3.2-1B,dtype=bfloat16"
            batch_size: 배치 크기 (auto 또는 정수)
            device: 디바이스 (cuda, cpu, mps 등)
            num_fewshot: Few-shot 예시 개수 (기본: 0)
            limit: 평가할 샘플 수 제한 (None이면 전체)
            output_path: 결과 저장 경로
            **kwargs: 추가 파라미터
        """
        self.model = model
        self.model_args = model_args
        self.batch_size = batch_size
        self.device = device
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.output_path = Path(output_path) if output_path else None
        self.kwargs = kwargs

        # Lazy loading
        self._lm_eval = None

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            import lm_eval
        except ImportError:
            raise ImportError(
                "lm-eval is required for LMEvalHarnessWrapper. "
                "Install it with: pip install lm-eval"
            )

        self._lm_eval = lm_eval

    def list_tasks(self, pattern: Optional[str] = None) -> List[str]:
        """
        사용 가능한 태스크 목록

        Args:
            pattern: 필터 패턴 (예: "mmlu", "arc")

        Returns:
            태스크 이름 리스트
        """
        self._check_dependencies()

        from lm_eval.tasks import TaskManager

        task_manager = TaskManager()
        all_tasks = task_manager.all_tasks

        if pattern:
            filtered_tasks = [t for t in all_tasks if pattern.lower() in t.lower()]
            return filtered_tasks

        return all_tasks

    def get_popular_tasks(self) -> Dict[str, str]:
        """
        인기 있는 벤치마크 태스크 목록

        Returns:
            {"task_name": "description", ...}
        """
        return self.POPULAR_TASKS.copy()

    def evaluate(
        self,
        tasks: Union[str, List[str]],
        num_fewshot: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        벤치마크 평가 실행

        Args:
            tasks: 태스크 이름 또는 리스트
                예: "mmlu", ["mmlu", "hellaswag"]
            num_fewshot: Few-shot 예시 개수 (None이면 기본값 사용)
            limit: 평가할 샘플 수 제한 (None이면 전체)
            **kwargs: 추가 평가 파라미터

        Returns:
            평가 결과 딕셔너리
            {
                "results": {
                    "mmlu": {"acc": 0.45, "acc_norm": 0.47},
                    "hellaswag": {"acc": 0.65, "acc_norm": 0.68}
                },
                "versions": {...},
                "config": {...}
            }

        Example:
            ```python
            # 단일 태스크
            results = evaluator.evaluate(tasks="mmlu", num_fewshot=5)

            # 여러 태스크
            results = evaluator.evaluate(
                tasks=["mmlu", "hellaswag", "arc_easy"],
                num_fewshot=5,
                limit=100  # 각 태스크당 100개 샘플만
            )
            ```
        """
        self._check_dependencies()

        from lm_eval import simple_evaluate

        # 파라미터 준비
        num_fewshot = num_fewshot if num_fewshot is not None else self.num_fewshot
        limit = limit if limit is not None else self.limit

        # tasks를 리스트로 변환
        if isinstance(tasks, str):
            tasks = [tasks]

        logger.info(
            f"LM Eval Harness: Evaluating {len(tasks)} tasks with "
            f"model={self.model}, num_fewshot={num_fewshot}"
        )

        # 평가 실행
        try:
            results = simple_evaluate(
                model=self.model,
                model_args=self.model_args,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=self.batch_size,
                device=self.device,
                limit=limit,
                **self.kwargs,
                **kwargs,
            )

            logger.info(f"LM Eval Harness: Evaluation completed for {len(tasks)} tasks")

            # 결과 저장 (옵션)
            if self.output_path:
                self._save_results(results, tasks)

            return results

        except Exception as e:
            logger.error(f"LM Eval Harness evaluation failed: {e}")
            raise

    def evaluate_mmlu(
        self,
        num_fewshot: int = 5,
        subjects: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        MMLU 평가 (57개 주제)

        Args:
            num_fewshot: Few-shot 예시 개수 (기본: 5)
            subjects: 평가할 주제 리스트 (None이면 전체)
            **kwargs: 추가 파라미터

        Returns:
            평가 결과

        Example:
            ```python
            # 전체 MMLU
            results = evaluator.evaluate_mmlu(num_fewshot=5)

            # 특정 주제만
            results = evaluator.evaluate_mmlu(
                subjects=["abstract_algebra", "anatomy"],
                num_fewshot=5
            )
            ```
        """
        if subjects:
            tasks = [f"mmlu_{subject}" for subject in subjects]
        else:
            tasks = ["mmlu"]

        return self.evaluate(tasks=tasks, num_fewshot=num_fewshot, **kwargs)

    def evaluate_suite(
        self,
        suite: str = "standard",
        num_fewshot: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        벤치마크 스위트 평가

        Args:
            suite: 스위트 이름
                - "standard": MMLU, HellaSwag, ARC, TruthfulQA
                - "reasoning": HellaSwag, ARC, WinoGrande, PIQA
                - "math": GSM8K, MATH
                - "coding": HumanEval, MBPP
                - "korean": KoBEST, KLUE
            num_fewshot: Few-shot 예시 개수
            **kwargs: 추가 파라미터

        Returns:
            평가 결과

        Example:
            ```python
            # 표준 벤치마크 스위트
            results = evaluator.evaluate_suite(suite="standard")

            # 수학 벤치마크
            results = evaluator.evaluate_suite(suite="math", num_fewshot=8)
            ```
        """
        suites = {
            "standard": ["mmlu", "hellaswag", "arc_easy", "arc_challenge", "truthfulqa_mc2"],
            "reasoning": ["hellaswag", "arc_challenge", "winogrande", "piqa"],
            "math": ["gsm8k", "math"],
            "coding": ["humaneval", "mbpp"],
            "korean": ["kobest", "klue"],
            "comprehensive": [
                "mmlu", "mmlu_pro", "hellaswag", "arc_challenge",
                "truthfulqa_mc2", "gsm8k", "humaneval"
            ],
        }

        if suite not in suites:
            raise ValueError(
                f"Unknown suite: {suite}. "
                f"Available: {list(suites.keys())}"
            )

        tasks = suites[suite]
        logger.info(f"Evaluating {suite} suite: {tasks}")

        return self.evaluate(tasks=tasks, num_fewshot=num_fewshot, **kwargs)

    def _save_results(self, results: Dict[str, Any], tasks: List[str]):
        """
        결과를 JSON 파일로 저장

        Args:
            results: 평가 결과
            tasks: 태스크 리스트
        """
        import json
        from datetime import datetime

        if not self.output_path:
            return

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tasks_str = "_".join(tasks[:3])  # 최대 3개 태스크 이름
        filename = f"lm_eval_{tasks_str}_{timestamp}.json"

        output_file = self.output_path / filename

        # 디렉토리 생성
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 저장
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"LM Eval Harness results saved to: {output_file}")

    def get_leaderboard_format(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        리더보드 형식으로 결과 변환

        Args:
            results: 평가 결과

        Returns:
            {"task": score, ...} 형식

        Example:
            ```python
            results = evaluator.evaluate(tasks=["mmlu", "hellaswag"])
            leaderboard = evaluator.get_leaderboard_format(results)
            print(leaderboard)
            # {"mmlu": 0.45, "hellaswag": 0.65}
            ```
        """
        leaderboard = {}

        if "results" in results:
            for task, metrics in results["results"].items():
                # acc_norm을 우선 사용, 없으면 acc
                score = metrics.get("acc_norm", metrics.get("acc", 0.0))
                leaderboard[task] = score

        return leaderboard

    def __repr__(self) -> str:
        return (
            f"LMEvalHarnessWrapper(model={self.model}, "
            f"num_fewshot={self.num_fewshot}, batch_size={self.batch_size})"
        )
