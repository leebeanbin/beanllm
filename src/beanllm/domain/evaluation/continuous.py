"""
Continuous Evaluation - 지속적 평가 시스템
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# apscheduler는 선택적 의존성
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore

from .evaluator import Evaluator
from .results import BatchEvaluationResult


@dataclass
class EvaluationTask:
    """평가 작업"""

    task_id: str
    name: str
    evaluator: Evaluator
    test_cases: List[Dict[str, Any]]  # [{"prediction": "...", "reference": "...", ...}]
    schedule: Optional[str] = None  # Cron 표현식 (예: "0 9 * * *" = 매일 9시)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRun:
    """평가 실행 결과"""

    run_id: str
    task_id: str
    timestamp: datetime
    results: List[BatchEvaluationResult]
    average_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinuousEvaluator:
    """
    지속적 평가 시스템

    정기적으로 평가를 실행하고 결과를 추적
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 결과 저장 경로 (선택적)
        """
        self.storage_path = storage_path
        self._tasks: Dict[str, EvaluationTask] = {}
        self._runs: List[EvaluationRun] = []
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._run_counter = 0

    def add_task(
        self,
        task_id: str,
        name: str,
        evaluator: Evaluator,
        test_cases: List[Dict[str, Any]],
        schedule: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationTask:
        """
        평가 작업 추가

        Args:
            task_id: 작업 ID
            name: 작업 이름
            evaluator: 평가기
            test_cases: 테스트 케이스 리스트
            schedule: Cron 표현식 (선택적, 없으면 수동 실행만)
            metadata: 추가 메타데이터

        Returns:
            EvaluationTask: 생성된 작업
        """
        task = EvaluationTask(
            task_id=task_id,
            name=name,
            evaluator=evaluator,
            test_cases=test_cases,
            schedule=schedule,
            metadata=metadata or {},
        )

        self._tasks[task_id] = task

        # 스케줄이 있으면 스케줄러에 추가
        if schedule:
            self._schedule_task(task)

        return task

    def remove_task(self, task_id: str) -> bool:
        """작업 제거"""
        if task_id not in self._tasks:
            return False

        self._tasks[task_id]
        del self._tasks[task_id]

        # 스케줄러에서도 제거
        if self._scheduler:
            try:
                self._scheduler.remove_job(f"eval_{task_id}")
            except Exception:
                pass

        return True

    async def run_task(self, task_id: str) -> EvaluationRun:
        """
        작업 실행

        Args:
            task_id: 작업 ID

        Returns:
            EvaluationRun: 실행 결과
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        if not task.enabled:
            raise ValueError(f"Task {task_id} is disabled")

        # 평가 실행
        results = []
        for test_case in task.test_cases:
            prediction = test_case.get("prediction", "")
            reference = test_case.get("reference", "")
            kwargs = {k: v for k, v in test_case.items() if k not in ["prediction", "reference"]}

            result = task.evaluator.evaluate(prediction, reference, **kwargs)
            results.append(result)

        # 평균 점수 계산
        if results:
            all_scores = []
            for result in results:
                all_scores.append(result.average_score)
            average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        else:
            average_score = 0.0

        # 실행 결과 생성
        run_id = f"run_{self._run_counter}"
        self._run_counter += 1

        run = EvaluationRun(
            run_id=run_id,
            task_id=task_id,
            timestamp=datetime.now(),
            results=results,
            average_score=average_score,
            metadata={
                "task_name": task.name,
                "test_cases_count": len(task.test_cases),
                "results_count": len(results),
            },
        )

        self._runs.append(run)
        self._save_if_needed()

        return run

    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """작업 조회"""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[EvaluationTask]:
        """모든 작업 조회"""
        return list(self._tasks.values())

    def get_runs(
        self,
        task_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[EvaluationRun]:
        """
        실행 결과 조회

        Args:
            task_id: 작업 ID 필터 (선택적)
            limit: 최대 개수 (선택적)

        Returns:
            List[EvaluationRun]: 실행 결과 리스트
        """
        runs = self._runs

        if task_id:
            runs = [r for r in runs if r.task_id == task_id]

        # 최신순 정렬
        runs.sort(key=lambda x: x.timestamp, reverse=True)

        if limit:
            runs = runs[:limit]

        return runs

    def get_latest_run(self, task_id: str) -> Optional[EvaluationRun]:
        """최신 실행 결과 조회"""
        runs = self.get_runs(task_id=task_id, limit=1)
        return runs[0] if runs else None

    def get_score_trend(
        self,
        task_id: str,
        window_days: int = 7,
    ) -> Dict[str, Any]:
        """
        점수 추이 분석

        Args:
            task_id: 작업 ID
            window_days: 분석 기간 (일)

        Returns:
            Dict[str, Any]: 추이 데이터
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        runs = [r for r in self._runs if r.task_id == task_id and r.timestamp >= cutoff_date]

        if not runs:
            return {
                "task_id": task_id,
                "window_days": window_days,
                "runs_count": 0,
                "average_score": None,
                "trend": "no_data",
            }

        scores = [r.average_score for r in runs]
        average_score = sum(scores) / len(scores)

        # 추이 계산 (선형 회귀 기울기)
        if len(scores) > 1:
            # 간단한 추이: 최근 점수와 초기 점수 비교
            recent_avg = sum(scores[: len(scores) // 2]) / (len(scores) // 2)
            early_avg = sum(scores[len(scores) // 2 :]) / (len(scores) - len(scores) // 2)
            trend = (
                "improving"
                if recent_avg > early_avg
                else "declining"
                if recent_avg < early_avg
                else "stable"
            )
        else:
            trend = "stable"

        return {
            "task_id": task_id,
            "window_days": window_days,
            "runs_count": len(runs),
            "average_score": average_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "trend": trend,
            "scores": scores,
            "timestamps": [r.timestamp.isoformat() for r in runs],
        }

    def start_scheduler(self):
        """스케줄러 시작"""
        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "apscheduler is required for scheduled tasks. "
                "Install it with: pip install beanllm[evaluation] or pip install apscheduler"
            )
        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler()
            self._scheduler.start()

    def stop_scheduler(self):
        """스케줄러 중지"""
        if self._scheduler:
            self._scheduler.shutdown()
            self._scheduler = None

    def _schedule_task(self, task: EvaluationTask):
        """작업 스케줄링"""
        if not task.schedule:
            return

        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "apscheduler is required for scheduled tasks. "
                "Install it with: pip install beanllm[evaluation] or pip install apscheduler"
            )

        if self._scheduler is None:
            self.start_scheduler()

        try:
            # Cron 표현식 파싱
            trigger = CronTrigger.from_crontab(task.schedule)

            # 작업 등록
            self._scheduler.add_job(
                self.run_task,
                trigger=trigger,
                args=[task.task_id],
                id=f"eval_{task.task_id}",
                replace_existing=True,
            )
        except Exception as e:
            raise ValueError(f"Invalid schedule format: {task.schedule} - {e}")

    def _save_if_needed(self):
        """필요시 저장 (파일 기반 저장 구현 예정)"""
        # TODO: 파일 기반 저장 구현
        pass
