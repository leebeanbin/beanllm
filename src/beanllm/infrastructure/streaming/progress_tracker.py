"""
Progress Tracker for Real-time Streaming

책임:
- 작업 진행 상황 추적
- WebSocket을 통한 실시간 업데이트 전송
- 다단계 작업 지원

SOLID:
- SRP: 진행 상황 추적만 담당
- OCP: 새로운 진행 상황 타입 쉽게 추가
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class ProgressUpdate:
    """진행 상황 업데이트"""

    stage: str  # Current stage name
    current: int  # Current step
    total: int  # Total steps
    message: str = ""  # Status message
    percentage: float = 0.0  # Completion percentage
    elapsed_time: float = 0.0  # Elapsed time in seconds
    estimated_remaining: Optional[float] = None  # Estimated remaining time
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    def __post_init__(self):
        if self.total > 0:
            self.percentage = (self.current / self.total) * 100
        else:
            self.percentage = 0.0


class ProgressTracker:
    """
    진행 상황 추적기

    작업의 진행 상황을 추적하고 WebSocket을 통해 실시간 업데이트 전송

    Example:
        ```python
        tracker = ProgressTracker(
            task_id="kg_build_123",
            total_steps=100,
            websocket_session=session,
        )

        await tracker.start("Building knowledge graph")

        for i in range(100):
            await tracker.update(
                current=i + 1,
                message=f"Processing document {i+1}/100"
            )
            # Do work...

        await tracker.complete(result={"nodes": 500, "edges": 1000})
        ```
    """

    def __init__(
        self,
        task_id: str,
        total_steps: int,
        websocket_session: Optional[Any] = None,
        stage: str = "processing",
        enable_time_estimation: bool = True,
    ):
        """
        초기화

        Args:
            task_id: 작업 ID
            total_steps: 전체 단계 수
            websocket_session: WebSocket 세션 (optional)
            stage: 현재 단계 이름
            enable_time_estimation: 남은 시간 추정 활성화
        """
        self.task_id = task_id
        self.total_steps = total_steps
        self.websocket_session = websocket_session
        self.stage = stage
        self.enable_time_estimation = enable_time_estimation

        # State
        self.current_step = 0
        self.start_time: Optional[float] = None
        self.is_complete = False
        self.is_cancelled = False

        # History for time estimation
        self._step_times: List[float] = []

    async def start(self, message: str = "Started"):
        """
        작업 시작

        Args:
            message: 시작 메시지
        """
        self.start_time = time.time()
        self.current_step = 0
        self.is_complete = False
        self.is_cancelled = False

        logger.info(f"Progress tracker started: {self.task_id}")

        await self._send_update(
            current=0,
            message=message,
        )

    async def update(
        self,
        current: Optional[int] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        increment: int = 1,
    ):
        """
        진행 상황 업데이트

        Args:
            current: 현재 단계 (None이면 increment 사용)
            message: 상태 메시지
            metadata: 추가 메타데이터
            increment: 증가량 (current가 None일 때)
        """
        if self.is_complete or self.is_cancelled:
            return

        # Update current step
        if current is not None:
            self.current_step = current
        else:
            self.current_step += increment

        # Track step time for estimation
        if self.enable_time_estimation and self.start_time:
            self._step_times.append(time.time())

        await self._send_update(
            current=self.current_step,
            message=message,
            metadata=metadata,
        )

    async def complete(
        self,
        message: str = "Completed",
        result: Optional[Dict[str, Any]] = None,
    ):
        """
        작업 완료

        Args:
            message: 완료 메시지
            result: 최종 결과
        """
        if self.is_complete:
            return

        self.is_complete = True
        self.current_step = self.total_steps

        logger.info(f"Progress tracker completed: {self.task_id}")

        # Send completion update
        await self._send_update(
            current=self.total_steps,
            message=message,
            metadata=result,
        )

        # Send completion message
        if self.websocket_session:
            try:
                await self.websocket_session.send_complete(
                    final_result={
                        "task_id": self.task_id,
                        "message": message,
                        "elapsed_time": self._get_elapsed_time(),
                        **(result or {}),
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to send completion message: {e}")

    async def error(
        self,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        에러 발생

        Args:
            error_message: 에러 메시지
            details: 에러 상세 정보
        """
        logger.error(f"Progress tracker error: {self.task_id} - {error_message}")

        if self.websocket_session:
            try:
                await self.websocket_session.send_error(
                    error=error_message,
                    details={
                        "task_id": self.task_id,
                        "current_step": self.current_step,
                        "total_steps": self.total_steps,
                        **(details or {}),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to send error message: {e}")

        self.is_cancelled = True

    async def cancel(self, message: str = "Cancelled"):
        """
        작업 취소

        Args:
            message: 취소 메시지
        """
        if self.is_complete or self.is_cancelled:
            return

        self.is_cancelled = True

        logger.info(f"Progress tracker cancelled: {self.task_id}")

        await self._send_update(
            current=self.current_step,
            message=message,
            metadata={"cancelled": True},
        )

    async def _send_update(
        self,
        current: int,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        진행 상황 업데이트 전송

        Args:
            current: 현재 단계
            message: 상태 메시지
            metadata: 추가 메타데이터
        """
        # Calculate progress
        elapsed = self._get_elapsed_time()
        estimated_remaining = self._estimate_remaining_time(current)

        # Create progress update
        update = ProgressUpdate(
            stage=self.stage,
            current=current,
            total=self.total_steps,
            message=message,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            metadata=metadata,
        )

        # Send via WebSocket
        if self.websocket_session:
            try:
                await self.websocket_session.send_progress(
                    current=current,
                    total=self.total_steps,
                    message=message,
                    metadata={
                        "stage": self.stage,
                        "elapsed_time": elapsed,
                        "estimated_remaining": estimated_remaining,
                        "percentage": update.percentage,
                        "task_id": self.task_id,
                        **(metadata or {}),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to send progress update: {e}")

    def _get_elapsed_time(self) -> float:
        """경과 시간 계산 (초)"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def _estimate_remaining_time(self, current: int) -> Optional[float]:
        """
        남은 시간 추정

        Args:
            current: 현재 단계

        Returns:
            추정 남은 시간 (초) 또는 None
        """
        if not self.enable_time_estimation:
            return None

        if current == 0 or self.total_steps == 0:
            return None

        elapsed = self._get_elapsed_time()
        if elapsed == 0:
            return None

        # Simple linear estimation
        avg_time_per_step = elapsed / current
        remaining_steps = self.total_steps - current
        return avg_time_per_step * remaining_steps

    def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 반환

        Returns:
            상태 딕셔너리
        """
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "current": self.current_step,
            "total": self.total_steps,
            "percentage": (self.current_step / self.total_steps * 100)
            if self.total_steps > 0
            else 0.0,
            "elapsed_time": self._get_elapsed_time(),
            "estimated_remaining": self._estimate_remaining_time(self.current_step),
            "is_complete": self.is_complete,
            "is_cancelled": self.is_cancelled,
        }


class MultiStageProgressTracker:
    """
    다단계 진행 상황 추적기

    여러 단계로 구성된 작업의 진행 상황 추적

    Example:
        ```python
        tracker = MultiStageProgressTracker(
            task_id="kg_build_123",
            stages=[
                ("extract_entities", 50),
                ("extract_relations", 30),
                ("build_graph", 20),
            ],
            websocket_session=session,
        )

        await tracker.start()

        # Stage 1
        await tracker.set_stage("extract_entities")
        for i in range(50):
            await tracker.update(message=f"Extracting entity {i+1}/50")

        # Stage 2
        await tracker.set_stage("extract_relations")
        for i in range(30):
            await tracker.update(message=f"Extracting relation {i+1}/30")

        # Stage 3
        await tracker.set_stage("build_graph")
        for i in range(20):
            await tracker.update(message=f"Building graph {i+1}/20")

        await tracker.complete()
        ```
    """

    def __init__(
        self,
        task_id: str,
        stages: List[tuple],  # [(stage_name, num_steps), ...]
        websocket_session: Optional[Any] = None,
    ):
        """
        초기화

        Args:
            task_id: 작업 ID
            stages: [(단계 이름, 단계 수), ...] 리스트
            websocket_session: WebSocket 세션
        """
        self.task_id = task_id
        self.stages = stages
        self.websocket_session = websocket_session

        # Calculate total steps
        self.total_steps = sum(num_steps for _, num_steps in stages)

        # Current state
        self.current_stage_index = 0
        self.current_stage_name = stages[0][0] if stages else ""
        self.current_stage_steps = stages[0][1] if stages else 0
        self.current_stage_progress = 0
        self.overall_progress = 0

        # Tracker
        self._current_tracker: Optional[ProgressTracker] = None
        self.start_time: Optional[float] = None

    async def start(self):
        """작업 시작"""
        self.start_time = time.time()
        self.current_stage_index = 0
        self.current_stage_name = self.stages[0][0]
        self.current_stage_steps = self.stages[0][1]

        # Create tracker for first stage
        self._current_tracker = ProgressTracker(
            task_id=self.task_id,
            total_steps=self.total_steps,
            websocket_session=self.websocket_session,
            stage=self.current_stage_name,
        )
        await self._current_tracker.start(message=f"Starting stage: {self.current_stage_name}")

    async def set_stage(self, stage_name: str):
        """
        현재 단계 설정

        Args:
            stage_name: 단계 이름
        """
        # Find stage
        for i, (name, num_steps) in enumerate(self.stages):
            if name == stage_name:
                self.current_stage_index = i
                self.current_stage_name = name
                self.current_stage_steps = num_steps
                self.current_stage_progress = 0

                # Update tracker stage
                if self._current_tracker:
                    self._current_tracker.stage = stage_name

                logger.info(f"Stage changed: {stage_name}")
                return

        raise ValueError(f"Unknown stage: {stage_name}")

    async def update(self, message: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        진행 상황 업데이트 (현재 단계 1 증가)

        Args:
            message: 상태 메시지
            metadata: 추가 메타데이터
        """
        if not self._current_tracker:
            return

        self.current_stage_progress += 1
        self.overall_progress += 1

        # Calculate percentage for current stage
        stage_percentage = (
            (self.current_stage_progress / self.current_stage_steps * 100)
            if self.current_stage_steps > 0
            else 0.0
        )

        # Update tracker
        await self._current_tracker.update(
            current=self.overall_progress,
            message=f"[{self.current_stage_name}] {message} ({stage_percentage:.1f}%)",
            metadata={
                "stage": self.current_stage_name,
                "stage_progress": self.current_stage_progress,
                "stage_total": self.current_stage_steps,
                "stage_percentage": stage_percentage,
                **(metadata or {}),
            },
        )

    async def complete(self, result: Optional[Dict[str, Any]] = None):
        """
        작업 완료

        Args:
            result: 최종 결과
        """
        if self._current_tracker:
            await self._current_tracker.complete(
                message="All stages completed",
                result=result,
            )

    async def error(self, error_message: str, details: Optional[Dict[str, Any]] = None):
        """
        에러 발생

        Args:
            error_message: 에러 메시지
            details: 에러 상세 정보
        """
        if self._current_tracker:
            await self._current_tracker.error(error_message, details)

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "task_id": self.task_id,
            "current_stage": self.current_stage_name,
            "current_stage_index": self.current_stage_index,
            "total_stages": len(self.stages),
            "overall_progress": self.overall_progress,
            "total_steps": self.total_steps,
            "overall_percentage": (self.overall_progress / self.total_steps * 100)
            if self.total_steps > 0
            else 0.0,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0.0,
        }
