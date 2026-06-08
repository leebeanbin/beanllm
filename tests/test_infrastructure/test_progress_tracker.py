"""
Tests for infrastructure/streaming/progress_tracker.py

Covers ProgressUpdate, ProgressTracker, and MultiStageProgressTracker.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.infrastructure.streaming.progress_tracker import (
    MultiStageProgressTracker,
    ProgressTracker,
    ProgressUpdate,
)

# ---------------------------------------------------------------------------
# ProgressUpdate tests
# ---------------------------------------------------------------------------


class TestProgressUpdate:
    def test_percentage_calculated_on_init(self):
        update = ProgressUpdate(stage="test", current=50, total=100)
        assert update.percentage == 50.0

    def test_percentage_zero_when_total_is_zero(self):
        update = ProgressUpdate(stage="test", current=0, total=0)
        assert update.percentage == 0.0

    def test_percentage_100_when_complete(self):
        update = ProgressUpdate(stage="test", current=10, total=10)
        assert update.percentage == 100.0

    def test_message_defaults_to_empty_string(self):
        update = ProgressUpdate(stage="test", current=1, total=10)
        assert update.message == ""

    def test_elapsed_time_defaults_to_zero(self):
        update = ProgressUpdate(stage="test", current=1, total=10)
        assert update.elapsed_time == 0.0

    def test_metadata_defaults_to_none(self):
        update = ProgressUpdate(stage="test", current=0, total=5)
        assert update.metadata is None

    def test_estimated_remaining_defaults_to_none(self):
        update = ProgressUpdate(stage="test", current=0, total=5)
        assert update.estimated_remaining is None

    def test_custom_message_is_stored(self):
        update = ProgressUpdate(stage="extracting", current=3, total=10, message="Processing")
        assert update.message == "Processing"


# ---------------------------------------------------------------------------
# ProgressTracker tests
# ---------------------------------------------------------------------------


def make_mock_session():
    """Return a fully mocked WebSocket session."""
    session = MagicMock()
    session.send_progress = AsyncMock(return_value=True)
    session.send_complete = AsyncMock(return_value=True)
    session.send_error = AsyncMock(return_value=True)
    return session


class TestProgressTrackerInit:
    def test_initializes_with_defaults(self):
        tracker = ProgressTracker(task_id="task-1", total_steps=100)
        assert tracker.task_id == "task-1"
        assert tracker.total_steps == 100
        assert tracker.current_step == 0
        assert tracker.is_complete is False
        assert tracker.is_cancelled is False
        assert tracker.start_time is None

    def test_initializes_with_custom_stage(self):
        tracker = ProgressTracker(task_id="t", total_steps=10, stage="custom")
        assert tracker.stage == "custom"

    def test_initializes_without_websocket(self):
        tracker = ProgressTracker(task_id="t", total_steps=5)
        assert tracker.websocket_session is None


class TestProgressTrackerStart:
    async def test_start_sets_start_time(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        assert tracker.start_time is not None

    async def test_start_resets_current_step(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        tracker.current_step = 5
        await tracker.start()
        assert tracker.current_step == 0

    async def test_start_sends_websocket_update(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=10, websocket_session=session)
        await tracker.start("Starting now")
        session.send_progress.assert_called_once()

    async def test_start_without_session_does_not_raise(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()  # Should not raise


class TestProgressTrackerUpdate:
    async def test_update_increments_current_step(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        await tracker.update()
        assert tracker.current_step == 1

    async def test_update_with_current_sets_absolute(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        await tracker.update(current=7)
        assert tracker.current_step == 7

    async def test_update_with_increment(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        await tracker.update(increment=3)
        assert tracker.current_step == 3

    async def test_update_sends_websocket_message(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=10, websocket_session=session)
        await tracker.start()
        session.send_progress.reset_mock()
        await tracker.update(message="step 1")
        session.send_progress.assert_called_once()

    async def test_update_skipped_when_complete(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.start()
        tracker.is_complete = True
        session.send_progress.reset_mock()
        await tracker.update()
        session.send_progress.assert_not_called()

    async def test_update_skipped_when_cancelled(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.start()
        tracker.is_cancelled = True
        session.send_progress.reset_mock()
        await tracker.update()
        session.send_progress.assert_not_called()

    async def test_update_tracks_step_times_when_enabled(self):
        tracker = ProgressTracker(task_id="t", total_steps=5, enable_time_estimation=True)
        await tracker.start()
        await tracker.update()
        assert len(tracker._step_times) == 1


class TestProgressTrackerComplete:
    async def test_complete_sets_is_complete(self):
        tracker = ProgressTracker(task_id="t", total_steps=5)
        await tracker.start()
        await tracker.complete()
        assert tracker.is_complete is True

    async def test_complete_sets_current_to_total(self):
        tracker = ProgressTracker(task_id="t", total_steps=5)
        await tracker.start()
        await tracker.complete()
        assert tracker.current_step == tracker.total_steps

    async def test_complete_sends_websocket_complete(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.start()
        await tracker.complete(result={"nodes": 10})
        session.send_complete.assert_called_once()

    async def test_complete_second_call_ignored(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.start()
        await tracker.complete()
        session.send_complete.reset_mock()
        await tracker.complete()
        session.send_complete.assert_not_called()

    async def test_complete_without_session_does_not_raise(self):
        tracker = ProgressTracker(task_id="t", total_steps=5)
        await tracker.start()
        await tracker.complete()


class TestProgressTrackerError:
    async def test_error_sets_is_cancelled(self):
        tracker = ProgressTracker(task_id="t", total_steps=5)
        await tracker.error("Something went wrong")
        assert tracker.is_cancelled is True

    async def test_error_sends_websocket_error(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.error("Oops")
        session.send_error.assert_called_once()

    async def test_error_without_session_does_not_raise(self):
        tracker = ProgressTracker(task_id="t", total_steps=5)
        await tracker.error("fail")

    async def test_error_with_details(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.error("fail", details={"code": 500})
        call_kwargs = session.send_error.call_args[1]
        assert "details" in call_kwargs


class TestProgressTrackerCancel:
    async def test_cancel_sets_is_cancelled(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        await tracker.cancel()
        assert tracker.is_cancelled is True

    async def test_cancel_when_complete_does_nothing(self):
        session = make_mock_session()
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.start()
        await tracker.complete()
        session.send_progress.reset_mock()
        await tracker.cancel()
        # Since is_complete is True, cancel should return early
        assert tracker.is_cancelled is False  # was not set by cancel


class TestProgressTrackerGetStatus:
    async def test_get_status_returns_dict(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        status = tracker.get_status()
        assert isinstance(status, dict)
        assert status["task_id"] == "t"

    async def test_get_status_percentage_correct(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        await tracker.update(current=5)
        status = tracker.get_status()
        assert status["percentage"] == 50.0

    async def test_get_status_zero_percentage_when_total_zero(self):
        tracker = ProgressTracker(task_id="t", total_steps=0)
        status = tracker.get_status()
        assert status["percentage"] == 0.0

    async def test_get_status_elapsed_time_positive(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        status = tracker.get_status()
        assert status["elapsed_time"] >= 0.0


class TestProgressTrackerTimeEstimation:
    async def test_estimate_remaining_none_before_start(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        result = tracker._estimate_remaining_time(0)
        assert result is None

    async def test_estimate_remaining_none_when_disabled(self):
        tracker = ProgressTracker(task_id="t", total_steps=10, enable_time_estimation=False)
        result = tracker._estimate_remaining_time(5)
        assert result is None

    async def test_get_elapsed_zero_before_start(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        assert tracker._get_elapsed_time() == 0.0

    async def test_estimate_remaining_returns_float_after_progress(self):
        tracker = ProgressTracker(task_id="t", total_steps=10)
        await tracker.start()
        # Simulate some time by setting start_time in the past
        tracker.start_time = time.time() - 2.0
        result = tracker._estimate_remaining_time(5)
        assert result is not None
        assert result >= 0.0


# ---------------------------------------------------------------------------
# WebSocket session error handling in _send_update
# ---------------------------------------------------------------------------


class TestSendUpdateErrorHandling:
    async def test_send_update_handles_websocket_exception(self):
        """_send_update should not propagate exceptions from websocket."""
        session = MagicMock()
        session.send_progress = AsyncMock(side_effect=Exception("WebSocket closed"))
        tracker = ProgressTracker(task_id="t", total_steps=5, websocket_session=session)
        await tracker.start()
        await tracker.update(message="test")  # Should not raise


# ---------------------------------------------------------------------------
# MultiStageProgressTracker tests
# ---------------------------------------------------------------------------


class TestMultiStageProgressTracker:
    def _make_tracker(self, stages=None, session=None):
        if stages is None:
            stages = [("extract", 10), ("build", 5), ("store", 5)]
        return MultiStageProgressTracker(
            task_id="multi-1",
            stages=stages,
            websocket_session=session,
        )

    def test_init_calculates_total_steps(self):
        tracker = self._make_tracker()
        assert tracker.total_steps == 20

    def test_init_sets_first_stage(self):
        tracker = self._make_tracker()
        assert tracker.current_stage_name == "extract"

    async def test_start_initializes_tracker(self):
        tracker = self._make_tracker()
        await tracker.start()
        assert tracker._current_tracker is not None
        assert tracker.start_time is not None

    async def test_update_increments_overall_progress(self):
        tracker = self._make_tracker()
        await tracker.start()
        await tracker.update(message="item 1")
        assert tracker.overall_progress == 1
        assert tracker.current_stage_progress == 1

    async def test_set_stage_changes_current_stage(self):
        tracker = self._make_tracker()
        await tracker.start()
        await tracker.set_stage("build")
        assert tracker.current_stage_name == "build"
        assert tracker.current_stage_steps == 5

    async def test_set_stage_unknown_raises_value_error(self):
        tracker = self._make_tracker()
        await tracker.start()
        with pytest.raises(ValueError, match="Unknown stage"):
            await tracker.set_stage("nonexistent")

    async def test_complete_marks_inner_tracker_complete(self):
        session = make_mock_session()
        tracker = self._make_tracker(session=session)
        await tracker.start()
        await tracker.complete(result={"ok": True})
        assert tracker._current_tracker.is_complete is True

    async def test_error_delegates_to_inner_tracker(self):
        tracker = self._make_tracker()
        await tracker.start()
        await tracker.error("Something failed")
        assert tracker._current_tracker.is_cancelled is True

    async def test_get_status_returns_dict(self):
        tracker = self._make_tracker()
        await tracker.start()
        status = tracker.get_status()
        assert isinstance(status, dict)
        assert status["task_id"] == "multi-1"
        assert "overall_progress" in status

    async def test_get_status_total_stages(self):
        tracker = self._make_tracker()
        await tracker.start()
        status = tracker.get_status()
        assert status["total_stages"] == 3

    async def test_update_without_tracker_does_not_raise(self):
        tracker = self._make_tracker()
        # Don't call start(), _current_tracker is None
        await tracker.update(message="should not raise")  # Should not raise

    async def test_set_stage_resets_stage_progress(self):
        tracker = self._make_tracker()
        await tracker.start()
        await tracker.update()
        await tracker.update()
        await tracker.set_stage("build")
        assert tracker.current_stage_progress == 0

    async def test_overall_percentage_after_progress(self):
        tracker = self._make_tracker()
        await tracker.start()
        for _ in range(10):
            await tracker.update()
        status = tracker.get_status()
        assert status["overall_percentage"] == 50.0

    async def test_elapsed_time_in_status(self):
        tracker = self._make_tracker()
        await tracker.start()
        status = tracker.get_status()
        assert status["elapsed_time"] >= 0.0

    async def test_elapsed_time_zero_before_start(self):
        tracker = self._make_tracker()
        status = tracker.get_status()
        assert status["elapsed_time"] == 0.0
