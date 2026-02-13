"""
Unified Evaluator - 평가 히스토리 저장/로드 (PersistenceMixin)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from beanllm.domain.evaluation.unified_models import EvalRecord

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class PersistenceMixin:
    """
    평가 히스토리 저장/로드 Mixin.

    사용처에서 persist_path, _records, _query_to_record 속성이 필요합니다.
    """

    persist_path: Path | None
    _records: Dict[str, EvalRecord]
    _query_to_record: Dict[str, str]

    def _save_history(self) -> None:
        """평가 히스토리 저장"""
        if not self.persist_path:
            return

        history_file = self.persist_path / "eval_history.json"

        data = {
            "records": [
                {
                    "record_id": r.record_id,
                    "query": r.query,
                    "response": r.response[:500],  # 응답 일부만 저장
                    "auto_scores": r.auto_scores,
                    "auto_avg_score": r.auto_avg_score,
                    "human_ratings": r.human_ratings,
                    "human_avg_rating": r.human_avg_rating,
                    "human_feedback_count": r.human_feedback_count,
                    "human_comments": r.human_comments,
                    "unified_score": r.unified_score,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self._records.values()
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved {len(self._records)} records to {history_file}")

    def _load_history(self) -> None:
        """평가 히스토리 로드"""
        if not self.persist_path:
            return

        history_file = self.persist_path / "eval_history.json"

        if not history_file.exists():
            return

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for r in data.get("records", []):
                record = EvalRecord(
                    record_id=r["record_id"],
                    query=r["query"],
                    response=r.get("response", ""),
                    contexts=[],
                    auto_scores=r.get("auto_scores", {}),
                    auto_avg_score=r.get("auto_avg_score", 0.0),
                    human_ratings=r.get("human_ratings", []),
                    human_avg_rating=r.get("human_avg_rating", 0.0),
                    human_feedback_count=r.get("human_feedback_count", 0),
                    human_comments=r.get("human_comments", []),
                    unified_score=r.get("unified_score", 0.0),
                    timestamp=datetime.fromisoformat(
                        r.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                )
                self._records[record.record_id] = record
                self._query_to_record[record.query] = record.record_id

            logger.info(f"Loaded {len(self._records)} records from {history_file}")

        except Exception as e:
            logger.error(f"Failed to load history: {e}")
