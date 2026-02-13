"""
Google Workspace 이벤트 로깅
관리자 모니터링을 위한 Google 서비스 사용 이벤트 추적
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from beanllm.infrastructure.distributed import get_event_logger
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


async def log_google_export(
    user_id: str, export_type: str, metadata: Dict[str, Any], session_id: Optional[str] = None
) -> None:
    """
    Google 서비스 내보내기 이벤트 로깅

    Args:
        user_id: 사용자 ID
        export_type: 내보내기 유형 (docs, drive, gmail, sheets)
        metadata: 추가 메타데이터 (doc_id, file_id, recipients 등)
        session_id: 세션 ID (선택)

    Example:
        await log_google_export(
            user_id="user123",
            export_type="docs",
            metadata={"doc_id": "abc123", "message_count": 50}
        )
    """
    event_logger = get_event_logger()

    event_data = {
        "user_id": user_id,
        "export_type": export_type,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        **metadata,
    }

    await event_logger.log_event(
        event_type=f"google.export.{export_type}", event_data=event_data, level="info"
    )

    logger.info(f"Google {export_type} export: user={user_id}, session={session_id}")


async def log_abnormal_activity(
    user_id: str, reason: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """
    이상 활동 로깅 (관리자 알림용)

    Args:
        user_id: 사용자 ID
        reason: 이상 활동 이유
        details: 추가 세부 정보

    Example:
        await log_abnormal_activity(
            user_id="user123",
            reason="rate_limit_exceeded",
            details={"count": 50, "threshold": 10}
        )
    """
    event_logger = get_event_logger()

    event_data = {
        "user_id": user_id,
        "reason": reason,
        "severity": "high",
        "timestamp": datetime.utcnow().isoformat(),
        **(details or {}),
    }

    await event_logger.log_event(
        event_type="security.abnormal_activity", event_data=event_data, level="warning"
    )

    logger.warning(f"Abnormal activity detected: user={user_id}, reason={reason}")


async def log_admin_action(
    admin_id: str,
    action: str,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    관리자 액션 로깅

    Args:
        admin_id: 관리자 ID
        action: 액션 (restrict_user, adjust_rate_limit, etc)
        target: 대상 (user_id 등)
        metadata: 추가 메타데이터

    Example:
        await log_admin_action(
            admin_id="admin",
            action="restrict_user",
            target="user123",
            metadata={"reason": "spam"}
        )
    """
    event_logger = get_event_logger()

    event_data = {
        "admin_id": admin_id,
        "action": action,
        "target": target,
        "timestamp": datetime.utcnow().isoformat(),
        **(metadata or {}),
    }

    await event_logger.log_event(
        event_type=f"admin.action.{action}", event_data=event_data, level="info"
    )

    logger.info(f"Admin action: admin={admin_id}, action={action}, target={target}")


async def get_google_export_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Google 서비스 내보내기 통계 조회

    Args:
        hours: 조회 기간 (시간)

    Returns:
        통계 정보 (서비스별 사용 횟수, 사용자별 통계 등)

    Example:
        stats = await get_google_export_stats(hours=24)
        # {
        #     "total_exports": 234,
        #     "by_service": {"docs": 120, "drive": 80, "gmail": 34},
        #     "top_users": [("user123", 50), ("user456", 30)]
        # }
    """
    import os
    from datetime import timedelta

    from motor.motor_asyncio import AsyncIOMotorClient

    mongo = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    db = mongo.beanllm

    since = datetime.utcnow() - timedelta(hours=hours)

    # 서비스별 통계
    by_service = {}
    for service in ["docs", "drive", "gmail", "sheets"]:
        count = await db.events.count_documents(
            {
                "event_type": f"google.export.{service}",
                "data.timestamp": {"$gte": since.isoformat()},
            }
        )
        by_service[service] = count

    total_exports = sum(by_service.values())

    # 사용자별 통계 (Top 10)
    pipeline = [
        {
            "$match": {
                "event_type": {"$regex": "^google.export"},
                "data.timestamp": {"$gte": since.isoformat()},
            }
        },
        {"$group": {"_id": "$data.user_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]

    top_users_cursor = db.events.aggregate(pipeline)
    top_users = [(doc["_id"], doc["count"]) async for doc in top_users_cursor]

    return {
        "total_exports": total_exports,
        "by_service": by_service,
        "top_users": top_users,
        "period_hours": hours,
    }


async def get_security_events(hours: int = 24, severity: str = "high") -> List[Dict[str, Any]]:
    """
    보안 이벤트 조회

    Args:
        hours: 조회 기간
        severity: 심각도 (low, medium, high)

    Returns:
        보안 이벤트 목록
    """
    import os
    from datetime import timedelta

    from motor.motor_asyncio import AsyncIOMotorClient

    mongo = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    db = mongo.beanllm

    since = datetime.utcnow() - timedelta(hours=hours)

    events = (
        await db.events.find(
            {
                "event_type": {"$regex": "^security"},
                "data.timestamp": {"$gte": since.isoformat()},
                "data.severity": severity,
            }
        )
        .sort("data.timestamp", -1)
        .limit(100)
        .to_list(length=100)
    )

    return [event.get("data", {}) for event in events]
