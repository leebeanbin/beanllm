"""
Admin CLI Commands - Google Workspace Monitoring
관리자용 CLI 명령어: Gemini를 활용한 모니터링 및 분석

Usage:
    beanllm admin analyze    - Gemini로 Google 서비스 사용 패턴 분석
    beanllm admin stats      - Google 서비스 통계 조회
    beanllm admin optimize   - Gemini로 비용 최적화 제안 생성
    beanllm admin security   - 보안 이벤트 조회 및 분석
    beanllm admin dashboard  - Streamlit 대시보드 실행
"""

from __future__ import annotations

import asyncio
import sys

from beanllm.utils.cli.admin_data_commands import (
    analyze_with_gemini,
    check_security,
    optimize_with_gemini,
    show_stats,
)
from beanllm.utils.cli.admin_system_commands import (
    launch_dashboard,
    print_error,
    print_help_admin,
)


async def main_admin() -> None:
    """Admin CLI 메인 엔트리포인트"""
    if len(sys.argv) < 3:
        print_help_admin()
        return

    subcommand = sys.argv[2]

    # 시간 옵션 파싱
    hours = 24
    if len(sys.argv) > 3 and sys.argv[3].startswith("--hours="):
        try:
            hours = int(sys.argv[3].split("=")[1])
        except ValueError:
            print_error("Invalid hours value")
            return

    if subcommand == "analyze":
        await analyze_with_gemini(hours=hours)
    elif subcommand == "stats":
        await show_stats(hours=hours)
    elif subcommand == "optimize":
        await optimize_with_gemini()
    elif subcommand == "security":
        await check_security(hours=hours)
    elif subcommand == "dashboard":
        launch_dashboard()
    else:
        print_help_admin()


# Re-export for backward compatibility
__all__ = [
    "analyze_with_gemini",
    "show_stats",
    "optimize_with_gemini",
    "check_security",
    "launch_dashboard",
    "print_help_admin",
    "main_admin",
]

if __name__ == "__main__":
    asyncio.run(main_admin())
