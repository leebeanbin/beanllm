"""
Admin System Commands - System/config commands

System and configuration-related admin CLI commands:
- Dashboard launcher
- Help text
- Console utilities
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

_console: Optional["Console"] = Console() if RICH_AVAILABLE else None


def get_console() -> "Console":
    """Get console instance (type-safe helper)"""
    assert _console is not None, "Rich is required for this operation"
    return _console


# Alias for backward compatibility
console = _console


def print_error(message: str) -> None:
    """에러 메시지 출력"""
    if RICH_AVAILABLE and _console is not None:
        get_console().print(f"[bold red]❌ Error:[/bold red] {message}")
    else:
        print(f"❌ Error: {message}")


def print_success(message: str) -> None:
    """성공 메시지 출력"""
    if RICH_AVAILABLE and _console is not None:
        get_console().print(f"[bold green]✅ Success:[/bold green] {message}")
    else:
        print(f"✅ Success: {message}")


def print_info(message: str) -> None:
    """정보 메시지 출력"""
    if RICH_AVAILABLE and _console is not None:
        get_console().print(f"[bold blue]ℹ️  Info:[/bold blue] {message}")
    else:
        print(f"ℹ️  Info: {message}")


def launch_dashboard() -> None:
    """
    Streamlit 대시보드 실행
    """
    try:
        # Streamlit이 설치되어 있는지 확인
        result = subprocess.run(["streamlit", "--version"], capture_output=True, text=True)

        if result.returncode != 0:
            print_error("Streamlit is not installed")
            print_info("Install with: pip install streamlit")
            return

        # 대시보드 파일 경로
        dashboard_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",  # Go up to project root
            "admin",
            "dashboard.py",
        )

        if not os.path.exists(dashboard_path):
            print_error(f"Dashboard not found: {dashboard_path}")
            print_info("Dashboard will be created in the next step")
            return

        print_info(f"Launching Streamlit dashboard: {dashboard_path}")
        print_info("Dashboard will open in your browser...")

        # Streamlit 실행
        subprocess.run(
            ["streamlit", "run", dashboard_path, "--server.port=8501", "--server.headless=true"]
        )

    except Exception as e:
        print_error(f"Failed to launch dashboard: {e}")


def print_help_admin() -> None:
    """Admin CLI 도움말 출력"""
    help_text = """
beanllm Admin CLI - Google Workspace Monitoring

Usage:
    beanllm admin <command> [options]

Commands:
    analyze     Gemini로 Google 서비스 사용 패턴 분석 (유료 API 키 사용, 추가 비용 없음)
    stats       Google 서비스 통계 조회
    optimize    Gemini로 비용 최적화 제안 생성
    security    보안 이벤트 조회 및 분석
    dashboard   Streamlit 대시보드 실행

Options:
    --hours=N   조회할 기간 (시간, 기본: 24)

Examples:
    beanllm admin analyze                 # 24시간 데이터 분석
    beanllm admin analyze --hours=168     # 7일 데이터 분석
    beanllm admin stats                   # 통계 조회
    beanllm admin optimize                # 비용 최적화 제안
    beanllm admin security --hours=72     # 72시간 보안 이벤트 확인
    beanllm admin dashboard               # 대시보드 실행

Requirements:
    - GEMINI_API_KEY: Gemini API 키 (analyze, optimize 명령에 필요)
    - MONGODB_URI: MongoDB 연결 URI (통계 조회에 필요)
    - Streamlit: 대시보드 실행에 필요 (pip install streamlit)
"""

    if RICH_AVAILABLE:
        get_console().print(Panel(help_text, title="Admin CLI Help", border_style="cyan"))
    else:
        print(help_text)
