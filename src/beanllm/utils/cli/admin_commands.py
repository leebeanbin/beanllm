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
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# beanllm imports
try:
    from beanllm.infrastructure.distributed.google_events import (
        get_google_export_stats,
        get_security_events
    )
    from beanllm.facade.core.client_facade import Client
    BEANLLM_AVAILABLE = True
except ImportError:
    BEANLLM_AVAILABLE = False


console = Console() if RICH_AVAILABLE else None


def print_error(message: str):
    """에러 메시지 출력"""
    if RICH_AVAILABLE:
        console.print(f"[bold red]❌ Error:[/bold red] {message}")
    else:
        print(f"❌ Error: {message}")


def print_success(message: str):
    """성공 메시지 출력"""
    if RICH_AVAILABLE:
        console.print(f"[bold green]✅ Success:[/bold green] {message}")
    else:
        print(f"✅ Success: {message}")


def print_info(message: str):
    """정보 메시지 출력"""
    if RICH_AVAILABLE:
        console.print(f"[bold blue]ℹ️  Info:[/bold blue] {message}")
    else:
        print(f"ℹ️  Info: {message}")


async def analyze_with_gemini(hours: int = 24) -> None:
    """
    Gemini를 사용하여 Google 서비스 사용 패턴 분석

    사용자가 유료 결제한 Gemini API 키를 활용하여 추가 비용 없이 분석

    Args:
        hours: 분석할 기간 (시간)
    """
    if not BEANLLM_AVAILABLE:
        print_error("beanllm is not installed or not available")
        return

    # Gemini API 키 확인
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print_error("GEMINI_API_KEY not found in environment variables")
        print_info("Set your Gemini API key: export GEMINI_API_KEY='your-key'")
        return

    try:
        # 1. 통계 조회
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading Google export statistics...", total=None)
                stats = await get_google_export_stats(hours=hours)
                progress.remove_task(task)
        else:
            print("Loading Google export statistics...")
            stats = await get_google_export_stats(hours=hours)

        # 2. Gemini 분석 요청
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🤖 Analyzing with Gemini...[/bold cyan]\n")
        else:
            print("\n🤖 Analyzing with Gemini...\n")

        # Gemini 프롬프트 생성
        prompt = f"""
다음은 지난 {hours}시간 동안의 Google Workspace 서비스 사용 통계입니다.

**총 내보내기 수**: {stats['total_exports']}

**서비스별 사용량**:
{json.dumps(stats['by_service'], indent=2, ensure_ascii=False)}

**상위 사용자 (Top 10)**:
{json.dumps([{"user": u, "count": c} for u, c in stats['top_users']], indent=2, ensure_ascii=False)}

위 데이터를 분석하여 다음 정보를 제공해주세요:

1. **사용 패턴 분석**:
   - 가장 많이 사용되는 서비스와 그 이유
   - 시간대별 사용 패턴 추론
   - 이상 징후 (unusual patterns)

2. **비용 최적화 제안**:
   - MongoDB/Redis 비용 절감 방안
   - API 호출 최적화 방법

3. **보안 권고사항**:
   - 비정상적인 사용 패턴
   - Rate limit 조정 필요 여부
   - 모니터링 강화가 필요한 영역

4. **사용자 경험 개선**:
   - 자주 사용되는 기능 강화 방안
   - UI/UX 개선 제안

간결하고 실행 가능한 조언을 제공해주세요.
"""

        # Gemini 호출 (유료 결제한 API 키 사용 - 추가 비용 없음)
        client = Client(model="gemini-2.0-flash-exp", provider="google")
        response = await client.chat([
            {"role": "user", "content": prompt}
        ])

        # 3. 결과 출력
        if RICH_AVAILABLE:
            # Rich 테이블로 통계 표시
            table = Table(title=f"Google Export Statistics (Last {hours} hours)")
            table.add_column("Service", style="cyan")
            table.add_column("Count", style="magenta", justify="right")

            for service, count in stats['by_service'].items():
                table.add_row(service.capitalize(), str(count))

            console.print(table)
            console.print()

            # Gemini 분석 결과
            panel = Panel(
                response.content,
                title="🤖 Gemini Analysis",
                border_style="green",
                padding=(1, 2)
            )
            console.print(panel)
        else:
            # Plain text 출력
            print("\n" + "=" * 60)
            print(f"Google Export Statistics (Last {hours} hours)")
            print("=" * 60)
            for service, count in stats['by_service'].items():
                print(f"{service.capitalize()}: {count}")
            print("\n" + "=" * 60)
            print("Gemini Analysis")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

    except Exception as e:
        print_error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


async def show_stats(hours: int = 24) -> None:
    """
    Google 서비스 통계 조회 (Gemini 사용 안 함)

    Args:
        hours: 조회할 기간 (시간)
    """
    if not BEANLLM_AVAILABLE:
        print_error("beanllm is not installed or not available")
        return

    try:
        # 통계 조회
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading statistics...", total=None)
                stats = await get_google_export_stats(hours=hours)
                progress.remove_task(task)
        else:
            print("Loading statistics...")
            stats = await get_google_export_stats(hours=hours)

        # Rich 테이블 출력
        if RICH_AVAILABLE:
            # 서비스별 통계
            service_table = Table(title=f"📊 Google Service Usage (Last {hours} hours)")
            service_table.add_column("Service", style="cyan", no_wrap=True)
            service_table.add_column("Exports", style="magenta", justify="right")
            service_table.add_column("Percentage", style="green", justify="right")

            total = stats['total_exports']
            for service, count in sorted(stats['by_service'].items(), key=lambda x: x[1], reverse=True):
                percentage = f"{(count / total * 100):.1f}%" if total > 0 else "0%"
                service_table.add_row(service.capitalize(), str(count), percentage)

            console.print()
            console.print(service_table)

            # 상위 사용자
            if stats['top_users']:
                user_table = Table(title=f"\n👥 Top Users (Last {hours} hours)")
                user_table.add_column("Rank", style="yellow", justify="right")
                user_table.add_column("User ID", style="cyan")
                user_table.add_column("Export Count", style="magenta", justify="right")

                for i, (user_id, count) in enumerate(stats['top_users'], 1):
                    user_table.add_row(str(i), user_id, str(count))

                console.print()
                console.print(user_table)
                console.print()

            # 요약
            summary_panel = Panel(
                f"[bold]Total Exports:[/bold] {stats['total_exports']}\n"
                f"[bold]Active Users:[/bold] {len(stats['top_users'])}\n"
                f"[bold]Most Popular:[/bold] {max(stats['by_service'], key=stats['by_service'].get) if stats['by_service'] else 'N/A'}",
                title="📈 Summary",
                border_style="blue"
            )
            console.print(summary_panel)
        else:
            # Plain text 출력
            print("\n" + "=" * 60)
            print(f"Google Service Usage (Last {hours} hours)")
            print("=" * 60)
            total = stats['total_exports']
            for service, count in sorted(stats['by_service'].items(), key=lambda x: x[1], reverse=True):
                percentage = f"{(count / total * 100):.1f}%" if total > 0 else "0%"
                print(f"{service.capitalize()}: {count} ({percentage})")

            print("\n" + "=" * 60)
            print(f"Top Users (Last {hours} hours)")
            print("=" * 60)
            for i, (user_id, count) in enumerate(stats['top_users'], 1):
                print(f"{i}. {user_id}: {count} exports")

            print("\n" + "=" * 60)
            print(f"Total Exports: {stats['total_exports']}")
            print(f"Active Users: {len(stats['top_users'])}")
            print("=" * 60)

    except Exception as e:
        print_error(f"Failed to load statistics: {e}")


async def optimize_with_gemini() -> None:
    """
    Gemini를 사용하여 비용 최적화 제안 생성
    """
    if not BEANLLM_AVAILABLE:
        print_error("beanllm is not installed or not available")
        return

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print_error("GEMINI_API_KEY not found")
        return

    try:
        # 24시간 통계
        stats_24h = await get_google_export_stats(hours=24)
        # 7일 통계
        stats_7d = await get_google_export_stats(hours=24 * 7)

        if RICH_AVAILABLE:
            console.print("\n[bold cyan]💰 Generating cost optimization recommendations with Gemini...[/bold cyan]\n")
        else:
            print("\n💰 Generating cost optimization recommendations with Gemini...\n")

        # Gemini 프롬프트
        prompt = f"""
현재 시스템 구성:
- MongoDB Atlas: Free tier (512MB)
- Upstash Redis: Free tier (10K commands/day)
- 사용자: ~100-1000명 예상

사용 통계:
- 24시간 내보내기: {stats_24h['total_exports']}건
- 7일 내보내기: {stats_7d['total_exports']}건
- 서비스별 (24h): {json.dumps(stats_24h['by_service'], ensure_ascii=False)}

비용 최적화 방안을 제안해주세요:
1. MongoDB/Redis 무료 티어 초과 방지 전략
2. API 호출 최적화 (rate limiting, batching)
3. 캐싱 전략 개선
4. 데이터 정리 정책 (TTL, archiving)
5. 예상 월간 비용 분석

구체적이고 실행 가능한 조언을 제공해주세요.
"""

        client = Client(model="gemini-2.0-flash-exp", provider="google")
        response = await client.chat([{"role": "user", "content": prompt}])

        # 결과 출력
        if RICH_AVAILABLE:
            panel = Panel(
                response.content,
                title="💰 Cost Optimization Recommendations",
                border_style="yellow",
                padding=(1, 2)
            )
            console.print(panel)
        else:
            print("\n" + "=" * 60)
            print("Cost Optimization Recommendations")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

    except Exception as e:
        print_error(f"Optimization analysis failed: {e}")


async def check_security(hours: int = 24) -> None:
    """
    보안 이벤트 조회 및 Gemini 분석

    Args:
        hours: 조회할 기간 (시간)
    """
    if not BEANLLM_AVAILABLE:
        print_error("beanllm is not installed or not available")
        return

    try:
        # 보안 이벤트 조회
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading security events...", total=None)
                events = await get_security_events(hours=hours, severity="high")
                progress.remove_task(task)
        else:
            print("Loading security events...")
            events = await get_security_events(hours=hours, severity="high")

        # 이벤트 출력
        if not events:
            print_success(f"No high-severity security events in the last {hours} hours")
            return

        if RICH_AVAILABLE:
            # 이벤트 테이블
            table = Table(title=f"🔒 Security Events (Last {hours} hours)")
            table.add_column("Time", style="cyan")
            table.add_column("User", style="yellow")
            table.add_column("Reason", style="red")
            table.add_column("Severity", style="magenta")

            for event in events[:10]:  # 최근 10개만
                timestamp = event.get("timestamp", "")
                user_id = event.get("user_id", "unknown")
                reason = event.get("reason", "")
                severity = event.get("severity", "")

                table.add_row(timestamp, user_id, reason, severity)

            console.print()
            console.print(table)
        else:
            print("\n" + "=" * 60)
            print(f"Security Events (Last {hours} hours)")
            print("=" * 60)
            for event in events[:10]:
                print(f"Time: {event.get('timestamp')}")
                print(f"User: {event.get('user_id')}")
                print(f"Reason: {event.get('reason')}")
                print(f"Severity: {event.get('severity')}")
                print("-" * 60)

        # Gemini 분석 (선택적)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and events:
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]🤖 Analyzing security events with Gemini...[/bold cyan]\n")
            else:
                print("\n🤖 Analyzing security events with Gemini...\n")

            prompt = f"""
다음은 최근 {hours}시간 동안 발생한 보안 이벤트입니다:

{json.dumps(events[:5], indent=2, ensure_ascii=False)}

보안 분석을 제공해주세요:
1. 위협 수준 평가
2. 대응 방안
3. 예방 조치
4. 모니터링 강화 필요 영역

간결하고 실행 가능한 조언을 제공해주세요.
"""

            client = Client(model="gemini-2.0-flash-exp", provider="google")
            response = await client.chat([{"role": "user", "content": prompt}])

            if RICH_AVAILABLE:
                panel = Panel(
                    response.content,
                    title="🛡️ Security Analysis",
                    border_style="red",
                    padding=(1, 2)
                )
                console.print(panel)
            else:
                print("\n" + "=" * 60)
                print("Security Analysis")
                print("=" * 60)
                print(response.content)
                print("=" * 60)

    except Exception as e:
        print_error(f"Security check failed: {e}")


def launch_dashboard() -> None:
    """
    Streamlit 대시보드 실행
    """
    try:
        import subprocess

        # Streamlit이 설치되어 있는지 확인
        result = subprocess.run(
            ["streamlit", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print_error("Streamlit is not installed")
            print_info("Install with: pip install streamlit")
            return

        # 대시보드 파일 경로
        dashboard_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",  # Go up to project root
            "admin", "dashboard.py"
        )

        if not os.path.exists(dashboard_path):
            print_error(f"Dashboard not found: {dashboard_path}")
            print_info("Dashboard will be created in the next step")
            return

        print_info(f"Launching Streamlit dashboard: {dashboard_path}")
        print_info("Dashboard will open in your browser...")

        # Streamlit 실행
        subprocess.run([
            "streamlit", "run", dashboard_path,
            "--server.port=8501",
            "--server.headless=true"
        ])

    except Exception as e:
        print_error(f"Failed to launch dashboard: {e}")


async def main_admin():
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


def print_help_admin():
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
        console.print(Panel(help_text, title="Admin CLI Help", border_style="cyan"))
    else:
        print(help_text)


if __name__ == "__main__":
    asyncio.run(main_admin())
