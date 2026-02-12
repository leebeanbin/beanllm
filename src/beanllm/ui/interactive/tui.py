"""
Interactive TUI - beanllm 진입점 (beantui 엔진 래퍼)

기존 API run_interactive_tui()를 유지하면서
내부적으로 beantui TUIEngine을 사용합니다.

이전 버전과 하위 호환성을 완벽히 유지합니다:
  - run_interactive_tui(working_dir=...) 동일 시그니처
  - main() CLI 진입점 동일
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


async def run_interactive_tui(working_dir: Optional[Path] = None) -> None:
    """beanllm 인터랙티브 TUI — beantui 엔진 기반

    Args:
        working_dir: 작업 디렉토리 (기본: cwd)
    """
    from beanllm.ui.interactive.backend import BeanllmBackend
    from beantui.config import TUIConfig
    from beantui.engine import TUIEngine

    config = TUIConfig.auto_discover()

    # beanllm 전용 오버라이드
    config.app.name = "beanllm"
    config.app.version = _get_beanllm_version()
    config.app.logo_lines = _get_beanllm_logo()

    engine = TUIEngine(
        config=config,
        backend=BeanllmBackend(),
        working_dir=working_dir,
    )
    await engine.run_async()


def _get_beanllm_version() -> str:
    """beanllm 버전 가져오기"""
    try:
        from beanllm import __version__

        return __version__
    except (ImportError, AttributeError):
        return "0.3.0"


def _get_beanllm_logo() -> list[str]:
    """beanllm ASCII 로고"""
    return [
        "██╗     ██╗     ███╗   ███╗██╗  ██╗██╗████████╗",
        "██║     ██║     ████╗ ████║██║ ██╔╝██║╚══██╔══╝",
        "██║     ██║     ██╔████╔██║█████╔╝ ██║   ██║   ",
        "██║     ██║     ██║╚██╔╝██║██╔═██╗ ██║   ██║   ",
        "███████╗███████╗██║ ╚═╝ ██║██║  ██╗██║   ██║   ",
        "╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ",
    ]


def main() -> None:
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=str, default=None, help="Working directory")
    parser.add_argument("--model", type=str, default=None, help="Default model")
    parser.add_argument("--theme", type=str, default="dark", help="Theme (dark/light)")
    args, _ = parser.parse_known_args()

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    asyncio.run(run_interactive_tui(working_dir=cwd))


if __name__ == "__main__":
    main()
