"""
CLI Tool - Beautiful Terminal UI (main entry point).
터미널 디자인 시스템 적용
"""
from __future__ import annotations

import asyncio
import sys

from beanllm.utils.cli.cli_analyze import analyze_model
from beanllm.utils.cli.cli_commands import (
    export_models,
    list_models,
    list_providers,
    print_help,
    show_model,
    show_summary,
)
from beanllm.utils.cli.cli_scan import scan_models
from beanllm.utils.cli.cli_utils import (
    ErrorPattern,
    RICH_AVAILABLE,
    console,
    create_hybrid_manager,
    get_console,
    get_model_registry,
    print_logo,
)


def main() -> None:
    """CLI main entry point."""
    if len(sys.argv) < 2 or sys.argv[1].startswith("--theme"):
        from beanllm.ui.interactive.themes import set_theme
        from beanllm.ui.interactive.tui import run_interactive_tui

        theme_name = "dark"
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--theme" and i + 1 < len(sys.argv):
                theme_name = sys.argv[i + 1]
            elif arg.startswith("--theme="):
                theme_name = arg.split("=", 1)[1]
        set_theme(theme_name)
        asyncio.run(run_interactive_tui())
        return

    command = sys.argv[1]

    if command == "admin":
        from beanllm.utils.cli.admin_commands import main_admin

        asyncio.run(main_admin())
        return

    if command in ["scan", "analyze"]:
        asyncio.run(async_main(command))
        return

    registry = get_model_registry()
    if command == "list":
        list_models(registry)
    elif command == "show":
        if len(sys.argv) < 3:
            ErrorPattern.render(
                "Usage: beanllm show <model_name>",
                error_type="MissingArgument",
                suggestion="Provide a model name to show details",
            )
            return
        show_model(registry, sys.argv[2])
    elif command == "providers":
        list_providers(registry)
    elif command == "export":
        export_models(registry)
    elif command == "summary":
        show_summary(registry)
    else:
        print_help()


async def async_main(command: str) -> None:
    """Async 명령어 처리."""
    if command == "scan":
        await scan_models()
    elif command == "analyze":
        if len(sys.argv) < 3:
            ErrorPattern.render(
                "Usage: beanllm analyze <model_name>",
                error_type="MissingArgument",
                suggestion="Provide a model name to analyze",
            )
            return
        await analyze_model(sys.argv[2])


__all__ = [
    "main",
    "async_main",
    "print_help",
    "list_models",
    "show_model",
    "list_providers",
    "export_models",
    "show_summary",
    "scan_models",
    "analyze_model",
    "console",
    "ErrorPattern",
    "print_logo",
    "get_console",
    "get_model_registry",
    "create_hybrid_manager",
    "RICH_AVAILABLE",
]

if __name__ == "__main__":
    main()
