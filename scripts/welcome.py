#!/usr/bin/env python3
"""
llmkit 환영 메시지 및 빠른 시작 가이드
디자인 시스템 적용
"""
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from llmkit.ui import (
        print_logo,
        OnboardingPattern,
        InfoPattern,
        get_console,
        Badge,
        StatusIcon
    )
    from rich.table import Table
    from rich import box
    use_ui = True
except ImportError:
    use_ui = False
    print("Rich 라이브러리가 필요합니다: pip install rich")


def print_welcome():
    """환영 메시지 출력 (디자인 시스템 적용)"""
    if not use_ui:
        print("=" * 70)
        print("🚀 Welcome to llmkit!")
        print("=" * 70)
        return
    
    # 로고 출력 (도움 패키지로서 커맨드 표시)
    print_logo(style="ascii", color="magenta", show_motto=True, show_commands=True)


def print_quick_start():
    """빠른 시작 가이드 (디자인 시스템 적용)"""
    if not use_ui:
        print("\n📚 Quick Start:")
        print("  1. Set environment variables: export OPENAI_API_KEY='your-key'")
        print("  2. Try: python -c \"from llmkit import get_registry; print(get_registry().get_available_models())\"")
        return
    
    # 온보딩 패턴 사용
    OnboardingPattern.render(
        "Quick Start Guide",
        steps=[
            {
                "title": "Set environment variables",
                "description": "export OPENAI_API_KEY='your-key'"
            },
            {
                "title": "Try it out",
                "description": "from llmkit import get_registry; r = get_registry()"
            },
            {
                "title": "Use CLI",
                "description": "llmkit list"
            },
            {
                "title": "Read docs",
                "description": "https://github.com/yourusername/llmkit"
            }
        ]
    )


def print_providers_status():
    """설치된 Provider 상태 표시 (디자인 시스템 적용)"""
    if not use_ui:
        return
    
    from rich.table import Table
    
    providers = []
    
    # OpenAI
    try:
        import openai
        providers.append((StatusIcon.success(), "OpenAI", Badge.success("Installed")))
    except ImportError:
        providers.append((StatusIcon.error(), "OpenAI", "Not installed"))
    
    # Anthropic
    try:
        import anthropic
        providers.append((StatusIcon.success(), "Anthropic", Badge.success("Installed")))
    except ImportError:
        providers.append((StatusIcon.error(), "Anthropic", "Not installed"))
    
    # Gemini
    try:
        import google.generativeai
        providers.append((StatusIcon.success(), "Gemini", Badge.success("Installed")))
    except ImportError:
        providers.append((StatusIcon.warning(), "Gemini", "Optional - pip install llmkit[gemini]"))
    
    # Ollama
    try:
        import ollama
        providers.append((StatusIcon.success(), "Ollama", Badge.success("Installed")))
    except ImportError:
        providers.append((StatusIcon.warning(), "Ollama", "Optional - pip install llmkit[ollama]"))
    
    console = get_console()
    table = Table(title="[bold cyan]📦 Provider Status[/bold cyan]", box=box.ROUNDED, show_header=True)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Provider", style="cyan", width=15)
    table.add_column("Info", style="dim")
    
    for status, provider, info in providers:
        table.add_row(status, provider, info)
    
    console.print(table)
    console.print()


def main():
    """메인 함수"""
    print_welcome()
    print_providers_status()
    print_quick_start()
    
    # 추가 정보
    if use_ui:
        InfoPattern.render(
            "💡 Tip: Set LLMKIT_SHOW_BANNER=true to see this on import",
            details=["📚 Docs: https://github.com/yourusername/llmkit"]
        )


if __name__ == "__main__":
    main()
