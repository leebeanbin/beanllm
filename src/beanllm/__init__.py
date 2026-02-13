"""
beanllm - Unified toolkit for managing and using multiple LLM providers
환경변수 기반 LLM 모델 활성화 및 관리 패키지
"""

import importlib

from beanllm._lazy_imports import (
    _LAZY_IMPORT_MAP,
    _OPTIONAL_LAZY_IMPORT_MAP,
    _PUBLIC_API,
)

__all__ = _PUBLIC_API

__version__ = "0.3.0"


# 설치 안내 메시지 (선택적 의존성)
def _check_optional_dependencies() -> None:
    """선택적 의존성 확인 및 안내 (디자인 시스템 적용)"""
    import sys

    # UI 모듈 import (에러 발생해도 import는 성공)
    try:
        from .ui import InfoPattern

        use_ui = True
    except ImportError:
        use_ui = False

    missing: list[str] = []

    # 선택적 의존성 체크 (import 없이)
    from importlib.util import find_spec

    if find_spec("google.generativeai") is None:
        missing.append("gemini")

    if find_spec("ollama") is None:
        missing.append("ollama")

    if missing and not hasattr(sys, "_beanllm_install_warned"):
        sys._beanllm_install_warned = True  # type: ignore[attr-defined]

        if use_ui:
            # 디자인 시스템 사용
            install_commands = []
            for pkg in missing:
                if pkg == "gemini":
                    install_commands.append("pip install beanllm[gemini]")
                elif pkg == "ollama":
                    install_commands.append("pip install beanllm[ollama]")

            InfoPattern.render(
                "Some provider SDKs are not installed",
                details=[f"Install: {cmd}" for cmd in install_commands]
                + ["Or install all: pip install beanllm[all]"],
            )
        else:
            # 기본 출력 (UI 없을 때)
            print("\n" + "=" * 60)
            print("📦 beanllm - Optional Provider SDKs")
            print("=" * 60)
            print("\nℹ️  Some provider SDKs are not installed:")
            for pkg in missing:
                if pkg == "gemini":
                    print("  • Gemini: pip install beanllm[gemini]")
                elif pkg == "ollama":
                    print("  • Ollama: pip install beanllm[ollama]")
            print("\nOr install all providers:")
            print("  pip install beanllm[all]")
            print("\n" + "=" * 60 + "\n")


def _print_welcome_banner() -> None:
    """환영 배너 출력 (선택적, 환경변수로 제어) - 디자인 시스템 적용"""
    import os

    # 환경변수로 제어 (기본값: False)
    if not os.getenv("LLMKIT_SHOW_BANNER", "false").lower() == "true":
        return

    try:
        from .ui import OnboardingPattern, print_logo
    except ImportError:
        return  # UI 없으면 출력 안 함

    # 로고 출력
    print_logo(style="minimal", color="magenta")

    # 온보딩 패턴
    OnboardingPattern.render(
        "Welcome to beanllm!",
        steps=[
            {
                "title": "Set environment variables",
                "description": "export OPENAI_API_KEY='your-key'",
            },
            {
                "title": "Try it out",
                "description": "from beanllm import get_registry; r = get_registry()",
            },
            {"title": "Use CLI", "description": "beanllm list"},
        ],
    )


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr = _LAZY_IMPORT_MAP[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    if name in _OPTIONAL_LAZY_IMPORT_MAP:
        mod_path, attr = _OPTIONAL_LAZY_IMPORT_MAP[name]
        try:
            mod = importlib.import_module(mod_path)
            val = getattr(mod, attr)
        except ImportError:
            val = None
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
