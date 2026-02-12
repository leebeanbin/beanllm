"""
Theme System - Dark/Light 테마 + 커스텀 컬러 팔레트

aichat / Gemini CLI 스타일 테마 시스템:
  - Dark / Light 프리셋
  - 역할별 색상 (user, assistant, system, error)
  - 모드별 프롬프트 색상 (chat, rag, shell, plan)
  - prompt_toolkit 스타일 연동
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ThemePalette:
    """컬러 팔레트 정의"""

    # --- 브랜드 ---
    brand_primary: str = "#a78bfa"  # violet-400
    brand_secondary: str = "#818cf8"  # indigo-400
    brand_accent: str = "#67e8f9"  # cyan-300

    # --- 역할 ---
    user_color: str = "#60a5fa"  # blue-400
    assistant_color: str = "#a78bfa"  # violet-400
    system_color: str = "#fbbf24"  # amber-400
    error_color: str = "#f87171"  # red-400
    warning_color: str = "#fb923c"  # orange-400
    success_color: str = "#4ade80"  # green-400
    info_color: str = "#67e8f9"  # cyan-300

    # --- 모드 프롬프트 ---
    chat_prompt: str = "#a78bfa"  # violet
    rag_prompt: str = "#34d399"  # emerald-400
    shell_prompt: str = "#fbbf24"  # amber
    plan_prompt: str = "#60a5fa"  # blue

    # --- UI 요소 ---
    border: str = "#4b5563"  # gray-600
    border_dim: str = "#374151"  # gray-700
    muted: str = "#6b7280"  # gray-500
    dim: str = "#9ca3af"  # gray-400
    text: str = "#e5e7eb"  # gray-200
    bg_panel: str = "#1f2937"  # gray-800
    bg_toolbar: str = "#111827"  # gray-900
    separator: str = "#374151"  # gray-700

    # --- 코드블록 ---
    code_bg: str = "#1e1e2e"  # dark code background
    code_border: str = "#45475a"  # catppuccin surface1

    # --- 그라디언트 (로고용) ---
    gradient: tuple[str, ...] = (
        "#c084fc",  # purple-400
        "#a78bfa",  # violet-400
        "#818cf8",  # indigo-400
        "#60a5fa",  # blue-400
        "#38bdf8",  # sky-400
        "#22d3ee",  # cyan-400
    )


@dataclass(frozen=True)
class LightPalette(ThemePalette):
    """라이트 테마 팔레트"""

    brand_primary: str = "#7c3aed"
    brand_secondary: str = "#6366f1"
    brand_accent: str = "#0891b2"

    user_color: str = "#2563eb"
    assistant_color: str = "#7c3aed"
    system_color: str = "#d97706"
    error_color: str = "#dc2626"
    warning_color: str = "#ea580c"
    success_color: str = "#16a34a"
    info_color: str = "#0891b2"

    chat_prompt: str = "#7c3aed"
    rag_prompt: str = "#059669"
    shell_prompt: str = "#d97706"
    plan_prompt: str = "#2563eb"

    border: str = "#d1d5db"
    border_dim: str = "#e5e7eb"
    muted: str = "#9ca3af"
    dim: str = "#6b7280"
    text: str = "#1f2937"
    bg_panel: str = "#f3f4f6"
    bg_toolbar: str = "#e5e7eb"
    separator: str = "#d1d5db"

    code_bg: str = "#f8f8f8"
    code_border: str = "#d1d5db"

    gradient: tuple[str, ...] = (
        "#9333ea",
        "#7c3aed",
        "#6366f1",
        "#2563eb",
        "#0284c7",
        "#0891b2",
    )


@dataclass
class Theme:
    """테마 설정"""

    name: str = "dark"
    palette: ThemePalette = field(default_factory=ThemePalette)

    # --- 아이콘 ---
    icons: Dict[str, str] = field(
        default_factory=lambda: {
            "user": "●",
            "assistant": "◆",
            "system": "▲",
            "thinking": "◐",
            "searching": "◉",
            "reading": "◎",
            "writing": "◈",
            "error": "✖",
            "warning": "▲",
            "success": "✔",
            "info": "ℹ",
            "file": "📎",
            "rag": "🔗",
            "chat": "💬",
            "shell": "$",
            "plan": "📋",
            "token": "⊛",
            "time": "⏱",
            "cost": "¢",
            "model": "⬡",
            "separator": "─",
            "prompt_chat": "❯",
            "prompt_rag": "⟐",
            "prompt_shell": "$",
            "prompt_plan": "◇",
        }
    )

    _pt_style_cache: Optional[dict[str, str]] = field(default=None, repr=False)

    @property
    def prompt_toolkit_style(self) -> dict[str, str]:
        """prompt_toolkit 스타일 딕셔너리 (캐싱)"""
        if self._pt_style_cache is not None:
            return self._pt_style_cache
        p = self.palette
        style = {
            "bottom-toolbar": f"bg:{p.bg_toolbar} {p.dim}",
            "bottom-toolbar.text": p.dim,
            "completion-menu": f"bg:{p.bg_panel} {p.text}",
            "completion-menu.completion": f"bg:{p.bg_panel} {p.text}",
            "completion-menu.completion.current": f"bg:{p.brand_primary} #ffffff",
            "completion-menu.meta.completion": f"bg:{p.bg_panel} {p.muted}",
            "completion-menu.meta.completion.current": f"bg:{p.brand_primary} #ffffff",
            "scrollbar.background": f"bg:{p.border_dim}",
            "scrollbar.button": f"bg:{p.muted}",
        }
        object.__setattr__(self, "_pt_style_cache", style)
        return style

    def get_prompt_color(self, mode: str) -> str:
        """모드에 따른 프롬프트 색상 반환 — getattr로 dict 재생성 회피"""
        attr = f"{mode}_prompt"
        return getattr(self.palette, attr, self.palette.chat_prompt)

    def get_prompt_icon(self, mode: str) -> str:
        """모드에 따른 프롬프트 아이콘 반환"""
        return self.icons.get(f"prompt_{mode}", self.icons["prompt_chat"])


# ---------------------------------------------------------------------------
# 싱글턴 테마 인스턴스
# ---------------------------------------------------------------------------

_current_theme: Theme = Theme()


def get_theme() -> Theme:
    """현재 활성 테마"""
    return _current_theme


def set_theme(name: str) -> Theme:
    """테마 전환 (dark / light)"""
    global _current_theme
    if name == "light":
        _current_theme = Theme(name="light", palette=LightPalette())
    else:
        _current_theme = Theme(name="dark", palette=ThemePalette())
    return _current_theme
