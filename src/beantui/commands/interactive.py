"""
Interactive Selection - prompt_toolkit 기반 화살표 키 메뉴

범용 interactive_select()와 역할/테마 전용 래퍼 제공.
"""

from __future__ import annotations

from typing import List, Tuple


async def interactive_select(
    title: str,
    items: List[Tuple[str, str, str, str]],
    default_idx: int = 0,
) -> str:
    """공통 인터랙티브 선택 메뉴 (화살표 키 + Enter)

    Args:
        title: 메뉴 타이틀
        items: [(value, icon, label, description), ...]
        default_idx: 초기 선택 인덱스

    Returns:
        선택된 value 문자열 (취소 시 빈 문자열)
    """
    from beantui.themes import get_theme

    theme = get_theme()
    p = theme.palette

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.styles import Style

        selected_idx = {"value": default_idx}
        cancelled = {"value": False}

        bindings = KeyBindings()

        @bindings.add(Keys.Up)
        def _up(event) -> None:  # type: ignore[no-untyped-def]
            if selected_idx["value"] > 0:
                selected_idx["value"] -= 1

        @bindings.add(Keys.Down)
        def _down(event) -> None:  # type: ignore[no-untyped-def]
            if selected_idx["value"] < len(items) - 1:
                selected_idx["value"] += 1

        @bindings.add(Keys.Enter)
        def _enter(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="")

        @bindings.add(Keys.Escape)
        def _esc(event) -> None:  # type: ignore[no-untyped-def]
            cancelled["value"] = True
            event.app.exit(result="")

        pt_style = Style.from_dict(theme.prompt_toolkit_style)

        def _build_menu() -> HTML:
            lines = [
                f"<b>  {title}</b>  "
                f"<style fg='{p.muted}'>(↑↓ navigate, Enter select, Esc cancel)</style>\n"
            ]
            for i, (_value, icon, label, desc) in enumerate(items):
                if i == selected_idx["value"]:
                    lines.append(
                        f"  <style fg='{p.brand_primary}'><b>▸ {icon} {label:<12}</b></style>"
                        f"  <style fg='{p.text}'>{desc}</style>"
                    )
                else:
                    lines.append(
                        f"  <style fg='{p.dim}'>  {icon} {label:<12}</style>"
                        f"  <style fg='{p.muted}'>{desc}</style>"
                    )
            return HTML("\n".join(lines))

        ps = PromptSession(key_bindings=bindings, style=pt_style)
        await ps.prompt_async(message=_build_menu, refresh_interval=0.1)

        if cancelled["value"]:
            return ""
        return items[selected_idx["value"]][0]

    except ImportError:
        return ""


async def interactive_role_select() -> str:
    """인터랙티브 역할 선택"""
    from beantui.roles import BUILTIN_ROLES

    items = [(name, role.icon, name, role.description) for name, role in BUILTIN_ROLES.items()]
    return await interactive_select("Select Role", items)


async def interactive_theme_select() -> str:
    """인터랙티브 테마 선택"""
    from beantui.themes import get_theme

    current = get_theme().name
    items = [
        ("dark", "🌙", "Dark Mode", "어두운 배경, 밝은 텍스트"),
        ("light", "☀️", "Light Mode", "밝은 배경, 어두운 텍스트"),
    ]
    default_idx = 0 if current == "dark" else 1
    return await interactive_select("Select Theme", items, default_idx=default_idx)
