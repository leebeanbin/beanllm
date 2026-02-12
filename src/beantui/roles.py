"""
Roles / Presets - 역할 프리셋 시스템

/role coder → 코딩 전문가 모드
/role writer → 글쓰기 모드
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Role:
    """역할 프리셋 정의"""

    name: str
    description: str
    system_prompt: str
    temperature: float = 0.7
    icon: str = "●"


BUILTIN_ROLES: Dict[str, Role] = {
    "default": Role(
        name="default",
        description="범용 어시스턴트",
        system_prompt="You are a helpful assistant.",
        icon="○",
    ),
    "coder": Role(
        name="coder",
        description="코딩 전문가",
        system_prompt=(
            "You are an expert software engineer. "
            "Write clean, efficient, and well-documented code. "
            "When explaining code, be concise and use code blocks. "
            "Follow best practices and suggest improvements."
        ),
        temperature=0.3,
        icon="⟨⟩",
    ),
    "writer": Role(
        name="writer",
        description="글쓰기 어시스턴트",
        system_prompt=(
            "You are a professional writer. "
            "Help with writing, editing, and proofreading. "
            "Focus on clarity, conciseness, and engaging prose."
        ),
        temperature=0.8,
        icon="✎",
    ),
    "translator": Role(
        name="translator",
        description="번역가",
        system_prompt=(
            "You are a professional translator. "
            "Translate text accurately while preserving meaning and tone. "
            "If the target language is not specified, translate to English."
        ),
        temperature=0.3,
        icon="🌐",
    ),
    "reviewer": Role(
        name="reviewer",
        description="코드 리뷰어",
        system_prompt=(
            "You are an expert code reviewer. "
            "Review code for bugs, security issues, performance, and readability. "
            "Suggest improvements with code examples. Be constructive and specific."
        ),
        temperature=0.3,
        icon="🔍",
    ),
    "planner": Role(
        name="planner",
        description="프로젝트 플래너",
        system_prompt=(
            "You are a project planning specialist. "
            "Help break down complex tasks into actionable steps. "
            "Create structured plans with timelines and priorities. "
            "Consider risks and dependencies."
        ),
        temperature=0.5,
        icon="📋",
    ),
    "shell": Role(
        name="shell",
        description="쉘 명령어 전문가",
        system_prompt=(
            "You are a shell command expert. "
            "Help with Unix/Linux/macOS shell commands. "
            "Provide clear command examples with explanations. "
            "Warn about potentially destructive commands."
        ),
        temperature=0.2,
        icon="$",
    ),
}


def get_role(name: str) -> Optional[Role]:
    """이름으로 역할 조회"""
    return BUILTIN_ROLES.get(name.lower())


def list_roles() -> Dict[str, Role]:
    """모든 역할 반환"""
    return dict(BUILTIN_ROLES)


def get_role_list_display() -> str:
    """역할 목록 Rich 포맷 문자열"""
    lines = ["  [bold cyan]Available Roles[/bold cyan]\n"]
    for name, role in BUILTIN_ROLES.items():
        lines.append(f"  [green]{role.icon} {name:<12}[/green] {role.description}")
    lines.append("\n  [dim]Usage: /role <name>[/dim]")
    return "\n".join(lines)
