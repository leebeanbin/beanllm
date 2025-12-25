"""
Prompts Enums - 프롬프트 관련 열거형
"""

from enum import Enum


class TemplateFormat(Enum):
    """템플릿 포맷"""

    F_STRING = "f-string"  # {variable}
    JINJA2 = "jinja2"  # {{ variable }}
    MUSTACHE = "mustache"  # {{variable}}
