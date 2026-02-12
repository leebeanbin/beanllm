"""
Roles - 하위 호환 래퍼

beantui.roles로 위임합니다.
"""

from beantui.roles import BUILTIN_ROLES, Role, get_role, get_role_list_display, list_roles

__all__ = ["Role", "BUILTIN_ROLES", "get_role", "list_roles", "get_role_list_display"]
