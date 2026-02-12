"""
TUI Configuration - TOML 기반 설정 시스템

beantui.toml 파일 하나로 모든 세팅을 관리:
  - 앱 메타 (이름, 버전, 로고)
  - 테마 기본값
  - 세션/히스토리 설정
  - 로깅 차단 목록
  - 플러그인 커맨드 모듈 경로
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    """앱 메타데이터"""

    name: str = "beantui"
    version: str = "0.1.0"
    motto: str = "Your AI Toolkit"
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    logo_lines: List[str] = field(default_factory=list)


@dataclass
class ThemeConfig:
    """테마 기본값"""

    default: str = "dark"


@dataclass
class SessionConfig:
    """세션 저장소"""

    db_path: str = "~/.beantui/sessions.db"
    history_limit: int = 200
    shell_timeout: int = 60


@dataclass
class LoggingConfig:
    """로그 차단"""

    blocked_prefixes: List[str] = field(default_factory=lambda: ["httpx", "httpcore", "urllib3"])


@dataclass
class CommandsConfig:
    """커맨드 플러그인"""

    builtin: bool = True
    plugins: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

_DEFAULT_LOGO = [
    "██████╗ ███████╗ █████╗ ███╗   ██╗████████╗██╗   ██╗██╗",
    "██╔══██╗██╔════╝██╔══██╗████╗  ██║╚══██╔══╝██║   ██║██║",
    "██████╔╝█████╗  ███████║██╔██╗ ██║   ██║   ██║   ██║██║",
    "██╔══██╗██╔══╝  ██╔══██║██║╚██╗██║   ██║   ██║   ██║██║",
    "██████╔╝███████╗██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║",
    "╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝",
]


@dataclass
class TUIConfig:
    """beantui 전체 설정"""

    app: AppConfig = field(default_factory=AppConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)

    # --- 팩토리 메서드 ---

    @classmethod
    def from_toml(cls, path: str | Path) -> "TUIConfig":
        """TOML 파일에서 설정 로드

        Args:
            path: TOML 파일 경로

        Returns:
            TUIConfig 인스턴스

        Raises:
            FileNotFoundError: 파일이 없는 경우
        """
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        raw = _load_toml(p)
        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TUIConfig":
        """딕셔너리에서 설정 로드"""
        return cls._from_dict(data)

    @classmethod
    def auto_discover(cls, search_dirs: Optional[Sequence[Path]] = None) -> "TUIConfig":
        """설정 파일 자동 탐색

        탐색 순서:
          1. search_dirs (지정된 경우)
          2. 현재 디렉토리 (beantui.toml)
          3. ~/.config/beantui/config.toml
          4. 기본값
        """
        candidates: list[Path] = []
        if search_dirs:
            for d in search_dirs:
                candidates.append(Path(d) / "beantui.toml")

        candidates.extend(
            [
                Path.cwd() / "beantui.toml",
                Path.home() / ".config" / "beantui" / "config.toml",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return cls.from_toml(candidate)

        return cls()  # 기본값

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "TUIConfig":
        """딕셔너리에서 내부 변환"""
        app_raw = data.get("app", {})
        logo = app_raw.get("logo_lines", _DEFAULT_LOGO)

        app = AppConfig(
            name=app_raw.get("name", "beantui"),
            version=app_raw.get("version", "0.1.0"),
            motto=app_raw.get("motto", "Your AI Toolkit"),
            default_model=app_raw.get("default_model", "gpt-4o-mini"),
            temperature=float(app_raw.get("temperature", 0.7)),
            logo_lines=logo if isinstance(logo, list) else _DEFAULT_LOGO,
        )

        theme_raw = data.get("theme", {})
        theme = ThemeConfig(default=theme_raw.get("default", "dark"))

        session_raw = data.get("session", {})
        session = SessionConfig(
            db_path=session_raw.get("db_path", "~/.beantui/sessions.db"),
            history_limit=int(session_raw.get("history_limit", 200)),
            shell_timeout=int(session_raw.get("shell_timeout", 60)),
        )

        logging_raw = data.get("logging", {})
        logging_cfg = LoggingConfig(
            blocked_prefixes=logging_raw.get("blocked_prefixes", ["httpx", "httpcore", "urllib3"]),
        )

        commands_raw = data.get("commands", {})
        commands = CommandsConfig(
            builtin=commands_raw.get("builtin", True),
            plugins=commands_raw.get("plugins", []),
        )

        return cls(
            app=app,
            theme=theme,
            session=session,
            logging=logging_cfg,
            commands=commands,
        )

    @property
    def resolved_db_path(self) -> Path:
        """DB 경로를 절대 경로로 해석"""
        return Path(self.session.db_path).expanduser()

    @property
    def logo_lines(self) -> List[str]:
        """로고 라인 (app 설정 또는 기본값)"""
        return self.app.logo_lines or _DEFAULT_LOGO


# ---------------------------------------------------------------------------
# TOML 로딩 (Python 3.11+ tomllib, fallback tomli)
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> Dict[str, Any]:
    """TOML 파일 로드 — Python 3.11+ 내장 tomllib 또는 tomli 사용"""
    text = path.read_bytes()

    try:
        import tomllib  # Python 3.11+

        return tomllib.loads(text.decode("utf-8"))
    except ImportError:
        pass

    try:
        import tomli  # fallback

        return tomli.loads(text.decode("utf-8"))
    except ImportError:
        pass

    raise ImportError(
        "TOML parsing requires Python 3.11+ (tomllib) or 'pip install tomli' for older versions."
    )
