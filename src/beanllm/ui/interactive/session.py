"""
Chat Session - 대화 상태 관리

메시지 히스토리, 모델, 첨부 컨텍스트(@file, !shell 결과) 관리
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AttachedContext:
    """첨부된 컨텍스트 (파일 내용, 명령 출력 등)"""

    source: str  # "file:path" or "shell:cmd"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """채팅 메시지"""

    role: str  # "user" | "assistant" | "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatSession:
    """
    대화 세션 상태

    OpenCode 스타일:
    - 메시지 히스토리 유지
    - @file로 첨부된 파일 내용
    - !cmd로 실행된 셸 출력
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        system: Optional[str] = None,
        working_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.system = system or "You are a helpful assistant."
        self.working_dir = working_dir or Path.cwd()

        self.messages: List[ChatMessage] = []
        self.attached_contexts: List[AttachedContext] = []
        self._rag: Dict[str, Any] = {}
        self.mode: str = "chat"
        self.verbose: bool = False
        self.role_name: str = "default"

    def add_user_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))

    def add_system_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="system", content=content))

    def attach_file(self, path: str, content: str) -> None:
        self.attached_contexts.append(
            AttachedContext(source=f"file:{path}", content=content, metadata={"path": path})
        )

    def attach_shell_output(self, cmd: str, output: str) -> None:
        self.attached_contexts.append(
            AttachedContext(source=f"shell:{cmd}", content=output, metadata={"cmd": cmd})
        )

    def build_messages_for_llm(self) -> List[Dict[str, str]]:
        if not self.messages:
            return []

        result: List[Dict[str, str]] = []
        for msg in self.messages:
            if msg.role == "system":
                continue
            result.append({"role": msg.role, "content": msg.content})

        if self.attached_contexts and result:
            context_block = self._format_attached_contexts()
            last = result[-1]
            if last["role"] == "user":
                last["content"] = f"{last['content']}\n\n{context_block}"

        return result

    def _format_attached_contexts(self) -> str:
        parts = []
        for ctx in self.attached_contexts:
            if ctx.source.startswith("file:"):
                path = ctx.source.replace("file:", "")
                parts.append(f"--- File: {path} ---\n{ctx.content}\n---")
            else:
                cmd = ctx.source.replace("shell:", "")
                parts.append(f"--- Shell: {cmd} ---\n{ctx.content}\n---")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.messages.clear()
        self.attached_contexts.clear()

    def set_model(self, model: str, provider: Optional[str] = None) -> None:
        self.model = model
        if provider is not None:
            self.provider = provider
