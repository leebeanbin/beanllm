"""
beanllm 전용 슬래시 커맨드 — beantui CommandRegistry에 등록

/rag, /agent, /eval, /search 등 beanllm 비즈니스 로직이 필요한 커맨드.
이 모듈은 beantui.toml의 plugins 목록에서 자동 import됩니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beantui.commands import register_command

if TYPE_CHECKING:
    from beantui.session import ChatSession


# ---------------------------------------------------------------------------
# /rag — RAG 검색
# ---------------------------------------------------------------------------


@register_command(
    "rag",
    description="RAG 검색",
    usage="/rag [path]",
    is_async=True,
)
async def cmd_rag(session: "ChatSession", args: str) -> str:
    """RAG 모드 — 경로 지정 시 문서 로드, 인자 없으면 상태 표시"""
    path = args.strip()

    if not path:
        if session._rag.get("chain"):
            doc_count = session._rag.get("doc_count", "?")
            return f"[green]RAG active[/green] ({doc_count} documents)"
        return "[dim]RAG inactive — /rag <path> to load documents[/dim]"

    try:
        from beanllm import RAGChain

        chain = RAGChain.from_documents(path)
        doc_count = chain.doc_count if hasattr(chain, "doc_count") else "?"
        session._rag["chain"] = chain
        session._rag["path"] = path
        session._rag["doc_count"] = doc_count
        return f"[green]✓ RAG loaded[/green] from {path} ({doc_count} documents)"
    except ImportError:
        return "[red]RAG not available — pip install beanllm[rag][/red]"
    except Exception as e:
        return f"[red]RAG error: {e}[/red]"


# ---------------------------------------------------------------------------
# /agent — 에이전트 실행
# ---------------------------------------------------------------------------


@register_command(
    "agent",
    description="에이전트 실행",
    usage="/agent <task>",
    is_async=True,
)
async def cmd_agent(session: "ChatSession", args: str) -> str:
    """에이전트 실행"""
    task = args.strip()
    if not task:
        return "[dim]Usage: /agent <task description>[/dim]"

    try:
        from beanllm import Agent

        agent = Agent(model=session.model)
        result = await agent.run(task)
        return str(result)
    except ImportError:
        return "[red]Agent not available — pip install beanllm[agent][/red]"
    except Exception as e:
        return f"[red]Agent error: {e}[/red]"


# ---------------------------------------------------------------------------
# /eval — 평가
# ---------------------------------------------------------------------------


@register_command(
    "eval",
    description="LLM 평가",
    usage="/eval <prompt>",
    is_async=True,
)
async def cmd_eval(session: "ChatSession", args: str) -> str:
    """LLM 평가 실행"""
    prompt = args.strip()
    if not prompt:
        return "[dim]Usage: /eval <prompt to evaluate>[/dim]"

    try:
        from beanllm import Client

        client = Client(model=session.model)
        result = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return f"[bold cyan]Eval Result:[/bold cyan]\n{result.content}"
    except Exception as e:
        return f"[red]Eval error: {e}[/red]"


# ---------------------------------------------------------------------------
# /search — 웹 검색
# ---------------------------------------------------------------------------


@register_command(
    "search",
    description="웹 검색",
    usage="/search <query>",
    is_async=True,
)
async def cmd_search(session: "ChatSession", args: str) -> str:
    """웹 검색"""
    query = args.strip()
    if not query:
        return "[dim]Usage: /search <query>[/dim]"

    try:
        from beanllm.domain.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        results = await tool.search(query)
        if not results:
            return "[yellow]No results found[/yellow]"

        lines = [f"[bold cyan]Search: {query}[/bold cyan]\n"]
        for r in results[:5]:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"  [green]• {title}[/green]")
            lines.append(f"    [dim]{url}[/dim]")
            if snippet:
                lines.append(f"    {snippet[:120]}")
            lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "[red]Web search not available — pip install beanllm[search][/red]"
    except Exception as e:
        return f"[red]Search error: {e}[/red]"
