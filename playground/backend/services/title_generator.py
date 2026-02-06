"""
Chat title generation using an open-source model (Ollama).

Generates a short title from the first message for session list display.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default model for title generation (small, fast)
DEFAULT_TITLE_MODEL = "qwen2.5:0.5b"

TITLE_PROMPT = """Summarize the following in 5-10 words as a chat title. Output only the title, no quotes or explanation.

Content:
{content}"""


async def generate_chat_title(
    content: str,
    model: str = DEFAULT_TITLE_MODEL,
    max_length: int = 50,
) -> Optional[str]:
    """
    Generate a short chat title from the first message using an open-source model.

    Args:
        content: First message (or preview) to summarize.
        model: Model name (e.g. qwen2.5:0.5b for Ollama).
        max_length: Maximum title length.

    Returns:
        Generated title, or None on failure.
    """
    if not (content and content.strip()):
        return None
    text = content.strip()[:500]
    prompt = TITLE_PROMPT.format(content=text)
    try:
        from beanllm.facade.core import Client

        # Use Ollama (open-source) for title generation
        client = Client(model=model, provider="ollama")
        response = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=60,
        )
        if not response or not getattr(response, "content", None):
            return None
        title = (response.content or "").strip()
        if not title:
            return None
        # Remove quotes if model wrapped the title
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1].strip()
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1].strip()
        return title[:max_length] if len(title) > max_length else title
    except Exception as e:
        logger.warning("Title generation failed (using fallback): %s", e)
        return None
