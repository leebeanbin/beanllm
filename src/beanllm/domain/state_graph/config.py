"""
GraphConfig - 그래프 설정
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GraphConfig:
    """그래프 설정"""

    max_iterations: int = 100  # 무한 루프 방지
    enable_checkpointing: bool = False
    checkpoint_dir: Optional[Path] = None
    debug: bool = False
