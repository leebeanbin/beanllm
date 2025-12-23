"""
Checkpoint - 상태 체크포인트
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Checkpoint:
    """상태 체크포인트"""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path(".checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save(self, execution_id: str, state: Dict[str, Any], node_name: str):
        """체크포인트 저장"""
        checkpoint_file = self.checkpoint_dir / f"{execution_id}_{node_name}.json"

        checkpoint_data = {
            "execution_id": execution_id,
            "node_name": node_name,
            "state": state,
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)

    def load(self, execution_id: str, node_name: str) -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        checkpoint_file = self.checkpoint_dir / f"{execution_id}_{node_name}.json"

        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        return checkpoint_data.get("state")

    def list_checkpoints(self, execution_id: str) -> List[str]:
        """체크포인트 목록"""
        pattern = f"{execution_id}_*.json"
        return [p.stem for p in self.checkpoint_dir.glob(pattern)]

    def clear(self, execution_id: Optional[str] = None):
        """체크포인트 삭제"""
        if execution_id:
            pattern = f"{execution_id}_*.json"
        else:
            pattern = "*.json"

        for p in self.checkpoint_dir.glob(pattern):
            p.unlink()
