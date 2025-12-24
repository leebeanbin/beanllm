"""
Prompts Versioning - 프롬프트 버전 관리
"""

import difflib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BasePromptTemplate


@dataclass
class PromptVersion:
    """프롬프트 버전 정보"""

    version: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    last_used: Optional[datetime] = None

    def add_metric(self, metric_name: str, value: float):
        """성능 메트릭 추가"""
        self.performance_metrics[metric_name] = value

    def record_usage(self):
        """사용 기록"""
        self.usage_count += 1
        self.last_used = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (저장용)"""
        return {
            "version": self.version,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """딕셔너리에서 생성 (로드용)"""
        return cls(
            version=data["version"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            performance_metrics=data.get("performance_metrics", {}),
            usage_count=data.get("usage_count", 0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
        )


class PromptVersioning:
    """프롬프트 버전 관리 (기존 클래스 - 하위 호환성 유지)"""

    def __init__(self):
        self.versions: Dict[str, List[tuple]] = {}  # name -> [(version, template)]

    def save(self, name: str, template: BasePromptTemplate, version: str) -> None:
        """템플릿 저장"""
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append((version, template))

    def load(self, name: str, version: Optional[str] = None) -> BasePromptTemplate:
        """템플릿 로드"""
        if name not in self.versions:
            raise ValueError(f"Template '{name}' not found")

        if version is None:
            # 최신 버전 반환
            return self.versions[name][-1][1]

        # 특정 버전 찾기
        for ver, template in self.versions[name]:
            if ver == version:
                return template

        raise ValueError(f"Version '{version}' not found for template '{name}'")

    def list_versions(self, name: str) -> List[str]:
        """템플릿의 모든 버전 나열"""
        if name not in self.versions:
            return []
        return [ver for ver, _ in self.versions[name]]


class PromptVersionManager:
    """프롬프트 버전 관리자 (확장된 기능)"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 파일 기반 저장소 경로 (None이면 메모리만 사용)
        """
        self.versions: Dict[str, List[PromptVersion]] = {}
        self.storage_path = storage_path
        if storage_path:
            self._load_from_storage()

    def create_version(
        self,
        name: str,
        content: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """
        새 버전 생성

        Args:
            name: 프롬프트 이름
            content: 프롬프트 내용
            version: 버전 번호 (None이면 자동 생성: v1, v2, ...)
            metadata: 추가 메타데이터

        Returns:
            PromptVersion: 생성된 버전
        """
        if name not in self.versions:
            self.versions[name] = []

        # 버전 번호 자동 생성
        if version is None:
            existing_versions = [v.version for v in self.versions[name]]
            version = f"v{len(existing_versions) + 1}"

        # 새 버전 생성
        prompt_version = PromptVersion(
            version=version,
            content=content,
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        self.versions[name].append(prompt_version)
        self._save_to_storage()

        return prompt_version

    def get_version(self, name: str, version: Optional[str] = None) -> PromptVersion:
        """
        버전 조회

        Args:
            name: 프롬프트 이름
            version: 버전 번호 (None이면 최신 버전)

        Returns:
            PromptVersion: 버전 정보
        """
        if name not in self.versions:
            raise ValueError(f"Prompt '{name}' not found")

        if version is None:
            # 최신 버전 반환
            return self.versions[name][-1]

        # 특정 버전 찾기
        for v in self.versions[name]:
            if v.version == version:
                return v

        raise ValueError(f"Version '{version}' not found for prompt '{name}'")

    def list_versions(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        버전 목록 조회

        Args:
            name: 프롬프트 이름 (None이면 모든 프롬프트)

        Returns:
            프롬프트별 버전 리스트 딕셔너리
        """
        if name:
            if name not in self.versions:
                return {}
            return {name: [v.version for v in self.versions[name]]}
        else:
            return {name: [v.version for v in versions] for name, versions in self.versions.items()}

    def compare_versions(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        버전 비교

        Args:
            name: 프롬프트 이름
            version1: 첫 번째 버전
            version2: 두 번째 버전

        Returns:
            {
                "content_diff": str,  # 내용 차이
                "metrics_diff": Dict[str, float],  # 메트릭 차이
                "usage_diff": int,  # 사용 횟수 차이
                "recommendation": str  # 추천 버전
            }
        """
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)

        # 내용 비교 (간단한 diff)
        content_diff = self._diff_content(v1.content, v2.content)

        # 메트릭 비교
        metrics_diff = {}
        all_metrics = set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys())
        for metric in all_metrics:
            val1 = v1.performance_metrics.get(metric, 0.0)
            val2 = v2.performance_metrics.get(metric, 0.0)
            metrics_diff[metric] = val2 - val1

        # 추천 버전 (accuracy 기준)
        recommendation = version1
        if "accuracy" in metrics_diff and metrics_diff["accuracy"] > 0:
            recommendation = version2

        return {
            "content_diff": content_diff,
            "metrics_diff": metrics_diff,
            "usage_diff": v2.usage_count - v1.usage_count,
            "recommendation": recommendation,
        }

    def _diff_content(self, content1: str, content2: str) -> str:
        """간단한 내용 차이 계산 (difflib 사용)"""
        diff = difflib.unified_diff(
            content1.splitlines(keepends=True),
            content2.splitlines(keepends=True),
            lineterm="",
        )
        return "".join(diff)

    def _save_to_storage(self):
        """파일 저장 (JSON)"""
        if not self.storage_path:
            return

        data = {}
        for name, versions in self.versions.items():
            data[name] = [v.to_dict() for v in versions]

        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_storage(self):
        """파일에서 로드"""
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for name, versions_data in data.items():
            self.versions[name] = [PromptVersion.from_dict(v) for v in versions_data]
