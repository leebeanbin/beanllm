"""NER data models - NEREntity, NERResult, BenchmarkSample, BenchmarkResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NEREntity:
    """NER 추출 결과"""

    text: str
    label: str  # PERSON, ORG, LOC, DATE, etc.
    start: int
    end: int
    confidence: float = 1.0
    source: str = ""  # 어떤 엔진에서 추출했는지


@dataclass
class NERResult:
    """NER 엔진 결과"""

    entities: List[NEREntity]
    engine_name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    """벤치마크 샘플"""

    text: str
    entities: List[Dict[str, Any]]  # [{"text": "...", "label": "PERSON", "start": 0, "end": 10}]


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""

    engine_name: str
    precision: float
    recall: float
    f1_score: float
    avg_latency_ms: float
    total_samples: int
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
