"""
Hybrid Model Manager - 하이브리드 모델 관리자 구현
"""

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from .types import HybridModelInfo

try:
    from ...infrastructure.models import MODELS
    from ...utils.logger import get_logger
    from ..inferrer import MetadataInferrer
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)

    class MetadataInferrer:
        def infer(self, provider: str, model_id: str) -> Dict:
            return {}

    MODELS = {}


logger = get_logger(__name__)


class HybridModelManager:
    """
    하이브리드 모델 관리자

    1. API 스캔 (ModelScanner)
    2. 로컬 메타데이터 (ModelConfig)
    3. 패턴 기반 추론 (MetadataInferrer)
    """

    def __init__(self):
        from ..scanner import ModelScanner

        self.scanner = ModelScanner()
        self.inferrer = MetadataInferrer()
        self.models: Dict[str, Dict[str, HybridModelInfo]] = {
            "openai": {},
            "anthropic": {},
            "google": {},
            "ollama": {},
        }
        self._loaded = False

    async def load(self, scan_api: bool = True) -> None:
        """
        모든 데이터 로드

        Args:
            scan_api: API 스캔 여부 (False면 로컬만)
        """
        logger.info("Loading hybrid model data...")

        # 1. 로컬 메타데이터 로드
        self._load_local_metadata()

        # 2. API 스캔 (선택적)
        if scan_api:
            await self._scan_and_merge()

        self._loaded = True
        logger.info(f"Loaded {self.get_total_count()} models")

    def _load_local_metadata(self) -> None:
        """로컬 ModelConfig 로드"""
        logger.info("Loading local metadata...")

        for model_id, config in MODELS.items():
            provider = config.get("provider", "unknown")

            if provider not in self.models:
                continue

            model_info = HybridModelInfo(
                model_id=model_id,
                provider=provider,
                display_name=config.get("model_name", model_id),
                supports_streaming=config.get("supports_streaming", True),
                supports_temperature=config.get("supports_temperature", True),
                supports_max_tokens=config.get("supports_max_tokens", True),
                uses_max_completion_tokens=config.get("uses_max_completion_tokens", False),
                max_tokens=config.get("max_tokens"),
                tier=config.get("tier"),
                speed=config.get("speed"),
                source="local",
                inference_confidence=1.0,
            )

            self.models[provider][model_id] = model_info

        logger.info(f"Loaded {sum(len(models) for models in self.models.values())} local models")

    async def _scan_and_merge(self) -> None:
        """API 스캔 및 병합"""
        logger.info("Scanning APIs...")

        try:
            scanned = await self.scanner.scan_all()

            for provider, models in scanned.items():
                if provider not in self.models:
                    continue

                for scanned_model in models:
                    model_id = scanned_model.model_id

                    # 이미 로컬에 있으면 스킵
                    if model_id in self.models[provider]:
                        # last_seen 업데이트
                        self.models[provider][model_id].last_seen = datetime.now().isoformat()
                        continue

                    # 신규 모델: 추론
                    inferred = self.inferrer.infer(provider, model_id)

                    model_info = HybridModelInfo(
                        model_id=model_id,
                        provider=provider,
                        display_name=getattr(scanned_model, "display_name", None) or model_id,
                        supports_streaming=inferred.get("supports_streaming", True),
                        supports_temperature=inferred.get("supports_temperature", True),
                        supports_max_tokens=inferred.get("supports_max_tokens", True),
                        uses_max_completion_tokens=inferred.get(
                            "uses_max_completion_tokens", False
                        ),
                        max_tokens=inferred.get("max_tokens"),
                        tier=inferred.get("tier"),
                        speed=inferred.get("speed"),
                        source="inferred",
                        inference_confidence=inferred.get("inference_confidence", 0.0),
                        matched_patterns=inferred.get("matched_patterns", []),
                        discovered_at=datetime.now().isoformat(),
                        last_seen=datetime.now().isoformat(),
                    )

                    self.models[provider][model_id] = model_info
                    logger.info(
                        f"New model discovered: {provider}/{model_id} (confidence: {model_info.inference_confidence:.2f})"
                    )

        except Exception as e:
            logger.error(f"Error scanning APIs: {e}")

    def get_model_info(
        self, model_id: str, provider: Optional[str] = None
    ) -> Optional[HybridModelInfo]:
        """
        모델 정보 가져오기

        Args:
            model_id: 모델 ID
            provider: Provider (없으면 모든 Provider 검색)
        """
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        if provider:
            return self.models.get(provider, {}).get(model_id)

        # 모든 Provider 검색
        for provider_models in self.models.values():
            if model_id in provider_models:
                return provider_models[model_id]

        return None

    def get_models_by_provider(self, provider: str) -> List[HybridModelInfo]:
        """Provider별 모델 목록"""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        return list(self.models.get(provider, {}).values())

    def get_all_models(self) -> List[HybridModelInfo]:
        """모든 모델 목록"""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        result = []
        for provider_models in self.models.values():
            result.extend(provider_models.values())
        return result

    def get_new_models(self) -> List[HybridModelInfo]:
        """신규 모델 목록 (source="inferred")"""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        return [model for model in self.get_all_models() if model.source == "inferred"]

    def get_local_models(self) -> List[HybridModelInfo]:
        """로컬 모델 목록 (source="local")"""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        return [model for model in self.get_all_models() if model.source == "local"]

    def get_total_count(self) -> int:
        """전체 모델 수"""
        return len(self.get_all_models())

    def get_provider_counts(self) -> Dict[str, int]:
        """Provider별 모델 수"""
        return {provider: len(models) for provider, models in self.models.items()}

    def search_models(
        self,
        query: str,
        provider: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[HybridModelInfo]:
        """
        모델 검색

        Args:
            query: 검색어 (모델 ID에 포함)
            provider: Provider 필터
            source: 소스 필터 ("local", "inferred")
            min_confidence: 최소 신뢰도
        """
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        results = []
        query_lower = query.lower()

        for model in self.get_all_models():
            # Provider 필터
            if provider and model.provider != provider:
                continue

            # Source 필터
            if source and model.source != source:
                continue

            # 신뢰도 필터
            if model.inference_confidence < min_confidence:
                continue

            # 검색어 필터
            if query_lower in model.model_id.lower() or query_lower in model.display_name.lower():
                results.append(model)

        return results

    def export_to_dict(self) -> Dict:
        """딕셔너리로 내보내기"""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        return {
            provider: {model_id: asdict(model_info) for model_id, model_info in models.items()}
            for provider, models in self.models.items()
        }

    def get_summary(self) -> Dict:
        """요약 정보"""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call await load() first.")

        all_models = self.get_all_models()
        new_models = self.get_new_models()
        local_models = self.get_local_models()

        return {
            "total": len(all_models),
            "by_provider": self.get_provider_counts(),
            "by_source": {"local": len(local_models), "inferred": len(new_models)},
            "new_models": len(new_models),
            "avg_confidence": (
                sum(m.inference_confidence for m in all_models) / len(all_models)
                if all_models
                else 0.0
            ),
        }


# 편의 함수
async def create_hybrid_manager(scan_api: bool = True) -> HybridModelManager:
    """HybridModelManager 생성 및 로드"""
    manager = HybridModelManager()
    await manager.load(scan_api=scan_api)
    return manager
