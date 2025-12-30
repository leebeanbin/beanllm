"""
STT 엔진 모듈

6개 STT 엔진 구현:
- Whisper V3 Turbo: OpenAI 최신 (6x faster, 99+ 언어)
- Distil-Whisper: 압축 모델 (6x faster, 정확도 유지)
- NVIDIA Parakeet TDT: 실시간 최적화 (RTFx >2000)
- Canary-1B: 다국어 + 번역 (4개 언어)
- Canary-1B-Flash: 초고속 추론 (RTFx >1000)
- Moonshine: 온디바이스 (초경량)
"""

from .base import BaseSTTEngine

__all__ = ["BaseSTTEngine"]

# Whisper V3 Turbo 엔진 (optional dependency)
try:
    from .whisper_engine import WhisperEngine

    __all__.append("WhisperEngine")
except ImportError:
    pass

# Distil-Whisper 엔진 (optional dependency)
try:
    from .distil_whisper_engine import DistilWhisperEngine

    __all__.append("DistilWhisperEngine")
except ImportError:
    pass

# NVIDIA Parakeet 엔진 (optional dependency)
try:
    from .parakeet_engine import ParakeetEngine

    __all__.append("ParakeetEngine")
except ImportError:
    pass

# Canary 엔진 (optional dependency)
try:
    from .canary_engine import CanaryEngine

    __all__.append("CanaryEngine")
except ImportError:
    pass

# Moonshine 엔진 (optional dependency)
try:
    from .moonshine_engine import MoonshineEngine

    __all__.append("MoonshineEngine")
except ImportError:
    pass
