"""
beanllm 전역 상수 정의

매직 넘버를 중앙에서 관리하여 일관성 및 유지보수성 향상.
변경 시 한 곳만 수정하면 전체 프로젝트에 반영됨.
"""

# ============================================================
# LLM 기본값
# ============================================================

DEFAULT_TEMPERATURE: float = 0.7
"""기본 생성 온도 (0.0~2.0)"""

MIN_TEMPERATURE: float = 0.0
"""최소 온도 (결정론적 생성)"""

MAX_TEMPERATURE: float = 2.0
"""최대 온도"""

CLAUDE_DEFAULT_MAX_TOKENS: int = 4096
"""Claude 모델 기본 max_tokens"""

DEFAULT_QUERY_EXPANSION_MAX_TOKENS: int = 512
"""Query expansion 기본 max_tokens"""

DEFAULT_MAX_CONTEXT_TOKENS: int = 4000
"""RAG 컨텍스트 최대 토큰 수"""

# ============================================================
# 재시도 (Retry)
# ============================================================

DEFAULT_MAX_RETRIES: int = 3
"""기본 최대 재시도 횟수"""

RESILIENT_MAX_RETRIES: int = 5
"""고가용성 재시도 횟수 (critical path)"""

DEFAULT_RETRY_INITIAL_DELAY: float = 1.0
"""재시도 초기 대기 시간 (초)"""

# ============================================================
# 배치 처리
# ============================================================

DEFAULT_EMBEDDING_BATCH_SIZE: int = 32
"""임베딩 기본 배치 크기"""

DEFAULT_VISION_BATCH_SIZE: int = 100
"""Vision RAG 배치 처리 크기"""

LOCAL_EMBEDDING_BATCH_SIZE: int = 64
"""로컬 임베딩 GPU 배치 크기"""

# ============================================================
# RAG / 검색
# ============================================================

DEFAULT_TOP_K: int = 5
"""기본 유사도 검색 결과 수"""

DEFAULT_CHUNK_SIZE: int = 1000
"""기본 문서 청크 크기 (문자)"""

DEFAULT_CHUNK_OVERLAP: int = 200
"""기본 문서 청크 오버랩 (문자)"""

# ============================================================
# 네트워크 / 인프라
# ============================================================

KAFKA_DEFAULT_RETRIES: int = 3
"""Kafka 기본 재시도 횟수"""

DEFAULT_TIMEOUT_SECONDS: int = 30
"""기본 네트워크 타임아웃 (초)"""

# ============================================================
# 검증 제한
# ============================================================

MAX_MESSAGE_LENGTH: int = 100_000
"""메시지 최대 길이 (문자)"""

MAX_FILE_SIZE: int = 50 * 1024 * 1024
"""파일 업로드 최대 크기 (50MB)"""

MAX_UPLOAD_CHUNK_SIZE: int = 1024 * 1024
"""파일 업로드 청크 크기 (1MB, 스트리밍 저장용)"""

HEALTH_CHECK_MAX_TOKENS: int = 5
"""Provider health check용 max_tokens"""

HEALTH_CHECK_MAX_TOKENS_LARGE: int = 10
"""Provider health check용 max_tokens (Perplexity/DeepSeek)"""
