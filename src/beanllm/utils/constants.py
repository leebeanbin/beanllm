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

LOW_TEMPERATURE: float = 0.3
"""낮은 온도 (사실적/결정적 작업용)"""

HIGH_TEMPERATURE: float = 0.9
"""높은 온도 (창의적 생성용)"""

MIN_TEMPERATURE: float = 0.0
"""최소 온도 (결정론적 생성)"""

MAX_TEMPERATURE: float = 2.0
"""최대 온도"""

DEFAULT_MAX_TOKENS: int = 4096
"""기본 max_tokens (일반 LLM 호출)"""

LARGE_MAX_TOKENS: int = 8192
"""큰 컨텍스트용 max_tokens"""

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

DEFAULT_RAG_CHUNK_SIZE: int = 500
"""RAG 파이프라인 기본 청크 크기 (문자)"""

MIN_CHUNK_SIZE: int = 100
"""최소 청크 크기 (문자)"""

DEFAULT_CHUNK_OVERLAP: int = 200
"""기본 문서 청크 오버랩 (문자)"""

# ============================================================
# 네트워크 / 인프라
# ============================================================

KAFKA_DEFAULT_RETRIES: int = 3
"""Kafka 기본 재시도 횟수"""

DEFAULT_TIMEOUT_SECONDS: int = 30
"""기본 네트워크 타임아웃 (초)"""

REDIS_TIMEOUT: float = 2.0
"""Redis 작업 기본 타임아웃 (초)"""

REDIS_SCAN_TIMEOUT: float = 5.0
"""Redis SCAN/대량 작업 타임아웃 (초)"""

LOCK_ACQUIRE_TIMEOUT: float = 30.0
"""분산 락 획득 기본 타임아웃 (초)"""

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

# ============================================================
# Optimizer 기본값
# ============================================================

DEFAULT_NUM_QUERIES: int = 50
"""벤치마크 기본 쿼리 수"""

DEFAULT_N_TRIALS: int = 50
"""최적화 기본 시행 횟수"""

# ============================================================
# RAG 세부 설정
# ============================================================

DEFAULT_RAG_CHUNK_OVERLAP: int = 50
"""RAG 파이프라인 기본 chunk overlap (문자)"""

# ============================================================
# ML / Vision
# ============================================================

DEFAULT_VLM_MAX_TOKENS: int = 1024
"""Vision-Language 모델 기본 max_new_tokens"""

DEFAULT_MAX_SEQ_LENGTH: int = 2048
"""Fine-tuning 기본 시퀀스 길이"""

# ============================================================
# 파일 I/O
# ============================================================

MMAP_THRESHOLD_BYTES: int = 10 * 1024 * 1024
"""mmap 사용 임계값 (10MB 이상 파일)"""

DEFAULT_STREAMING_CHUNK_SIZE: int = 1024 * 1024
"""스트리밍 읽기 청크 크기 (1MB)"""

# ============================================================
# 캐시
# ============================================================

DEFAULT_CACHE_MAX_SIZE: int = 1000
"""LRU/TTL 캐시 기본 최대 항목 수"""

# ============================================================
# 실험 / 디버그
# ============================================================

DEFAULT_CHUNK_SIZES: list[int] = [256, 512, 1000, 2000]
"""청킹 실험 기본 크기 목록"""

MS_PER_SECOND: int = 1000
"""밀리초 ↔ 초 변환 상수"""
