# 데이터베이스 인덱싱 및 최적화 파이프라인

## 🎯 목표

각 DB별로 인덱싱 전략과 데이터 최적화 파이프라인을 구축하여 성능 향상

---

## 📊 현재 상태

### MongoDB
- ✅ 기본 인덱스 생성됨 (`create_session_indexes()`)
- ⚠️ 최적화 파이프라인 없음
- ⚠️ 데이터 정리/압축 없음

### Vector DB (Chroma)
- ✅ 기본 임베딩 인덱스
- ⚠️ 명시적 인덱싱 전략 없음
- ⚠️ 최적화 파이프라인 없음

### Redis
- ✅ 기본 캐싱
- ⚠️ 인덱싱 전략 없음
- ⚠️ 데이터 정리 파이프라인 없음

---

## ✅ 개선 방안

### 1. MongoDB 인덱싱 및 최적화

#### A. 현재 인덱스 상태

```python
# playground/backend/database.py
async def create_session_indexes():
    """세션 컬렉션 인덱스 생성"""
    # ✅ 이미 구현됨
    - session_id (unique)
    - updated_at
    - feature_mode
    - 복합 인덱스: feature_mode + updated_at
    - total_tokens, message_count, created_at, title
```

#### B. 추가 인덱싱 전략

**1. 메시지 컬렉션 인덱스 (신규)**
```python
# playground/backend/database.py (추가)
async def create_message_indexes():
    """메시지 관련 컬렉션 인덱스 생성"""
    db = get_mongodb_database()
    if db is None:
        return
    
    # media_cache 컬렉션 인덱스
    await db.media_cache.create_index("hash", unique=True, background=True)
    await db.media_cache.create_index("session_id", background=True)
    await db.media_cache.create_index("created_at", background=True)
    await db.media_cache.create_index([("session_id", 1), ("created_at", -1)], background=True)
    
    # multimodal_context 컬렉션 인덱스
    await db.multimodal_context.create_index("session_id", background=True)
    await db.multimodal_context.create_index("image_hash", background=True)
    await db.multimodal_context.create_index([("session_id", 1), ("created_at", -1)], background=True)
    
    # session_databases 컬렉션 인덱스 (클라우드 서비스 연결)
    await db.session_databases.create_index("session_id", background=True)
    await db.session_databases.create_index([("session_id", 1), ("service_type", 1)], background=True)
    
    logger.info("✅ Message-related indexes created")
```

**2. TTL 인덱스 (자동 정리)**
```python
async def create_ttl_indexes():
    """TTL 인덱스 생성 (자동 데이터 정리)"""
    db = get_mongodb_database()
    if db is None:
        return
    
    # media_cache: 30일 후 자동 삭제
    await db.media_cache.create_index(
        "created_at",
        expireAfterSeconds=30 * 24 * 3600,  # 30일
        background=True
    )
    
    # multimodal_context: 90일 후 자동 삭제
    await db.multimodal_context.create_index(
        "created_at",
        expireAfterSeconds=90 * 24 * 3600,  # 90일
        background=True
    )
    
    logger.info("✅ TTL indexes created")
```

**3. 텍스트 검색 인덱스**
```python
async def create_text_search_indexes():
    """텍스트 검색 인덱스 생성"""
    db = get_mongodb_database()
    if db is None:
        return
    
    # 제목 및 메시지 내용 검색
    await db.chat_sessions.create_index(
        [("title", "text"), ("messages.content_preview", "text")],
        background=True
    )
    
    logger.info("✅ Text search indexes created")
```

#### C. 데이터 최적화 파이프라인

**1. 주기적 데이터 정리**
```python
# playground/backend/services/db_optimization_service.py (신규)
class DatabaseOptimizationService:
    """
    데이터베이스 최적화 서비스
    
    주기적으로 데이터를 정리하고 최적화
    """
    
    async def optimize_mongodb(self):
        """MongoDB 최적화"""
        db = get_mongodb_database()
        if db is None:
            return
        
        # 1. 오래된 세션 정리 (90일 이상 미사용)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        result = await db.chat_sessions.delete_many({
            "updated_at": {"$lt": cutoff_date},
            "message_count": 0  # 메시지가 없는 세션만
        })
        logger.info(f"✅ Cleaned up {result.deleted_count} old sessions")
        
        # 2. 인덱스 재구축 (주기적)
        await db.chat_sessions.reindex()
        
        # 3. 통계 수집 (쿼리 최적화)
        await db.command("collStats", "chat_sessions")
    
    async def compact_collections(self):
        """컬렉션 압축 (디스크 공간 최적화)"""
        db = get_mongodb_database()
        if db is None:
            return
        
        # compact 명령 실행 (디스크 공간 회수)
        await db.command({"compact": "chat_sessions"})
        await db.command({"compact": "media_cache"})
        await db.command({"compact": "multimodal_context"})
        
        logger.info("✅ Collections compacted")
```

**2. 배치 최적화 작업**
```python
async def run_optimization_pipeline():
    """최적화 파이프라인 실행"""
    from services.db_optimization_service import db_optimization_service
    
    # 1. 인덱스 재구축
    await db_optimization_service.rebuild_indexes()
    
    # 2. 오래된 데이터 정리
    await db_optimization_service.cleanup_old_data()
    
    # 3. 통계 업데이트
    await db_optimization_service.update_statistics()
    
    # 4. 컬렉션 압축
    await db_optimization_service.compact_collections()
```

---

### 2. Vector DB (Chroma) 인덱싱 및 최적화

#### A. ChromaDB 인덱싱 전략

**1. 컬렉션별 인덱스 관리**
```python
# playground/backend/services/vector_db_optimization_service.py (신규)
class VectorDBOptimizationService:
    """
    Vector DB 최적화 서비스
    
    ChromaDB 컬렉션 인덱싱 및 최적화
    """
    
    async def optimize_collection(self, collection_name: str):
        """컬렉션 최적화"""
        from beanllm.domain.vector_stores.local.chroma import ChromaVectorStore
        
        collection = ChromaVectorStore(
            collection_name=collection_name,
            embedding_function=embedding_func
        )
        
        # 1. 중복 문서 제거
        await self._remove_duplicates(collection)
        
        # 2. 임베딩 재계산 (필요시)
        await self._recompute_embeddings(collection)
        
        # 3. 메타데이터 정리
        await self._cleanup_metadata(collection)
    
    async def _remove_duplicates(self, collection):
        """중복 문서 제거"""
        # 해시 기반 중복 감지
        # 동일한 content를 가진 문서 제거
        pass
    
    async def _recompute_embeddings(self, collection):
        """임베딩 재계산 (모델 업데이트 시)"""
        # 오래된 임베딩 재계산
        pass
    
    async def _cleanup_metadata(self, collection):
        """메타데이터 정리"""
        # 불필요한 메타데이터 제거
        # 메타데이터 크기 최적화
        pass
```

**2. 배치 인덱싱**
```python
async def batch_index_messages(messages: List[Dict[str, Any]]):
    """메시지 배치 인덱싱 (성능 최적화)"""
    from services.message_vector_store import message_vector_store
    
    # 배치로 임베딩 생성 (한 번에 여러 메시지)
    texts = [msg["content"] for msg in messages]
    embeddings = await asyncio.to_thread(embedding_func, texts)
    
    # 배치로 Vector DB에 저장
    await asyncio.to_thread(
        _message_vector_store.collection.upsert,
        ids=[msg["message_id"] for msg in messages],
        embeddings=embeddings,
        documents=texts,
        metadatas=[msg.get("metadata", {}) for msg in messages]
    )
```

#### B. ChromaDB 최적화 파이프라인

**1. 주기적 최적화**
```python
async def optimize_vector_db():
    """Vector DB 최적화"""
    # 1. 중복 제거
    await vector_db_optimization_service.remove_duplicates()
    
    # 2. 오래된 임베딩 재계산
    await vector_db_optimization_service.recompute_stale_embeddings()
    
    # 3. 메타데이터 정리
    await vector_db_optimization_service.cleanup_metadata()
    
    # 4. 컬렉션 통계 업데이트
    await vector_db_optimization_service.update_statistics()
```

**2. 인덱스 재구축**
```python
async def rebuild_vector_indexes():
    """Vector 인덱스 재구축"""
    # ChromaDB는 자동으로 인덱스를 관리하지만
    # 대량 데이터 추가 후 재구축 필요할 수 있음
    pass
```

---

### 3. Redis 인덱싱 및 최적화

#### A. Redis 인덱싱 전략

**1. 키 네임스페이스 최적화**
```python
# playground/backend/services/redis_optimization_service.py (신규)
class RedisOptimizationService:
    """
    Redis 최적화 서비스
    
    키 네임스페이스 및 인덱싱 최적화
    """
    
    def __init__(self):
        from services.session_cache import get_redis_client
        self.redis = get_redis_client()
    
    async def optimize_key_namespaces(self):
        """키 네임스페이스 최적화"""
        # 현재 키 구조:
        # - sessions:{session_id}
        # - sessions:list:{user_id}:{filter}
        # - summary:{session_id}
        # - cache:{key}
        
        # 최적화: 해시 기반 키 분산
        # sessions:{hash(session_id)[:8]}:{session_id}
        pass
    
    async def create_secondary_indexes(self):
        """보조 인덱스 생성 (Sorted Set)"""
        # 세션 목록을 Sorted Set으로 관리 (정렬 최적화)
        # sessions:list:sorted:{user_id} -> ZADD score=updated_at
        pass
```

**2. 메모리 최적화**
```python
async def optimize_redis_memory(self):
    """Redis 메모리 최적화"""
    # 1. 오래된 키 정리
    await self._cleanup_expired_keys()
    
    # 2. 큰 값 압축
    await self._compress_large_values()
    
    # 3. 메모리 사용량 모니터링
    await self._monitor_memory_usage()
```

#### B. Redis 최적화 파이프라인

**1. 주기적 정리**
```python
async def cleanup_redis():
    """Redis 정리"""
    # 1. 만료된 키 정리
    # 2. 메모리 사용량 확인
    # 3. 큰 값 압축
    # 4. 통계 업데이트
    pass
```

**2. 캐시 워밍업**
```python
async def warmup_cache():
    """자주 사용되는 데이터 캐시 워밍업"""
    # 최근 세션 목록 캐시
    # 자주 사용되는 모델 정보 캐시
    pass
```

---

### 4. 통합 최적화 파이프라인

#### A. 스케줄러 설정

```python
# playground/backend/services/optimization_scheduler.py (신규)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class OptimizationScheduler:
    """최적화 작업 스케줄러"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._setup_jobs()
    
    def _setup_jobs(self):
        """최적화 작업 스케줄 설정"""
        # 매일 새벽 2시: MongoDB 최적화
        self.scheduler.add_job(
            self._optimize_mongodb,
            trigger="cron",
            hour=2,
            minute=0
        )
        
        # 매주 일요일 새벽 3시: Vector DB 최적화
        self.scheduler.add_job(
            self._optimize_vector_db,
            trigger="cron",
            day_of_week="sun",
            hour=3,
            minute=0
        )
        
        # 매시간: Redis 정리
        self.scheduler.add_job(
            self._cleanup_redis,
            trigger="cron",
            minute=0
        )
    
    async def _optimize_mongodb(self):
        """MongoDB 최적화 실행"""
        from services.db_optimization_service import db_optimization_service
        await db_optimization_service.optimize_mongodb()
    
    async def _optimize_vector_db(self):
        """Vector DB 최적화 실행"""
        from services.vector_db_optimization_service import vector_db_optimization_service
        await vector_db_optimization_service.optimize_all_collections()
    
    async def _cleanup_redis(self):
        """Redis 정리 실행"""
        from services.redis_optimization_service import redis_optimization_service
        await redis_optimization_service.cleanup_redis()
    
    def start(self):
        """스케줄러 시작"""
        self.scheduler.start()
    
    def shutdown(self):
        """스케줄러 종료"""
        self.scheduler.shutdown()
```

#### B. 수동 최적화 엔드포인트

```python
# playground/backend/routers/optimization_router.py (신규)
@router.post("/optimize/mongodb")
async def optimize_mongodb():
    """MongoDB 수동 최적화"""
    from services.db_optimization_service import db_optimization_service
    await db_optimization_service.optimize_mongodb()
    return {"status": "success", "message": "MongoDB optimized"}

@router.post("/optimize/vector_db")
async def optimize_vector_db():
    """Vector DB 수동 최적화"""
    from services.vector_db_optimization_service import vector_db_optimization_service
    await vector_db_optimization_service.optimize_all_collections()
    return {"status": "success", "message": "Vector DB optimized"}

@router.post("/optimize/redis")
async def optimize_redis():
    """Redis 수동 최적화"""
    from services.redis_optimization_service import redis_optimization_service
    await redis_optimization_service.optimize_redis_memory()
    return {"status": "success", "message": "Redis optimized"}

@router.post("/optimize/all")
async def optimize_all():
    """모든 DB 최적화"""
    # 순차 실행
    await optimize_mongodb()
    await optimize_vector_db()
    await optimize_redis()
    return {"status": "success", "message": "All databases optimized"}
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] MongoDB 기본 인덱스 (`database.py`의 `create_session_indexes`)
- [x] 세션 Vector DB 인덱싱 (`session_search_service.py`의 `index_session`)

### ❌ 미구현
- [ ] **메시지 관련 컬렉션 인덱스 추가**
  - **통합 위치**: `database.py`의 `create_session_indexes()` 함수 확장
  - **구현 방향**: 문서의 "B. 추가 인덱싱 전략" 섹션 참조
  - **방법**: `create_message_indexes()` 함수 추가
- [ ] **TTL 인덱스 생성 (자동 정리)**
  - **통합 위치**: `database.py`의 `create_ttl_indexes()` 함수
  - **구현 방향**: 문서의 "C. 데이터 최적화 파이프라인" 섹션 참조
  - **방법**: `createIndex({ "created_at": 1 }, { expireAfterSeconds: 7776000 })` (90일)
- [ ] **텍스트 검색 인덱스 생성**
  - **통합 위치**: `database.py`의 `create_text_search_indexes()` 함수
  - **구현 방향**: MongoDB Text Search 인덱스
  - **방법**: 문서의 "C. 텍스트 검색 인덱스" 섹션 참조
- [ ] **데이터 최적화 파이프라인 구현**
  - **파일**: `playground/backend/services/db_optimization_service.py` (신규 생성 필요)
  - **구현 방향**: 문서의 "C. 데이터 최적화 파이프라인" 섹션 참조
  - **방법**: 주기적으로 오래된 데이터 정리, 인덱스 재구축
- [ ] **주기적 정리 작업 스케줄링**
  - **통합 위치**: `OptimizationScheduler` 서비스
  - **방법**: `asyncio` 또는 `APScheduler` 활용
- [ ] **컬렉션별 인덱스 관리**
  - **파일**: `playground/backend/services/vector_db_optimization_service.py` (신규 생성 필요)
  - **구현 방향**: 문서의 "2. Vector DB 인덱싱 및 최적화" 섹션 참조
- [ ] **중복 문서 제거**
  - **통합 위치**: `vector_db_optimization_service.py`
  - **방법**: 문서 해시 기반 중복 감지
- [ ] **배치 인덱싱 최적화**
  - **통합 위치**: `vector_db_optimization_service.py`
  - **방법**: 여러 메시지를 한 번에 임베딩 생성 후 배치 저장
- [ ] **임베딩 재계산 파이프라인**
  - **통합 위치**: `vector_db_optimization_service.py`
  - **방법**: 오래된 임베딩 식별 후 재계산
- [ ] **메타데이터 정리**
  - **통합 위치**: `vector_db_optimization_service.py`
  - **방법**: 불필요한 메타데이터 필드 제거
- [ ] **키 네임스페이스 최적화**
  - **파일**: `playground/backend/services/redis_optimization_service.py` (신규 생성 필요)
  - **구현 방향**: 문서의 "3. Redis 인덱싱 및 최적화" 섹션 참조
- [ ] **보조 인덱스 생성 (Sorted Set)**
  - **통합 위치**: `redis_optimization_service.py`
  - **방법**: Sorted Set으로 정렬된 키 관리
- [ ] **메모리 최적화**
  - **통합 위치**: `redis_optimization_service.py`
  - **방법**: 큰 값 압축, 만료된 키 정리
- [ ] **주기적 정리 파이프라인**
  - **통합 위치**: `redis_optimization_service.py`
  - **방법**: 주기적으로 만료된 키 삭제
- [ ] **캐시 워밍업**
  - **통합 위치**: `redis_optimization_service.py`
  - **방법**: 자주 사용되는 데이터 사전 로드
- [ ] **OptimizationScheduler 생성**
  - **파일**: `playground/backend/services/optimization_scheduler.py` (신규 생성 필요)
  - **구현 방향**: 문서의 "4. 통합 최적화 파이프라인" 섹션 참조
  - **방법**: 모든 최적화 작업을 스케줄링
- [ ] **스케줄러 작업 설정**
  - **통합 위치**: `optimization_scheduler.py`
  - **방법**: 매시간, 매일, 매주 작업 설정
- [ ] **수동 최적화 엔드포인트**
  - **위치**: `routers/optimization_router.py` (신규 생성 필요)
  - **방법**: 문서의 "B. 수동 최적화 엔드포인트" 섹션 참조
- [ ] **최적화 진행 상황 모니터링**
  - **통합 위치**: `optimization_scheduler.py`
  - **방법**: SSE 또는 로그로 진행 상황 전달

---

## 🎯 최적화 전략 요약

### MongoDB
1. **인덱싱**: 모든 쿼리 패턴에 맞는 인덱스 생성
2. **TTL**: 오래된 데이터 자동 삭제
3. **압축**: 주기적 컬렉션 압축
4. **통계**: 쿼리 최적화를 위한 통계 수집

### Vector DB
1. **배치 처리**: 여러 메시지를 한 번에 인덱싱
2. **중복 제거**: 동일한 내용의 문서 제거
3. **재계산**: 오래된 임베딩 재계산
4. **메타데이터 정리**: 불필요한 메타데이터 제거

### Redis
1. **키 최적화**: 효율적인 키 네임스페이스
2. **인덱싱**: Sorted Set으로 정렬 최적화
3. **메모리 관리**: 큰 값 압축, 만료된 키 정리
4. **워밍업**: 자주 사용되는 데이터 사전 로드

---

## 💡 핵심 원칙

1. **자동화**: 주기적 최적화 작업 자동 실행
2. **모니터링**: 최적화 진행 상황 및 결과 모니터링
3. **점진적**: 대량 작업은 배치로 처리
4. **안전성**: 최적화 중 데이터 손실 방지

---

## 🔗 관련 문서

- [03_CONTEXT_MANAGEMENT.md](./03_CONTEXT_MANAGEMENT.md): 요약 저장 시 인덱싱 활용
- [04_SESSION_RAG.md](./04_SESSION_RAG.md): RAG 컬렉션 인덱싱
- [08_MULTIMODAL_CONTEXT.md](./08_MULTIMODAL_CONTEXT.md): 이미지 메타데이터 인덱싱
