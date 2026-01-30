# 멀티모달 컨텍스트 관리

## 🎯 목표

이미지/파일 캐싱 및 AI 메모리 시스템으로 이전 대화의 이미지를 기억하고 참조 가능

---

## 📊 현재 문제점

- ❌ 이미지/파일 캐싱 없음 (매번 재전송)
- ❌ AI가 이미지를 기억하지 못함
- ❌ 이미지와 텍스트의 연관성 관리 부족

---

## ✅ 개선 방안

### 1. 이미지/파일 캐싱

**옵션 1: Firebase Storage (권장)**
- 무료 티어: 5GB
- CDN 지원
- 실시간 업데이트

**옵션 2: AWS S3**
- 확장성
- 저렴한 비용

**옵션 3: Google Cloud Storage**
- Google 통합

### 2. AI 메모리 시스템

```python
# playground/backend/services/multimodal_context_service.py
class MultimodalContextService:
    """이미지-텍스트 연관성 저장"""
    
    async def save_image_context(
        self,
        session_id: str,
        image_hash: str,
        image_url: str,
        user_message: str,
        ai_response: str
    ):
        """이미지 컨텍스트 저장"""
        # 이미지 임베딩 + 텍스트 임베딩 저장
        # Vector DB에 연관성 저장
```

### 3. 채팅에서 이미지 캐싱 통합

```python
# 이미지 해시로 중복 확인
# 캐시된 이미지 URL 사용
# 이전 대화의 관련 이미지 가져오기
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] MongoDB `multimodal_context` 컬렉션 구조 (database.py에 정의됨)
- [x] 이미지 업로드 기본 기능 (chat_router.py)

### ❌ 미구현
- [ ] **MediaCacheService 생성**
  - **파일**: `playground/backend/services/media_cache_service.py` (신규 생성 필요)
  - **구현 방향**:
    1. 이미지 해시 계산 (MD5 또는 SHA256)
    2. Firebase Storage/S3/GCS 선택 (환경변수로 설정)
    3. 이미지 URL 캐싱 (MongoDB `media_cache` 컬렉션)
  - **방법**:
    ```python
    class MediaCacheService:
        async def cache_image(
            self, 
            image_data: bytes, 
            session_id: str
        ) -> str:
            """이미지 캐싱 및 URL 반환"""
            # 1. 해시 계산
            image_hash = hashlib.md5(image_data).hexdigest()
            
            # 2. 캐시 확인
            cached = await db.media_cache.find_one({"hash": image_hash})
            if cached:
                return cached["url"]
            
            # 3. 클라우드 스토리지 업로드
            url = await self._upload_to_storage(image_data, image_hash)
            
            # 4. MongoDB에 저장
            await db.media_cache.insert_one({
                "hash": image_hash,
                "url": url,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc)
            })
            
            return url
    ```
- [ ] **Firebase Storage/S3/GCS 통합**
  - **선택 기준**: 비용, 확장성, Google 통합 여부
  - **구현 방향**: 환경변수로 선택 가능하도록
  - **방법**: 각 스토리지별 어댑터 클래스 생성
- [ ] **MultimodalContextService 생성**
  - **파일**: `playground/backend/services/multimodal_context_service.py` (신규 생성 필요)
  - **구현 방향**:
    1. 이미지 임베딩 생성 (Vision 모델 활용)
    2. 텍스트 임베딩 생성 (기존 임베딩 모델)
    3. Vector DB에 연관성 저장
  - **방법**:
    ```python
    class MultimodalContextService:
        async def save_image_context(
            self,
            session_id: str,
            image_hash: str,
            user_message: str,
            ai_response: str
        ):
            """이미지-텍스트 연관성 저장"""
            # 1. 이미지 임베딩
            image_embedding = await self._embed_image(image_hash)
            
            # 2. 텍스트 임베딩
            text_embedding = await self._embed_text(f"{user_message} {ai_response}")
            
            # 3. Vector DB에 저장
            await message_vector_store.save_multimodal_context(
                session_id=session_id,
                image_hash=image_hash,
                image_embedding=image_embedding,
                text_embedding=text_embedding,
                user_message=user_message,
                ai_response=ai_response
            )
    ```
- [ ] **이미지-텍스트 연관성 저장**
  - **통합 위치**: `orchestrator.py`의 `_handle_chat` 또는 `_handle_vision`
  - **방법**: 이미지가 포함된 메시지 처리 후 `MultimodalContextService.save_image_context()` 호출
- [ ] **채팅에서 이미지 캐싱 통합**
  - **통합 위치**: `routers/chat_router.py`의 `/api/chat` 또는 `/api/chat/agentic`
  - **방법**:
    1. 요청에 이미지가 있으면 `MediaCacheService.cache_image()` 호출
    2. 캐시된 URL을 메시지에 포함
    3. 이전 대화의 관련 이미지 조회 (`MultimodalContextService.get_related_images()`)
- [ ] MongoDB 인덱싱 (multimodal_context 컬렉션) - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조
- [ ] Vector DB 인덱싱 (이미지-텍스트 연관성) - [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md) 참조

---

## 🎯 우선순위

**중간**: 멀티모달 지원 강화

---

## 🔗 관련 문서

- [13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md): 이미지 메타데이터 인덱싱 및 최적화
