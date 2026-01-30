# 검색엔진 통합

## 🎯 목표

검색 결과를 인덱싱하고 캐싱하여 AI 검색 강화

---

## 📊 현재 문제점

- ❌ AI 검색 결과 캐싱 없음
- ❌ 검색 결과 기반 컨텍스트 관리 부족

---

## ✅ 개선 방안

### 1. 검색엔진 옵션

**Meilisearch (권장)**
- 오픈소스
- 빠르고 가벼움
- 무제한

**Algolia**
- 빠른 검색
- API 간단
- 무료 10K records

### 2. 검색 결과 인덱싱

```python
# playground/backend/services/search_engine_service.py
class SearchEngineService:
    """검색 결과 인덱싱 및 캐싱"""
    
    async def index_search_result(
        self,
        session_id: str,
        query: str,
        results: List[Dict[str, Any]],
        ai_summary: str
    ):
        """검색 결과를 인덱싱"""
        # Meilisearch에 저장
```

### 3. 검색 결과 캐싱

```python
# 이전 검색 결과 확인
# 캐시된 결과 사용
# 새 검색 시 인덱싱
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] 웹 검색 기본 기능 (`routers/web_router.py`, `services/orchestrator.py`의 `_handle_web_search`)
- [x] MCP tool `web_search` 사용

### ❌ 미구현
- [ ] **SearchEngineService 생성**
  - **파일**: `playground/backend/services/search_engine_service.py` (신규 생성 필요)
  - **구현 방향**:
    1. Meilisearch 클라이언트 초기화
    2. 검색 결과를 인덱싱
    3. 검색 결과 캐싱 (Redis 또는 MongoDB)
  - **방법**:
    ```python
    from meilisearch import Client as MeiliClient
    
    class SearchEngineService:
        def __init__(self):
            self.meili = MeiliClient(
                url=os.getenv("MEILISEARCH_URL", "http://localhost:7700"),
                api_key=os.getenv("MEILISEARCH_API_KEY")
            )
            self.index_name = "search_results"
        
        async def index_search_result(
            self,
            session_id: str,
            query: str,
            results: List[Dict[str, Any]],
            ai_summary: Optional[str] = None
        ):
            """검색 결과 인덱싱"""
            documents = [{
                "id": f"{session_id}_{query}_{i}",
                "session_id": session_id,
                "query": query,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
                "ai_summary": ai_summary,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            } for i, r in enumerate(results)]
            
            await self.meili.index(self.index_name).add_documents(documents)
    ```
- [ ] **Meilisearch/Algolia 통합**
  - **선택**: Meilisearch (오픈소스, 무료)
  - **설치**: Docker로 Meilisearch 실행 또는 클라우드 서비스
  - **의존성**: `pyproject.toml`에 `meilisearch` 추가
- [ ] **검색 결과 인덱싱**
  - **통합 위치**: `orchestrator.py`의 `_handle_web_search` 메서드
  - **방법**: 웹 검색 실행 후 `SearchEngineService.index_search_result()` 호출
- [ ] **검색 결과 캐싱**
  - **구현 방향**:
    1. 동일 쿼리 검색 시 캐시 확인
    2. 캐시 히트 시 인덱싱된 결과 반환
    3. 캐시 미스 시 새 검색 및 인덱싱
  - **방법**:
    ```python
    async def search_with_cache(
        self,
        query: str,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """캐시된 검색 결과 조회"""
        # 1. 캐시 확인
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        cached = await redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # 2. Meilisearch에서 검색
        results = await self.meili.index(self.index_name).search(query)
        
        # 3. 캐시 저장
        await redis.setex(cache_key, 3600, json.dumps(results))
        
        return results
    ```

---

## 🎯 우선순위

**낮음**: 검색 기능 강화

---

## ⚠️ 중요: 내부 DB 검색과의 구분

**이 검색엔진은 내부 DB 검색과 별개입니다:**

- **내부 DB 검색** (이미 구현됨): 채팅 세션/메시지 검색
  - 위치: `session_search_service.py`, `message_vector_store.py`
  - 기술: MongoDB + Vector DB
  - 목적: 사용자의 대화 내용 검색

- **외부 검색엔진** (이 문서): 웹 검색 결과 관리
  - 기술: Meilisearch/Algolia
  - 목적: 인터넷 검색 결과 인덱싱/캐싱

**관련 문서**: [14_SEARCH_ARCHITECTURE.md](./14_SEARCH_ARCHITECTURE.md)
