# 클라우드 서비스 연동

## 🎯 목표

다양한 클라우드 서비스를 데이터베이스/파일 저장소로 활용 (비용 절감)

---

## 📊 서비스 옵션 비교

### 데이터베이스/스프레드시트

| 서비스 | 비용 | 제한 | 통합 난이도 | 우선순위 |
|--------|------|------|------------|---------|
| **Google Sheets** | 무료 | 1,000만 셀 | ⭐ 쉬움 | 1순위 |
| **Notion** | 무료/유료 | 무제한 (유료) | ⭐⭐ 보통 | 2순위 |
| **Airtable** | 무료/유료 | 1,200 레코드 | ⭐⭐ 보통 | 3순위 |

### 파일 저장

| 서비스 | 비용 | 제한 | 통합 난이도 | 우선순위 |
|--------|------|------|------------|---------|
| **Google Drive** | 무료 | 15GB | ⭐ 쉬움 | 1순위 |
| **Dropbox** | 무료/유료 | 2GB | ⭐⭐ 보통 | 2순위 |

---

## ✅ 구현 방안

### 1. Google Sheets 데이터베이스

```python
# playground/backend/services/google_sheets_db_service.py
class GoogleSheetsDBService:
    """Google Sheets를 데이터베이스로 사용"""
    
    async def query_sheet(
        self,
        session_id: str,
        natural_language_query: str
    ) -> Dict[str, Any]:
        """자연어 쿼리를 Google Sheets 데이터로 변환"""
        # Google Sheets API로 데이터 읽기
        # LLM으로 필터링 조건 분석
        # 결과 반환
```

### 2. Notion 데이터베이스

```python
# playground/backend/services/notion_db_service.py
class NotionDBService:
    """Notion을 데이터베이스로 사용"""
    
    async def query_database(
        self,
        session_id: str,
        natural_language_query: str
    ) -> Dict[str, Any]:
        """자연어 쿼리를 Notion 쿼리로 변환"""
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] Google OAuth 서비스 (`google_oauth_service.py`)
- [x] Google Drive 핸들러 (`orchestrator.py`의 `_handle_google_drive`)
- [x] Google Docs 핸들러 (`orchestrator.py`의 `_handle_google_docs`)
- [x] Google Gmail 핸들러 (`orchestrator.py`의 `_handle_google_gmail`)
- [x] MCP Google tools (`mcp_server/tools/google_tools.py`)

### ⚠️ 부분 구현
- [ ] **Google Sheets 핸들러**
  - **현재**: `orchestrator.py`의 `_handle_google_sheets`에 TODO 주석
  - **필요**: Google Sheets API 연동 (시트 생성/데이터 입력/조회)
  - **통합 위치**: `orchestrator.py`의 `_handle_google_sheets` 메서드
  - **방법**:
    ```python
    # Google Sheets API 사용
    from googleapiclient.discovery import build
    service = build('sheets', 'v4', credentials=credentials)
    
    # 시트 생성 또는 데이터 읽기/쓰기
    # MCP tool로 래핑하여 사용
    ```

### ❌ 미구현
- [ ] **CloudServiceFactory 생성**
  - **파일**: `playground/backend/services/cloud_service_factory.py` (신규 생성 필요)
  - **구현 방향**: 여러 클라우드 서비스를 통합 관리하는 팩토리
  - **방법**: 각 서비스별 어댑터 패턴으로 구현
- [ ] **Google Sheets 데이터베이스 연동**
  - **통합 위치**: `orchestrator.py`의 `_handle_google_sheets` 또는 별도 서비스
  - **구현 방향**:
    1. Google Sheets API로 데이터 읽기/쓰기
    2. 자연어 쿼리를 Sheets 쿼리로 변환 (LLM 활용)
    3. RAG에 Sheets 데이터 인덱싱 (선택적)
  - **방법**: 문서의 "1. Google Sheets 데이터베이스" 섹션 참조
- [ ] **Notion 데이터베이스 연동 (선택)**
  - **파일**: `playground/backend/services/notion_db_service.py` (신규 생성 필요)
  - **의존성**: `notion-client` 패키지 추가 필요
  - **방법**: 문서의 "2. Notion 데이터베이스" 섹션 참조
- [ ] **Airtable 데이터베이스 연동 (선택)**
  - **파일**: `playground/backend/services/airtable_db_service.py` (신규 생성 필요)
  - **의존성**: `pyairtable` 패키지 추가 필요
- [ ] **Dropbox 파일 저장 연동 (선택)**
  - **파일**: `playground/backend/services/dropbox_service.py` (신규 생성 필요)
  - **의존성**: `dropbox` 패키지 추가 필요

---

## 🎯 우선순위

**낮음**: 비용 절감 목적
