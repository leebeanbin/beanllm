# Playground Backend 구조 분석 및 개선 제안

## 📊 현재 구조

```
playground/backend/
├── main.py                    # ~970줄 - 메인 애플리케이션
├── common.py                  # 공통 유틸리티 (get_client, get_kg 등)
├── database.py                # MongoDB 연결 관리
├── .env.example
├── requirements.txt
│
├── routers/                   # 17개 라우터 (✅ 잘 정리됨)
│   ├── __init__.py            # ⚠️ 일부만 export
│   ├── config_router.py
│   ├── chat_router.py
│   ├── rag_router.py
│   └── ...
│
├── services/                  # 10개 서비스 (✅ 잘 정리됨)
│   ├── __init__.py            # ✅ 모든 서비스 export
│   ├── config_service.py
│   ├── encryption_service.py
│   └── ...
│
├── schemas/                   # 스키마 파일들 (✅ 잘 정리됨)
│   ├── __init__.py            # ✅ 모든 스키마 export
│   ├── chat.py
│   ├── rag.py
│   └── ...
│
├── monitoring/                # 모니터링 관련
│   ├── __init__.py
│   ├── middleware.py
│   └── dashboard.py
│
├── scripts/                   # 스크립트 파일들 (✅ 정리됨)
│   ├── setup_and_build.sh
│   ├── auto_setup_and_test.sh
│   └── quick_test.sh
│
├── docs/                      # 문서 파일들 (✅ 정리됨)
│   ├── CLEANUP_ANALYSIS.md
│   └── MCP_INTEGRATION_ANALYSIS.md
│
└── tests/                     # 테스트 파일들
    └── ...
```

## 🔍 발견된 문제점

### 1. 루트 디렉토리 파일 과다 (해결됨 ✅)
**현재:**
- `main.py` - 메인 애플리케이션 (필수)
- `common.py` - 공통 유틸리티
- `database.py` - DB 연결
- ~~`chat_history.py`~~ → `routers/history_router.py`로 이동됨 ✅
- ~~`models.py`~~ → `schemas/database.py`로 이동됨 ✅
- ~~`mcp_streaming.py`~~ → 삭제됨 (2025-01-25) ✅

**문제:**
- 루트에 파일이 너무 많아 가독성 저하
- `chat_history.py`는 라우터인데 `routers/`에 없음
- `models.py`는 스키마인데 `schemas/`에 없음

### 2. routers/__init__.py 불완전
**현재 export:**
```python
__all__ = [
    "config_router",
    "chat_router",
    "rag_router",
    "kg_router",
    "models_router",
    "agent_router",
]
```

**실제 라우터 수:** 17개
**누락된 라우터:** 11개 (audio, chain, evaluation, finetuning, google_auth, monitoring, ocr, optimizer, vision, web 등)

### 3. 파일 분류 문제
- `chat_history.py`: 라우터인데 루트에 있음
- `models.py`: 스키마인데 루트에 있음
- `common.py`: 유틸리티인데 적절한 위치 없음

### 4. 디렉토리 구조 개선 필요
현재는 기능별로 잘 정리되어 있지만, 일부 파일들이 적절한 위치에 없음

## 💡 개선 제안

### Option 1: 최소 변경 (권장)
```
playground/backend/
├── main.py                    # 메인 애플리케이션만
├── core/                      # ✨ 새로 생성
│   ├── __init__.py
│   ├── common.py             # 공통 유틸리티
│   ├── database.py           # DB 연결
│   └── config.py             # 설정 (향후)
├── routers/
│   ├── chat_history.py       # chat_history.py 이동
│   └── ...
├── schemas/
│   ├── models.py             # models.py 이동
│   └── ...
└── ...
```

**장점:**
- 최소한의 변경
- 루트 디렉토리 깔끔해짐
- 기존 import 경로는 `core/` 추가로 수정

### Option 2: 완전 재구성
```
playground/backend/
├── main.py
├── core/                      # 핵심 인프라
│   ├── database.py
│   ├── config.py
│   └── common.py
├── api/                       # API 레이어
│   ├── routers/
│   └── schemas/
├── services/                  # 비즈니스 로직
├── infrastructure/             # 인프라
│   └── monitoring/
└── ...
```

**장점:**
- 더 명확한 레이어 분리
- 확장성 좋음

**단점:**
- 많은 import 경로 수정 필요

## 📋 우선순위별 개선 작업

### 높음 (즉시)
1. **routers/__init__.py 완성**
   - 모든 17개 라우터 export 추가

2. **chat_history.py 이동**
   - `routers/chat_history_router.py`로 이동
   - 또는 `routers/history_router.py`로 이름 변경 후 이동

3. **models.py 이동**
   - `schemas/models.py`로 이동
   - 또는 `schemas/database.py`로 이름 변경 (DB 모델이므로)

### 중간 (선택적)
4. **common.py, database.py 정리**
   - `core/` 디렉토리 생성 후 이동
   - 또는 `utils/` 디렉토리 생성

5. **루트 디렉토리 최소화**
   - `main.py`만 남기기

### 낮음 (완료 ✅)
6. **mcp_streaming.py 제거** ✅ (2025-01-25)
   - MCP Client Service로 대체됨
   - orchestrator.py가 모든 Tool 실행 담당

## 🎯 권장 작업 순서

1. ✅ **routers/__init__.py 완성** (5분)
2. ✅ **chat_history.py → routers/ 이동** (10분)
3. ✅ **models.py → schemas/ 이동** (10분)
4. ⚠️ **core/ 디렉토리 생성 및 common.py, database.py 이동** (15분, import 경로 수정 필요)

---

## 📊 현재 vs 개선 후

### 이전 루트 파일
- main.py
- common.py
- database.py
- ~~chat_history.py~~ (이동됨)
- ~~models.py~~ (이동됨)
- ~~mcp_streaming.py~~ (삭제됨)
**총 6개**

### ✅ 개선 후 루트 파일 (2025-01-25)
- main.py
- common.py
- database.py
**총 3개 (50% 감소)**

### ✅ 완료된 작업 (2025-01-24)
1. **routers/__init__.py 완성** - 모든 17개 라우터 export ✅
2. **chat_history.py → routers/history_router.py 이동** ✅
3. **models.py → schemas/database.py 이동** ✅
4. **requirements.txt 삭제** - pyproject.toml의 `web` 옵션에 통합 ✅
5. **pyproject.toml 업데이트** - playground backend 의존성 추가 ✅
6. **scripts/ 디렉토리 생성** - 스크립트 파일들 정리 ✅
7. **docs/ 디렉토리 생성** - 문서 파일들 정리 ✅
8. **README.md 생성** - playground/backend/README.md 생성 및 루트 README.md 업데이트 ✅

### 📁 최종 구조 (2025-01-25)
```
playground/backend/
├── main.py                    # 메인 애플리케이션 (~970줄)
├── common.py                  # 공통 유틸리티
├── database.py                # DB 연결
├── routers/                   # 18개 라우터 (history_router 포함)
├── schemas/                   # 스키마 (database.py 포함)
├── services/                  # 12개 서비스 (mcp_client, context_manager 포함)
├── scripts/                   # 스크립트
└── docs/                      # 문서
```
