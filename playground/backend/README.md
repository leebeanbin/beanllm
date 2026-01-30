# beanllm Playground Backend

FastAPI 기반 백엔드 서버로 beanllm의 모든 기능을 제공하는 통합 API입니다.

## 📁 디렉토리 구조

```
playground/backend/
├── main.py                    # FastAPI 애플리케이션 진입점
├── common.py                  # 공통 유틸리티 및 상태 관리
├── database.py                # MongoDB 연결 관리
├── .env.example               # 환경 변수 템플릿
│
├── routers/                   # API 라우터 (18개)
│   ├── __init__.py            # 모든 라우터 export
│   ├── config_router.py      # 설정 및 API 키 관리
│   ├── chat_router.py        # 채팅 API
│   ├── history_router.py      # 채팅 히스토리 관리
│   ├── rag_router.py         # RAG 파이프라인
│   ├── kg_router.py          # Knowledge Graph
│   ├── agent_router.py       # Agent 및 Multi-Agent
│   ├── chain_router.py       # Chain 워크플로우
│   ├── vision_router.py      # Vision RAG
│   ├── audio_router.py        # Audio 처리
│   ├── evaluation_router.py  # 평가 및 벤치마크
│   ├── finetuning_router.py  # Fine-tuning
│   ├── ocr_router.py         # OCR
│   ├── web_router.py         # Web Search
│   ├── optimizer_router.py   # 성능 최적화
│   ├── models_router.py      # 모델 관리
│   ├── monitoring_router.py  # 모니터링
│   └── google_auth_router.py  # Google OAuth
│
├── services/                  # 비즈니스 로직 서비스 (10개)
│   ├── __init__.py            # 모든 서비스 export
│   ├── config_service.py     # 런타임 설정 관리
│   ├── encryption_service.py # API 키 암호화
│   ├── key_validator.py      # API 키 검증
│   ├── intent_classifier.py  # 의도 분류 (Agentic)
│   ├── tool_registry.py      # 도구 관리
│   ├── orchestrator.py       # Agentic 오케스트레이터
│   ├── google_oauth_service.py # Google OAuth
│   ├── message_vector_store.py # Vector DB 메시지 저장
│   ├── session_search_service.py # 하이브리드 세션 검색
│   └── session_cache.py      # Redis 세션 캐싱
│
├── schemas/                   # Pydantic 스키마 (요청/응답 모델)
│   ├── __init__.py            # 모든 스키마 export
│   ├── database.py           # DB 모델 (ChatSession, ApiKey 등)
│   ├── chat.py               # 채팅 요청/응답
│   ├── rag.py                # RAG 요청/응답
│   ├── kg.py                 # Knowledge Graph 요청/응답
│   ├── agent.py              # Agent 요청/응답
│   ├── multi_agent.py       # Multi-Agent 요청/응답
│   ├── chain.py             # Chain 요청/응답
│   ├── vision.py            # Vision 요청/응답
│   ├── audio.py             # Audio 요청/응답
│   ├── evaluation.py        # Evaluation 요청/응답
│   ├── finetuning.py        # Fine-tuning 요청/응답
│   ├── optimizer.py         # Optimizer 요청/응답
│   ├── web.py               # Web Search 요청/응답
│   └── responses/           # 응답 전용 스키마
│       ├── agent.py
│       ├── kg.py
│       └── rag.py
│
├── monitoring/                # 모니터링 미들웨어
│   ├── __init__.py
│   ├── middleware.py         # 요청/응답 로깅
│   └── dashboard.py          # Streamlit 대시보드
│
├── scripts/                   # 유틸리티 스크립트
│   ├── setup_and_build.sh    # 전체 설정 및 빌드
│   ├── auto_setup_and_test.sh # 자동 설정 및 테스트
│   └── quick_test.sh         # 빠른 테스트
│
├── docs/                      # 문서
│   ├── CLEANUP_ANALYSIS.md   # 코드 정리 분석
│   ├── MCP_INTEGRATION_ANALYSIS.md # MCP 통합 분석
│   └── STRUCTURE_ANALYSIS.md # 구조 분석 및 개선 제안
│
└── tests/                     # 테스트 파일들
    └── ...
```

## 🚀 빠른 시작

### 방법 1: Docker 모드 (권장)

```bash
cd playground/backend

# 1. 환경 변수 설정
cp .env.example .env

# 2. 백엔드 시작 (Docker Compose로 MongoDB/Redis 자동 시작)
./start_backend.sh
```

### 방법 2: 로컬 모드

```bash
cd playground/backend

# 1. MongoDB와 Redis 설치 및 시작
# macOS:
brew services start mongodb-community@7.0
brew services start redis

# 2. 환경 변수 설정
cp .env.example .env

# 3. 백엔드 시작 (로컬 서비스 사용)
./start_backend.sh --local
```

**자세한 설정은 [START_GUIDE.md](./START_GUIDE.md) 및 [LOCAL_SETUP.md](./LOCAL_SETUP.md) 참고**

### 4. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📦 의존성 관리

이 프로젝트는 **Poetry**를 사용합니다. 의존성은 `pyproject.toml`의 `[project.optional-dependencies]` 섹션에 정의되어 있습니다.

### 주요 의존성 (web 옵션)

- `fastapi>=0.100.0` - 웹 프레임워크
- `uvicorn>=0.23.0` - ASGI 서버
- `websockets>=11.0.0` - WebSocket 지원
- `python-multipart>=0.0.6` - 파일 업로드
- `motor>=3.3.0` - Async MongoDB 드라이버
- `pymongo>=4.0.0` - MongoDB 드라이버
- `google-api-python-client>=2.100.0` - Google Workspace API
- `streamlit>=1.29.0` - Admin Dashboard
- `plotly>=5.18.0` - 차트

### 설치

```bash
# Poetry 사용
poetry install -E web

# pip 사용
pip install -e ".[web]"
```

## 🏗️ 아키텍처

### 레이어 구조

```
FastAPI (main.py)
    ↓
Routers (API 엔드포인트)
    ↓
Services (비즈니스 로직)
    ↓
beanllm Facades (Core 라이브러리)
```

### 주요 컴포넌트

1. **Routers**: API 엔드포인트 정의 및 요청 검증
2. **Services**: 비즈니스 로직 및 상태 관리
3. **Schemas**: Pydantic 모델 (요청/응답 검증)
4. **Monitoring**: 요청/응답 로깅 및 메트릭 수집

## 🔧 설정

### 환경 변수

`.env.example`을 참고하여 다음 변수들을 설정하세요:

```bash
# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=false

# MongoDB
MONGODB_URI=mongodb://localhost:27017/beanllm

# Redis (선택적)
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka (선택적, 분산 모드)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# LLM Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
# ... 기타
```

## 📝 주요 기능

### 1. 채팅 API
- 일반 채팅 (`/api/chat`)
- 스트리밍 채팅 (`/api/chat/stream`)
- 채팅 히스토리 관리 (`/api/chat/sessions`)

### 2. RAG (Retrieval-Augmented Generation)
- 문서 로드 및 벡터화
- 유사도 검색
- RAG 파이프라인 실행
- RAG 디버깅

### 3. Knowledge Graph
- 그래프 구축
- Cypher 쿼리
- Graph RAG

### 4. Agent & Multi-Agent
- 단일 Agent 실행
- Multi-Agent 협업
- Tool 호출 및 오케스트레이션

### 5. 기타 기능
- Vision RAG (이미지 기반 검색)
- Audio 처리 (STT, TTS, Audio RAG)
- OCR (광학 문자 인식)
- Web Search (다중 검색 엔진)
- Fine-tuning
- 평가 및 벤치마크

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/

# 특정 테스트
pytest tests/test_chat_comprehensive.py

# 커버리지 포함
pytest --cov=. --cov-report=html tests/
```

## 📚 문서

- [코드 정리 분석](./docs/CLEANUP_ANALYSIS.md)
- [MCP 통합 분석](./docs/MCP_INTEGRATION_ANALYSIS.md)
- [구조 분석 및 개선 제안](./docs/STRUCTURE_ANALYSIS.md)

## 🔄 최근 변경사항 (2025-01-24)

### 구조 개선
- ✅ `routers/__init__.py` 완성 - 모든 17개 라우터 export
- ✅ `chat_history.py` → `routers/history_router.py` 이동
- ✅ `models.py` → `schemas/database.py` 이동
- ✅ `requirements.txt` 삭제 - `pyproject.toml`의 `web` 옵션으로 통합
- ✅ 루트 디렉토리 정리 (6개 → 4개 파일)

### 코드 정리
- ✅ 중복 엔드포인트 제거 (11개)
- ✅ 사용되지 않는 import 제거 (15개)
- ✅ 중복 전역 상태 통일
- ✅ `main.py` 크기 감소 (2,704줄 → 1,161줄, 57% 감소)

## ⚠️ 알려진 이슈

- `orchestrator.py`: 일부 TODO 항목 (MCP 통합 후 구현 예정)

## 📞 지원

문제가 발생하면 이슈를 생성하거나 문서를 참고하세요.
