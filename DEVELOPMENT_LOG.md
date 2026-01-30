# beanllm Playground Development Log

## 개발 진행 상황 상세 기록

---

## Phase 1: 환경 설정 & 인프라 (완료)

### 생성된 파일

#### 1. `docker-compose.yml`
**목적**: 개발 환경 인프라 통합
**내용**:
- MongoDB 7.0 (포트 27017) - 채팅 히스토리, API 키 저장
- Redis 7.2 (포트 6379) - 캐싱, Rate Limiting, 메트릭
- Zookeeper (포트 2181) - Kafka 코디네이션
- Kafka (포트 9092) - 이벤트 스트리밍, 멀티에이전트 통신
- 선택적 UI: Kafka UI (8080), Mongo Express (8081), Redis Commander (8082)

**사용법**:
```bash
docker-compose up -d                    # 기본 서비스 시작
docker-compose --profile ui up -d       # UI 포함 시작
docker-compose down                     # 중지
```

#### 2. `scripts/mongo-init.js`
**목적**: MongoDB 초기화 (컬렉션, 인덱스)
**생성된 컬렉션**:
- `chat_sessions` - 채팅 세션 및 메시지
- `api_keys` - 암호화된 API 키 저장
- `google_oauth_tokens` - Google OAuth 토큰
- `request_logs` - 요청 로그 (30일 TTL)
- `rag_documents` - RAG 문서 저장 (선택적)

#### 3. `playground/backend/.env.example`
**목적**: 백엔드 환경 변수 문서화
**주요 설정**:
- 서버 설정 (HOST, PORT, DEBUG)
- MongoDB/Redis/Kafka 연결
- LLM Provider API 키 (OpenAI, Anthropic, Google, etc.)
- Vector Store API 키 (Pinecone, Qdrant, Weaviate)
- Google OAuth 설정
- 기능 플래그 및 Rate Limiting

#### 4. `playground/frontend/.env.local.example`
**목적**: 프론트엔드 환경 변수
**주요 설정**:
- `NEXT_PUBLIC_API_URL` - 백엔드 API URL
- `NEXT_PUBLIC_WS_URL` - WebSocket URL
- 기능 플래그 (Agentic Mode, Google Services, etc.)
- 기본 모델 설정

#### 5. `scripts/check-env.sh`
**목적**: 환경 검증 스크립트
**기능**:
- 필수 명령어 확인 (docker, python, node, etc.)
- .env 파일 존재 확인
- 인프라 서비스 연결 테스트 (MongoDB, Redis, Kafka, Ollama)
- 환경 변수 검증 (필수/선택)
- Python/Node 의존성 확인
- `--fix` 옵션으로 .env 파일 자동 생성

---

## Phase 2: Dynamic Config System (완료)

### 생성된 파일

#### 1. `playground/backend/services/encryption_service.py`
**목적**: API 키 암호화/복호화
**기술**: Fernet 대칭 암호화 (AES-128-CBC + HMAC)
**주요 기능**:
```python
class EncryptionService:
    def encrypt(self, plaintext: str) -> str       # 암호화
    def decrypt(self, ciphertext: str) -> str      # 복호화
    def get_key_hint(self, api_key: str) -> str    # 마지막 4자리
    def mask_key(self, api_key: str) -> str        # "sk-****...7890"
```
**설정**: `ENCRYPTION_KEY` 환경변수 사용 (없으면 임시 키 생성)

#### 2. `playground/backend/services/key_validator.py`
**목적**: API 키 유효성 검증
**특징**: beanllm의 기존 provider 인프라 활용
**주요 기능**:
```python
class KeyValidator:
    async def validate(provider: str, api_key: str) -> ApiKeyValidationResult
    # 지원 Provider: openai, anthropic, google, gemini, deepseek, perplexity, ollama
    # 기타 Provider: tavily, serpapi, pinecone, qdrant, weaviate
```
**동작**:
1. beanllm provider의 `health_check()` 메서드 활용
2. 환경변수 임시 설정 → provider 생성 → 검증 → 복원
3. 검증 성공 시 사용 가능한 모델 목록 반환

#### 3. `playground/backend/services/config_service.py`
**목적**: 런타임 환경변수 관리
**주요 기능**:
```python
class ConfigService:
    async def load_keys_from_db(db) -> int         # MongoDB에서 키 로드
    def set_key(provider: str, api_key: str)       # 키 설정 및 EnvConfig 갱신
    def remove_key(provider: str)                  # 키 제거
    def get_config_status() -> Dict                # 현재 설정 상태

async def init_config_on_startup(db)               # 앱 시작 시 호출
```
**특징**: beanllm의 `EnvConfig` 클래스 자동 갱신

#### 4. `playground/backend/models.py` (수정)
**추가된 모델**:
```python
# API Key 관련
class ApiKeyBase, ApiKeyCreate, ApiKeyInDB, ApiKeyResponse
class ApiKeyListResponse, ApiKeyValidationResult
class ProviderInfo, ProviderListResponse
PROVIDER_CONFIG = {...}  # 13개 Provider 설정

# Google OAuth 관련
class GoogleOAuthToken, GoogleAuthStatus

# 모니터링 관련
class RequestLog
```

#### 5. `playground/backend/routers/config_router.py` (수정)
**추가된 엔드포인트**:
```
GET    /api/config/keys              # 모든 API 키 목록
GET    /api/config/keys/{provider}   # 특정 Provider 키 조회
POST   /api/config/keys              # 키 저장/업데이트
DELETE /api/config/keys/{provider}   # 키 삭제
POST   /api/config/keys/{provider}/validate  # 키 검증
GET    /api/config/providers/all     # 모든 Provider 상태
POST   /api/config/keys/load-all     # MongoDB에서 모든 키 로드
```

#### 6. `playground/frontend/src/components/ui/dialog.tsx`
**목적**: Radix Dialog 래퍼 컴포넌트
**사용**: API Key 모달에서 사용

#### 7. `playground/frontend/src/components/ApiKeyModal.tsx`
**목적**: API 키 관리 UI
**기능**:
- Provider별 키 입력/저장/삭제
- 키 유효성 검증 (Validate 버튼)
- 상태 표시 (Valid/Invalid/Not validated)
- Provider 문서 링크
- LLM Provider와 기타 서비스 그룹 분리

---

## Phase 3: Agentic Router (진행 중)

### 목표
사용자의 자연어 입력을 분석하여 적절한 기능(Chat, RAG, Agent, etc.)을 자동으로 선택하고 실행

### 아키텍처 설계

```
User Input
    ↓
┌─────────────────────────────────┐
│       Intent Classifier          │
│  - 키워드 분석                    │
│  - LLM 기반 의도 분류             │
│  - 필요 도구/기능 추출            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│        Tool Registry             │
│  - 사용 가능한 도구 목록          │
│  - 도구별 필요 조건 (API 키 등)   │
│  - 도구 실행 함수 매핑            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│     Agentic Orchestrator         │
│  - 도구 실행 순서 결정            │
│  - 병렬/순차 실행                │
│  - 결과 통합                     │
│  - 스트리밍 응답                 │
└─────────────────────────────────┘
    ↓
SSE Streaming Response
```

### 구현 계획

#### 1. Intent Classifier (`intent_classifier.py`)
**역할**: 사용자 입력 분석 및 의도 분류
**구현 방식**:
- 규칙 기반 분류 (키워드 매칭) - 빠른 응답
- LLM 기반 분류 (복잡한 경우) - 정확한 분류
- 하이브리드 (규칙 우선, 불확실하면 LLM)

**분류 카테고리**:
```python
class IntentType(Enum):
    CHAT = "chat"                    # 일반 대화
    RAG = "rag"                      # 문서 검색/질의
    WEB_SEARCH = "web_search"        # 웹 검색
    AGENT = "agent"                  # 도구 사용 에이전트
    MULTI_AGENT = "multi_agent"      # 멀티 에이전트
    KNOWLEDGE_GRAPH = "kg"           # 지식 그래프
    GOOGLE_DRIVE = "google_drive"    # Google Drive
    GOOGLE_DOCS = "google_docs"      # Google Docs
    GOOGLE_GMAIL = "google_gmail"    # Gmail
    GOOGLE_CALENDAR = "google_calendar"  # Calendar
    GOOGLE_SHEETS = "google_sheets"  # Sheets
    AUDIO = "audio"                  # 음성 처리
    OCR = "ocr"                      # 이미지 텍스트 추출
    CODE = "code"                    # 코드 생성/분석
    EVALUATION = "evaluation"        # 평가
```

#### 2. Tool Registry (`tool_registry.py`)
**역할**: 사용 가능한 도구 관리
**구조**:
```python
class Tool:
    name: str                    # 도구 이름
    description: str             # 설명
    intent_types: List[IntentType]  # 지원하는 의도
    required_keys: List[str]     # 필요한 API 키
    handler: Callable            # 실행 함수

class ToolRegistry:
    def register(tool: Tool)
    def get_tools_for_intent(intent: IntentType) -> List[Tool]
    def check_requirements(tool: Tool) -> Tuple[bool, List[str]]
```

#### 3. Agentic Orchestrator (`orchestrator.py`)
**역할**: 도구 실행 및 결과 통합
**기능**:
```python
class AgenticOrchestrator:
    async def process(
        query: str,
        intent: IntentResult,
        tools: List[Tool],
        stream: bool = True
    ) -> AsyncGenerator[AgenticEvent, None]
```

**이벤트 타입**:
```python
class AgenticEvent:
    type: Literal["intent", "tool_start", "tool_progress", "tool_result", "text", "error", "done"]
    data: Dict[str, Any]
```

#### 4. Agentic Router (`agentic_router.py`)
**엔드포인트**:
```
POST /api/chat/agentic
  - 자연어 입력 → 자동 라우팅 → 스트리밍 응답
  - SSE (Server-Sent Events) 형식

GET /api/agentic/tools
  - 사용 가능한 도구 목록

GET /api/agentic/status
  - 현재 설정 상태 (어떤 기능 사용 가능한지)
```

### 현재 진행 상황

- [x] 아키텍처 설계 완료
- [ ] Intent Classifier 구현 중
- [ ] Tool Registry 구현 예정
- [ ] Agentic Orchestrator 구현 예정
- [ ] Agentic Router 구현 예정

### beanllm 기존 코드 활용

**확인된 관련 코드**:
- `src/beanllm/infrastructure/routing/` - 모델 라우팅 규칙 (참고용)
- `src/beanllm/facade/` - 각 기능의 Facade 클래스
- `src/beanllm/providers/` - LLM Provider (health_check 활용)

---

## Phase 4: Google Services 통합 (예정)

### 계획
- OAuth 2.0 인증 플로우
- Google Drive, Docs, Gmail, Calendar, Sheets API 연동
- beanllm의 기존 Google 도구 활용

---

## Phase 5: Unified Chat API (예정)

### 계획
- Frontend lib 파일 구현 (beanllm-client.ts, mcp-client.ts 등)
- /api/chat 엔드포인트 완성
- Streaming 지원

---

## Phase 6: Clean Chat UI (완료)

### 완료된 작업

- [x] Chat 페이지 전면 리디자인
- [x] 모바일 반응형 UI 구현
- [x] Settings 패널 → Popover로 변경
- [x] Navigation 모바일 메뉴 추가
- [x] Feature Badge 컴포넌트 추가
- [x] 불필요한 코드 정리
- [x] Playground Backend 코드 정리 (2025-01-24) ✅

### 생성된 파일

#### UI 컴포넌트

**1. `playground/frontend/src/components/ui/popover.tsx`**
- shadcn/ui Popover 컴포넌트 (Radix UI)

**2. `playground/frontend/src/components/ui/slider.tsx`**
- shadcn/ui Slider 컴포넌트 (Radix UI)

**3. `playground/frontend/src/components/ChatSettingsPopover.tsx`**
- 컴팩트한 설정 팝오버 (Temperature, Max Tokens, Top P, Penalties)
- 탭 구조: Parameters, System Prompt
- Reset to defaults 버튼

**4. `playground/frontend/src/components/FeatureBadge.tsx`**
- Feature 모드 표시 배지
- 모드별 색상 및 아이콘

### 수정된 파일

**1. `playground/frontend/src/app/chat/page.tsx`**
- 전면 리디자인 (1383줄 → 1009줄, 27% 감소)
- 모바일 반응형 (sm:, lg: breakpoints)
- ChatSettingsPopover 통합
- FeatureBadge 추가
- More 메뉴 (⋯) 로 내보내기/가져오기/초기화 통합
- 메시지 버블 모바일 최적화

**2. `playground/frontend/src/components/Navigation.tsx`**
- 모바일 헤더 추가 (햄버거 메뉴)
- 모바일 드롭다운 메뉴
- 데스크톱 사이드바 간소화
- 불필요한 Feature 목록 제거

**3. `playground/frontend/src/components/PageLayout.tsx`**
- 모바일 헤더 높이 (pt-14) 적용
- 반응형 패딩/마진

**4. `playground/frontend/package.json`**
- `@radix-ui/react-popover` 추가
- `@radix-ui/react-slider` 추가

### 모바일 반응형 개선 사항

```
| 요소 | 모바일 | 데스크톱 |
|------|--------|----------|
| 네비게이션 | 상단 헤더 + 드롭다운 | 좌측 사이드바 |
| 설정 | Popover | Popover |
| 메시지 버블 | 85% 너비, 작은 텍스트 | 75% 너비, 일반 텍스트 |
| 입력 영역 | 작은 버튼, 한 줄 | 큰 버튼, 여러 줄 |
| 아바타 | 28px | 32px |
| 헤더 | 2줄 레이아웃 | 1줄 레이아웃 |
```

### 제거된 요소

- 기존 Settings 패널 (200줄+ 제거)
- OnboardingGuide 관련 코드
- 중복된 상태 변수들
- 사용하지 않는 Import

---

## Phase 7: Monitoring & Observability (완료)

### 생성된 파일

#### 1. `playground/backend/routers/monitoring_router.py`
**목적**: 실시간 모니터링 API 엔드포인트
**주요 엔드포인트**:
- `GET /api/monitoring/health` - 시스템 헬스 체크 (Redis, Kafka 연결 상태)
- `GET /api/monitoring/summary` - 메트릭 요약 (요청수, 에러율, 응답시간)
- `GET /api/monitoring/trend` - 요청 트렌드 (분 단위)
- `GET /api/monitoring/endpoints` - 엔드포인트별 통계
- `GET /api/monitoring/tokens` - 모델별 토큰 사용량
- `GET /api/monitoring/dashboard` - 전체 대시보드 데이터
- `POST /api/monitoring/clear` - 메트릭 초기화

**Response 모델**:
```python
class MetricsSummary:
    total_requests: int
    total_errors: int
    error_rate: float
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

class SystemHealth:
    status: str  # healthy, degraded, unhealthy
    redis_connected: bool
    kafka_connected: bool
    uptime_seconds: float
```

#### 2. `playground/frontend/src/app/monitoring/page.tsx`
**목적**: React 기반 모니터링 대시보드 UI
**특징**:
- 모바일 반응형 디자인 (sm:, lg: breakpoints)
- 자동 새로고침 (10초 간격)
- 실시간 요청 트렌드 차트
- 엔드포인트별 성능 테이블
- 모델별 토큰 사용량
- 응답 시간 분포 (min, p50, p95, p99, max)

**주요 컴포넌트**:
- `StatCard` - 핵심 메트릭 카드
- `HealthIndicator` - 시스템 상태 표시
- `RequestTrendChart` - 요청 트렌드 바 차트
- `EndpointTable` - 엔드포인트 통계 테이블
- `TokenUsageTable` - 토큰 사용량 테이블

### 수정된 파일

#### 1. `playground/backend/main.py`
- monitoring_router 등록

#### 2. `playground/frontend/src/components/Navigation.tsx`
- Monitoring 페이지 네비게이션 추가 (Activity 아이콘)

### 기존 모니터링 인프라 활용

#### `monitoring/middleware.py` (기존)
- `MonitoringMiddleware` - HTTP 요청/응답 로깅, 메트릭 수집
- `ChatMonitoringMixin` - LLM 호출 상세 로깅
- Redis에 저장되는 메트릭:
  - `metrics:response_time` - 응답 시간 (Sorted Set)
  - `metrics:requests:{minute}` - 분당 요청 수
  - `metrics:errors:{minute}` - 분당 에러 수
  - `metrics:endpoint:{method}:{path}` - 엔드포인트별 통계
  - `metrics:tokens:{model}` - 모델별 토큰 사용량

---

## 파일 변경 이력

| 날짜 | 파일 | 변경 유형 | 설명 |
|------|------|----------|------|
| 2026-01-23 | docker-compose.yml | 생성 | 인프라 설정 |
| 2026-01-23 | scripts/mongo-init.js | 생성 | MongoDB 초기화 |
| 2026-01-23 | scripts/check-env.sh | 생성 | 환경 검증 |
| 2026-01-23 | playground/backend/.env.example | 생성 | 환경변수 문서화 |
| 2026-01-23 | playground/frontend/.env.local.example | 생성 | 환경변수 문서화 |
| 2026-01-23 | playground/backend/services/encryption_service.py | 생성 | 암호화 서비스 |
| 2026-01-23 | playground/backend/services/key_validator.py | 생성 | 키 검증 서비스 |
| 2026-01-23 | playground/backend/services/config_service.py | 생성 | 설정 서비스 |
| 2026-01-23 | playground/backend/services/__init__.py | 생성 | 서비스 모듈 |
| 2026-01-23 | playground/backend/models.py | 수정 | API Key 모델 추가 |
| 2026-01-23 | playground/backend/routers/config_router.py | 수정 | 키 관리 엔드포인트 |
| 2026-01-23 | playground/frontend/src/components/ui/dialog.tsx | 생성 | Dialog 컴포넌트 |
| 2026-01-23 | playground/frontend/src/components/ApiKeyModal.tsx | 생성 | API Key 모달 |

---

## 코드 리뷰 & 수정 (2026-01-24)

### 리뷰 결과

이전 세션에서 작업한 코드들을 검토한 결과, 대부분 beanllm 패턴을 잘 따르고 있음:

**✅ 잘 구현된 부분:**
- `message_vector_store.py`: beanllm의 ChromaVectorStore, OllamaEmbedding, HuggingFaceEmbedding 정상 사용
- `session_cache.py`: beanllm의 get_redis_client 정상 사용
- `chat_history.py`: Redis 캐싱, Vector DB 메시지 저장 통합 잘 됨
- `mcp_streaming.py`: Clean Architecture 준수 (Facade만 사용)
- `rag_service_impl.py`: Rate limiting 데코레이터 수정 정상

**🔧 수정된 부분:**

1. **`session_search_service.py`**:
   - 하드코딩된 임베딩 모델 → 환경변수 사용으로 변경
   - `message_vector_store.py`와 동일한 패턴 적용 (Ollama → HuggingFace fallback)

2. **`datetime.utcnow()` Deprecation 수정**:
   Python 3.12+에서 deprecated된 `datetime.utcnow()` → `datetime.now(timezone.utc)` 로 일괄 변경:
   - `models.py`: `utc_now()` 헬퍼 함수 추가
   - `chat_history.py`
   - `mcp_streaming.py`
   - `config_router.py`
   - `session_search_service.py`

### 수정된 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `session_search_service.py` | 환경변수 기반 임베딩 모델 선택, datetime 수정 |
| `models.py` | `utc_now()` 헬퍼 함수 추가, default_factory 수정 |
| `chat_history.py` | `datetime.now(timezone.utc)` 사용 |
| `mcp_streaming.py` | `datetime.now(timezone.utc)` 사용 |
| `config_router.py` | `datetime.now(timezone.utc)` 사용 |

---

## Phase 3: Agentic Router (완료)

### 완료된 작업

- [x] Intent Classifier 구현 (`services/intent_classifier.py`)
- [x] Tool Registry 구현 (`services/tool_registry.py`)
- [x] Agentic Orchestrator 구현 (`services/orchestrator.py`)
- [x] Agentic Router 엔드포인트 구현 (`routers/chat_router.py`)

### 생성된 파일

#### 1. `playground/backend/services/tool_registry.py`
**목적**: beanllm의 기능들을 도구로 래핑하고 관리

**주요 클래스**:
```python
class Tool:
    name: str                     # 도구 이름
    description: str              # 설명
    description_ko: str           # 한국어 설명
    intent_types: List[IntentType]  # 지원하는 의도
    requirements: ToolRequirement   # 필요 API 키, 패키지, 서비스
    facade_class: Optional[str]     # beanllm Facade 경로

class ToolRegistry:
    get_tool(name) -> Tool
    get_tools_for_intent(intent_type) -> List[Tool]
    check_requirements(tool) -> ToolCheckResult
    get_best_tool_for_intent(intent_type) -> ToolCheckResult
```

**등록된 도구** (14개):
- `chat`: 기본 LLM 대화
- `rag`: RAG 기반 문서 Q&A
- `agent`: 도구 사용 에이전트
- `multi_agent`: 멀티 에이전트 토론/협업
- `web_search`: 웹 검색 (Tavily/SerpAPI)
- `knowledge_graph`: 지식 그래프 (Neo4j)
- `google_drive`, `google_docs`, `google_gmail`, `google_calendar`, `google_sheets`: Google 서비스
- `audio_transcribe`: 음성 전사 (Whisper)
- `vision`: 이미지 분석
- `ocr`: OCR 텍스트 추출
- `code`: 코드 생성/분석
- `evaluation`: 모델/RAG 평가

#### 2. `playground/backend/services/orchestrator.py`
**목적**: Intent와 도구를 받아 실행하고 SSE 이벤트 스트리밍

**이벤트 타입**:
- `intent`: 의도 분류 결과
- `tool_select`: 도구 선택
- `tool_start`: 도구 실행 시작
- `tool_progress`: 진행 상황
- `tool_result`: 실행 결과
- `text`: 텍스트 청크 (스트리밍)
- `text_done`: 텍스트 완료
- `error`: 오류
- `done`: 전체 완료

**핸들러 구현 상태**:
- ✅ `chat`: 완전 구현 (스트리밍)
- ✅ `rag`: 완전 구현
- 🚧 기타: 스켈레톤 (TODO)

#### 3. `playground/backend/routers/chat_router.py` (업데이트)
**추가된 엔드포인트**:
```
POST /api/chat          # 기본 채팅 (비스트리밍)
POST /api/chat/stream   # 기본 채팅 (스트리밍)
POST /api/chat/agentic  # Agentic 채팅 (자동 라우팅, SSE)
POST /api/chat/classify # Intent 분류만
GET  /api/chat/tools    # 도구 목록 및 상태
GET  /api/chat/tools/{name}  # 특정 도구 상태
GET  /api/chat/intents  # 지원 Intent 목록
```

### 아키텍처 플로우

```
User Input ("문서에서 AI 찾아줘")
    ↓
┌─────────────────────────────────────┐
│     /api/chat/agentic               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Intent Classifier                │
│  → primary_intent: RAG              │
│  → confidence: 0.85                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Tool Registry                    │
│  → get_best_tool_for_intent(RAG)    │
│  → check_requirements(rag_tool)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Agentic Orchestrator             │
│  → execute(context)                  │
│  → yield intent event               │
│  → yield tool_select event          │
│  → yield tool_progress events       │
│  → yield text chunks                │
│  → yield tool_result event          │
│  → yield done event                 │
└─────────────────────────────────────┘
    ↓
SSE Stream Response
```

---

## Phase 4: Google Services 통합 (완료)

### 완료된 작업

- [x] Google OAuth 2.0 인증 서비스 구현 (`services/google_oauth_service.py`)
- [x] Google OAuth 라우터 구현 (`routers/google_auth_router.py`)
- [x] Orchestrator Google 서비스 핸들러 구현
- [x] MCP Server Google 도구 연동

### 생성된 파일

#### 1. `playground/backend/services/google_oauth_service.py`
**목적**: Google OAuth 2.0 인증 플로우 관리

**주요 기능**:
```python
class GoogleOAuthService:
    def get_authorization_url(services, user_id) -> Dict
    async def handle_callback(code, state, db) -> Dict
    async def get_valid_access_token(user_id, db) -> Optional[str]
    async def _refresh_token(user_id, refresh_token_encrypted, db)
    async def get_auth_status(user_id, db) -> Dict
    async def revoke_token(user_id, db) -> bool
```

**지원 서비스 스코프**:
- `drive`: Google Drive 파일 관리
- `docs`: Google Docs 문서 관리
- `gmail`: Gmail 이메일 전송/읽기
- `calendar`: Google Calendar 일정 관리
- `sheets`: Google Sheets 스프레드시트 관리

**보안 기능**:
- Fernet 암호화로 토큰 저장 (encryption_service 활용)
- MongoDB에 암호화된 토큰 저장
- 만료 10분 전 자동 갱신

#### 2. `playground/backend/routers/google_auth_router.py`
**목적**: Google OAuth 엔드포인트 제공

**엔드포인트**:
```
GET  /api/auth/google/services    # 사용 가능한 Google 서비스 목록
POST /api/auth/google/start       # OAuth 인증 시작 (Auth URL 생성)
GET  /api/auth/google/callback    # OAuth 콜백 처리
GET  /api/auth/google/status      # 인증 상태 확인
POST /api/auth/google/logout      # 로그아웃 (토큰 취소)
GET  /api/auth/google/token       # 액세스 토큰 확인 (내부용)
```

**사용 플로우**:
```
1. Frontend → POST /start (services: ["drive", "docs"])
2. Frontend → 사용자를 auth_url로 리다이렉트
3. Google → GET /callback (code, state)
4. Backend → 토큰 교환, 암호화 저장
5. Backend → Frontend로 리다이렉트 (success)
```

#### 3. `playground/backend/services/orchestrator.py` (업데이트)
**업데이트된 핸들러**:

- ✅ `_handle_google_drive`: 파일 목록 조회, 파일 저장
- ✅ `_handle_google_docs`: Google Docs 문서 생성/내보내기
- ✅ `_handle_google_gmail`: Gmail 이메일 전송
- 🚧 `_handle_google_calendar`: 스켈레톤 (인증만 확인)
- 🚧 `_handle_google_sheets`: 스켈레톤 (인증만 확인)

**MCP Server 도구 연동**:
- `mcp_server/tools/google_tools.py`의 기존 함수 활용
- `export_to_google_docs()` - Docs 내보내기
- `save_to_google_drive()` - Drive 저장
- `share_via_gmail()` - Gmail 공유
- `list_google_drive_files()` - Drive 파일 목록

### Google OAuth 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (Settings)                        │
│  [Google 로그인] → /api/auth/google/start → auth_url 리다이렉트  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Google OAuth Server                        │
│  사용자 로그인 → 스코프 동의 → callback으로 리다이렉트            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (Callback)                         │
│  code 수신 → 토큰 교환 → 암호화 → MongoDB 저장 → Frontend 리다이렉트│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator Handlers                      │
│  사용자 요청 → 토큰 조회 → Google API 호출 → 결과 스트리밍        │
└─────────────────────────────────────────────────────────────────┘
```

### 환경 변수 설정

```env
# Google OAuth 2.0
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
GOOGLE_OAUTH_REDIRECT_URI=http://localhost:8000/api/auth/google/callback
```

---

## Phase 5: Frontend Integration (완료)

### 완료된 작업

- [x] Settings 페이지 Google OAuth UI 구현
- [x] Agentic Chat UI 구현
- [x] Google Services Export UI 구현
- [x] 누락된 lib 파일들 생성
- [x] UI 컴포넌트 추가

### 생성된 파일

#### Lib Files (필수 유틸리티)

**1. `playground/frontend/src/lib/utils.ts`**
- `cn()` 함수: Tailwind CSS 클래스 병합 (clsx + tailwind-merge)

**2. `playground/frontend/src/lib/api-client.ts`**
- Backend API 클라이언트
- Google OAuth 관련: `getGoogleServices()`, `startGoogleAuth()`, `getGoogleAuthStatus()`, `logoutGoogle()`
- Agentic Chat: `streamAgenticChat()`, `classifyIntent()`, `getTools()`

**3. `playground/frontend/src/lib/beanllm-client.ts`**
- BeanLLM 채팅 클라이언트
- `createBeanLLMClient()`: chat/stream 메서드 제공
- SSE 스트리밍 지원

**4. `playground/frontend/src/lib/error-messages.ts`**
- 사용자 친화적 에러 메시지 포맷팅
- 네트워크, 인증, Rate Limit, 모델, Context, 타임아웃, Google OAuth 에러 처리

**5. `playground/frontend/src/lib/mcp-client.ts`**
- MCP/SSE 스트리밍 클라이언트
- Agentic Chat용 이벤트 스트리밍
- Tool Call 진행 상황 처리

#### UI Components

**6. `playground/frontend/src/components/ui/checkbox.tsx`**
- shadcn/ui Checkbox 컴포넌트 (Radix UI)

**7. `playground/frontend/src/components/ui/dropdown-menu.tsx`**
- shadcn/ui Dropdown Menu 컴포넌트 (Radix UI)

**8. `playground/frontend/src/components/GoogleOAuthCard.tsx`**
- Google OAuth 연결 UI
- 서비스별 체크박스 (Drive, Docs, Gmail, Calendar, Sheets)
- 로그인/로그아웃 버튼
- 연결 상태 표시

**9. `playground/frontend/src/components/AgenticIntentDisplay.tsx`**
- Intent 분류 결과 표시
- Primary Intent 아이콘/라벨
- Confidence 퍼센트 표시
- Secondary Intents, 추출된 엔티티, 추론 과정 표시

**10. `playground/frontend/src/components/GoogleExportMenu.tsx`**
- Google 서비스 내보내기 드롭다운 메뉴
- Google Docs로 내보내기 (제목 입력)
- Google Drive에 저장 (파일명 입력)
- Gmail로 공유 (수신자, 제목, 메시지 입력)
- SSE 응답 파싱하여 결과 표시

#### Page Components

**11. `playground/frontend/src/app/settings/page.tsx`**
- Settings 페이지 (탭 기반)
- API Keys 탭: API 키 관리
- Google 탭: Google OAuth 연결
- About 탭: 시스템 정보

#### 수정된 파일

**12. `playground/frontend/src/components/Navigation.tsx`**
- Settings 링크 추가

**13. `playground/frontend/src/components/ToolCallDisplay.tsx`**
- Google 서비스 결과 렌더러 추가
- Drive, Docs, Gmail 결과 포맷팅
- Tool 이름 한국어 매핑 확장

**14. `playground/frontend/src/app/chat/page.tsx`**
- GoogleExportMenu 통합
- 채팅 내보내기 버튼 옆에 Google 내보내기 메뉴 추가

**15. `playground/frontend/package.json`**
- `@radix-ui/react-checkbox` 의존성 추가
- `@radix-ui/react-dropdown-menu` 의존성 추가

### Frontend 아키텍처

```
src/
├── app/
│   ├── chat/page.tsx          # 메인 채팅 페이지 (Agentic 지원)
│   └── settings/page.tsx      # 설정 페이지 (API 키, Google OAuth)
├── components/
│   ├── ui/                    # shadcn/ui 컴포넌트
│   │   ├── checkbox.tsx
│   │   ├── dialog.tsx
│   │   └── dropdown-menu.tsx
│   ├── GoogleOAuthCard.tsx    # Google 연결 UI
│   ├── GoogleExportMenu.tsx   # Google 내보내기 메뉴
│   ├── AgenticIntentDisplay.tsx  # Intent 분류 표시
│   ├── ToolCallDisplay.tsx    # 도구 실행 진행 상황
│   └── ApiKeyModal.tsx        # API 키 관리 모달
└── lib/
    ├── utils.ts               # 유틸리티 (cn)
    ├── api-client.ts          # Backend API 클라이언트
    ├── beanllm-client.ts      # BeanLLM 채팅 클라이언트
    ├── mcp-client.ts          # MCP/SSE 스트리밍
    └── error-messages.ts      # 에러 메시지 포맷팅
```

### Google Export 플로우

```
사용자 → GoogleExportMenu 클릭
    ↓
┌─────────────────────────────────┐
│   Google 인증 상태 확인         │
│   (getGoogleAuthStatus)         │
└─────────────────────────────────┘
    ↓ (인증됨)
┌─────────────────────────────────┐
│   내보내기 다이얼로그 표시       │
│   (Docs: 제목, Drive: 파일명,   │
│    Gmail: 수신자/제목/메시지)   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   /api/chat/agentic 호출        │
│   (force_intent 사용)           │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   SSE 응답 파싱                  │
│   (tool_result 이벤트)          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   결과 표시                      │
│   (성공: 링크, 실패: 에러)       │
└─────────────────────────────────┘
```

---

## 다음 작업 (Phase 6: Clean Chat UI)

1. **단일 Chat 화면 통합**
   - 모든 기능을 하나의 Chat 화면에서 접근
   - FeatureSelector로 모드 전환

2. **사이드바/모달 정리**

---

## Phase 7: Playground Backend 코드 정리 (2025-01-24) ✅

### 목적
MCP 통합을 위한 코드베이스 정리 및 중복 코드 제거

### 완료된 작업

#### 1. 중복 엔드포인트 제거
- `main.py`에서 11개 중복 엔드포인트 제거
  - RAG Debug, Optimizer, Multi-Agent, Orchestrator, Chain, VisionRAG, Audio, Evaluation, Fine-tuning, OCR, Google Workspace
  - 모든 기능은 이미 각각의 router에 구현되어 있음

#### 2. 중복 전역 상태 통일
- `common.py`와 `main.py`의 중복 전역 변수 통일
  - `_rag_debugger`, `_downloaded_models` 등
  - `common.py`로 통일하여 단일 진실의 원천 확보

#### 3. 빈 파일 정리
- `ml_router.py` 삭제 (빈 파일, 다른 routers에 구현됨)
- `notebooks/` 디렉토리 정리 (`.gitignore`에 추가)

#### 4. 사용되지 않는 import 제거
- `main.py`에서 15개 사용되지 않는 beanllm facade import 제거
  - KnowledgeGraph, RAGChain, RAGBuilder, Agent, ChainBuilder, PromptChain, WebSearch, SearchEngine, RAGDebug, Optimizer, MultiAgentCoordinator, Orchestrator, VisionRAG, MultimodalRAG, WhisperSTT, TextToSpeech, AudioRAG, EvaluatorFacade, FineTuningManagerFacade, beanOCR, OCRConfig
- `Chain`만 유지 (실제 사용: `_chains: Dict[str, Chain]`)

#### 5. 레거시 코드 표시
- `mcp_streaming.py`에 레거시 경고 주석 추가
  - "⚠️ 레거시 코드: MCP 통합 후 제거 예정"

#### 6. 불필요한 주석 제거
- 구분선 주석(`# =====`) 제거
- "Moved to routers/..." 같은 단순 설명 주석 제거
- 구체적인 로직 설명 주석은 유지

### 결과

**코드 감소:**
- `main.py`: **2,704줄 → 1,161줄** (57% 감소, 1,543줄 감소)
- `ml_router.py`: 삭제

**문서:**
- `CLEANUP_ANALYSIS.md`: 체크리스트 및 진행 상황 업데이트
- `.gitignore`: `playground/backend/notebooks/` 추가

### 남은 작업 (MCP 통합 후)

- `/api/chat/stream` 엔드포인트를 `chat_router.py`로 이동
- `orchestrator.py`에서 `_rag_instances` import 제거
- MCP Client Service 생성
- `orchestrator.py`의 TODO 항목들 구현

---

## Phase 8: Playground Backend 구조 개선 (2025-01-24) ✅

### 목적
디렉토리 구조 정리 및 파일 분류 개선

### 완료된 작업

#### 1. 디렉토리 구조 정리
- `scripts/` 디렉토리 생성 및 스크립트 파일 이동
  - `setup_and_build.sh` → `scripts/setup_and_build.sh`
  - `auto_setup_and_test.sh` → `scripts/auto_setup_and_test.sh`
  - `quick_test.sh` → `scripts/quick_test.sh`
  - 모든 스크립트의 경로 수정 (`cd "$SCRIPT_DIR/.."`)

- `docs/` 디렉토리 생성 및 문서 파일 이동
  - `CLEANUP_ANALYSIS.md` → `docs/CLEANUP_ANALYSIS.md`
  - `MCP_INTEGRATION_ANALYSIS.md` → `docs/MCP_INTEGRATION_ANALYSIS.md`
  - `STRUCTURE_ANALYSIS.md` 생성 (구조 분석 문서)

#### 2. routers/__init__.py 완성
- 모든 17개 라우터 export 추가
- 누락된 라우터: audio, chain, evaluation, finetuning, google_auth, monitoring, ocr, optimizer, vision, web

#### 3. 파일 이동 및 정리
- `chat_history.py` → `routers/history_router.py` 이동
  - 모든 import 경로 수정 (main.py, scripts/*.sh)
  
- `models.py` → `schemas/database.py` 이동
  - 모든 import 경로 수정:
    - `routers/config_router.py`
    - `routers/history_router.py`
    - `services/config_service.py`
    - `services/key_validator.py`
    - `scripts/auto_setup_and_test.sh`
    - `scripts/setup_and_build.sh`
  - `schemas/__init__.py`에 database 모델 export 추가

#### 4. 의존성 관리 정리
- `requirements.txt` 삭제 (Poetry 사용)
- `pyproject.toml`의 `web` 옵션에 의존성 통합:
  - `python-multipart>=0.0.6`
  - `motor>=3.3.0`, `pymongo>=4.0.0`
  - `google-api-python-client>=2.100.0`
  - `google-auth-oauthlib>=1.1.0`
  - `google-auth-httplib2>=0.1.1`
  - `streamlit>=1.29.0`
  - `plotly>=5.18.0`

#### 5. 문서화
- `playground/backend/README.md` 생성
  - 디렉토리 구조 설명
  - 빠른 시작 가이드
  - 의존성 관리 (Poetry)
  - 아키텍처 설명
  - 주요 기능 목록
  - 최근 변경사항 기록

- 루트 `README.md` 업데이트
  - Documentation 섹션에 Playground Backend 링크 추가

### 결과

**구조 개선:**
- 루트 파일: 6개 → 4개 (33% 감소)
- `routers/`: 17개 → 18개 라우터 (history_router 추가)
- `schemas/`: database 모델 추가

**파일 이동:**
- `chat_history.py` → `routers/history_router.py`
- `models.py` → `schemas/database.py`
- 스크립트 3개 → `scripts/` 디렉토리
- 문서 2개 → `docs/` 디렉토리

**의존성:**
- `requirements.txt` 삭제
- `pyproject.toml` 업데이트 (web 옵션)

**문서:**
- `playground/backend/README.md` 생성
- 루트 `README.md` 업데이트

### 최종 디렉토리 구조

```
playground/backend/
├── main.py                    # FastAPI 애플리케이션
├── common.py                  # 공통 유틸리티
├── database.py                # MongoDB 연결
├── mcp_streaming.py           # 레거시 (향후 제거)
├── .env.example
├── README.md                  # ✨ 새로 생성
│
├── routers/                   # 18개 라우터
│   ├── __init__.py            # ✅ 모든 라우터 export
│   ├── history_router.py      # ✨ 이동됨
│   └── ...
│
├── schemas/                   # Pydantic 스키마
│   ├── __init__.py            # ✅ database 모델 export
│   ├── database.py            # ✨ 이동됨
│   └── ...
│
├── services/                  # 비즈니스 로직
├── scripts/                   # ✨ 스크립트 정리
├── docs/                      # ✨ 문서 정리
└── tests/                     # 테스트
```

---

## Phase 6: UI 개선 및 리디자인 (2025-01-24) ✅

### 완료된 작업

#### 1. 리디자인
- ✅ Input Area: Mode dropdown 제거 → 배지로 변경
- ✅ Empty State: Gemini 스타일 미니멀 디자인 적용
- ✅ Message Bubbles: Usage info 카드 스타일, 코드 블록 스타일 개선
- ✅ InfoPanel: Settings 탭 통합, Monitor 탭 메트릭 추가
- ✅ 불필요한 컴포넌트 제거 (ChatSettingsPopover, GoogleExportMenu 등)

#### 2. UI 개선
- ✅ Tooltip 강화: 모든 주요 버튼에 Tooltip 추가 (7개)
- ✅ SVG Icon 재배치: 간격 최적화 (`gap-1` → `gap-1.5`), 크기 통일
- ✅ 모델 진행 상황 시각화 강화:
  - ThinkMode: "Model Thinking Process" + 설명 추가
  - ToolCallDisplay: 진행률 퍼센트 표시, Current Step 카드 스타일
  - Loading Indicator: 진행률 바 추가 (애니메이션)
- ✅ 그래프 노드 시각화 통합: PipelineVisualization 컴포넌트 통합 (n8n-like)
- ✅ 데이터 동기화 UI: InfoPanel에 Data Sync Status 추가

### 변경된 파일
- `playground/frontend/src/app/chat/page.tsx`: Tooltip 추가, Pipeline 시각화 통합, 리디자인
- `playground/frontend/src/components/ToolCallDisplay.tsx`: 진행률 표시 강화, 영어화
- `playground/frontend/src/components/ThinkMode.tsx`: 설명 강화
- `playground/frontend/src/components/InfoPanel.tsx`: 데이터 동기화 UI 추가, Settings 통합

### 생성된 문서
- `playground/frontend/CHANGELOG_UI.md`: UI 변경 로그
- `playground/frontend/UI_WORK_SUMMARY.md`: UI 작업 완료 요약

### 정리된 파일
- ❌ `REDESIGN_STEP_BY_STEP.md` (삭제)
- ❌ `REDESIGN_ANALYSIS.md` (삭제)
- ❌ `REDESIGN_PLAN_2025.md` (삭제)
- ❌ `ENHANCEMENT_PLAN.md` (삭제)
- ❌ `IMPROVEMENT_CHECKLIST.md` (삭제)

**상태**: 모든 UI 개선 및 리디자인 작업 완료 ✅

---

## Phase 7: Frontend 파일 정리 (2025-01-24) ✅

### 삭제된 파일 (16개)

#### Components (9개)
- ❌ `ChatSettingsPopover.tsx` - InfoPanel에 통합됨
- ❌ `DocumentPreviewSidebar.tsx` - 사용되지 않음
- ❌ `DocumentPropertiesSidebar.tsx` - 사용되지 않음
- ❌ `GoogleExportMenu.tsx` - 사용되지 않음
- ❌ `OnboardingGuide.tsx` - 사용되지 않음
- ❌ `AgenticIntentDisplay.tsx` - 사용되지 않음
- ❌ `ModelSettingsPanel.tsx` - 사용되지 않음
- ❌ `SessionList.tsx` - 사용되지 않음
- ❌ `ParameterTooltip.tsx` - ModelSettingsPanel에서만 사용

#### Hooks (3개)
- ❌ `use-file-upload.tsx` - 사용되지 않음
- ❌ `useMediaQuery.tsx` - 사용되지 않음
- ❌ `useSessionManager.ts` - SessionList에서만 사용

#### Providers (1개)
- ❌ `Thread.tsx` - 사용되지 않음

#### Icons (3개)
- ❌ `ChatIcon.tsx` - 사용되지 않음
- ❌ `github.tsx` - 사용되지 않음
- ❌ `langgraph.tsx` - 사용되지 않음

### 생성된 문서
- `playground/frontend/CLEANUP_ANALYSIS.md`: 정리 분석
- `playground/frontend/CLEANUP_COMPLETE.md`: 정리 완료 요약

### 결과
- **삭제된 파일**: 16개
- **코드베이스 크기 감소**: 약 15-20% 감소
- **유지된 파일**: 필수 컴포넌트만 유지

**상태**: Frontend 파일 정리 완료 ✅

### 복구된 파일 (2025-01-24)
- ✅ `use-file-upload.tsx` - Phase 2 파일 업로드 UI 구현 시 필요
- ✅ `SessionList.tsx` - Phase 2 세션별 RAG 관리 시 필요
- ✅ `useSessionManager.ts` - SessionList와 함께 사용

**이유**: `CHAT_IMPROVEMENT_PLANS/` 문서에서 Phase 2 (높음 우선순위) 구현 계획에 포함됨

**의존성 확인**:
- ✅ `multimodal-utils.ts` - 이미 존재 (의존성 정상)
- ✅ 모든 import 경로 정상
- ✅ TypeScript 에러 없음

### 생성된 문서
- `playground/frontend/CLEANUP_ANALYSIS.md`: 정리 분석
- `playground/frontend/CLEANUP_COMPLETE.md`: 정리 완료 요약
- `playground/frontend/CLEANUP_FINAL.md`: 최종 보고서

### 정리 통계
- **삭제된 파일**: 16개
- **삭제된 코드 크기**: 약 112KB
- **코드베이스 감소**: 약 15-20%

---

## 커밋 그룹 기록 (2026-01-30, 완료)

그룹별 커밋 1–7 모두 완료됨.

### 그룹 요약

| 순서 | 그룹 | 커밋 타입 | 요약 |
|------|------|-----------|------|
| 1 | docs | docs | CHAT_IMPROVEMENT_PLANS, DEVELOPMENT_LOG, .claude 삭제 |
| 2 | chore | chore | docker-compose, scripts (check-env, mongo-init), .gitignore, Makefile |
| 3+4 | playground/backend | refactor(playground) | schemas, scripts, docs, monitoring, 라우터/서비스 |
| 5 | playground/frontend | refactor(playground) | Clean Chat UI, Settings/Monitoring, 새 컴포넌트 |
| 6 | beanllm core + MCP | fix(beanllm) | RAG handler, Neo4j, Ollama, mcp_server, pyproject/poetry |
| 7 | README | docs | 루트 README 최신화 |

### 참고

- **CHAT_IMPROVEMENT_PLANS/00_INDEX.md**: Phase 10 (MCP), 10.5 (코드 정리), 10.6 (스키마 분리), Phase 0 완료
- **DEVELOPMENT_LOG.md**: Phase 1–8 (인프라, Dynamic Config, Agentic, Google OAuth, Frontend, Backend 정리/구조, UI/파일 정리)

---
