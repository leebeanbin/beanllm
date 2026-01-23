# 🚀 beanllm Admin Dashboard

관리자용 Google Workspace 모니터링 및 분석 대시보드입니다.

## 주요 기능

### 1. 실시간 통계 (Overview)
- Google 서비스별 사용량 (Docs, Drive, Gmail)
- 상위 사용자 TOP 10
- 시간대별 사용 패턴
- 실시간 메트릭

### 2. AI 분석 (AI Analysis)
- **Gemini 기반 사용 패턴 분석**
- 이상 징후 탐지
- 최적화 권장사항
- 비용 예측

### 3. 보안 모니터링 (Security)
- 고위험 이벤트 실시간 모니터링
- Gemini 기반 위협 분석
- Rate limit 초과 감지
- 비정상 활동 알림

### 4. 비용 최적화 (Cost)
- MongoDB/Redis 무료 티어 사용량 분석
- API 호출 최적화 제안
- 예상 월간 비용 계산
- 비용 절감 방안

### 5. 설정 (Settings)
- 환경 변수 확인
- 관리자 ID 관리
- 대시보드 설정

---

## 📦 설치

### 1. 필수 패키지 설치

```bash
# Streamlit 및 관련 패키지
pip install streamlit>=1.29.0 pandas>=2.0.0

# beanllm (Google 이벤트 로깅 포함)
pip install -e .

# MongoDB 드라이버
pip install motor>=3.3.0

# Google API 클라이언트 (선택적)
pip install google-api-python-client google-auth-oauthlib
```

또는 한 번에:

```bash
cd playground/backend
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성 (또는 `.env.example` 복사):

```bash
cp playground/backend/.env.example .env
```

**필수 환경 변수**:

```bash
# MongoDB (통계 저장용)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/beanllm

# Gemini API (AI 분석용 - 이미 유료 결제한 키 사용)
GEMINI_API_KEY=your-gemini-api-key
```

**선택적 환경 변수**:

```bash
# Redis (세션 관리용)
REDIS_URL=rediss://default:password@hostname:6379

# Ollama (로컬 모델)
OLLAMA_HOST=http://localhost:11434
```

---

## 🚀 실행 방법

### 방법 1: Streamlit 직접 실행

```bash
streamlit run admin/dashboard.py
```

브라우저가 자동으로 열리며 http://localhost:8501 에서 대시보드에 접근할 수 있습니다.

### 방법 2: beanllm CLI 사용

```bash
beanllm admin dashboard
```

### 방법 3: 포트 지정

```bash
streamlit run admin/dashboard.py --server.port=8502
```

---

## 📊 CLI 명령어

대시보드 외에도 CLI로 관리 작업을 수행할 수 있습니다.

### 1. 사용 패턴 분석 (Gemini)

```bash
# 24시간 데이터 분석
beanllm admin analyze

# 7일 데이터 분석
beanllm admin analyze --hours=168
```

**출력 예시**:
```
📊 Google Export Statistics (Last 24 hours)
┏━━━━━━━━━┳━━━━━━━┓
┃ Service ┃ Count ┃
┡━━━━━━━━━╇━━━━━━━┩
│ Docs    │ 120   │
│ Drive   │ 80    │
│ Gmail   │ 34    │
└─────────┴───────┘

🤖 Gemini Analysis:
주요 사용 패턴:
- Google Docs가 가장 인기 (51% 사용)
- 오후 2-4시 사이 피크 타임
- user123이 전체의 15% 사용 (모니터링 필요)

권장 조치:
- Rate limit 조정 고려 (user123)
- 캐싱 TTL 증가로 API 호출 감소
```

### 2. 통계 조회 (Gemini 없이)

```bash
# 24시간 통계
beanllm admin stats

# 72시간 통계
beanllm admin stats --hours=72
```

### 3. 비용 최적화 제안

```bash
beanllm admin optimize
```

**출력 예시**:
```
💰 Cost Optimization Recommendations:

1. MongoDB 무료 티어 상태: 안전 (사용량 40%)
2. Redis 무료 티어 상태: 주의 (일일 8,500 commands)
3. 예상 월간 비용: $5-10

권장 조치:
- 세션 TTL을 30분 → 20분으로 단축
- 배치 처리로 Redis 호출 30% 감소 가능
- 예상 절감: $3/month
```

### 4. 보안 이벤트 확인

```bash
# 24시간 내 고위험 이벤트
beanllm admin security

# 72시간 내 고위험 이벤트
beanllm admin security --hours=72
```

---

## 🏗️ 아키텍처

### 데이터 흐름

```
사용자 → Ollama Chat → Google Workspace 공유
                             ↓
                    log_google_export()
                             ↓
                    MongoDB (events 컬렉션)
                             ↓
                    Admin Dashboard / CLI
                             ↓
                    Gemini 분석 (선택적)
```

### 구성 요소

```
beanllm/
├── infrastructure/distributed/
│   └── google_events.py          # 이벤트 로깅 (log_google_export 등)
│
├── utils/cli/
│   └── admin_commands.py         # CLI 명령어 (analyze, stats, optimize, security)
│
├── admin/
│   ├── dashboard.py              # Streamlit 대시보드 (이 파일)
│   └── README.md                 # 이 문서
│
└── playground/backend/
    └── main.py                   # FastAPI (Google Workspace API 통합)
```

---

## 🔧 커스터마이징

### 1. 분석 프롬프트 수정

`admin/dashboard.py` 또는 `utils/cli/admin_commands.py`에서 Gemini 프롬프트를 수정할 수 있습니다:

```python
prompt = f"""
다음은 지난 {hours}시간 동안의 통계입니다:
...

[여기에 원하는 분석 질문 추가]
"""
```

### 2. 대시보드 테마 변경

`.streamlit/config.toml` 생성:

```toml
[theme]
primaryColor = "#10b981"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f3f4f6"
textColor = "#1f2937"
font = "sans serif"
```

### 3. 새로운 메트릭 추가

`google_events.py`에 새로운 로깅 함수 추가:

```python
async def log_custom_event(
    user_id: str,
    event_type: str,
    metadata: Dict[str, Any]
) -> None:
    event_logger = get_event_logger()
    await event_logger.log_event(
        event_type=f"custom.{event_type}",
        data={"user_id": user_id, **metadata},
        level="info"
    )
```

---

## 🔒 보안

### 환경 변수 보호

- `.env` 파일은 **절대 Git에 커밋하지 마세요**
- `.gitignore`에 `.env` 추가 확인
- 프로덕션에서는 환경 변수를 시스템 레벨에서 설정

### MongoDB 접근 제어

```javascript
// MongoDB Atlas에서 IP 화이트리스트 설정
// Network Access → Add IP Address → Add Current IP Address
```

### Streamlit 인증 (선택적)

프로덕션 환경에서는 `.streamlit/config.toml`에 인증 추가:

```toml
[server]
enableCORS = false
enableXsrfProtection = true

[client]
showErrorDetails = false
```

또는 리버스 프록시(Nginx)로 Basic Auth 추가:

```nginx
location /admin {
    auth_basic "Admin Area";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8501;
}
```

---

## 📈 비용 예상

### 무료 티어 (월 $0)

**전제 조건**:
- MongoDB Atlas Free: 512MB
- Upstash Redis Free: 10,000 commands/day
- Gemini API: 이미 유료 결제한 키 사용 (추가 비용 없음)

**사용 가능 범위**:
- 일일 사용자: ~100명
- 일일 내보내기: ~500건
- 세션 저장: ~1,000개 (30일 보관)

### 유료 전환 시 (월 $10-20)

**MongoDB Atlas**:
- M2 Shared: $9/month (2GB, 충분함)

**Upstash Redis**:
- Pro: $5/month (100K commands/day)

**예상 사용자 규모**:
- 일일 사용자: ~1,000명
- 일일 내보내기: ~5,000건
- 세션 저장: ~10,000개 (30일 보관)

---

## 🐛 트러블슈팅

### 1. "beanllm not available" 에러

```bash
# beanllm 설치
pip install -e .

# 또는
cd /path/to/llmkit
pip install -e .
```

### 2. "MONGODB_URI not set" 경고

```bash
# .env 파일에 MongoDB URI 추가
echo "MONGODB_URI=mongodb+srv://..." >> .env

# 또는 환경 변수로 설정
export MONGODB_URI="mongodb+srv://..."
```

### 3. Gemini 분석 실패

```bash
# Gemini API 키 확인
echo $GEMINI_API_KEY

# API 키 설정
export GEMINI_API_KEY="your-key"

# API 키 유효성 확인
beanllm admin analyze --hours=1
```

### 4. Streamlit 포트 충돌

```bash
# 다른 포트로 실행
streamlit run admin/dashboard.py --server.port=8502
```

### 5. MongoDB 연결 실패

```bash
# 네트워크 확인
ping cluster.mongodb.net

# IP 화이트리스트 확인 (MongoDB Atlas)
# Network Access → Add Current IP Address
```

---

## 📚 참고 자료

### MongoDB Atlas 설정

1. [MongoDB Atlas 가입](https://www.mongodb.com/cloud/atlas/register)
2. Free tier 클러스터 생성
3. Database User 생성 (username/password)
4. Network Access에서 IP 화이트리스트 추가
5. Connect → Connect your application → URI 복사

### Upstash Redis 설정

1. [Upstash 가입](https://upstash.com/)
2. Redis 데이터베이스 생성 (Free tier)
3. Details → Redis Connect URL 복사
4. `.env`에 `REDIS_URL` 추가

### Gemini API 키 발급

1. [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API Key
3. `.env`에 `GEMINI_API_KEY` 추가

---

## 🤝 기여

이슈나 개선 사항이 있으면 GitHub Issues에 올려주세요!

---

## 📝 License

MIT License - 자유롭게 사용하세요.

---

**Built with ❤️ by beanllm team**
