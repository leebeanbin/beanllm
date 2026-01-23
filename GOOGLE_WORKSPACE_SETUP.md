# 🌐 Google Workspace Integration Guide

beanllm에 Google Workspace 통합 기능을 추가하여 사용자가 Ollama 채팅을 Google 서비스(Docs, Drive, Gmail)로 쉽게 공유하고, 관리자가 Gemini로 모니터링할 수 있습니다.

---

## 📋 목차

1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [설치 및 설정](#설치-및-설정)
4. [사용자 기능](#사용자-기능)
5. [관리자 기능](#관리자-기능)
6. [비용 분석](#비용-분석)
7. [보안 고려사항](#보안-고려사항)
8. [FAQ](#faq)

---

## 개요

### 사용자 레이어

- **Ollama 채팅** → Google Workspace 공유
- 채팅 내역을 Google Docs로 내보내기
- 채팅 내역을 Google Drive에 저장
- 채팅 내역을 Gmail로 공유

### 관리자 레이어

- **Gemini 기반 모니터링** (유료 결제한 API 키 사용, 추가 비용 없음)
- 실시간 사용 패턴 분석
- 비용 최적화 제안
- 보안 이벤트 감지
- Streamlit 대시보드

### 데이터 저장

- **MongoDB Atlas**: 세션 장기 저장, 이벤트 로깅 (Free: 512MB)
- **Upstash Redis**: 실시간 세션 캐시 (Free: 10K commands/day)

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 레이어                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frontend (Next.js)                                             │
│       ↓                                                         │
│  Backend (FastAPI) - main.py                                    │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   Ollama    │ → │  Google      │ → │   Event      │       │
│  │   Chat      │    │  Workspace   │    │   Logging    │       │
│  │   (Local)   │    │  API         │    │             │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│                           ↓                     ↓              │
│                     ┌──────────┐         ┌──────────┐         │
│                     │  Google  │         │ MongoDB  │         │
│                     │  Docs/   │         │  Events  │         │
│                     │  Drive/  │         │          │         │
│                     │  Gmail   │         └──────────┘         │
│                     └──────────┘                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        관리자 레이어                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────┐         ┌────────────────┐                │
│  │   CLI          │         │  Streamlit     │                │
│  │   Commands     │         │  Dashboard     │                │
│  └────────────────┘         └────────────────┘                │
│         ↓                           ↓                          │
│  ┌──────────────────────────────────────────┐                 │
│  │         MongoDB Events 조회               │                 │
│  │   (get_google_export_stats,              │                 │
│  │    get_security_events)                  │                 │
│  └──────────────────────────────────────────┘                 │
│         ↓                                                      │
│  ┌──────────────────────────────────────────┐                 │
│  │         Gemini 분석 (선택적)              │                 │
│  │   - 사용 패턴 분석                         │                 │
│  │   - 비용 최적화                            │                 │
│  │   - 보안 위협 탐지                         │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 설치 및 설정

### 1. 필수 패키지 설치

```bash
# 프로젝트 루트에서
cd playground/backend

# 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install google-api-python-client google-auth-oauthlib
pip install motor  # MongoDB async driver
pip install streamlit pandas  # Admin dashboard
```

### 2. MongoDB Atlas 설정 (무료)

1. **가입 및 클러스터 생성**
   - https://www.mongodb.com/cloud/atlas/register
   - "Create a Free Cluster" 선택
   - 지역: 가장 가까운 AWS/GCP 리전 선택

2. **Database User 생성**
   - Security → Database Access → Add New Database User
   - Username: `beanllm_admin`
   - Password: 강력한 비밀번호 생성

3. **네트워크 접근 허용**
   - Security → Network Access → Add IP Address
   - "Allow Access from Anywhere" 또는 특정 IP 추가

4. **연결 URI 복사**
   - Database → Connect → Connect your application
   - Driver: Python, Version: 3.12 or later
   - URI 복사: `mongodb+srv://beanllm_admin:<password>@cluster0.xxxxx.mongodb.net/beanllm`

### 3. Upstash Redis 설정 (무료)

1. **가입 및 데이터베이스 생성**
   - https://upstash.com/
   - "Create Database" → Region: 가장 가까운 지역 선택

2. **연결 URL 복사**
   - Details → Redis Connect URL
   - 형식: `rediss://default:<password>@<hostname>:6379`

### 4. 환경 변수 설정

`.env` 파일 생성 (프로젝트 루트 또는 `playground/backend/`):

```bash
# ============================================================================
# MongoDB (세션 저장 + 이벤트 로깅)
# ============================================================================
MONGODB_URI=mongodb+srv://beanllm_admin:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/beanllm

# ============================================================================
# Redis (실시간 세션 캐시)
# ============================================================================
REDIS_URL=rediss://default:YOUR_PASSWORD@hostname:6379

# ============================================================================
# Gemini API (관리자 모니터링용 - 이미 유료 결제한 키)
# ============================================================================
GEMINI_API_KEY=your-gemini-api-key

# ============================================================================
# Ollama (사용자용 - 무료 오픈소스 모델)
# ============================================================================
OLLAMA_HOST=http://localhost:11434

# ============================================================================
# Google Workspace (선택적)
# ============================================================================
# 프론트엔드에서 OAuth 2.0 처리
# GOOGLE_CLIENT_ID=...
# GOOGLE_CLIENT_SECRET=...
```

### 5. 환경 변수 로드 확인

```bash
# 환경 변수 확인
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('MONGODB_URI:', os.getenv('MONGODB_URI')[:30] + '...')
print('REDIS_URL:', 'Set' if os.getenv('REDIS_URL') else 'Not set')
print('GEMINI_API_KEY:', 'Set' if os.getenv('GEMINI_API_KEY') else 'Not set')
"
```

---

## 사용자 기능

### 1. Ollama 채팅 사용

```bash
# Ollama 실행 확인
ollama list

# 모델 다운로드 (예: Qwen 0.5B)
ollama pull qwen2.5:0.5b

# Backend 시작
cd playground/backend
uvicorn main:app --reload

# Frontend 시작 (별도 터미널)
cd playground/frontend
pnpm dev
```

브라우저에서 http://localhost:3000 접속하여 채팅 시작.

### 2. Google Workspace 공유

#### A. Google Docs로 내보내기

**API 엔드포인트**: `POST /api/chat/export/docs`

**프론트엔드에서 호출 예시**:

```typescript
// Google OAuth 2.0으로 access_token 받기
const accessToken = await getGoogleAccessToken();

// API 호출
const response = await fetch('http://localhost:8000/api/chat/export/docs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: currentSessionId,
    user_id: userId,
    title: "My beanllm Chat",
    access_token: accessToken
  })
});

const data = await response.json();
// { doc_id: "...", doc_url: "https://docs.google.com/document/d/...", ... }

// 사용자에게 링크 표시
window.open(data.doc_url, '_blank');
```

#### B. Google Drive에 저장

**API 엔드포인트**: `POST /api/chat/save/drive`

```typescript
const response = await fetch('http://localhost:8000/api/chat/save/drive', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: currentSessionId,
    user_id: userId,
    title: "beanllm_chat_2026-01-21.txt",
    access_token: accessToken
  })
});

const data = await response.json();
// { file_id: "...", file_url: "https://drive.google.com/file/d/...", ... }
```

#### C. Gmail로 공유

**API 엔드포인트**: `POST /api/chat/share/email`

```typescript
const response = await fetch('http://localhost:8000/api/chat/share/email', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: currentSessionId,
    user_id: userId,
    to_email: "friend@example.com",
    subject: "Check out this AI conversation!",
    message: "Here's an interesting chat I had with beanllm:",
    access_token: accessToken
  })
});

const data = await response.json();
// { message_id: "...", to_email: "friend@example.com", ... }
```

### 3. Google OAuth 2.0 설정 (프론트엔드)

#### A. Google Cloud Console 설정

1. https://console.cloud.google.com/
2. 프로젝트 생성: "beanllm-playground"
3. APIs & Services → Enabled APIs & services
4. "Google Docs API", "Google Drive API", "Gmail API" 활성화
5. OAuth consent screen 설정:
   - User Type: External
   - App name: "beanllm Playground"
   - Scopes: `docs`, `drive.file`, `gmail.send`
6. Credentials → Create Credentials → OAuth 2.0 Client ID
   - Application type: Web application
   - Authorized redirect URIs: `http://localhost:3000/auth/callback`
   - Client ID/Secret 복사

#### B. 프론트엔드 구현 (Next.js)

```typescript
// lib/google-auth.ts
import { GoogleAuthProvider, signInWithPopup } from 'firebase/auth';

export async function getGoogleAccessToken(): Promise<string> {
  const provider = new GoogleAuthProvider();
  provider.addScope('https://www.googleapis.com/auth/documents');
  provider.addScope('https://www.googleapis.com/auth/drive.file');
  provider.addScope('https://www.googleapis.com/auth/gmail.send');

  const result = await signInWithPopup(auth, provider);
  const credential = GoogleAuthProvider.credentialFromResult(result);
  return credential!.accessToken!;
}
```

---

## 관리자 기능

### 1. CLI 명령어

#### A. 사용 패턴 분석 (Gemini)

```bash
# 24시간 분석
beanllm admin analyze

# 7일 분석
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
주요 발견:
- Google Docs가 전체의 51% 차지 (가장 인기)
- 오후 2-4시에 피크 타임 (office hours)
- user123이 비정상적으로 많은 사용 (50건, 전체의 15%)

권장 조치:
1. user123 Rate limit 조정 필요 (현재 100 req/min → 50 req/min)
2. 캐싱 TTL 증가로 API 호출 30% 감소 가능 (1h → 2h)
3. 오후 피크 타임에 대비하여 Redis 모니터링 강화
```

#### B. 통계 조회

```bash
# 빠른 통계 (Gemini 사용 안 함)
beanllm admin stats

# 72시간 통계
beanllm admin stats --hours=72
```

#### C. 비용 최적화

```bash
beanllm admin optimize
```

**출력 예시**:

```
💰 Cost Optimization Recommendations

현재 상태:
- MongoDB: 120MB / 512MB (23% 사용) ✅ 안전
- Redis: 8,500 commands/day (85% 사용) ⚠️ 주의

예상 월간 비용: $0 (무료 티어 내)

권장 조치:
1. Redis 사용량 감소:
   - 세션 TTL 30분 → 20분 단축
   - 배치 get/set으로 호출 30% 감소
   - 예상 효과: 8,500 → 6,000 commands/day

2. MongoDB 최적화:
   - 30일 이후 이벤트 자동 삭제 (TTL 인덱스)
   - 압축으로 저장 공간 20% 절감

예상 절감: 무료 티어 유지 + 여유 확보
```

#### D. 보안 이벤트

```bash
# 고위험 이벤트 확인
beanllm admin security

# 72시간 이벤트
beanllm admin security --hours=72
```

### 2. Streamlit 대시보드

#### 실행 방법

```bash
# 방법 1: CLI
beanllm admin dashboard

# 방법 2: Streamlit 직접
streamlit run admin/dashboard.py

# 방법 3: 포트 지정
streamlit run admin/dashboard.py --server.port=8502
```

#### 대시보드 기능

- **Overview**: 실시간 통계, 차트, 상위 사용자
- **AI Analysis**: Gemini 기반 심층 분석
- **Security**: 보안 이벤트 모니터링
- **Cost**: 비용 최적화 제안
- **Settings**: 환경 변수, 관리자 설정

---

## 비용 분석

### 무료 티어 (월 $0)

| 서비스 | 무료 한도 | 예상 사용량 | 상태 |
|--------|----------|-------------|------|
| MongoDB Atlas | 512MB | ~120MB | ✅ 안전 |
| Upstash Redis | 10K commands/day | ~8.5K | ⚠️ 주의 |
| Gemini API | 유료 키 사용 | 월 ~100회 | ✅ 추가 비용 없음 |
| **총 비용** | **$0** | - | ✅ |

**지원 가능 규모**:
- 일일 사용자: ~100명
- 일일 채팅 세션: ~500개
- 일일 Google 내보내기: ~500건
- 세션 보관: 30일 (자동 삭제)

### 유료 전환 시 (월 $10-20)

| 서비스 | 플랜 | 비용 | 확장 규모 |
|--------|------|------|----------|
| MongoDB Atlas | M2 Shared (2GB) | $9/month | ~10,000 세션 |
| Upstash Redis | Pro (100K commands/day) | $5/month | ~1,000 사용자/일 |
| Gemini API | 기존 키 사용 | $0 | 무제한 (수동 호출) |
| **총 비용** | **$14/month** | | |

**지원 가능 규모**:
- 일일 사용자: ~1,000명
- 일일 채팅 세션: ~5,000개
- 일일 Google 내보내기: ~5,000건

---

## 보안 고려사항

### 1. API 키 보호

```bash
# .env 파일은 Git에 커밋하지 않기
echo ".env" >> .gitignore

# 환경 변수로 설정 (프로덕션)
export MONGODB_URI="..."
export GEMINI_API_KEY="..."
```

### 2. MongoDB 접근 제어

- **IP 화이트리스트**: Network Access에서 특정 IP만 허용
- **강력한 비밀번호**: 최소 20자, 특수문자 포함
- **읽기 전용 사용자**: 분석용 별도 계정 생성

### 3. Google OAuth 2.0

- **Redirect URI 제한**: 정확한 도메인만 허용
- **Scope 최소화**: 필요한 권한만 요청
- **Access Token 만료**: 1시간 후 자동 만료 (Refresh Token 사용)

### 4. Rate Limiting

```python
# 사용자별 Rate Limit 설정
from beanllm.infrastructure.distributed import get_rate_limiter

rate_limiter = get_rate_limiter()
await rate_limiter.acquire(
    key=f"google_export:{user_id}",
    max_requests=10,  # 10회
    window_seconds=60  # 1분
)
```

### 5. 민감 정보 로깅 방지

```python
# 이메일 주소 마스킹
masked_email = email.replace(email.split('@')[0], '***')

# 사용자 ID 해싱
import hashlib
hashed_user = hashlib.sha256(user_id.encode()).hexdigest()[:8]
```

---

## FAQ

### Q1. Gemini API 비용이 추가로 발생하나요?

**A**: 아니요. 이미 유료 결제한 Gemini API 키를 CLI/대시보드에서 **수동으로** 호출하므로 추가 비용이 없습니다. 자동화하면 비용이 발생할 수 있으니 주의하세요.

### Q2. MongoDB 무료 티어가 부족하면 어떻게 하나요?

**A**:
1. **이벤트 자동 삭제**: 30일 이후 이벤트 삭제 (TTL 인덱스)
2. **압축**: MongoDB 압축으로 20% 절감
3. **유료 전환**: M2 Shared ($9/month, 2GB)

### Q3. Google OAuth 2.0 설정이 어려워요.

**A**: 프론트엔드에서 Firebase Authentication을 사용하면 쉽게 설정할 수 있습니다:

```bash
npm install firebase
```

```typescript
import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = { /* Firebase Console에서 복사 */ };
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
```

### Q4. Streamlit 대시보드를 프로덕션에 배포하려면?

**A**: Streamlit Cloud (무료) 또는 Docker 컨테이너로 배포:

```bash
# Streamlit Cloud
streamlit deploy admin/dashboard.py

# Docker
docker build -t beanllm-admin .
docker run -p 8501:8501 --env-file .env beanllm-admin
```

### Q5. Redis 무료 티어를 초과하면 어떻게 되나요?

**A**: Upstash는 자동으로 요청을 제한합니다 (throttling). 해결 방법:
1. TTL 단축 (30분 → 20분)
2. 배치 처리로 호출 감소
3. Pro 플랜으로 업그레이드 ($5/month)

---

## 다음 단계

### 1. 프론트엔드 OAuth 구현

`playground/frontend/` 에 Google OAuth 2.0 추가:

```typescript
// components/GoogleShareButton.tsx
export function GoogleShareButton({ sessionId }: { sessionId: string }) {
  const handleShare = async () => {
    const accessToken = await getGoogleAccessToken();
    const response = await fetch('/api/chat/export/docs', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, access_token: accessToken })
    });
    const data = await response.json();
    window.open(data.doc_url, '_blank');
  };

  return <button onClick={handleShare}>📄 Export to Google Docs</button>;
}
```

### 2. 세션 관리 고도화

MongoDB + Redis 하이브리드 세션 관리 구현:

```python
# infrastructure/session_manager.py
class HybridSessionManager:
    async def get_session(self, session_id: str):
        # 1. Redis 먼저 확인 (빠름)
        cached = await redis.get(f"session:{session_id}")
        if cached:
            return json.loads(cached)

        # 2. MongoDB에서 조회 (느림)
        session = await mongodb.sessions.find_one({"id": session_id})
        if session:
            # Redis에 다시 캐싱
            await redis.setex(f"session:{session_id}", 3600, json.dumps(session))

        return session
```

### 3. 알림 시스템 추가

보안 이벤트 발생 시 이메일/Slack 알림:

```python
# infrastructure/distributed/google_events.py
async def log_abnormal_activity(user_id: str, reason: str):
    # 기존 로깅
    await event_logger.log_event(...)

    # 추가: 알림 전송
    if reason == "rate_limit_exceeded":
        await send_slack_alert(
            f"⚠️ User {user_id} exceeded rate limit"
        )
```

---

**Built with ❤️ for the beanllm community**
