# Cursor Agent를 활용한 통합 테스트 자동화 가이드

## 🎯 왜 Cursor Agent인가?

### 장점

1. **즉시 사용 가능**: Cursor를 이미 사용 중이면 추가 설정 불필요
2. **백그라운드 실행**: AI가 독립적으로 작업 수행, 개발자는 다른 일 가능
3. **대화 추적**: Agent Conversation API로 진행 상황 모니터링
4. **자동화**: 코드 변경 시 자동으로 테스트 재실행 가능
5. **자연어 제어**: "통합 테스트 실행해줘"만으로 모든 작업 수행

---

## 🚀 구현 방법

### 방법 1: Cursor Agent에게 직접 요청 (가장 간단)

**사용법**:
```
Cursor 채팅에서:
"백엔드-프론트엔드 통합 테스트를 실행하고 결과를 리포트해줘"
```

**Agent가 자동으로 수행**:
1. 백엔드 서버 상태 확인
2. 프론트엔드 서버 상태 확인
3. 서버가 없으면 자동 시작
4. Playwright 테스트 실행
5. 결과 분석 및 리포트 생성

---

### 방법 2: 커스텀 Agent 생성 (고급)

**파일**: `.claude/agents/test-runner.md`

```markdown
# Test Runner Agent

## Agent Description

자동으로 백엔드-프론트엔드 통합 테스트를 실행하고 결과를 리포트합니다.

## Capabilities

- 백엔드 서버 시작/중지
- 프론트엔드 서버 시작/중지
- Playwright 테스트 실행
- 테스트 결과 분석
- 리포트 생성

## Workflow

1. Check if backend is running on port 8000
2. If not, start backend: `cd playground/backend && python -m uvicorn main:app --reload`
3. Check if frontend is running on port 3000
4. If not, start frontend: `cd playground/frontend && pnpm dev`
5. Wait for both servers to be ready
6. Run Playwright tests: `cd playground/frontend && npx playwright test`
7. Analyze test results
8. Generate report with:
   - Passed tests count
   - Failed tests count
   - Screenshots of failures
   - Performance metrics
   - Recommendations

## Output Format

```json
{
  "status": "success|failure",
  "summary": {
    "total": 10,
    "passed": 8,
    "failed": 2,
    "duration": "45.2s"
  },
  "failures": [
    {
      "test": "chat.spec.ts:15",
      "error": "...",
      "screenshot": "test-results/chat-failure.png"
    }
  ],
  "recommendations": [
    "Fix selector in chat.spec.ts line 15",
    "Add retry logic for flaky test"
  ]
}
```

## Tools Available

- File system access
- Terminal commands
- HTTP requests (for health checks)
- Screenshot analysis (if needed)
```

**사용법**:
```
/test-runner
또는
"test-runner 에이전트에게 통합 테스트 실행 요청"
```

---

### 방법 3: Command로 래핑 (권장)

**파일**: `.claude/commands/run-e2e-tests.md`

```markdown
# /run-e2e-tests - E2E 테스트 실행

**트리거**: `/run-e2e-tests`
**모델**: sonnet
**설명**: 백엔드-프론트엔드 통합 테스트 자동 실행

## Command Description

백엔드와 프론트엔드 서버를 자동으로 시작하고 Playwright E2E 테스트를 실행합니다.

## Usage

```
/run-e2e-tests
/run-e2e-tests --headless
/run-e2e-tests --ui
/run-e2e-tests --browser chromium
```

## Options

- `--headless`: 헤드리스 모드 (기본값)
- `--ui`: Playwright UI 모드로 실행
- `--browser`: 브라우저 선택 (chromium, firefox, webkit)
- `--port-backend`: 백엔드 포트 (기본: 8000)
- `--port-frontend`: 프론트엔드 포트 (기본: 3000)

## Execution Steps

### 1. 서버 상태 확인

```bash
# Backend health check
curl http://localhost:8000/health

# Frontend health check
curl http://localhost:3000
```

### 2. 서버 시작 (필요시)

```bash
# Backend
cd playground/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &

# Frontend
cd playground/frontend
pnpm dev &
```

### 3. 서버 준비 대기

```bash
# 최대 30초 대기
timeout=30
while ! curl -s http://localhost:8000/health > /dev/null; do
  sleep 1
  timeout=$((timeout-1))
  [ $timeout -eq 0 ] && exit 1
done
```

### 4. Playwright 테스트 실행

```bash
cd playground/frontend
npx playwright test
```

### 5. 결과 분석 및 리포트

- 테스트 결과 파싱
- 실패한 테스트 스크린샷 확인
- 성능 메트릭 수집 (응답 시간, 로딩 시간)
- 리포트 생성

## Output

테스트 결과를 마크다운 형식으로 출력:

```markdown
# E2E 테스트 결과

## 요약
- 총 테스트: 15
- 통과: 13 ✅
- 실패: 2 ❌
- 실행 시간: 45.2초

## 실패한 테스트

### 1. chat.spec.ts:15 - 메시지 전송
- **에러**: Element not found: textarea[placeholder*="message"]
- **스크린샷**: test-results/chat-failure.png
- **권장사항**: 셀렉터 업데이트 필요

### 2. rag.spec.ts:8 - 파일 업로드
- **에러**: Timeout waiting for upload-success
- **스크린샷**: test-results/rag-failure.png
- **권장사항**: 타임아웃 증가 또는 업로드 로직 확인

## 성능 메트릭

- 평균 응답 시간: 1.2초
- 최대 응답 시간: 3.5초
- 페이지 로딩 시간: 0.8초
```

## Error Handling

- 서버 시작 실패 시 에러 메시지와 해결 방법 제시
- 테스트 실패 시 스크린샷과 로그 제공
- 타임아웃 발생 시 재시도 제안
```

---

## 📋 실제 사용 시나리오

### 시나리오 1: 개발 중 자동 테스트

```
개발자가 코드 수정
↓
Cursor Agent가 자동 감지
↓
"변경사항이 있으니 통합 테스트를 실행할까요?"
↓
사용자: "네"
↓
Agent가 자동으로:
  1. 서버 재시작
  2. 테스트 실행
  3. 결과 리포트
```

### 시나리오 2: 커밋 전 검증

```
git commit 하기 전
↓
"/run-e2e-tests" 실행
↓
모든 테스트 통과 확인
↓
커밋 진행
```

### 시나리오 3: PR 생성 시

```
PR 생성
↓
Cursor Agent가 자동으로:
  1. 변경된 파일 분석
  2. 영향받는 테스트 식별
  3. 관련 테스트만 실행
  4. PR 코멘트에 결과 추가
```

---

## 🔧 고급 활용

### 1. Agent Conversation API 활용

**프로그래밍 방식으로 Agent 제어**:

```python
# Python 예시 (참고용)
import requests

# Agent에게 테스트 실행 요청
response = requests.post(
    "https://api.cursor.com/agent/conversation",
    json={
        "message": "백엔드-프론트엔드 통합 테스트 실행",
        "context": {
            "project": "llmkit",
            "type": "e2e_test"
        }
    }
)

# 대화 내역 추적
conversation_id = response.json()["conversation_id"]

# 진행 상황 확인
status = requests.get(
    f"https://api.cursor.com/agent/conversation/{conversation_id}"
)
```

### 2. MCP 도구와 통합

**MCP 서버에 테스트 도구 추가**:

```python
# mcp_server/tools/test_tools.py
@mcp.tool()
async def run_e2e_tests(
    headless: bool = True,
    browser: str = "chromium"
) -> dict:
    """E2E 테스트 실행"""
    # Playwright 테스트 실행
    # 결과 반환
```

**Cursor Agent가 MCP 도구 호출**:
```
Agent: "MCP의 run_e2e_tests 도구를 사용하여 테스트 실행"
```

### 3. 자동 리포트 생성

**테스트 결과를 자동으로 문서화**:

```markdown
# Agent가 자동 생성하는 리포트

## 테스트 실행 시간
2026-01-XX XX:XX:XX

## 결과 요약
- ✅ 통과: 13/15
- ❌ 실패: 2/15
- ⏱️ 실행 시간: 45.2초

## 실패 상세
[스크린샷과 로그 포함]

## 추천 조치
1. chat.spec.ts의 셀렉터 업데이트
2. rag.spec.ts의 타임아웃 증가
```

---

## 🎯 프로젝트 적용 체크리스트

### 즉시 사용 가능
- [x] Cursor 설치됨
- [x] 프로젝트 구조 이해
- [ ] `/run-e2e-tests` 커맨드 생성
- [ ] Playwright 설정 확인

### 단계별 구현
1. **1단계**: `/run-e2e-tests` 커맨드 생성
2. **2단계**: test-runner Agent 생성 (선택)
3. **3단계**: MCP 테스트 도구 추가 (선택)
4. **4단계**: 자동 리포트 시스템 구축 (선택)

---

## 💡 팁

### 1. Agent에게 명확한 지시

**❌ 나쁜 예**:
```
"테스트 해줘"
```

**✅ 좋은 예**:
```
"백엔드(포트 8000)와 프론트엔드(포트 3000) 서버를 확인하고,
없으면 시작한 후 Playwright E2E 테스트를 실행하고,
결과를 마크다운 리포트로 생성해줘"
```

### 2. 컨텍스트 제공

```
"다음 파일들이 변경되었어:
- playground/frontend/src/app/chat/page.tsx
- playground/backend/main.py

이 변경사항에 영향을 받는 통합 테스트만 실행해줘"
```

### 3. 결과 활용

```
"지난번 테스트에서 실패한 chat.spec.ts:15를 다시 실행해줘"
```

---

## 📚 참고 자료

- [Cursor Agent Conversation API](https://docs.cursor.com/ko/background-agent/api/agent-conversation)
- [Playwright 공식 문서](https://playwright.dev/)
- 프로젝트 내: `.claude/commands/test-gen.md` (참고)

---

## 🚀 다음 단계

1. **지금 바로 시도**: Cursor 채팅에서 "통합 테스트 실행해줘" 입력
2. **커맨드 생성**: `.claude/commands/run-e2e-tests.md` 파일 생성
3. **Agent 생성**: `.claude/agents/test-runner.md` 파일 생성 (선택)
