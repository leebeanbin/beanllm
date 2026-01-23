# 백엔드-프론트엔드 통합 테스트 (E2E) 전략

## 현재 상태

### 기존 테스트
- `playground/backend/tests/test_frontend_backend_integration.py`
  - **API만 테스트** (requests 사용)
  - 백엔드 엔드포인트 직접 호출
  - 프론트엔드 UI 테스트 없음

### 필요한 것
- **브라우저 기반 E2E 테스트**
- 프론트엔드 UI + 백엔드 API 통합 검증
- 실제 사용자 플로우 테스트

## 방법론 비교

### 옵션 1: Playwright ✅ (권장)

### 왜 Playwright?

1. **Next.js 공식 지원**: `@playwright/test` 이미 설치됨
2. **빠르고 안정적**: Selenium보다 빠름
3. **멀티 브라우저**: Chromium, Firefox, WebKit 지원
4. **자동 대기**: 네트워크, DOM 변경 자동 대기
5. **스크린샷/비디오**: 실패 시 자동 캡처

### 대안 비교

| 도구 | 장점 | 단점 |
|------|------|------|
| **Playwright** ✅ | 빠름, 안정적, Next.js 지원 | - |
| Cypress | 사용하기 쉬움 | 느림, Chrome만 |
| Selenium | 오래됨, 널리 사용 | 느림, 불안정 |

**결론: Playwright 권장**

## 구현 방법

### 1. Playwright 설정

**파일**: `playground/frontend/playwright.config.ts`

```typescript
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: [
    {
      command: 'cd ../backend && python -m uvicorn main:app --reload',
      port: 8000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'pnpm dev',
      port: 3000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
```

### 2. E2E 테스트 예시

**파일**: `playground/frontend/e2e/chat.spec.ts`

```typescript
import { test, expect } from '@playwright/test';

test.describe('Chat Integration', () => {
  test('사용자가 메시지를 보내고 응답을 받는다', async ({ page }) => {
    // 1. 채팅 페이지 접속
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');

    // 2. 모델 선택 (필요시)
    // await page.selectOption('select[name="model"]', 'qwen2.5:0.5b');

    // 3. 메시지 입력
    const textarea = page.locator('textarea[placeholder*="message"]');
    await textarea.fill('안녕하세요! 테스트 메시지입니다.');

    // 4. 전송 버튼 클릭
    await page.click('button:has-text("Send")');

    // 5. 응답 대기 (스트리밍)
    await page.waitForSelector('.message-assistant', { timeout: 30000 });

    // 6. 응답 내용 검증
    const response = page.locator('.message-assistant').last();
    await expect(response).toBeVisible();
    
    const responseText = await response.textContent();
    expect(responseText).toBeTruthy();
    expect(responseText!.length).toBeGreaterThan(0);
  });

  test('여러 메시지 대화 플로우', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');

    // 첫 번째 메시지
    await page.fill('textarea[placeholder*="message"]', '1+1은?');
    await page.click('button:has-text("Send")');
    await page.waitForSelector('.message-assistant', { timeout: 30000 });

    // 두 번째 메시지
    await page.fill('textarea[placeholder*="message"]', '2+2는?');
    await page.click('button:has-text("Send")');
    await page.waitForSelector('.message-assistant', { timeout: 30000 });

    // 메시지 개수 확인
    const messages = page.locator('.message-assistant');
    await expect(messages).toHaveCount(2);
  });
});
```

### 3. RAG 통합 테스트

**파일**: `playground/frontend/e2e/rag.spec.ts`

```typescript
import { test, expect } from '@playwright/test';

test.describe('RAG Integration', () => {
  test('문서 업로드 후 RAG 쿼리', async ({ page }) => {
    await page.goto('/rag');
    await page.waitForLoadState('networkidle');

    // 1. 파일 업로드
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('This is a test document about AI and machine learning.'),
    });

    // 2. 빌드 대기
    await page.waitForSelector('.upload-success', { timeout: 10000 });

    // 3. 쿼리 입력
    await page.fill('input[placeholder*="Ask"]', 'What is AI?');
    await page.click('button:has-text("Search")');

    // 4. 결과 대기
    await page.waitForSelector('.rag-response', { timeout: 30000 });

    // 5. 응답 검증
    const response = page.locator('.rag-response');
    await expect(response).toBeVisible();
  });
});
```

## 실행 방법

### 1. Playwright 설치

```bash
cd playground/frontend
pnpm add -D @playwright/test
npx playwright install chromium
```

### 2. 테스트 실행

```bash
# 모든 E2E 테스트 실행
npx playwright test

# 특정 테스트만
npx playwright test e2e/chat.spec.ts

# UI 모드 (디버깅)
npx playwright test --ui

# 헤드 모드 (브라우저 보기)
npx playwright test --headed
```

### 3. 서버 자동 시작

`playwright.config.ts`의 `webServer` 설정으로:
- 백엔드 (포트 8000) 자동 시작
- 프론트엔드 (포트 3000) 자동 시작
- 테스트 종료 시 자동 종료

## 테스트 구조

```
playground/
├── frontend/
│   ├── e2e/
│   │   ├── chat.spec.ts          # 채팅 통합 테스트
│   │   ├── rag.spec.ts           # RAG 통합 테스트
│   │   ├── agent.spec.ts        # Agent 통합 테스트
│   │   └── multi-agent.spec.ts  # Multi-Agent 통합 테스트
│   └── playwright.config.ts      # Playwright 설정
│
└── backend/
    └── tests/
        └── test_frontend_backend_integration.py  # API만 테스트 (기존)
```

## CI/CD 통합

**파일**: `.github/workflows/e2e.yml`

```yaml
name: E2E Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install backend dependencies
        run: |
          cd playground/backend
          pip install -r requirements.txt
      
      - name: Install frontend dependencies
        run: |
          cd playground/frontend
          pnpm install
      
      - name: Install Playwright
        run: |
          cd playground/frontend
          npx playwright install --with-deps chromium
      
      - name: Run E2E tests
        run: |
          cd playground/frontend
          npx playwright test
      
      - name: Upload test results
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playground/frontend/playwright-report/
```

## 장점

1. **실제 사용자 플로우 검증**: 브라우저에서 실제 동작 확인
2. **자동화**: CI/CD에서 자동 실행
3. **디버깅**: 스크린샷, 비디오, 트레이스 자동 저장
4. **빠른 피드백**: 개발 중 빠르게 실행 가능

## 다음 단계

1. `playground/frontend/playwright.config.ts` 생성
2. `playground/frontend/e2e/` 디렉토리 생성
3. 첫 번째 테스트 작성 (`chat.spec.ts`)
4. CI/CD 워크플로우 추가
