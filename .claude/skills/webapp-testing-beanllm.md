# Web Application Testing for beanllm Playground

**마켓플레이스 스킬**: `webapp-testing` (Anthropic Agent Skills)
**자동 활성화**: "E2E 테스트", "integration test", "Playwright", "playground/frontend" 키워드 감지 시
**모델**: sonnet

## Skill Description

beanllm playground (FastAPI 백엔드 + Next.js 프론트엔드)의 E2E 통합 테스트를 자동화합니다.

## When to Use

- playground/frontend UI 테스트
- FastAPI 백엔드 + Next.js 프론트엔드 통합 테스트
- 채팅 스트리밍 기능 검증
- RAG 파이프라인 E2E 테스트
- 멀티 에이전트 대화 UI 테스트

## beanllm Specific Usage

### 1. FastAPI + Next.js 통합 테스트

```python
# scripts/test_playground_integration.py
from playwright.sync_api import sync_playwright

# with_server.py가 이미 서버 시작/종료 관리
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 1. Next.js 프론트엔드 접속
    page.goto('http://localhost:3000')
    page.wait_for_load_state('networkidle')

    # 2. 채팅 인터페이스 테스트
    page.fill('textarea[placeholder*="message"]', 'Explain RAG in simple terms')
    page.click('button:has-text("Send")')

    # 3. 스트리밍 응답 대기
    page.wait_for_selector('.message-assistant', timeout=10000)

    # 4. 응답 내용 검증
    response = page.locator('.message-assistant').inner_text()
    assert 'Retrieval' in response or 'RAG' in response

    # 5. 스크린샷 저장
    page.screenshot(path='test-results/chat-response.png')

    browser.close()
```

**실행:**
```bash
cd /Users/leejungbin/Downloads/llmkit

# 백엔드 + 프론트엔드 자동 시작 후 테스트
python scripts/with_server.py \
  --server "cd playground/backend && uvicorn main:app --reload" --port 8000 \
  --server "cd playground/frontend && pnpm dev" --port 3000 \
  -- python scripts/test_playground_integration.py
```

### 2. RAG 파이프라인 E2E 테스트

```python
# scripts/test_rag_e2e.py
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto('http://localhost:3000/rag')
    page.wait_for_load_state('networkidle')

    # 1. 문서 업로드
    page.set_input_files('input[type="file"]', 'test_documents/sample.pdf')
    page.wait_for_selector('.upload-success', timeout=5000)

    # 2. RAG 쿼리 실행
    page.fill('input[placeholder*="Ask"]', 'What is the main topic?')
    page.click('button:has-text("Search")')

    # 3. 검색 결과 확인
    page.wait_for_selector('.search-results', timeout=10000)
    results = page.locator('.search-result').count()
    assert results > 0

    # 4. 컨텍스트 포함 응답 확인
    page.wait_for_selector('.rag-response', timeout=15000)
    response = page.locator('.rag-response').inner_text()

    # 5. 스크린샷 저장
    page.screenshot(path='test-results/rag-results.png', full_page=True)

    browser.close()
```

### 3. 멀티 에이전트 대화 테스트

```python
# scripts/test_multi_agent_ui.py
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto('http://localhost:3000/multi-agent')
    page.wait_for_load_state('networkidle')

    # 1. 에이전트 패턴 선택
    page.select_option('select[name="pattern"]', 'debate')

    # 2. 토픽 입력
    page.fill('input[name="topic"]', 'Best practices for RAG systems')
    page.click('button:has-text("Start Debate")')

    # 3. 에이전트 응답 대기 (최대 3개 에이전트)
    for i in range(3):
        page.wait_for_selector(f'.agent-{i+1}-message', timeout=20000)

    # 4. 모든 에이전트 메시지 수집
    messages = page.locator('.agent-message').all_inner_texts()
    assert len(messages) >= 3

    # 5. 애니메이션 캡처 (GIF)
    page.screenshot(path='test-results/multi-agent.png', full_page=True)

    browser.close()
```

### 4. 브라우저 로그 캡처 (디버깅)

```python
# scripts/test_with_console_logs.py
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 콘솔 로그 수집
    console_logs = []
    page.on('console', lambda msg: console_logs.append(f'[{msg.type}] {msg.text}'))

    # 에러 수집
    errors = []
    page.on('pageerror', lambda err: errors.append(str(err)))

    page.goto('http://localhost:3000')
    page.wait_for_load_state('networkidle')

    # 테스트 실행...

    # 결과 출력
    print('\n=== Console Logs ===')
    for log in console_logs:
        print(log)

    print('\n=== Errors ===')
    for error in errors:
        print(error)

    browser.close()
```

## Test Results Directory

테스트 결과는 `playground/test-results/`에 저장:

```
playground/test-results/
├── chat-response.png
├── rag-results.png
├── multi-agent.png
└── console-logs.txt
```

## Common Selectors for beanllm Playground

### Chat Interface
- 메시지 입력: `textarea[placeholder*="message"]`
- 전송 버튼: `button:has-text("Send")`
- 사용자 메시지: `.message-user`
- 어시스턴트 메시지: `.message-assistant`
- 스트리밍 커서: `.cursor`

### RAG Interface
- 파일 업로드: `input[type="file"]`
- 쿼리 입력: `input[placeholder*="Ask"]`
- 검색 버튼: `button:has-text("Search")`
- 검색 결과: `.search-result`
- RAG 응답: `.rag-response`

### Multi-Agent Interface
- 패턴 선택: `select[name="pattern"]`
- 토픽 입력: `input[name="topic"]`
- 시작 버튼: `button:has-text("Start")`
- 에이전트 메시지: `.agent-message`

## Best Practices for beanllm

1. **항상 `networkidle` 대기**: Next.js 하이드레이션 완료 후 테스트
2. **스트리밍 타임아웃**: 채팅/RAG 응답은 최소 10-15초 타임아웃
3. **스크린샷 저장**: 실패 시 디버깅용 스크린샷 필수
4. **콘솔 로그 캡처**: 프론트엔드 에러 추적
5. **서버 로그 확인**: FastAPI 백엔드 에러도 함께 확인

## Integration with CI/CD

```yaml
# .github/workflows/e2e-tests.yml
name: E2E Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          pip install -r playground/backend/requirements.txt
          cd playground/frontend && pnpm install

      - name: Install Playwright
        run: pip install playwright && playwright install chromium

      - name: Run E2E tests
        run: |
          python scripts/with_server.py \
            --server "cd playground/backend && uvicorn main:app" --port 8000 \
            --server "cd playground/frontend && pnpm build && pnpm start" --port 3000 \
            -- python scripts/test_playground_integration.py

      - name: Upload screenshots
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: playground/test-results/
```

## Related Documents

- `.claude/skills/frontend-patterns.md` - React/Next.js 패턴
- `playground/backend/test_all_apis.py` - API 단위 테스트
- `playground/frontend/` - Next.js 앱 구조

## Upstream Skill

이 스킬은 Anthropic Agent Skills 마켓플레이스의 `webapp-testing` 스킬을 beanllm 프로젝트에 맞게 커스터마이징한 것입니다.

**원본 스킬 위치**: `~/.claude/plugins/marketplaces/anthropic-agent-skills/skills/webapp-testing/`
