# 🫘 beanllm - Claude Code 설정 가이드

이 디렉토리는 Claude Code를 beanllm 프로젝트에 최적화하기 위한 설정을 포함합니다.

## 📁 디렉토리 구조

```
.claude/
├── rules/                    # 항상 따르는 규칙
│   ├── clean-architecture.md # Clean Architecture 의존성 규칙 ⭐ CRITICAL
│   ├── code-quality.md       # 코드 품질 규칙 (중복 제거, 최적화)
│   ├── security.md           # 보안 규칙 (API 키, XSS, SQL Injection)
│   └── testing.md            # 테스트 규칙 (TDD, 80% 커버리지)
│
├── skills/                   # 자동 활성화 스킬 (6개)
│   ├── python-clean-architecture.md  # Clean Architecture 패턴
│   ├── decorator-pattern.md          # 데코레이터 패턴 (중복 코드 제거)
│   ├── backend-patterns.md           # FastAPI, Redis, PostgreSQL 패턴
│   ├── frontend-patterns.md          # React 19, Next.js 15 패턴
│   ├── webapp-testing-beanllm.md     # Playwright E2E 테스트 🆕
│   └── frontend-design-beanllm.md    # 프로덕션급 UI 디자인 🆕
│
├── commands/                 # 수동 트리거 커맨드 (10개)
│   ├── plan.md               # /plan - 기능 계획 수립
│   ├── tdd.md                # /tdd - TDD 워크플로우 (Red-Green-Refactor)
│   ├── arch-check.md         # /arch-check - 아키텍처 검증
│   ├── dedup.md              # /dedup - 중복 코드 찾기 및 리팩토링
│   ├── test-gen.md           # /test-gen - 테스트 자동 생성
│   ├── code-review.md        # /code-review - 종합 코드 리뷰 (Opus)
│   ├── update-docs.md        # /update-docs - 문서 자동 업데이트
│   ├── build-fix.md          # /build-fix - 빌드 에러 자동 수정
│   ├── commit.md             # /commit - 스마트 커밋 (도메인별 자동 분할) 🆕
│   └── pr.md                 # /pr - GitHub PR 자동 생성 🆕
│
├── agents/                   # 독립 작업 위임 에이전트
│   ├── code-reviewer.md      # Opus - 코드 품질/보안 종합 검토
│   ├── architecture-fixer.md # Sonnet - Clean Architecture 위반 자동 수정
│   └── performance-optimizer.md # Sonnet - 알고리즘 최적화
│
├── settings.json             # Hooks, MCP 설정
├── settings.local.json       # 로컬 permissions 설정 (기존)
└── README.md                 # 이 파일
```

## 🚀 빠른 시작

### 1. Claude Code 확인

```bash
# Claude Code가 .claude 디렉토리를 인식하는지 확인
# 프로젝트 루트에서 Claude Code 시작 시 자동으로 로드됩니다
```

### 2. 완전한 개발 워크플로우 (Affaan Mustafa 가이드)

```bash
# 1. /plan "사용자 인증 기능 구현"
#    → Planner가 단계별 계획 수립

/plan "HyDE query expansion 추가"

# 2. /tdd
#    → TDD-Guide가 Red-Green-Refactor 사이클 안내

/tdd

# 3. [코드 작성]
#    → PostToolUse 훅이 자동으로 Black/Ruff 실행

# 4. 중복 코드 제거
/dedup

# 5. 아키텍처 검증
/arch-check

# 6. /code-review
#    → Code-Reviewer 에이전트(Opus)가 품질 검토

/code-review

# 7. /update-docs
#    → Doc-Updater가 문서 동기화

/update-docs

# 8. /commit 🆕
#    → 변경된 파일을 도메인별로 자동 분할하여 여러 커밋 생성

/commit

# 9. /pr 🆕
#    → GitHub PR 자동 생성

/pr
```

### 3. 개별 커맨드

```bash
# 기능 계획
/plan "기능 설명"

# TDD 워크플로우
/tdd

# 아키텍처 검증
/arch-check

# 중복 코드 찾기
/dedup

# 테스트 자동 생성
/test-gen --path src/beanllm/facade/core/client_facade.py

# 종합 코드 리뷰 (Opus)
/code-review

# 문서 업데이트
/update-docs

# 빌드 에러 수정
/build-fix

# 스마트 커밋 (도메인별 자동 분할) 🆕
/commit

# GitHub PR 생성 🆕
/pr
```

### 3. Rules (자동 적용)

Rules는 모든 코드 변경 시 자동으로 적용됩니다:

- **clean-architecture.md** ⭐ CRITICAL
  - Handler → Service (인터페이스만)
  - Domain → 외부 의존성 없음
  - 절대 경로 import

- **code-quality.md**
  - 중복 코드 85-90% 감소 목표
  - 알고리즘 최적화 (O(n) → O(1))
  - 타입 힌트 + Docstring 필수

- **security.md**
  - API 키 하드코딩 금지
  - SQL Injection 방지
  - 입력 검증

- **testing.md**
  - TDD (Test-Driven Development)
  - 80% 커버리지 목표 (현재 61%)
  - 엣지 케이스 + 에러 처리 테스트

### 4. Skills (자동 활성화)

특정 키워드 감지 시 자동으로 활성화:

- **python-clean-architecture.md**
  - 키워드: "facade", "handler", "service", "domain", "리팩토링"
  - 모델: sonnet
  - 의존성 방향 검증 및 리팩토링

- **decorator-pattern.md**
  - 키워드: "중복", "캐싱", "rate limiting", "데코레이터"
  - 모델: sonnet
  - 중복 코드 → 데코레이터 패턴 리팩토링

### 5. Hooks (이벤트 기반 자동화)

#### PreToolUse 훅

- **테스트 실행 전**: `pytest` 실행 시 경고 메시지
- **코드 품질 체크 전**: `black`, `ruff`, `mypy` 실행 시 안내

#### PostToolUse 훅

- **Python 파일 수정 후**: 자동으로 Black, Ruff 포매팅 실행

#### Stop 훅 (응답 완료 후)

- **디버그 코드 감사**: `print()`, `console.log()` 확인
- **Clean Architecture 검증**: 의존성 방향, 상대 경로 import 확인

### 6. Subagents (작업 위임)

#### code-reviewer (Opus)

```
/code-review
/code-review --file src/beanllm/facade/core/client_facade.py
```

**검토 항목**:
- Clean Architecture 준수
- 코드 품질 (중복, 최적화)
- 보안 취약점
- 성능 분석
- 테스트 커버리지

**출력**:
- Critical issues (즉시 수정 필요)
- Warnings (권장 수정)
- Suggestions (개선 제안)
- Before/After 코드 예시

#### architecture-fixer (Sonnet)

```
/arch-fix
/arch-fix --auto
/arch-fix --preview-only
```

**자동 수정**:
- Handler → Service 구현체 → 인터페이스로 변경
- 상대 경로 → 절대 경로
- 순환 의존 → Protocol로 분리
- Factory 패턴 생성

#### performance-optimizer (Sonnet)

```
/optimize
/optimize --path src/beanllm/domain/retrieval/
/optimize --benchmark
```

**최적화 패턴**:
- O(n) → O(1): 딕셔너리 캐싱
- O(n log n) → O(n log k): heapq.nlargest()
- O(n×m×p) → O(n×m): 정규표현식 사전 컴파일
- 반복 계산 제거
- Generator 사용 (메모리 최적화)
- 배치 처리 (I/O 최적화)

## 📋 워크플로우 예시

### 새 기능 추가

```
1. /plan "RAG 쿼리 확장 기능 추가"
   → Planner 에이전트가 단계별 계획 수립

2. 코드 작성 (TDD)
   - /test-gen --path src/beanllm/service/impl/core/rag_service_impl.py
   - 테스트 작성
   - 구현

3. /arch-check
   → Clean Architecture 규칙 준수 확인

4. /dedup
   → 중복 코드 제거

5. /code-review
   → 종합 코드 리뷰 (Opus)

6. /optimize
   → 성능 최적화

7. pytest --cov=src/beanllm --cov-report=html
   → 테스트 커버리지 확인
```

### 버그 수정

```
1. 재현 가능한 테스트 작성
   /test-gen --type unit

2. 최소 변경으로 수정

3. /arch-check
   → 아키텍처 위반 없는지 확인

4. /code-review
   → 코드 리뷰

5. pytest
   → 회귀 테스트
```

### 리팩토링

```
1. /arch-check
   → 위반 사항 확인

2. /dedup
   → 중복 코드 85-90% 감소

3. /optimize
   → 알고리즘 최적화

4. /test-gen
   → 리팩토링 후 테스트 추가

5. pytest --cov=src/beanllm
   → 커버리지 유지 확인
```

## 🎯 모델 선택 전략

### Opus (최고 품질, 높은 비용)

- **code-reviewer**: 종합 코드 리뷰 (보안, 성능, 품질)
- **보안 검토**: 취약점 심층 분석
- **복잡한 리팩토링**: 대규모 아키텍처 변경

### Sonnet (균형)

- **architecture-fixer**: Clean Architecture 위반 수정
- **performance-optimizer**: 알고리즘 최적화
- **skills**: python-clean-architecture, decorator-pattern
- **commands**: arch-check, dedup, test-gen
- **일반 코딩**: 기능 구현, 버그 수정

### Haiku (빠르고 저렴)

- **문서 업데이트**: README, API 문서
- **간단한 수정**: 오타, 스타일 변경
- **테스트 리뷰**: 단순 테스트 검증

## ⚙️ 설정 커스터마이징

### Hooks 비활성화

```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": []  // 자동 포매팅 비활성화
  }
}
```

### MCP 서버 활성화

```json
// .claude/settings.json
{
  "mcpServers": {
    "github": {
      "_enabled": true,  // false → true로 변경
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token"
      }
    }
  }
}
```

### Rule 우선순위 변경

Rules는 파일명 알파벳 순으로 적용됩니다:
- `clean-architecture.md` (최우선)
- `code-quality.md`
- `security.md`
- `testing.md`

## 🐛 트러블슈팅

### 문제: Hooks가 실행되지 않음

**해결**:
1. `.claude/settings.json` 문법 확인 (유효한 JSON)
2. 파일 권한 확인: `chmod +x .claude/settings.json`

### 문제: Subagents가 작동하지 않음

**해결**:
1. `.claude/agents/*.md` 파일 존재 확인
2. 파일 내용에 `**모델**`, `**허용 도구**` 메타데이터 포함 확인

### 문제: Skills가 자동 활성화되지 않음

**해결**:
1. `.claude/skills/*.md` 파일 확인
2. `## When to Use` 섹션의 키워드 확인
3. 대화에서 해당 키워드 사용

## 📚 참고 문서

### 프로젝트 문서
- **CLAUDE.md** - 프로젝트 전체 컨텍스트 (먼저 읽기)
- **DEPENDENCY_RULES.md** - Clean Architecture 의존성 규칙 (상세)
- **ARCHITECTURE.md** - 아키텍처 상세 설명
- **.cursorrules** - 코딩 스타일, 패턴 (Claude Code Rules와 유사)

### Claude Code 문서
- Rules: https://github.com/anthropics/claude-code#rules
- Skills: https://github.com/anthropics/claude-code#skills
- Commands: https://github.com/anthropics/claude-code#commands
- Subagents: https://github.com/anthropics/claude-code#subagents
- Hooks: https://github.com/anthropics/claude-code#hooks

## 💡 팁

### 1. 컨텍스트 관리

- 불필요한 MCP 서버 비활성화 (컨텍스트 절약)
- Subagents는 제한된 도구만 허용 (집중된 실행)

### 2. 비용 최적화

- 간단한 작업: Haiku
- 일반 작업: Sonnet
- 중요한 리뷰: Opus

### 3. 병렬 작업

- Git worktrees로 병렬 작업
- 별도 Claude Code 세션 실행

```bash
git worktree add ../feature-branch feature-branch
cd ../feature-branch
claude  # 새 세션
```

## 🔌 마켓플레이스 스킬 (Anthropic Agent Skills)

beanllm 프로젝트에는 Anthropic Agent Skills 마켓플레이스의 스킬 2개가 통합되어 있습니다.

### webapp-testing (E2E 테스트 자동화)

**자동 활성화 키워드**:
- "E2E 테스트", "integration test"
- "Playwright", "frontend test"
- "playground/frontend"

**주요 기능**:
```python
# playground/backend + frontend 통합 테스트
python scripts/with_server.py \
  --server "cd playground/backend && uvicorn main:app" --port 8000 \
  --server "cd playground/frontend && pnpm dev" --port 3000 \
  -- python scripts/test_playground_integration.py
```

**사용 사례**:
- 채팅 UI 스트리밍 테스트
- RAG 검색 결과 UI 검증
- 멀티 에이전트 대화 시각화 테스트

**상세 가이드**: `.claude/skills/webapp-testing-beanllm.md`

### frontend-design (프로덕션급 UI 디자인)

**자동 활성화 키워드**:
- "UI 디자인", "컴포넌트 생성"
- "React component", "Tailwind"
- "playground/frontend"

**디자인 시스템**: Technical Elegance
- **컬러**: Deep Tech Green (Emerald), Data Amber, Insight Blue
- **타이포그래피**: JetBrains Mono, Work Sans, Fira Code
- **애니메이션**: Framer Motion (staggered, streaming, pulse)

**금지 패턴** (AI Slop 회피):
- ❌ Inter, Roboto, Arial 폰트
- ❌ Purple gradient on white
- ❌ 중앙 정렬 남발

**컴포넌트 예시**:
- ChatMessage (그라디언트 border + 스트리밍 커서)
- RAGSearchResults (relevance score visualization)
- MultiAgentDebate (animated timeline)

**상세 가이드**: `.claude/skills/frontend-design-beanllm.md`

### 설정 위치

`.claude/settings.json` → `skills` 섹션:

```json
{
  "skills": {
    "webapp-testing": {
      "enabled": true,
      "autoActivate": true
    },
    "frontend-design": {
      "enabled": true,
      "autoActivate": true
    }
  }
}
```

### 추가 스킬 (비활성화)

필요시 `.claude/settings.json`에서 활성화:

- **web-artifacts-builder**: React + shadcn/ui 프로토타입 제작
- **mcp-builder**: beanllm을 MCP 서버로 확장

## 🎉 시작하기

```
1. 프로젝트 루트에서 Claude Code 시작
2. /arch-check 실행 (Clean Architecture 검증)
3. /dedup 실행 (중복 코드 확인)
4. /code-review 실행 (종합 리뷰)
5. 코드 작성 시작!
```

**🫘 Built with beanllm - The unified LLM framework**
