# /code-review - Comprehensive Code Review

**트리거**: `/code-review`
**모델**: opus (via code-reviewer agent)
**설명**: code-reviewer 에이전트에게 종합 코드 리뷰 위임

## Command Description

이 커맨드는 **code-reviewer 에이전트**(Opus)에게 코드 리뷰를 위임합니다.

## Usage

```bash
/code-review
/code-review --path src/beanllm/service/impl/core/rag_service_impl.py
/code-review --full
```

## What This Command Does

1. **code-reviewer 에이전트 호출** (`.claude/agents/code-reviewer.md`)
2. 에이전트가 다음을 자동 수행:
   - Clean Architecture 검증
   - 보안 취약점 스캔
   - 성능 분석
   - 테스트 커버리지 확인
   - 코드 품질 평가
   - 종합 리포트 생성

## Delegation Flow

```
You: /code-review
  ↓
Command: code-review.md (이 파일)
  ↓
Agent: code-reviewer.md (Opus 모델)
  ↓
- Read 변경된 파일
- Grep 패턴 검색
- Bash git diff 실행
  ↓
Review Report 생성
```

## Review Scope

code-reviewer 에이전트가 다음을 검토합니다:

### 1. Clean Architecture ⭐
- 의존성 방향 (Facade → Handler → Service → Domain)
- 순환 의존 검사
- 절대 경로 import 확인

### 2. Security 🔒
- API 키 하드코딩
- SQL Injection 취약점
- XSS 취약점
- 입력 검증

### 3. Code Quality 📊
- 중복 코드 (목표: < 10%)
- 타입 힌트 + Docstring
- 알고리즘 복잡도 (Cyclomatic < 10)
- 파일/함수 크기

### 4. Performance ⚡
- 알고리즘 복잡도 최적화
- 캐싱 활용
- N+1 쿼리 방지

### 5. Test Coverage 🧪
- Domain: 100%
- Service: 90%+
- Handler: 80%+
- 엣지 케이스 테스트

## Example Output

```markdown
# Code Review Report (by Opus)

## Summary
- ✅ Overall: APPROVED with recommendations
- 🎯 Quality Score: 87/100
- ⚠️ Warnings: 3
- 💡 Suggestions: 5

## Clean Architecture: ✅ PASS
- ✅ Correct dependencies
- ✅ No circular imports

## Security: ✅ PASS
- ✅ No hardcoded secrets
- 💡 Add rate limiting

## Code Quality: ⚠️ NEEDS ATTENTION
- ⚠️ Duplicate code at lines 45-67, 89-111
- ⚠️ High complexity in query() method (12 > 10)

## Performance: ✅ GOOD
- 💡 Use heapq.nlargest() for 5.7× speedup

## Test Coverage: ✅ EXCELLENT (92%)
- ✅ All critical paths tested

## Action Items
1. Extract cache logic to decorator
2. Reduce query() complexity
3. Add rate limiting
```

## Cost & Model Selection

| Scope | Model | Cost | When to Use |
|-------|-------|------|-------------|
| Full project | **Opus** | ~$2-5 | Before production |
| Single file | **Opus** | ~$0.50 | Critical changes |
| Quick check | Sonnet | ~$0.08 | Daily review |

**This command uses Opus** for highest quality review.

## Integration with Workflow

```bash
# Complete TDD workflow
/plan "Add HyDE to RAG"      # 1. Plan
/tdd                          # 2. TDD cycle
# [Write code]
/dedup                        # 3. Remove duplication
/arch-check                   # 4. Verify architecture
/code-review                  # 5. Comprehensive review (Opus) ⭐
/update-docs                  # 6. Update documentation
```

## Related Documents

- **`.claude/agents/code-reviewer.md`** ← 실제 리뷰 로직 (이 커맨드가 호출)
- `.claude/commands/arch-check.md` - Architecture only
- `.claude/commands/dedup.md` - Duplication only
- `.claude/rules/clean-architecture.md` - Architecture rules
- `.claude/rules/security.md` - Security standards

## Quick Reference

| Command | Scope | Model | Purpose |
|---------|-------|-------|---------|
| `/arch-check` | Architecture | Sonnet | Fast architecture check |
| `/dedup` | Code quality | Sonnet | Find duplicates |
| **`/code-review`** | **Comprehensive** | **Opus** | **Full review (all aspects)** |

---

**💡 Remember**: This command **delegates** to the code-reviewer agent (Opus). See `.claude/agents/code-reviewer.md` for implementation details.

**🎯 Use Case**: Run before creating PR or deploying to production
