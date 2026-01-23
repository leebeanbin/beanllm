# TDD Workflow Skill

**자동 활성화**: "TDD", "test", "테스트" 키워드 감지 시
**모델**: sonnet

## Skill Description

Test-Driven Development (TDD) 방법론을 적용한 개발 워크플로우를 제공합니다. Red-Green-Refactor 사이클을 자동화하고, 80% 테스트 커버리지 달성을 지원합니다.

## TDD Cycle

```
1. Red (실패하는 테스트 작성)
   ↓
2. Green (최소한의 코드로 테스트 통과)
   ↓
3. Refactor (코드 개선)
   ↓
(반복)
```

## 워크플로우 파일

- `01-red.md` - Red 단계: 실패하는 테스트 작성
- `02-green.md` - Green 단계: 최소 구현
- `03-refactor.md` - Refactor 단계: 코드 개선
- `best-practices.md` - TDD 베스트 프랙티스

## 사용법

1. `/tdd` 커맨드로 TDD 워크플로우 시작
2. 각 단계별로 자동 안내
3. 테스트 커버리지 자동 추적

## 관련 문서

- `.claude/rules/testing.md` - 테스트 규칙
- `.claude/commands/test-gen.md` - 테스트 자동 생성
