# Claude Code 모델 선택 가이드

**목적**: 작업 특성에 따라 최적의 Claude 모델을 선택하여 비용 효율성과 품질을 동시에 달성

## 모델 개요

| 모델 | 가격 (1M tokens) | 특징 | 사용 사례 |
|------|------------------|------|-----------|
| **Opus** | Input: $15, Output: $75 | 최고 품질, 복잡한 추론 | 코드 리뷰, 보안 분석 |
| **Sonnet** | Input: $3, Output: $15 | 균형잡힌 성능/가격 | 일반 코딩, 리팩토링 |
| **Haiku** | Input: $0.25, Output: $1.25 | 빠르고 저렴 | 문서 작성, 간단한 수정 |

## 작업별 모델 선택

### 1. 코드 리뷰 및 보안 분석 → **Opus**

**왜 Opus?**
- 보안 취약점 심층 분석 (XSS, SQL Injection, API 키 노출)
- 성능 병목 지점 정확한 파악
- 복잡한 아키텍처 패턴 이해
- 미묘한 버그 감지 (경쟁 조건, 메모리 누수)

**비용 대비 가치**:
- 보안 취약점 1개 = 잠재적 손실 ≫ Opus 비용
- 성능 최적화 = 장기적 비용 절감
- 높은 품질의 리뷰 = 코드 베이스 건강성 향상

**사용 예시**:
```bash
# 전체 PR 리뷰
/code-review

# 특정 파일 심층 분석
/code-review --file src/beanllm/facade/core/client_facade.py --verbose
```

**예상 비용** (1,000줄 코드 리뷰):
- Input: ~3,000 tokens = $0.045
- Output: ~5,000 tokens = $0.375
- **총 비용: ~$0.42**

### 2. 리팩토링 및 최적화 → **Sonnet**

**왜 Sonnet?**
- Clean Architecture 위반 자동 수정 가능
- 중복 코드 85-90% 감소 달성
- 알고리즘 최적화 (O(n) → O(1)) 정확히 수행
- 데코레이터 패턴 자동 적용

**비용 대비 가치**:
- Opus 대비 5배 저렴
- 품질은 Opus의 90% 수준
- 대부분의 리팩토링 작업에 충분

**사용 예시**:
```bash
# Clean Architecture 위반 자동 수정
/arch-fix

# 중복 코드 제거
/dedup

# 성능 최적화
/optimize
```

**예상 비용** (1,000줄 리팩토링):
- Input: ~3,000 tokens = $0.009
- Output: ~10,000 tokens = $0.15
- **총 비용: ~$0.16** (Opus 대비 62% 저렴)

### 3. 일반 코딩 (기능 구현, 버그 수정) → **Sonnet**

**왜 Sonnet?**
- 복잡한 비즈니스 로직 구현 가능
- TDD 워크플로우 완벽 지원
- 타입 힌트 + Docstring 자동 생성
- 에러 처리 패턴 적용

**비용 대비 가치**:
- 일반 코딩에는 Opus 불필요
- Sonnet으로 충분한 품질
- Haiku는 복잡한 로직에 부족

**사용 예시**:
```bash
# 기능 계획 수립
/plan "HyDE query expansion 추가"

# 테스트 자동 생성
/test-gen --path src/beanllm/domain/retrieval/hyde.py

# 아키텍처 검증
/arch-check
```

**예상 비용** (새 기능 500줄):
- Input: ~2,000 tokens = $0.006
- Output: ~5,000 tokens = $0.075
- **총 비용: ~$0.08**

### 4. 테스트 작성 → **Sonnet**

**왜 Sonnet?**
- 엣지 케이스 발견 능력
- 적절한 Mock/Fixture 생성
- pytest parametrize 활용
- 80% 커버리지 달성

**사용 예시**:
```bash
# 단위 테스트 자동 생성
/test-gen --type unit --path src/beanllm/facade/core/client_facade.py

# 통합 테스트 생성
/test-gen --type integration
```

**예상 비용** (50개 테스트 생성):
- Input: ~1,000 tokens = $0.003
- Output: ~8,000 tokens = $0.12
- **총 비용: ~$0.12**

### 5. 문서 작성 → **Haiku**

**왜 Haiku?**
- 문서는 복잡한 추론 불필요
- 빠른 응답 시간
- 10배 저렴한 비용

**비용 대비 가치**:
- Sonnet 대비 12배 저렴
- 문서 품질은 충분함
- 대량 문서 작업에 최적

**사용 예시**:
```bash
# API 문서 업데이트
/update-docs --path src/beanllm/facade/core/

# README 작성
/write-readme

# Docstring 추가
/add-docstrings --path src/beanllm/domain/
```

**예상 비용** (10페이지 문서):
- Input: ~500 tokens = $0.000125
- Output: ~10,000 tokens = $0.0125
- **총 비용: ~$0.013** (Sonnet 대비 92% 저렴)

### 6. 간단한 수정 (오타, 포매팅) → **Haiku**

**왜 Haiku?**
- 단순 작업에는 Haiku로 충분
- 즉각적인 응답
- 비용 거의 없음

**사용 예시**:
```bash
# Black/Ruff 자동 포매팅
/build-fix

# 오타 수정
# (간단한 수정은 직접 명령)
```

**예상 비용** (100개 파일 포매팅):
- Input: ~100 tokens = $0.000025
- Output: ~1,000 tokens = $0.00125
- **총 비용: ~$0.001** (거의 무료)

## 비용 최적화 전략

### 1. 단계적 모델 사용

```
1단계: Haiku로 초기 분석
   ↓
2단계: Sonnet으로 구현
   ↓
3단계: Opus로 최종 리뷰
```

**예시**: RAG 기능 추가
1. **Haiku**: 요구사항 정리, 파일 목록 작성 (~$0.01)
2. **Sonnet**: 코드 구현, 테스트 작성 (~$0.20)
3. **Opus**: 보안/성능 리뷰 (~$0.50)
**총 비용: ~$0.71**

### 2. Batch 작업

여러 파일을 한 번에 처리하여 컨텍스트 재사용:

```bash
# ❌ Bad: 각 파일마다 새 요청
/test-gen --path file1.py  # $0.12
/test-gen --path file2.py  # $0.12
/test-gen --path file3.py  # $0.12
# 총 비용: $0.36

# ✅ Good: 한 번에 처리
/test-gen --path "src/beanllm/domain/**/*.py"
# 총 비용: $0.15 (컨텍스트 재사용으로 58% 절감)
```

### 3. 캐싱 활용

반복되는 작업은 캐싱:

```bash
# 첫 번째 요청: Full cost
/code-review --file client.py  # $0.42

# 같은 파일 재리뷰: 50% 할인 (프롬프트 캐싱)
/code-review --file client.py  # $0.21
```

### 4. Subagent 도구 제한

Subagents는 필요한 도구만 허용하여 컨텍스트 절약:

```markdown
# code-reviewer agent
**허용 도구**: Read, Grep, Bash (git)
**금지 도구**: Edit, Write
```

→ 컨텍스트 30% 절약

## 실전 워크플로우 예시

### 예시 1: 새 기능 추가 (HyDE Query Expansion)

```bash
# 1. 계획 수립 (Sonnet)
/plan "HyDE query expansion 추가"
# 비용: $0.05

# 2. 테스트 작성 (Sonnet)
/test-gen --path src/beanllm/domain/retrieval/hyde.py
# 비용: $0.12

# 3. 구현 (Sonnet)
# (직접 코딩 + Claude Code 도움)
# 비용: $0.20

# 4. 아키텍처 검증 (Sonnet)
/arch-check
# 비용: $0.08

# 5. 중복 코드 제거 (Sonnet)
/dedup
# 비용: $0.10

# 6. 최종 리뷰 (Opus)
/code-review
# 비용: $0.50

총 비용: $1.05
```

### 예시 2: 버그 수정 (Rate Limit 에러)

```bash
# 1. 버그 분석 (Sonnet)
# (로그 확인, 재현)
# 비용: $0.05

# 2. 테스트 작성 (Sonnet)
/test-gen --type unit
# 비용: $0.08

# 3. 수정 (Sonnet)
# (에러 처리 추가)
# 비용: $0.10

# 4. 빌드 체크 (Haiku)
/build-fix
# 비용: $0.01

총 비용: $0.24
```

### 예시 3: 문서 업데이트

```bash
# 1. API 문서 업데이트 (Haiku)
/update-docs --path src/beanllm/facade/
# 비용: $0.02

# 2. README 업데이트 (Haiku)
# (새 기능 추가)
# 비용: $0.01

# 3. Changelog 생성 (Haiku)
# (git log 기반)
# 비용: $0.005

총 비용: $0.035
```

## 월간 예상 비용

**가정**:
- 주 5일 개발
- 하루 평균: 기능 1개 + 버그 수정 2개 + 문서 업데이트

**일일 비용**:
- 기능 추가 1개: $1.05
- 버그 수정 2개: $0.48
- 문서 업데이트: $0.04
**일일 총: $1.57**

**월간 비용** (20일):
$1.57 × 20 = **$31.40**

**최적화 후** (Batch, 캐싱 활용):
**월간 비용: ~$20-25**

## 결론

### 황금 규칙

1. **보안/성능 리뷰**: 무조건 **Opus** (비용 < 리스크)
2. **일반 코딩**: **Sonnet** (90% 품질, 80% 저렴)
3. **문서 작성**: **Haiku** (충분한 품질, 10배 저렴)

### ROI (Return on Investment)

| 작업 | 수동 시간 | Claude 시간 | 시간 절감 | 비용 | ROI |
|------|-----------|-------------|-----------|------|-----|
| 코드 리뷰 | 2시간 | 5분 | 115분 | $0.50 | **14,000%** |
| 테스트 작성 | 1시간 | 2분 | 58분 | $0.12 | **29,000%** |
| 리팩토링 | 3시간 | 10분 | 170분 | $0.20 | **51,000%** |
| 문서 작성 | 30분 | 1분 | 29분 | $0.02 | **87,000%** |

*ROI 계산: (시간 절감 × 시급 $60) / 비용*

### 최종 권장사항

- **프로토타입**: Sonnet 위주 사용
- **프로덕션**: Opus로 최종 리뷰 필수
- **문서**: 항상 Haiku 사용
- **학습**: Opus로 복잡한 패턴 이해

**월 $25 투자로 200+ 시간 절약 가능**

## 관련 문서

- `CLAUDE.md` - 프로젝트 컨텍스트
- `.claude/README.md` - Claude Code 설정 가이드
- `.claude/agents/` - 각 에이전트별 모델 설정
