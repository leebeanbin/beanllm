# Chat 개선 계획 문서

## 🚀 시작하기

### Claude Code에게 작업 위임 시

**필수 읽기** (3개 문서만, 45분):
1. **[QUICK_START.md](./QUICK_START.md)** ⚡ (5분)
2. **[CURRENT_STATE_ANALYSIS.md](./CURRENT_STATE_ANALYSIS.md)** (10분)
3. **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** ⭐ (30분)

**그 다음**: 구현 시작

---

## 📚 문서 구조 (간소화)

### 핵심 문서 (즉시 구현용) - 5개

1. **[QUICK_START.md](./QUICK_START.md)** ⚡ - 5분 빠른 시작
2. **[CURRENT_STATE_ANALYSIS.md](./CURRENT_STATE_ANALYSIS.md)** - 현재 상태 분석
3. **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** ⭐ - 구현 가이드 (마스터)
4. **[IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)** - 구현 상태
5. **[IMPLEMENTATION_STATUS_SUMMARY.md](./IMPLEMENTATION_STATUS_SUMMARY.md)** - 구현 상태 요약 (각 Phase별 진행률)

### 참고 문서 (선택) - 3개

6. **[15_ARCHITECTURE_REVIEW.md](./15_ARCHITECTURE_REVIEW.md)** - 아키텍처 재검토
7. **[00_INDEX.md](./00_INDEX.md)** - 전체 인덱스 (Phase 0-8 문서 목록)
8. **[14_SEARCH_ARCHITECTURE.md](./14_SEARCH_ARCHITECTURE.md)** - 검색 시스템 구조

### 기능별 개선 계획 (나중에 구현) - 15개

**Phase 0-10 문서들** (01-17번):
- 필요시 [00_INDEX.md](./00_INDEX.md) 참고
- Phase 10 (Plan 모드, 시각적 워크플로우) 우선순위 높음 ⭐

---

## 🎯 우선순위

### 완료된 작업 (Phase 10) ✅
**MCP 서버를 통한 중앙 관리** (2025-01-25)
- ✅ MCP Client Service 생성 (`mcp_client_service.py`)
- ✅ Orchestrator MCP Client 통합 (7개 핸들러)
- ✅ Facade 직접 호출 제거 (Chat 제외)
- 결과: Single Source of Truth 달성

### 완료된 작업 (Phase 10.5) ✅
**코드 정리 및 구조 개선** (2025-01-24)
- ✅ 중복 엔드포인트 제거 (11개)
- ✅ 파일 구조 정리 (scripts/, docs/)
- ✅ 의존성 관리 정리 (Poetry)
- ✅ 문서화 완료
- 결과: `main.py` 57% 감소, 루트 파일 33% 감소

### 완료된 작업 (Phase 10.6) ✅
**main.py 스키마 정리** (2025-01-25)
- ✅ 22개 Pydantic 모델 schemas/로 이동
- ✅ main.py 994줄로 감소

### 나중에 구현 (Phase 0-8)
- 기능별 개선 계획
- 필요시 [00_INDEX.md](./00_INDEX.md) 참고

---

## 📋 사용 방법

### Claude Code에게 작업 위임 시

**3개 문서만 읽기**:
1. [QUICK_START.md](./QUICK_START.md) ⚡
2. [CURRENT_STATE_ANALYSIS.md](./CURRENT_STATE_ANALYSIS.md)
3. [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) ⭐

**그 다음**: 구현 시작

### 기능별 개선 계획 확인

- [00_INDEX.md](./00_INDEX.md)에서 전체 계획 확인
- 필요시 Phase별 문서 읽기

---

## 💡 핵심 원칙

1. **Single Source of Truth**: MCP 서버의 tools가 유일한 구현
2. **Clean Architecture**: Facade 직접 호출 금지
3. **점진적 개선**: 우선순위에 따라 단계별 구현
