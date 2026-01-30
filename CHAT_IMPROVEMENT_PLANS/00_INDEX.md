# Chat 개선 계획 인덱스

## 📋 개요

**⚠️ 이 문서는 나중에 구현할 기능들의 인덱스입니다**

**즉시 구현**: [README.md](./README.md) 참고 (3개 문서만 읽기)

**이 문서**: Phase 0-8 기능별 개선 계획 (나중에 구현)

---

## 📚 문서 구조

### Phase 0: 현재 구현 활용 (즉시 가능)
- **[01_CURRENT_IMPLEMENTATION.md](./01_CURRENT_IMPLEMENTATION.md)**: 이미 구현된 기능 활용
  - 메모리 시스템 통합
  - 진행 상황 추적 개선
  - 병렬 처리 진행 상황 표시

### Phase 1: 기본 구조
- **[02_AGENTIC_MODE.md](./02_AGENTIC_MODE.md)**: 기본 모드를 Agentic으로 변경
- **[03_CONTEXT_MANAGEMENT.md](./03_CONTEXT_MANAGEMENT.md)**: 컨텍스트 정리 및 메모리 관리
  - 요약 전략 (어떻게 요약할지)
  - 저장 전략 (어떻게 저장할지)
  - 전달 전략 (어떻게 전달할지)

### Phase 2: RAG 및 문서
- **[04_SESSION_RAG.md](./04_SESSION_RAG.md)**: 세션별 RAG 자동 관리
- **[05_DOCUMENT_UPLOAD.md](./05_DOCUMENT_UPLOAD.md)**: 문서 업로드 자동 처리

### Phase 3: 특화 기능
- **[06_SPECIAL_FEATURES.md](./06_SPECIAL_FEATURES.md)**: 특화 기능 처리 전략
- **[07_INTENT_CLASSIFIER.md](./07_INTENT_CLASSIFIER.md)**: Intent Classifier 개선
  - 쿼리 재구성 (사용자 피드백 기반)
  - 프롬프트 재구성 (동적 구성)
  - Ensemble Prompting (GenQREnsemble 방식)

### Phase 4: 멀티모달 및 검색
- **[08_MULTIMODAL_CONTEXT.md](./08_MULTIMODAL_CONTEXT.md)**: 멀티모달 컨텍스트 관리
- **[09_SEARCH_ENGINE.md](./09_SEARCH_ENGINE.md)**: 검색엔진 통합

### Phase 5: 클라우드 서비스
- **[10_CLOUD_SERVICES.md](./10_CLOUD_SERVICES.md)**: 클라우드 서비스 연동 (Google Sheets, Notion, Airtable 등)

### Phase 6: UI/UX
- **[11_UI_UX.md](./11_UI_UX.md)**: UI/UX 개선 방안

### Phase 7: MCP 통합
- **[12_MCP_INTEGRATION.md](./12_MCP_INTEGRATION.md)**: 오픈소스 MCP 활용

### Phase 8: DB 최적화
- **[13_DB_OPTIMIZATION.md](./13_DB_OPTIMIZATION.md)**: 데이터베이스 인덱싱 및 최적화 파이프라인

### Phase 9: 아키텍처 정리
- **[14_SEARCH_ARCHITECTURE.md](./14_SEARCH_ARCHITECTURE.md)**: 검색 시스템 구조 및 Playground 기능 활용 현황
- **[15_ARCHITECTURE_REVIEW.md](./15_ARCHITECTURE_REVIEW.md)**: 아키텍처 재검토 및 최종 픽스 방안 ⭐

### Phase 10: 고급 기능
- **[16_PLAN_MODE.md](./16_PLAN_MODE.md)**: Plan 모드 (Claude 스타일 계획 검토) ⭐
- **[17_VISUAL_WORKFLOW.md](./17_VISUAL_WORKFLOW.md)**: 시각적 워크플로우 구성 (n8n 스타일) ⭐

### Phase 10.0: MCP 서버 통합 (완료 ✅)
- **MCP Client Service** (2025-01-25): `mcp_client_service.py` 생성, 편의 메서드 구현
- **Orchestrator 통합** (2025-01-25): 7개 핸들러 MCP Client 사용으로 변경
- **결과**: Single Source of Truth 달성, 중복 코드 제거

### Phase 10.5: 코드 정리 및 구조 개선 (완료 ✅)
- **코드 정리** (2025-01-24): 중복 엔드포인트 제거, import 정리, 주석 정리
- **구조 개선** (2025-01-24): 파일 이동, 디렉토리 정리, 문서화
- **결과**: `main.py` 57% 감소 (2,704줄 → 1,161줄), 루트 파일 33% 감소

### Phase 10.6: main.py 스키마 정리 (완료 ✅)
- **스키마 분리** (2025-01-25): 22개 Pydantic 모델을 schemas/로 이동
- **결과**: `main.py` 994줄로 감소

---

## 🎯 우선순위

1. **Phase 10** (완료 ✅): MCP 서버를 통한 중앙 관리
   - MCP Client Service 생성
   - Orchestrator MCP Client 통합
   - 단일 진실의 원천 확보
   - **상태**: 완료 (2025-01-25)
2. **Phase 10.5** (완료 ✅): 코드 정리 및 구조 개선
   - 중복 엔드포인트 제거 (11개)
   - 파일 구조 정리 (scripts/, docs/)
   - 의존성 관리 정리 (Poetry)
   - **상태**: 완료 (2025-01-24)
3. **Phase 10.6** (완료 ✅): main.py 스키마 정리
   - 22개 Pydantic 모델 schemas/로 이동
   - **상태**: 완료 (2025-01-25)
4. **Phase 0** (완료 ✅): 현재 구현 활용
   - ContextManager 생성 (beanllm 메모리 시스템)
   - mcp_streaming.py 삭제 (레거시 정리)
   - 병렬 처리 진행 상황 개선 (execute_parallel)
   - **상태**: 완료 (2025-01-26)
5. **Phase 1** (다음 권장 ⭐): 기본 모드 변경, 컨텍스트 관리
6. **Phase 2** (높음): 세션별 RAG, 문서 업로드
7. **Phase 3** (중간): 특화 기능, Intent Classifier
8. **Phase 4** (중간): 멀티모달, 검색엔진
9. **Phase 5** (낮음): 클라우드 서비스
10. **Phase 6** (낮음): UI/UX
11. **Phase 7** (선택): MCP 통합
12. **Phase 8** (중간): DB 최적화 파이프라인
13. **Phase 9** (참고): 아키텍처 정리 문서
14. **Phase 10** (높음 ⭐): 고급 기능
    - Plan 모드 (계획 검토 및 승인)
    - 시각적 워크플로우 구성 (n8n 스타일)

---

## 📝 사용 방법

### Claude Code에게 작업 위임 시

1. **필수 읽기**:
   - [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) ⭐ (마스터 가이드)
   - [15_ARCHITECTURE_REVIEW.md](./15_ARCHITECTURE_REVIEW.md) (아키텍처 재검토)
   - [IMPLEMENTATION_STATUS_SUMMARY.md](./IMPLEMENTATION_STATUS_SUMMARY.md) (구현 상태 요약)

2. **우선순위**:
   - Phase 10 (완료 ✅): MCP 서버를 통한 중앙 관리
   - Phase 3 (높음): QueryRefiner & PromptBuilder (다른 Phase 의존성)
   - Phase 2 (높음): SessionRAGService, 파일 업로드
   - Phase 0-9: 기능별 개선

3. **구현 순서**:
   - 각 Phase 문서의 "구현 체크리스트 및 상태" 섹션 확인
   - ✅ 구현됨 항목은 스킵
   - ❌ 미구현 항목의 "구현 방향" 및 "방법" 참조하여 구현

### 일반 사용

1. 각 문서는 독립적으로 검토 가능
2. 필요 없는 기능은 해당 문서만 삭제/무시
3. 우선순위에 따라 순차적으로 구현
4. 각 문서는 구현 전략, 코드 예시, 체크리스트 포함
