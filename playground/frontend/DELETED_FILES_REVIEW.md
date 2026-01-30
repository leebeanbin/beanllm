# 삭제된 파일 재검토 - 향후 사용 계획

## ⚠️ 향후 사용 가능성이 있는 파일들

### 🔴 높은 우선순위 (복구 고려)

#### 1. **use-file-upload.tsx** 🔴
**이유**: 
- `CHAT_IMPROVEMENT_PLANS/05_DOCUMENT_UPLOAD.md`에서 파일 업로드 UI가 **미구현** 상태
- Phase 2 (높음 우선순위)에 포함됨
- 문서 업로드 자동 처리 기능에 필요

**계획된 사용**:
```typescript
// 05_DOCUMENT_UPLOAD.md 참조
const handleFileUpload = async (files: FileList) => {
  // 파일 업로드 및 RAG 인덱싱
};
```

**결정**: ✅ **복구 권장** - Phase 2 구현 시 필요

---

#### 2. **SessionList.tsx + useSessionManager.ts** 🔴
**이유**:
- `CHAT_IMPROVEMENT_PLANS/04_SESSION_RAG.md`에서 세션별 RAG 관리 계획
- `CHAT_IMPROVEMENT_PLANS/13_DB_OPTIMIZATION.md`에서 세션 목록 최적화 계획
- 세션 관리 기능에 필요

**계획된 사용**:
- 세션별 RAG 자동 관리
- 세션 목록 표시 및 관리
- 세션별 문서 관리

**결정**: ✅ **복구 권장** - 세션 관리 기능 구현 시 필요

---

### 🟡 중간 우선순위 (선택적 복구)

#### 3. **OnboardingGuide.tsx** 🟡
**이유**:
- 사용자 온보딩은 일반적으로 유용한 기능
- 하지만 현재 계획 문서에 명시적으로 언급되지 않음

**결정**: ⚠️ **선택적** - 필요 시 재구현 가능

---

#### 4. **AgenticIntentDisplay.tsx** 🟡
**이유**:
- `CHAT_IMPROVEMENT_PLANS/07_INTENT_CLASSIFIER.md`에서 Intent Classifier 개선 계획
- 에이전트 의도 표시는 유용할 수 있음
- 하지만 현재 구현 계획에 명시되지 않음

**결정**: ⚠️ **선택적** - Intent Classifier 개선 시 필요할 수 있음

---

#### 5. **Thread.tsx** 🟡
**이유**:
- LangGraph SDK 통합이 계획되어 있다면 필요
- 하지만 현재 명시적인 계획 없음

**결정**: ⚠️ **선택적** - LangGraph 통합 시 필요할 수 있음

---

#### 6. **useMediaQuery.tsx** 🟡
**이유**:
- 반응형 디자인에 유용
- 하지만 현재는 직접 구현으로 대체 가능

**결정**: ⚠️ **선택적** - 필요 시 재구현 가능 (간단함)

---

### 🟢 낮은 우선순위 (복구 불필요)

#### 7. **ChatSettingsPopover.tsx** 🟢
- ✅ InfoPanel에 완전히 통합됨
- 복구 불필요

#### 8. **DocumentPreviewSidebar.tsx** 🟢
- ✅ InfoPanel의 Documents 탭으로 대체됨
- 복구 불필요

#### 9. **DocumentPropertiesSidebar.tsx** 🟢
- ✅ InfoPanel에 통합됨
- 복구 불필요

#### 10. **GoogleExportMenu.tsx** 🟢
- ✅ InfoPanel의 Settings 탭에 통합됨
- 복구 불필요

#### 11. **ModelSettingsPanel.tsx + ParameterTooltip.tsx** 🟢
- ✅ InfoPanel의 Settings 탭에 통합됨
- 복구 불필요

#### 12. **Icons (ChatIcon, github, langgraph)** 🟢
- lucide-react 또는 다른 아이콘으로 대체 가능
- 복구 불필요

---

## 📋 복구 권장 사항

### 즉시 복구 (Phase 2 구현 시 필요)
1. ✅ **use-file-upload.tsx** - 파일 업로드 기능
2. ✅ **SessionList.tsx** - 세션 관리 UI
3. ✅ **useSessionManager.ts** - 세션 관리 로직

### 선택적 복구 (필요 시)
4. ⚠️ **OnboardingGuide.tsx** - 사용자 온보딩
5. ⚠️ **AgenticIntentDisplay.tsx** - Intent 표시
6. ⚠️ **Thread.tsx** - LangGraph 통합
7. ⚠️ **useMediaQuery.tsx** - 반응형 디자인

---

## 🎯 결론

**즉시 복구 권장**: 3개 파일
- `use-file-upload.tsx`
- `SessionList.tsx`
- `useSessionManager.ts`

**이유**: Phase 2 (높음 우선순위) 구현 계획에 포함되어 있음

**나머지**: 필요 시 재구현 가능하거나 계획에 명시되지 않음

---

**검토 날짜**: 2025-01-24
**상태**: 복구 권장 파일 식별 완료
