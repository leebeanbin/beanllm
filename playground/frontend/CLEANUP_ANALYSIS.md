# Frontend 파일 정리 분석

## 📊 분석 결과

### ❌ 사용되지 않는 파일 (삭제 대상)

#### Components
1. **ChatSettingsPopover.tsx** ❌
   - 상태: InfoPanel의 Settings 탭에 통합됨
   - import: 없음
   - 삭제 가능

2. **DocumentPreviewSidebar.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

3. **DocumentPropertiesSidebar.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

4. **GoogleExportMenu.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

5. **OnboardingGuide.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

6. **AgenticIntentDisplay.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

7. **ModelSettingsPanel.tsx** ❌
   - 상태: 사용되지 않음 (ParameterTooltip만 사용하지만 ModelSettingsPanel 자체는 사용 안 됨)
   - import: 없음
   - 삭제 가능

8. **SessionList.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

#### Hooks
9. **use-file-upload.tsx** ❌
   - 상태: 사용되지 않음
   - import: 없음
   - 삭제 가능

10. **useMediaQuery.tsx** ❌
    - 상태: 사용되지 않음
    - import: 없음
    - 삭제 가능

#### Providers
11. **Thread.tsx** ❌
    - 상태: LangGraph SDK 타입만 import하지만 실제 사용 안 됨
    - import: 없음
    - 삭제 가능

#### Icons
12. **ChatIcon.tsx** ❌
    - 상태: 사용되지 않음
    - import: 없음
    - 삭제 가능

13. **github.tsx** ❌
    - 상태: 사용되지 않음
    - import: 없음
    - 삭제 가능

14. **langgraph.tsx** ❌
    - 상태: 사용되지 않음
    - import: 없음
    - 삭제 가능

---

### ✅ 사용되는 파일 (유지)

#### Components
- ✅ ApiKeyModal.tsx - settings/page.tsx에서 사용
- ✅ BrowserTabs.tsx - PageLayout.tsx에서 사용
- ✅ FeatureBadge.tsx - 사용됨
- ✅ FeatureSelector.tsx - 사용됨
- ✅ GoogleOAuthCard.tsx - settings/page.tsx에서 사용
- ✅ GoogleServiceSelector.tsx - 사용됨
- ✅ InfoPanel.tsx - chat/page.tsx에서 사용
- ✅ ModelSelectorSimple.tsx - 사용됨
- ✅ Navigation.tsx - layout.tsx에서 사용
- ✅ PageLayout.tsx - 사용됨
- ✅ ParameterTooltip.tsx - ModelSettingsPanel에서 사용 (하지만 ModelSettingsPanel이 사용 안 됨)
- ✅ ThinkMode.tsx - chat/page.tsx에서 사용
- ✅ ToolCallDisplay.tsx - chat/page.tsx에서 사용
- ✅ Visualization.tsx - chat/page.tsx에서 사용

#### Hooks
- ✅ useSessionManager.ts - SessionList에서 사용 (하지만 SessionList가 사용 안 됨)

#### Icons
- ✅ BeanIcon.tsx - Navigation.tsx에서 사용

---

## 🎯 정리 계획

### Phase 1: 명확히 사용되지 않는 파일 삭제
1. ChatSettingsPopover.tsx
2. DocumentPreviewSidebar.tsx
3. DocumentPropertiesSidebar.tsx
4. GoogleExportMenu.tsx
5. OnboardingGuide.tsx
6. AgenticIntentDisplay.tsx
7. ModelSettingsPanel.tsx (ParameterTooltip도 함께 확인)
8. SessionList.tsx
9. use-file-upload.tsx
10. useMediaQuery.tsx
11. Thread.tsx
12. ChatIcon.tsx
13. github.tsx
14. langgraph.tsx

### Phase 2: 의존성 확인
- ParameterTooltip.tsx: ModelSettingsPanel에서만 사용 → ModelSettingsPanel 삭제 시 함께 삭제 가능
- useSessionManager.ts: SessionList에서만 사용 → SessionList 삭제 시 함께 삭제 가능

---

## 📝 정리 후 예상 구조

```
src/
├── app/
│   ├── chat/page.tsx
│   ├── monitoring/page.tsx
│   ├── settings/page.tsx
│   └── ...
├── components/
│   ├── ApiKeyModal.tsx ✅
│   ├── BrowserTabs.tsx ✅
│   ├── FeatureBadge.tsx ✅
│   ├── FeatureSelector.tsx ✅
│   ├── GoogleOAuthCard.tsx ✅
│   ├── GoogleServiceSelector.tsx ✅
│   ├── InfoPanel.tsx ✅
│   ├── ModelSelectorSimple.tsx ✅
│   ├── Navigation.tsx ✅
│   ├── PageLayout.tsx ✅
│   ├── ThinkMode.tsx ✅
│   ├── ToolCallDisplay.tsx ✅
│   ├── Visualization.tsx ✅
│   ├── icons/
│   │   └── BeanIcon.tsx ✅
│   └── ui/ ✅
├── hooks/ (비어있음 또는 필요한 것만)
├── providers/ (비어있음)
└── types/
    └── chat.ts ✅
```

---

**분석 날짜**: 2025-01-24
**상태**: 정리 준비 완료
