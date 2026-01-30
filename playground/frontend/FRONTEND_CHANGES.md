# 프론트엔드 변경사항 정리

> playground/frontend 기준 변경·삭제·추가 내역 요약

---

## 1. 수정된 파일 (Modified)

### 앱 레이아웃·페이지
| 파일 | 변경 개요 |
|------|-----------|
| `src/app/layout.tsx` | 레이아웃 구조/메타 조정 |
| `src/app/page.tsx` | 랜딩/홈 페이지 구조·콘텐츠 변경 |
| `src/app/chat/page.tsx` | 채팅 페이지 대폭 수정 (통합 UI, InfoPanel·모달·에러 문구 영어화, 패널 토글 상단 탭 바로 이동 등) |
| `src/app/globals.css` | 글로벌 스타일·토큰 조정 |

### 컴포넌트
| 파일 | 변경 개요 |
|------|-----------|
| `src/components/Navigation.tsx` | 네비 항목·라벨 영어화, aria-label 등 접근성 |
| `src/components/PageLayout.tsx` | 푸터에서 Settings 링크 제거, `headerTrailing` 슬롯 추가 (탭 바 오른쪽 영역) |
| `src/components/FeatureSelector.tsx` | 기능 선택 UI· copy 정리 |
| `src/components/GoogleServiceSelector.tsx` | Google 서비스 선택 UI· copy 정리 |
| `src/components/ModelSelectorSimple.tsx` | 모델 선택·다운로드 UI, 토스트/알림/라벨 영어화 |
| `src/components/ThinkMode.tsx` | Think 모드 표시 방식 정리 |
| `src/components/ToolCallDisplay.tsx` | 툴 콜 결과 표시 (이메일 전송·Drive 목록 등), 라벨 영어화 |
| `src/components/Visualization.tsx` | 시각화 컴포넌트 정리 |

### UI 프리미티브
| 파일 | 변경 개요 |
|------|-----------|
| `src/components/ui/tooltip.tsx` | 툴팁 동작·스타일 조정 |

### 타입·설정
| 파일 | 변경 개요 |
|------|-----------|
| `src/types/chat.ts` | 채팅 관련 타입 필드 추가/정리 |
| `tsconfig.json` | TS 빌드 옵션 조정 |
| `package.json` | 의존성 추가/변경 |
| `pnpm-lock.yaml` | lockfile 갱신 |

---

## 2. 삭제된 파일 (Deleted)

### 페이지
| 파일 | 비고 |
|------|------|
| `src/app/chat/page_with_sessions.tsx` | 세션 기반 채팅 페이지 제거, `chat/page.tsx`로 통합 |

### 채팅·세션용 컴포넌트
| 파일 | 비고 |
|------|------|
| `src/components/AssistantInputPanel.tsx` | 어시스턴트 입력 패널 제거 |
| `src/components/AssistantSelector.tsx` | 어시스턴트 선택 UI 제거 |
| `src/components/OnboardingGuide.tsx` | 온보딩 가이드 제거 |
| `src/components/ModelSelector.tsx` | 기존 모델 선택기 제거 → ModelSelectorSimple로 통일 |
| `src/components/ModelSettingsPanel.tsx` | 모델 설정 패널 제거 (InfoPanel 등으로 대체) |
| `src/components/ParameterTooltip.tsx` | 파라미터 툴팁 제거 |

### 스레드/히스토리 UI (thread)
| 파일 | 비고 |
|------|------|
| `src/components/thread/index.tsx` | 스레드 진입점 제거 |
| `src/components/thread/artifact.tsx` | 아티팩트 렌더 제거 |
| `src/components/thread/history/index.tsx` | 히스토리 목록 제거 |
| `src/components/thread/ContentBlocksPreview.tsx` | 콘텐츠 블록 미리보기 제거 |
| `src/components/thread/MultimodalPreview.tsx` | 멀티모달 미리보기 제거 |
| `src/components/thread/markdown-text.tsx` | 마크다운 전용 텍스트 제거 |
| `src/components/thread/markdown-styles.css` | 마크다운 전용 스타일 제거 |
| `src/components/thread/messages/ai.tsx` | AI 메시지 뷰 제거 |
| `src/components/thread/messages/human.tsx` | Human 메시지 뷰 제거 |
| `src/components/thread/messages/shared.tsx` | 공통 메시지 뷰 제거 |
| `src/components/thread/messages/tool-calls.tsx` | 툴 콜 메시지 뷰 제거 |
| `src/components/thread/messages/generic-interrupt.tsx` | 인터럽트 UI 제거 |
| `src/components/thread/syntax-highlighter.tsx` | 문법 하이라이트 제거 |
| `src/components/thread/tooltip-icon-button.tsx` | 툴팁 아이콘 버튼 제거 |
| `src/components/thread/utils.ts` | 스레드 유틸 제거 |

### 아이콘
| 파일 | 비고 |
|------|------|
| `src/components/icons/ChatIcon.tsx` | 채팅 아이콘 제거 |
| `src/components/icons/github.tsx` | 깃헙 아이콘 제거 |
| `src/components/icons/langgraph.tsx` | 랭그래프 아이콘 제거 |

### 프로바이더·훅
| 파일 | 비고 |
|------|------|
| `src/providers/Stream.tsx` | 스트림 프로바이더 제거 |
| `src/providers/Thread.tsx` | 스레드 프로바이더 제거 |
| `src/hooks/useMediaQuery.tsx` | 미디어 쿼리 훅 제거 |

---

## 3. 추가된 파일 (New / Untracked)

### 페이지·라우트
| 파일/경로 | 개요 |
|-----------|------|
| `src/app/settings/page.tsx` | 설정 페이지 (API 키, Google, 시스템 정보) |
| `src/app/monitoring/page.tsx` | 모니터링 페이지 |

### 모달·패널
| 파일 | 개요 |
|------|------|
| `src/components/ApiKeyModal.tsx` | API 키 관리 모달 (일괄 저장, .env import, 유효 입력 초록 표시, 검증 버튼 제거·저장 시 자동 검증, UI 영어화) |
| `src/components/GoogleConnectModal.tsx` | Google 연동 모달 (OAuth 설정·연결 해제, 유효 입력 초록 표시, UI 영어화) |
| `src/components/GoogleOAuthCard.tsx` | Google OAuth 카드 (설정·연결 상태·서비스 선택, UI 영어화) |
| `src/components/InfoPanel.tsx` | 우측 정보 패널 (Quickstart/Models/Session/Settings 탭, 접기/펼치기 부모 제어·탭 바 trailing으로 이동, 보더 제거·폭 조정) |
| `src/components/BrowserTabs.tsx` | 상단 탭 바 (Chat / Monitoring / Settings + trailing 슬롯) |
| `src/components/PackageInstallModal.tsx` | Provider SDK 설치 모달 |
| `src/components/ProviderWarning.tsx` | Provider 미설치 경고 배너 |

### 채팅·스트리밍 UI
| 파일 | 개요 |
|------|------|
| `src/components/StreamingText.tsx` | 스트리밍 텍스트 표시 |
| `src/components/TypingIndicator.tsx` | 타이핑/생성 중 인디케이터 |
| `src/components/FeatureBadge.tsx` | 기능 뱃지 (RAG, Agent 등) |

### UI 프리미티브 (shadcn/ui 스타일)
| 파일 | 개요 |
|------|------|
| `src/components/ui/dialog.tsx` | 다이얼로그 (모달 닫힐 때 body 정리 포함) |
| `src/components/ui/alert.tsx` | 알림 박스 |
| `src/components/ui/alert-dialog.tsx` | 확인 다이얼로그 |
| `src/components/ui/checkbox.tsx` | 체크박스 |
| `src/components/ui/dropdown-menu.tsx` | 드롭다운 메뉴 |
| `src/components/ui/popover.tsx` | 팝오버 |
| `src/components/ui/scroll-area.tsx` | 스크롤 영역 |
| `src/components/ui/select.tsx` | 셀렉트 |
| `src/components/ui/slider.tsx` | 슬라이더 |

### 설정·환경
| 파일 | 개요 |
|------|------|
| `src/app/../.env.local.example` | 프론트 환경변수 예시 |
| `tsconfig.tsbuildinfo` | TS 빌드 캐시 (보통 .gitignore 대상) |

### E2E
| 파일/경로 | 개요 |
|-----------|------|
| `e2e/chat.spec.ts` | 채팅 플로우 E2E 테스트 |

---

## 4. 주제별 요약

| 주제 | 변경 요약 |
|------|-----------|
| **레이아웃·내비게이션** | PageLayout 푸터 단순화, BrowserTabs + headerTrailing으로 탭·패널 토글 분리 |
| **채팅 화면** | 세션 페이지 제거 후 단일 chat/page.tsx, InfoPanel·모달·설정 접근 통합 |
| **설정·API 키** | ApiKeyModal(일괄 저장·import·자동 검증·영어 UI), 설정 페이지·Settings 탭 추가 |
| **Google 연동** | GoogleConnectModal·GoogleOAuthCard 추가, 입력 유효 시 초록 강조·영어 UI |
| **패널·탭** | InfoPanel 접기/펼치기 제어를 상단 탭 바 오른쪽으로 이동, 보더 제거·폭 조정 |
| **스레드 UI 제거** | thread/*, Stream/Thread 프로바이더, OnboardingGuide 등 세션/스레드 중심 UI 삭제 |
| **모델·기능 선택** | ModelSelectorSimple만 사용, FeatureBadge·FeatureSelector·ThinkMode 정리 |
| **다국어** | 사용자 대면 문구·토스트·라벨 등을 영어로 통일 |
| **UI 키트** | dialog, alert, select, dropdown, slider 등 공통 UI 컴포넌트 추가 |
| **모니터링·E2E** | monitoring 페이지, e2e/chat.spec.ts 추가 |

---

## 5. 숫자 요약

| 구분 | 개수 |
|------|------|
| 수정된 파일 | 18 |
| 삭제된 파일 | 27 |
| 추가된 파일/디렉터리 | 약 28 (md 제외 시 약 20) |
| diff 규모 (tracked만) | +2,151 / -6,230 lines |

---

## 6. 최근 동작·UX 개선 (이번 세션 반영)

| 항목 | 내용 |
|------|------|
| **모달 닫힌 뒤 동작** | `dialog.tsx`에서 `onCloseAutoFocus` 시 body의 `pointer-events`/`overflow`/`padding-right`/`inert` 초기화, 150ms 후 한 번 더 정리. ApiKeyModal·GoogleConnectModal에서도 `open === false`일 때 body 스타일 정리 유지. |
| **패널 접기/펼치기** | InfoPanel에 `isCollapsed?: boolean` prop 추가 → chat 페이지에서 `isCollapsed={isInfoPanelCollapsed}` 전달해 상단 state와 동기화. 토글 버튼은 **탭 바 오른쪽**(`BrowserTabs` trailing)에만 두고, 패널 내부 버튼 제거. |
| **API 키 모달** | 개별 저장 제거 → `getKeysToSave()` + `handleBatchSave()`로 일괄 저장. 유효한 입력란에 초록 테두리(`border-green-500/50` 등). 검증 버튼 제거, 저장 성공 시 해당 프로바이더 `/api/config/keys/{provider}/validate` 호출 후 목록 갱신. |
| **Google 모달** | GoogleConnectModal·GoogleOAuthCard에도 “유효하면 초록” 스타일 적용. |
| **영어화** | 사용자 대면 문구(토스트·라벨·버튼·placeholder·Badge·설명) 영어로 통일. |

---

*이 문서는 `playground/frontend` 기준 변경사항을 정리한 것입니다. MD 문서(*.md)는 제외했고, 필요 시 커밋 범위에서도 제외할 수 있습니다.*
