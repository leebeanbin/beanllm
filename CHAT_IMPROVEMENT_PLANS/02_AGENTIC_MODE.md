# 기본 모드를 Agentic으로 변경

## 🎯 목표

ChatGPT/Claude처럼 단일 채팅 인터페이스에서 자동으로 기능을 선택하고 실행

---

## 📊 현재 문제점

- ❌ FeatureSelector로 수동 선택 필요
- ❌ 기본 모드가 "chat" (Agentic이 아님)
- ❌ 사용자가 매번 기능을 선택해야 함

---

## ✅ 개선 방안

### 1. 기본 모드 변경

#### 변경 사항
```typescript
// playground/frontend/src/app/chat/page.tsx
// Before
const [selectedFeature, setSelectedFeature] = useState<FeatureMode>("chat");

// After
// 항상 Agentic 모드 사용 (FeatureSelector 제거 또는 특화 기능만)
const [selectedFeature, setSelectedFeature] = useState<FeatureMode>("agentic");
```

#### 플로우 변경
```
사용자 입력
    ↓
항상 /api/chat/agentic 호출
    ↓
자동 Intent 분류
    ↓
자동 도구 선택 및 실행
```

---

### 2. FeatureSelector 변경

#### 옵션 A: 완전 제거
```typescript
// FeatureSelector 컴포넌트 제거
// 항상 Agentic 모드만 사용
```

#### 옵션 B: 특화 기능만 선택 (권장)
```typescript
// 특화 기능만 선택 가능
const [showAdvanced, setShowAdvanced] = useState(false);

<div>
  {/* 일반 기능은 자동 감지 */}
  <p>자동 모드 (기본)</p>
  
  {/* 특화 기능만 선택 */}
  <button onClick={() => setShowAdvanced(!showAdvanced)}>
    ⚙️ 특화 기능 ▼
  </button>
  {showAdvanced && (
    <div>
      <button onClick={() => setIntent("multi_agent")}>
        🤝 Multi-Agent
      </button>
      <button onClick={() => setIntent("knowledge_graph")}>
        📊 Knowledge Graph
      </button>
      {/* ... */}
    </div>
  )}
</div>
```

---

## 📋 구현 체크리스트 및 상태

### ✅ 구현됨
- [x] 프론트엔드 기본 모드를 Agentic으로 변경 ✅ (2025-01-26)
- [x] 항상 `/api/chat/agentic` 사용 ✅ (이미 백엔드 구현됨)
- [x] 자동 Intent 분류 확인 ✅ (IntentClassifier 이미 구현됨)
- [x] Orchestrator 구현 (`services/orchestrator.py`)
- [x] Tool Registry 구현 (`services/tool_registry.py`)

### ❌ 미구현
- [ ] **FeatureSelector를 "특화 기능 선택기"로 변경 또는 제거**
  - **위치**: `playground/frontend/src/app/chat/page.tsx`
  - **구현 방향**:
    - 옵션 A: 완전 제거 (권장) - 항상 Agentic 모드만 사용
    - 옵션 B: 특화 기능만 선택 가능 - Multi-Agent, Knowledge Graph 등만
  - **방법**:
    ```typescript
    // 옵션 A: 제거
    // FeatureSelector 컴포넌트 삭제
    
    // 옵션 B: 특화 기능만
    const [showAdvanced, setShowAdvanced] = useState(false);
    const advancedFeatures = ["multi_agent", "knowledge_graph", "evaluation"];
    
    {showAdvanced && (
      <div>
        {advancedFeatures.map(feature => (
          <button onClick={() => setForceIntent(feature)}>
            {feature}
          </button>
        ))}
      </div>
    )}
    ```
  - **통합**: `force_intent` 파라미터로 특화 기능 강제 지정

---

## 🎯 우선순위

**높음**: 즉시 구현 가능, 사용자 경험 개선
