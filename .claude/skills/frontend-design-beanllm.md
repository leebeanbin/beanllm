# Frontend Design for beanllm Playground

**마켓플레이스 스킬**: `frontend-design` (Anthropic Agent Skills)
**자동 활성화**: "UI 디자인", "컴포넌트 생성", "React component", "Tailwind", "playground/frontend" 키워드 감지 시
**모델**: sonnet

## Skill Description

beanllm playground (Next.js 15 + React 19 + Tailwind CSS)를 위한 독창적이고 프로덕션급 UI 컴포넌트를 생성합니다.

## beanllm Design System

### Core Aesthetic: **Technical Elegance**

**컨셉**: LLM 툴킷의 정교함과 데이터의 흐름을 시각화한 **기술적 미니멀리즘**

**디자인 원칙**:
- **정밀함**: 코드의 정확성을 UI에 반영
- **깊이감**: 레이어와 그림자로 아키텍처 표현
- **데이터 중심**: 정보 계층 명확하게 구분
- **유동성**: LLM 스트리밍의 흐름을 애니메이션으로

## Typography

### 금지된 폰트 (Generic AI Slop)
❌ Inter, Roboto, Arial, Space Grotesk, System UI

### 추천 폰트

**Display (헤더, 제목)**:
```css
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
/* 또는 */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
```

**Body (본문)**:
```css
@import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;500&display=swap');
/* 또는 */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&display=swap');
```

**Code (코드 블록)**:
```css
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');
```

## Color Palette

### 금지된 색상 조합
❌ Purple gradient on white background
❌ Default Tailwind purple (`purple-600`)
❌ Generic blue (`blue-500`)

### beanllm 브랜드 컬러

```css
:root {
  /* Primary: Deep Tech Green (LLM = 자연어) */
  --color-primary-50: #f0fdf4;
  --color-primary-500: #10b981; /* Emerald */
  --color-primary-700: #047857;
  --color-primary-900: #064e3b;

  /* Secondary: Data Amber (정보의 흐름) */
  --color-secondary-50: #fffbeb;
  --color-secondary-500: #f59e0b; /* Amber */
  --color-secondary-700: #b45309;

  /* Accent: Insight Blue (통찰) */
  --color-accent-500: #0ea5e9; /* Sky */
  --color-accent-700: #0369a1;

  /* Neutrals: Refined Grays */
  --color-neutral-50: #fafaf9;
  --color-neutral-100: #f5f5f4;
  --color-neutral-800: #292524;
  --color-neutral-900: #1c1917;

  /* Semantic */
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
}
```

**Dark Mode** (기본):
```css
.dark {
  --bg-primary: #0a0a0a;
  --bg-secondary: #171717;
  --bg-tertiary: #262626;
  --text-primary: #fafaf9;
  --text-secondary: #a8a29e;
  --border: #404040;
}
```

## Component Patterns

### 1. Chat Message Component

```typescript
// components/ChatMessage.tsx
'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  isStreaming?: boolean;
  timestamp?: Date;
}

export function ChatMessage({
  role,
  content,
  isStreaming = false,
  timestamp,
}: ChatMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={cn(
        'group relative mb-4 rounded-2xl p-6 backdrop-blur-sm',
        'border border-neutral-800/50',
        role === 'user'
          ? 'ml-12 bg-gradient-to-br from-primary-900/20 to-primary-800/10'
          : 'mr-12 bg-gradient-to-br from-neutral-900/80 to-neutral-800/40'
      )}
    >
      {/* 역할 뱃지 */}
      <div className="mb-3 flex items-center justify-between">
        <span
          className={cn(
            'inline-flex items-center gap-2 rounded-full px-3 py-1',
            'text-xs font-medium tracking-wide',
            'font-mono uppercase',
            role === 'user'
              ? 'bg-primary-500/20 text-primary-400'
              : 'bg-accent-500/20 text-accent-400'
          )}
        >
          <span
            className={cn(
              'h-1.5 w-1.5 rounded-full',
              role === 'user' ? 'bg-primary-500' : 'bg-accent-500'
            )}
          />
          {role}
        </span>

        {timestamp && (
          <time className="text-xs text-neutral-500 font-mono">
            {timestamp.toLocaleTimeString('ko-KR', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </time>
        )}
      </div>

      {/* 메시지 내용 */}
      <div className="prose prose-invert max-w-none">
        <p className="text-sm leading-relaxed text-neutral-100 font-['Work_Sans']">
          {content}
        </p>
      </div>

      {/* 스트리밍 커서 */}
      {isStreaming && (
        <motion.span
          animate={{ opacity: [1, 0, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
          className="ml-1 inline-block h-4 w-1.5 bg-primary-500"
        />
      )}

      {/* Gradient Border Glow Effect */}
      <div
        className={cn(
          'absolute inset-0 rounded-2xl opacity-0 transition-opacity duration-300',
          'group-hover:opacity-100',
          'bg-gradient-to-br from-primary-500/10 via-transparent to-accent-500/10',
          'pointer-events-none'
        )}
      />
    </motion.div>
  );
}
```

### 2. RAG Search Results Component

```typescript
// components/RAGSearchResults.tsx
'use client';

import { motion } from 'framer-motion';
import { FileText, Star } from 'lucide-react';

interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: {
    source: string;
    page?: number;
  };
}

export function RAGSearchResults({ results }: { results: SearchResult[] }) {
  return (
    <div className="space-y-3">
      {results.map((result, idx) => (
        <motion.div
          key={result.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: idx * 0.1 }}
          className="group relative overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 backdrop-blur-sm"
        >
          {/* Score Bar */}
          <div className="absolute left-0 top-0 h-full w-1 bg-gradient-to-b from-primary-500 to-accent-500">
            <div
              className="bg-primary-400 transition-all duration-500"
              style={{ height: `${result.score * 100}%` }}
            />
          </div>

          <div className="ml-4">
            {/* Header */}
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-primary-500" />
                <span className="text-xs font-mono text-neutral-400">
                  {result.metadata.source}
                  {result.metadata.page && ` • Page ${result.metadata.page}`}
                </span>
              </div>

              {/* Relevance Score */}
              <div className="flex items-center gap-1">
                <Star className="h-3 w-3 fill-amber-500 text-amber-500" />
                <span className="text-xs font-mono font-semibold text-amber-500">
                  {(result.score * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Content Preview */}
            <p className="text-sm leading-relaxed text-neutral-200 line-clamp-3">
              {result.content}
            </p>

            {/* Hover Overlay */}
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/5 to-accent-500/5 opacity-0 transition-opacity group-hover:opacity-100" />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
```

### 3. Multi-Agent Debate Visualization

```typescript
// components/MultiAgentDebate.tsx
'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Bot, Sparkles } from 'lucide-react';

interface AgentMessage {
  agentId: number;
  agentName: string;
  content: string;
  timestamp: Date;
}

export function MultiAgentDebate({ messages }: { messages: AgentMessage[] }) {
  return (
    <div className="relative min-h-screen bg-neutral-950 p-8">
      {/* Background Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#404040_1px,transparent_1px),linear-gradient(to_bottom,#404040_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-10" />

      {/* Agent Timeline */}
      <div className="relative mx-auto max-w-4xl">
        <AnimatePresence mode="popLayout">
          {messages.map((msg, idx) => (
            <motion.div
              key={`${msg.agentId}-${idx}`}
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ delay: idx * 0.2 }}
              className={cn(
                'mb-8 flex gap-4',
                msg.agentId % 2 === 0 ? 'flex-row' : 'flex-row-reverse'
              )}
            >
              {/* Agent Avatar */}
              <motion.div
                animate={{
                  scale: [1, 1.1, 1],
                  rotate: [0, 5, -5, 0],
                }}
                transition={{ duration: 2, repeat: Infinity }}
                className={cn(
                  'flex h-12 w-12 items-center justify-center rounded-full',
                  'bg-gradient-to-br shadow-lg',
                  msg.agentId % 3 === 0
                    ? 'from-primary-500 to-primary-700'
                    : msg.agentId % 3 === 1
                      ? 'from-accent-500 to-accent-700'
                      : 'from-secondary-500 to-secondary-700'
                )}
              >
                <Bot className="h-6 w-6 text-white" />
              </motion.div>

              {/* Message Bubble */}
              <div
                className={cn(
                  'flex-1 rounded-2xl border border-neutral-800 p-6',
                  'bg-gradient-to-br from-neutral-900/80 to-neutral-800/40',
                  'backdrop-blur-sm'
                )}
              >
                <div className="mb-2 flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary-400" />
                  <span className="font-mono text-sm font-semibold text-primary-400">
                    {msg.agentName}
                  </span>
                </div>

                <p className="text-sm leading-relaxed text-neutral-200">
                  {msg.content}
                </p>

                <time className="mt-3 block text-xs font-mono text-neutral-500">
                  {msg.timestamp.toLocaleTimeString()}
                </time>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
```

## Motion Patterns

### Staggered Fade In (리스트 항목)

```typescript
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};
```

### Streaming Text Animation (채팅 스트리밍)

```typescript
const streamVariants = {
  initial: { width: 0 },
  animate: {
    width: '100%',
    transition: {
      duration: 0.5,
      ease: 'easeOut',
    },
  },
};
```

### Pulse (진행 중 표시)

```typescript
<motion.div
  animate={{
    scale: [1, 1.2, 1],
    opacity: [0.5, 1, 0.5],
  }}
  transition={{
    duration: 2,
    repeat: Infinity,
    ease: 'easeInOut',
  }}
  className="h-2 w-2 rounded-full bg-primary-500"
/>
```

## Layout Patterns

### Dashboard Grid (비대칭)

```tsx
<div className="grid grid-cols-12 gap-6">
  {/* Main Content */}
  <div className="col-span-12 lg:col-span-8">
    <ChatInterface />
  </div>

  {/* Sidebar */}
  <div className="col-span-12 lg:col-span-4 space-y-6">
    <ModelSelector />
    <ParameterControls />
    <UsageStats />
  </div>
</div>
```

### Card with Gradient Border

```tsx
<div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-primary-500 via-accent-500 to-secondary-500">
  <div className="rounded-2xl bg-neutral-900 p-6">
    {/* Content */}
  </div>
</div>
```

## Accessibility

- **키보드 네비게이션**: 모든 인터랙티브 요소는 Tab으로 접근 가능
- **ARIA 레이블**: 스크린 리더용 명확한 레이블
- **색상 대비**: WCAG AA 준수 (최소 4.5:1)
- **Focus 표시**: `focus-visible:ring-2 ring-primary-500`

## Performance

- **CSS-only 애니메이션 우선**: Framer Motion은 복잡한 케이스만
- **이미지 최적화**: Next.js `<Image>` 컴포넌트 사용
- **Code Splitting**: 큰 컴포넌트는 `dynamic import`
- **Memoization**: `React.memo()`, `useMemo()`, `useCallback()`

## Related Documents

- `.claude/skills/frontend-patterns.md` - React/Next.js 패턴
- `playground/frontend/tailwind.config.ts` - Tailwind 설정
- `.claude/rules/coding-standards.md` - TypeScript 표준

## Upstream Skill

이 스킬은 Anthropic Agent Skills 마켓플레이스의 `frontend-design` 스킬을 beanllm 프로젝트 디자인 시스템에 맞게 커스터마이징한 것입니다.

**원본 스킬 위치**: `~/.claude/plugins/marketplaces/anthropic-agent-skills/skills/frontend-design/`
