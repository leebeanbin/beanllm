# beanllm REPL - Interactive CLI

beanllm의 모든 기능을 터미널에서 대화형으로 사용할 수 있는 REPL(Read-Eval-Print Loop) 인터페이스입니다.

## 빠른 시작

### 실행 방법

```bash
# Python 모듈로 실행
python -m beanllm.ui.repl

# 또는 Python 코드에서
python
>>> from beanllm.ui.repl import repl_main
>>> repl_main()
```

### 첫 화면

```
┌─ Welcome ──────────────────────────────────────────┐
│ beanllm REPL                                       │
│                                                     │
│ Unified LLM Framework with Clean Architecture     │
│                                                     │
│ 📚 Type help to see available commands            │
│ 🚀 Type status to check system status             │
│ 👋 Type exit to quit                              │
└────────────────────────────────────────────────────┘

Loading command modules...
✓ Knowledge Graph commands loaded
✓ RAG Debug commands loaded
✓ Optimizer commands loaded
✓ Orchestrator commands loaded

Ready!

beanllm>
```

## 주요 명령어

### 일반 명령어 (General)

| 명령어 | 설명 | 사용법 |
|--------|------|--------|
| `help` | 사용 가능한 명령어 표시 | `help [command]` |
| `exit` / `quit` | REPL 종료 | `exit` |
| `clear` | 화면 지우기 | `clear` |
| `version` | beanllm 버전 정보 | `version` |
| `status` | REPL 상태 확인 | `status` |
| `config` | 환경 설정 표시 | `config` |

### Knowledge Graph 명령어

| 명령어 | 설명 |
|--------|------|
| `build_graph` | Knowledge Graph 구축 |
| `query` | 그래프 쿼리 실행 |
| `graph_rag` | Graph-based RAG 질의 |
| `visualize` | 그래프 시각화 |
| `entities` | 엔티티 추출 |
| `relations` | 관계 추출 |

### RAG Debug 명령어

| 명령어 | 설명 |
|--------|------|
| `start_debug` | RAG 디버그 세션 시작 |
| `analyze_embeddings` | 임베딩 분석 |
| `validate_chunks` | 청크 검증 |
| `test_similarity` | 유사도 테스트 |
| `tune_parameters` | 파라미터 튜닝 |

### Optimizer 명령어

| 명령어 | 설명 |
|--------|------|
| `benchmark` | 성능 벤치마크 |
| `optimize` | 자동 최적화 |
| `profile` | 프로파일링 |
| `compare` | A/B 테스트 |

### Orchestrator 명령어

| 명령어 | 설명 |
|--------|------|
| `create_workflow` | 워크플로우 생성 |
| `visualize_workflow` | 워크플로우 시각화 |
| `run_workflow` | 워크플로우 실행 |
| `monitor` | 실시간 모니터링 |

## 사용 예제

### 예제 1: Help 확인

```
beanllm> help

Available Commands

┌─ General ──────────────────────────────────────────┐
│ Command    │ Description                           │
├────────────┼───────────────────────────────────────┤
│ clear      │ Clear the screen                      │
│ config     │ Show configuration                    │
│ exit       │ Exit the REPL                         │
│ help       │ Show help for commands                │
│ quit       │ Exit the REPL (alias for exit)       │
│ status     │ Show REPL status                      │
│ version    │ Show beanllm version info             │
└────────────┴───────────────────────────────────────┘

┌─ Knowledge Graph ──────────────────────────────────┐
│ Command       │ Description                        │
├───────────────┼────────────────────────────────────┤
│ build_graph   │ Build Knowledge Graph from docs    │
│ entities      │ Extract entities from text         │
│ graph_rag     │ Graph-based RAG query              │
│ query         │ Query the knowledge graph          │
│ relations     │ Extract relations from text        │
│ visualize     │ Visualize the knowledge graph      │
└───────────────┴────────────────────────────────────┘

Type 'help <command>' for detailed information.
```

### 예제 2: Knowledge Graph 구축

```
beanllm> build_graph

Enter documents (one per line, empty line to finish):
> Apple was founded by Steve Jobs in 1976.
> Microsoft was founded by Bill Gates in 1975.
>

Enter graph ID (default: auto-generated):
> tech_companies

Building Knowledge Graph...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 2/2 docs

┌─ Knowledge Graph Built ────────────────────────────┐
│ Graph ID: tech_companies                           │
│ Nodes: 6 entities                                  │
│ Edges: 4 relations                                 │
│ Processing time: 3.24s                             │
└────────────────────────────────────────────────────┘
```

### 예제 3: Status 확인

```
beanllm> status

┌─ REPL Status ──────────────────────────────────────┐
│ Component         │ Status                         │
├───────────────────┼────────────────────────────────┤
│ REPL Shell        │ ✅ Running                     │
│ Commands Loaded   │ ✅ 20                          │
│ Redis             │ ✅ Connected                   │
│ Kafka             │ ⚠️  Not configured             │
└───────────────────┴────────────────────────────────┘
```

### 예제 4: 특정 명령어 Help

```
beanllm> help build_graph

build_graph
Category: Knowledge Graph
Description: Build Knowledge Graph from documents

Usage:
  build_graph [options]

Options:
  --docs <file>      Load documents from file
  --graph-id <id>    Specify graph ID
  --entity-types     Specify entity types
  --relation-types   Specify relation types
```

## 기능

### ✅ 구현된 기능

- **명령어 자동 등록**: 모든 `cmd_*` 메서드 자동 인식
- **카테고리별 정리**: help에서 카테고리별로 명령어 표시
- **에러 처리**: 우아한 에러 처리 및 표시
- **Rich UI**: 색상, 테이블, 패널 등 Rich 라이브러리 활용
- **비동기 지원**: 비동기 명령어 자동 처리
- **모듈형 구조**: 새로운 명령어 모듈 쉽게 추가 가능

### 🎯 특징

- **간단함**: 복잡한 설정 없이 바로 사용
- **확장성**: 새로운 명령어 모듈 쉽게 추가
- **일관성**: 모든 명령어가 동일한 인터페이스
- **시각화**: Rich 라이브러리로 아름다운 출력

## 커스텀 명령어 추가

### 새로운 명령어 모듈 만들기

```python
# my_commands.py
class MyCommands:
    \"\"\"My custom commands\"\"\"

    def __init__(self, client=None):
        self.client = client

    def cmd_hello(self, args=None):
        \"\"\"Say hello\"\"\"
        name = args[0] if args else "World"
        print(f"Hello, {name}!")

    async def cmd_async_example(self, args=None):
        \"\"\"Async command example\"\"\"
        import asyncio
        await asyncio.sleep(1)
        print("Async command completed!")
```

### REPL에 등록하기

```python
from beanllm.ui.repl import REPLShell
from my_commands import MyCommands

shell = REPLShell()

# Register custom module
my_commands = MyCommands()
shell.register_module("my", my_commands, "My Commands")

# Run
shell.run()
```

## 환경 설정

### 환경 변수

```bash
# Distributed features
export USE_DISTRIBUTED=true
export REDIS_HOST=localhost
export REDIS_PORT=6379
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# LLM provider
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
```

### 프로그래밍 방식 설정

```python
from beanllm.ui.repl import REPLShell
from beanllm import Client

# Create client
client = Client(provider="openai", api_key="your-key")

# Create shell with client
shell = REPLShell()
shell.client = client

# Run
shell.run()
```

## 단축키

- `Ctrl+C`: 현재 명령 중단 (REPL은 계속 실행)
- `Ctrl+D` 또는 `EOF`: REPL 종료
- `exit` 또는 `quit`: REPL 종료

## 문제 해결

### 명령어 모듈 로딩 실패

```
⚠ Knowledge Graph commands unavailable: ...
```

**해결**: 해당 모듈의 dependencies가 설치되어 있는지 확인

```bash
pip install beanllm[advanced]
```

### Redis/Kafka 연결 실패

```
❌ Redis: Disconnected
```

**해결**:
1. 분산 기능 비활성화: `export USE_DISTRIBUTED=false`
2. 또는 Redis/Kafka 설치 및 실행

### 명령어 찾을 수 없음

```
Unknown command: xyz
```

**해결**: `help`로 사용 가능한 명령어 확인

## 개발자 정보

### 디렉토리 구조

```
src/beanllm/ui/repl/
├── __init__.py                      # Exports
├── __main__.py                      # CLI entry point
├── README.md                        # This file
├── repl_shell.py                    # Main REPL shell
├── common_commands.py               # Common commands
├── knowledge_graph_commands.py      # KG commands
├── rag_commands.py                  # RAG debug commands
├── optimizer_commands.py            # Optimizer commands
└── orchestrator_commands.py         # Orchestrator commands
```

### 명령어 등록 흐름

```
┌──────────────────┐
│ Command Module   │
│ (cmd_* methods)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ register_module()│
│ (scan cmd_*)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CommonCommands   │
│ (command registry│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ REPLShell        │
│ (execute)        │
└──────────────────┘
```

## 향후 계획

### 가능한 개선사항

- ⏳ **Tab completion**: prompt_toolkit 통합
- ⏳ **Command history**: 이전 명령어 기록 및 재사용
- ⏳ **Syntax highlighting**: 입력 중 하이라이팅
- ⏳ **Multi-line input**: 여러 줄 입력 지원
- ⏳ **Configuration file**: .beanllmrc 설정 파일

## 라이선스

beanllm과 동일

## 기여

버그 리포트나 기능 제안은 GitHub Issues로 제출해주세요.
