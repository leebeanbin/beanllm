# 📚 beanllm 문서 가이드

## 📋 핵심 문서

### 사용자 가이드
- **[../README.md](../README.md)**: 프로젝트 개요 및 주요 기능
- **[../QUICK_START.md](../QUICK_START.md)**: 빠른 시작 가이드
- **[API_REFERENCE.md](API_REFERENCE.md)**: 완전한 API 문서
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: 배포 가이드

### 개발자 가이드
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)**: 아키텍처 상세 설명
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)**: 개발 가이드 (Claude Code용)
- **[../DEPENDENCY_RULES.md](../DEPENDENCY_RULES.md)**: 의존성 규칙 상세 가이드
- **[DISTRIBUTED_ARCHITECTURE_PERFORMANCE.md](DISTRIBUTED_ARCHITECTURE_PERFORMANCE.md)**: 분산 아키텍처 성능 가이드

### 이론 문서 (선택적)
- **[theory/](theory/)**: 주제별 이론 문서 (RAG, Embeddings, Graph, Multi-Agent 등)
  - 각 주제별로 `00_overview.md`, `practice_*.md`, `study_*.md` 제공
  - 석사 수준의 수학적 엄밀성과 실무 가이드 포함

### 튜토리얼
- **[tutorials/](tutorials/)**: 실행 가능한 Python 튜토리얼 코드

---

## 🚀 빠른 시작

1. **처음 사용**: [../QUICK_START.md](../QUICK_START.md)부터 시작
2. **API 사용**: [API_REFERENCE.md](API_REFERENCE.md) 참고
3. **아키텍처 이해**: [../ARCHITECTURE.md](../ARCHITECTURE.md) 읽기
4. **이론 학습**: [theory/](theory/) 폴더의 주제별 문서

---

## 📖 문서 구조

```
docs/
├── README.md              # 이 파일 (문서 가이드)
├── API_REFERENCE.md       # API 참조 문서
├── DEPLOYMENT.md          # 배포 가이드
├── DEVELOPMENT_GUIDE.md   # 개발 가이드
├── DISTRIBUTED_ARCHITECTURE_PERFORMANCE.md  # 분산 아키텍처 성능 가이드
├── theory/                # 이론 문서 (주제별)
│   ├── embeddings/        # 임베딩
│   ├── rag/               # RAG
│   ├── graph/             # Graph Workflows
│   ├── multi_agent/       # Multi-Agent
│   ├── vision/            # Vision RAG
│   ├── tools/             # Tool Calling
│   ├── web_search/        # Web Search
│   ├── audio/             # Audio Processing
│   ├── ml_models/         # ML Models
│   └── production/        # Production Features
└── tutorials/             # 튜토리얼 코드
```

---

**최종 업데이트**: 2026-01-XX
