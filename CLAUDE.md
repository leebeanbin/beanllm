# beanllm Agent Rules

Unified LLM framework — 8 providers, Clean Architecture, PyPI 배포 라이브러리.

## 아키텍처 소스 오브 트루스

Clean Architecture — 의존성은 항상 안쪽으로만 흐른다:

```
Facade → Handler → Service (interface) → Domain (interface) ← Infrastructure
```

계층을 건너뛰거나 역방향 의존성을 만들지 않는다. 자세한 규칙: [`.claude/rules/`](.claude/rules/)

## 코드 작성 전 반드시 확인

- **아키텍처 규칙**: [`.claude/rules/python-clean-architecture.md`](.claude/rules/python-clean-architecture.md)
- **시스템 아키텍처**: [`docs/architecture/system-overview.md`](docs/architecture/system-overview.md)
- **ADR (설계 결정)**: [`docs/adr/`](docs/adr/)
- **커밋 워크플로**: [`.claude/COMMIT_WORKFLOW.md`](.claude/COMMIT_WORKFLOW.md)

## 패키지 구조 (신규 코드 위치 결정 기준)

```
src/beanllm/
  facade/       공개 API (Client, RAGChain, Agent, StateGraph)
  handler/      요청 검증 · 데코레이터 적용
  service/      비즈니스 로직 인터페이스 + impl/
  domain/       핵심 엔티티 · 규칙 (LLMResponse, ModelParameterStrategy)
  infrastructure/ 외부 연동 (providers, vector stores, HTTP clients)
  providers/    LLM 프로바이더 구현체 (8개)
  dto/          데이터 전송 객체
  decorators/   에러 처리 · 검증 · 로깅 데코레이터
```

새 프로바이더 추가 시: `providers/` 에 `BaseLLMProvider` 상속 구현체 + `provider_factory.py` 등록.

## 절대 규칙

- **커밋에 `Co-Authored-By` 줄을 절대 넣지 않는다.** 작성자는 사용자 단독.
- 커밋은 논리 단위로 분리한다 (기능 하나 = 커밋 하나).
- 코드 변경 후 반드시 `make quality` (ruff + mypy + bandit + pytest) 통과 확인.
- 새 의존성은 `pyproject.toml`의 optional extras에 추가한다. 기본 install을 무겁게 만들지 않는다.

## 테스트

```bash
pytest                          # 전체 (6,340 tests)
pytest --cov=src/beanllm        # 커버리지 (목표 80%)
make quality                    # 전체 품질 파이프라인
```

테스트는 레이어별로 격리한다. Infrastructure 레이어는 HTTP 모킹, 상위 레이어는 Service 인터페이스 모킹.
