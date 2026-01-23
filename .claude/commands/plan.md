# /plan - Feature Planning

**트리거**: `/plan`
**모델**: sonnet
**설명**: 기능 구현 전 단계별 계획 수립

## Command Description

새 기능을 구현하기 전에 아키텍처 설계, 파일 변경 사항, 테스트 전략 등을 단계별로 계획합니다.

## Usage

```
/plan "HyDE query expansion 기능 추가"
/plan "Multi-agent debate pattern 구현"
/plan "Knowledge graph RAG 통합"
```

## Execution Steps

### 1. 요구사항 분석

```
사용자 입력: "HyDE query expansion 기능 추가"

분석:
- HyDE (Hypothetical Document Embeddings)
- RAG 정확도 향상 목적
- 쿼리 대신 가상 답변을 임베딩
```

### 2. 아키텍처 설계

```
레이어별 변경 사항:

Domain Layer:
  - 새 파일: src/beanllm/domain/retrieval/hyde.py
    * HyDEQueryExpander 클래스
    * _generate_hypothetical_answer() 메서드

Service Layer:
  - 수정: src/beanllm/service/impl/core/rag_service_impl.py
    * HyDEQueryExpander 통합
    * query() 메서드 수정

DTO Layer:
  - 수정: src/beanllm/dto/request/core/rag_request.py
    * enable_hyde: bool 필드 추가

Facade Layer:
  - 수정: src/beanllm/facade/core/rag_facade.py
    * RAGChain.query()에 enable_hyde 파라미터 추가
```

### 3. 파일 체크리스트

```
생성할 파일:
  [ ] src/beanllm/domain/retrieval/hyde.py
  [ ] tests/domain/retrieval/test_hyde.py

수정할 파일:
  [ ] src/beanllm/service/impl/core/rag_service_impl.py
  [ ] src/beanllm/dto/request/core/rag_request.py
  [ ] src/beanllm/facade/core/rag_facade.py
  [ ] tests/facade/core/test_rag_facade.py
  [ ] docs/API_REFERENCE.md

검증할 사항:
  [ ] Clean Architecture 준수 (의존성 방향)
  [ ] 테스트 커버리지 80% 이상
  [ ] 성능 벤치마크 (before/after)
```

### 4. 구현 순서

```
1. Domain Layer (TDD)
   a. tests/domain/retrieval/test_hyde.py 작성
   b. src/beanllm/domain/retrieval/hyde.py 구현
   c. 테스트 통과 확인

2. DTO Layer
   a. RAGRequest에 enable_hyde 필드 추가
   b. Validation 규칙 추가

3. Service Layer
   a. tests/service/impl/core/test_rag_service_impl.py 수정
   b. RAGServiceImpl에 HyDE 통합
   c. 테스트 통과 확인

4. Facade Layer
   a. tests/facade/core/test_rag_facade.py 수정
   b. RAGChain.query()에 enable_hyde 파라미터 추가
   c. 테스트 통과 확인

5. 문서화
   a. docs/API_REFERENCE.md 업데이트
   b. 사용 예시 추가
```

### 5. 테스트 전략

```
Unit Tests:
  - HyDEQueryExpander._generate_hypothetical_answer()
  - HyDEQueryExpander.expand_query()
  - RAGServiceImpl.query() with enable_hyde=True

Integration Tests:
  - 전체 RAG 파이프라인 with HyDE
  - Ollama 모델과 통합 테스트

Performance Tests:
  - 정확도 벤치마크 (HyDE vs 기본)
  - 레이턴시 측정 (추가 LLM 호출로 인한 증가)

Coverage Goal:
  - 85% 이상 (새 코드 기준)
```

### 6. 성능 고려사항

```
Impact 분석:
  - 추가 LLM 호출: 쿼리당 +1 call
  - 예상 레이턴시 증가: ~1-2초
  - 정확도 향상: ~20% (예상)

최적화 방안:
  - 가상 답변 캐싱 (동일 쿼리 재사용)
  - 배치 처리 (여러 쿼리 동시 처리)
```

## Output Format

```markdown
=================================================
📋 Feature Implementation Plan
=================================================

Feature: HyDE Query Expansion for RAG
Complexity: Medium
Estimated Time: 4-6 hours

=================================================
📐 Architecture Design
=================================================

Layer Changes:

Domain Layer (NEW):
  ✅ src/beanllm/domain/retrieval/hyde.py
     - HyDEQueryExpander class
     - _generate_hypothetical_answer() method
     - expand_query() method

Service Layer (MODIFY):
  ✅ src/beanllm/service/impl/core/rag_service_impl.py
     - Integrate HyDEQueryExpander
     - Update query() method

DTO Layer (MODIFY):
  ✅ src/beanllm/dto/request/core/rag_request.py
     - Add enable_hyde: bool field

Facade Layer (MODIFY):
  ✅ src/beanllm/facade/core/rag_facade.py
     - Add enable_hyde parameter to query()

=================================================
📋 Implementation Checklist
=================================================

Phase 1: Domain Layer (TDD)
  [ ] Write tests/domain/retrieval/test_hyde.py
  [ ] Implement src/beanllm/domain/retrieval/hyde.py
  [ ] Verify tests pass

Phase 2: DTO Layer
  [ ] Add enable_hyde field to RAGRequest
  [ ] Add validation

Phase 3: Service Layer
  [ ] Update RAGServiceImpl tests
  [ ] Integrate HyDE into query()
  [ ] Verify tests pass

Phase 4: Facade Layer
  [ ] Update RAGChain tests
  [ ] Add enable_hyde parameter
  [ ] Verify tests pass

Phase 5: Documentation
  [ ] Update docs/API_REFERENCE.md
  [ ] Add usage examples

=================================================
🧪 Test Strategy
=================================================

Unit Tests (Target: 85% coverage):
  - test_generate_hypothetical_answer()
  - test_expand_query()
  - test_rag_service_with_hyde()

Integration Tests:
  - test_rag_end_to_end_with_hyde()
  - test_hyde_with_ollama()

Performance Tests:
  - Benchmark accuracy improvement
  - Measure latency impact

=================================================
⚡ Performance Impact
=================================================

Latency:
  Before: ~500ms
  After: ~1.5s (+1s for hypothetical answer generation)
  Mitigation: Cache hypothetical answers

Accuracy:
  Expected improvement: +20%
  Will benchmark on 100 test queries

=================================================
💡 Next Steps
=================================================

1. Review this plan
2. Start with Phase 1 (Domain Layer TDD)
3. Proceed step by step
4. Run tests after each phase
5. Document as you go

Ready to start? (y/n)
```

## Related Commands

- `/test-gen` - 테스트 자동 생성
- `/arch-check` - 아키텍처 검증

## Related Agents

- `code-reviewer` - 계획 검토
- `architecture-fixer` - 아키텍처 자동 수정

## Invocation Example

```
User: /plan "HyDE query expansion 추가"

Claude: [Generates detailed plan as shown above]

User: y

Claude: [Starts Phase 1: Domain Layer TDD]
  1. Creating tests/domain/retrieval/test_hyde.py...
  2. Writing failing tests (Red)...
  3. Implementing HyDEQueryExpander (Green)...
  4. Running tests...
  5. All tests pass! ✅

  Phase 1 complete. Ready for Phase 2? (y/n)
```
