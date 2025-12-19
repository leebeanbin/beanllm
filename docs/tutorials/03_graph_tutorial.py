"""
Graph Workflows Tutorial
========================

이 튜토리얼은 llmkit의 Graph 시스템을 실습합니다.

Topics:
1. Basic Graph 구조
2. ConditionalNode (조건부 분기)
3. LoopNode (반복 실행)
4. ParallelNode (병렬 실행)
5. Advanced Patterns (복합 워크플로우)
"""

import asyncio
import time
from typing import Dict, Any
import random

# llmkit imports (실제 환경에서는 이렇게 사용)
# from llmkit import (
#     Graph, GraphState, FunctionNode, ConditionalNode,
#     LoopNode, ParallelNode, Client
# )

# 시뮬레이션을 위한 간단한 클래스들
print("="*80)
print("Graph Workflows Tutorial")
print("="*80)


# =============================================================================
# Part 1: Basic Graph
# =============================================================================

print("\n" + "="*80)
print("Part 1: Basic Graph - 기본 그래프 구조")
print("="*80)

"""
Theory:
    그래프는 DAG (Directed Acyclic Graph)로 표현됩니다.

    수식:
        output = f₃(f₂(f₁(input)))
"""


async def demo_basic_graph():
    """기본 그래프 예제"""
    from llmkit import Graph, GraphState, FunctionNode

    # 노드 함수 정의
    def process_input(state: GraphState) -> Dict[str, Any]:
        """입력 처리"""
        text = state.get("input", "")
        processed = text.upper()
        print(f"  [Process] {text} -> {processed}")
        return {"processed": processed}

    def add_prefix(state: GraphState) -> Dict[str, Any]:
        """접두어 추가"""
        text = state.get("processed", "")
        result = f"PROCESSED: {text}"
        print(f"  [Add Prefix] {text} -> {result}")
        return {"output": result}

    def count_words(state: GraphState) -> Dict[str, Any]:
        """단어 개수 세기"""
        text = state.get("output", "")
        count = len(text.split())
        print(f"  [Count Words] {count} words")
        return {"word_count": count}

    # 그래프 생성
    graph = Graph()

    # 노드 추가
    graph.add_function_node("process", process_input)
    graph.add_function_node("prefix", add_prefix)
    graph.add_function_node("count", count_words)

    # 엣지 추가 (순차적 연결)
    graph.add_edge("process", "prefix")
    graph.add_edge("prefix", "count")

    # 시작점 설정
    graph.set_entry_point("process")

    # 실행
    print("\nExecuting graph...")
    result = await graph.run(
        {"input": "hello world from llmkit"},
        verbose=True
    )

    print(f"\nFinal Output: {result['output']}")
    print(f"Word Count: {result['word_count']}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_basic_graph())
    pass


# =============================================================================
# Part 2: ConditionalNode - 조건부 분기
# =============================================================================

print("\n" + "="*80)
print("Part 2: ConditionalNode - 조건부 실행")
print("="*80)

"""
Theory:
    조건부 노드는 state에 따라 다른 경로를 실행합니다.

    수식:
        result = {
            f_true(s)  if c(s) = True
            f_false(s) if c(s) = False
        }
"""


async def demo_conditional_node():
    """조건부 노드 예제: 품질 검사"""
    from llmkit import Graph, GraphState, FunctionNode, ConditionalNode

    # 노드 함수들
    def analyze_text(state: GraphState) -> Dict[str, Any]:
        """텍스트 분석 (품질 점수 부여)"""
        text = state.get("input", "")

        # 간단한 품질 점수 (단어 개수 기반)
        word_count = len(text.split())
        score = min(word_count * 10, 100)

        print(f"  [Analyze] '{text}' -> Score: {score}")
        return {"text": text, "quality_score": score}

    def high_quality_process(state: GraphState) -> Dict[str, Any]:
        """고품질 텍스트 처리"""
        text = state.get("text", "")
        result = f"✓ APPROVED: {text}"
        print(f"  [High Quality] {result}")
        return {"output": result, "status": "approved"}

    def low_quality_process(state: GraphState) -> Dict[str, Any]:
        """저품질 텍스트 처리"""
        text = state.get("text", "")
        result = f"✗ REJECTED: {text} (needs improvement)"
        print(f"  [Low Quality] {result}")
        return {"output": result, "status": "rejected"}

    # 조건 함수
    def is_high_quality(state: GraphState) -> bool:
        """품질 점수가 50 이상이면 True"""
        score = state.get("quality_score", 0)
        return score >= 50

    # 그래프 생성
    graph = Graph()

    # 분석 노드
    graph.add_function_node("analyze", analyze_text)

    # 조건부 노드
    high_quality_node = FunctionNode("high_quality", high_quality_process)
    low_quality_node = FunctionNode("low_quality", low_quality_process)

    conditional_node = ConditionalNode(
        "quality_router",
        condition=is_high_quality,
        true_node=high_quality_node,
        false_node=low_quality_node
    )
    graph.add_node(conditional_node)

    # 엣지
    graph.add_edge("analyze", "quality_router")
    graph.set_entry_point("analyze")

    # 테스트 케이스
    test_cases = [
        "This is a comprehensive and detailed text with many words",
        "Short",
        "Machine learning and AI are transforming technology"
    ]

    for text in test_cases:
        print(f"\n--- Testing: '{text}' ---")
        result = await graph.run({"input": text}, verbose=False)
        print(f"Status: {result['status']}")
        print(f"Output: {result['output']}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_conditional_node())
    pass


# =============================================================================
# Part 3: LoopNode - 반복 실행
# =============================================================================

print("\n" + "="*80)
print("Part 3: LoopNode - 반복 실행과 종료 조건")
print("="*80)

"""
Theory:
    LoopNode는 종료 조건이 만족될 때까지 반복 실행합니다.

    수식:
        s₀ = initial_state
        sₙ₊₁ = f(sₙ)  while not c_term(sₙ)

    Halting Problem:
        일반적으로 종료 여부를 결정할 수 없으므로
        max_iterations로 무한 루프를 방지합니다.
"""


async def demo_loop_node():
    """루프 노드 예제: 반복 개선"""
    from llmkit import Graph, GraphState, FunctionNode, LoopNode

    # 개선 노드
    def improve_text(state: GraphState) -> Dict[str, Any]:
        """텍스트를 조금씩 개선"""
        text = state.get("text", "")
        iteration = state.get("iteration", 0)

        # 간단한 개선: 단어 추가
        improvements = [
            "enhanced",
            "optimized",
            "refined",
            "polished",
            "perfected"
        ]

        if iteration < len(improvements):
            improved = f"{improvements[iteration]} {text}"
        else:
            improved = text

        print(f"  [Iteration {iteration + 1}] {text} -> {improved}")

        return {
            "text": improved,
            "iteration": iteration + 1,
            "length": len(improved.split())
        }

    # 종료 조건: 5개 이상의 단어
    def should_terminate(state: GraphState) -> bool:
        length = state.get("length", 0)
        iteration = state.get("iteration", 0)

        # 5개 이상의 단어 OR 5번 반복
        terminate = length >= 5 or iteration >= 5

        if terminate:
            print(f"  [Termination] Reached {length} words after {iteration} iterations")

        return terminate

    # 그래프 생성
    graph = Graph()

    # 초기화 노드
    def initialize(state: GraphState) -> Dict[str, Any]:
        text = state.get("input", "text")
        print(f"  [Initialize] Starting with: '{text}'")
        return {"text": text, "iteration": 0, "length": len(text.split())}

    graph.add_function_node("init", initialize)

    # 루프 노드
    improve_node = FunctionNode("improve", improve_text)
    loop_node = LoopNode(
        "improvement_loop",
        body_node=improve_node,
        termination_condition=should_terminate,
        max_iterations=10  # 안전장치
    )
    graph.add_node(loop_node)

    # 엣지
    graph.add_edge("init", "improvement_loop")
    graph.set_entry_point("init")

    # 실행
    print("\nRunning improvement loop...")
    result = await graph.run({"input": "text"}, verbose=False)

    print(f"\nFinal Text: {result['text']}")
    print(f"Total Iterations: {result['improvement_loop_iterations']}")
    print(f"Terminated: {result['improvement_loop_terminated']}")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_loop_node())
    pass


# =============================================================================
# Part 4: ParallelNode - 병렬 실행
# =============================================================================

print("\n" + "="*80)
print("Part 4: ParallelNode - 병렬 실행과 결과 집계")
print("="*80)

"""
Theory:
    ParallelNode는 여러 노드를 병렬로 실행합니다.

    수식:
        result = aggregate([f₁(s), f₂(s), ..., fₙ(s)])

    시간 복잡도:
        T_parallel = max(T₁, T₂, ..., Tₙ)
        vs T_sequential = T₁ + T₂ + ... + Tₙ

    Speedup:
        S = T_sequential / T_parallel
"""


async def demo_parallel_node():
    """병렬 노드 예제: 다중 분석"""
    from llmkit import Graph, GraphState, FunctionNode, ParallelNode

    # 여러 분석 노드들
    def sentiment_analysis(state: GraphState) -> Dict[str, Any]:
        """감성 분석 (시뮬레이션)"""
        text = state.get("text", "")
        print(f"  [Sentiment] Analyzing: '{text}'")

        # 시뮬레이션: 지연 시간
        time.sleep(0.5)

        # 간단한 감성 분석
        positive_words = ["good", "great", "awesome", "excellent"]
        negative_words = ["bad", "terrible", "awful", "poor"]

        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)

        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        print(f"  [Sentiment] Result: {sentiment}")
        return {"sentiment": sentiment}

    def entity_extraction(state: GraphState) -> Dict[str, Any]:
        """개체명 인식 (시뮬레이션)"""
        text = state.get("text", "")
        print(f"  [Entities] Extracting from: '{text}'")

        time.sleep(0.3)

        # 간단한 개체명 추출 (대문자로 시작하는 단어)
        entities = [word for word in text.split() if word[0].isupper()]

        print(f"  [Entities] Found: {entities}")
        return {"entities": entities}

    def keyword_extraction(state: GraphState) -> Dict[str, Any]:
        """키워드 추출 (시뮬레이션)"""
        text = state.get("text", "")
        print(f"  [Keywords] Extracting from: '{text}'")

        time.sleep(0.4)

        # 간단한 키워드 추출 (길이 > 4인 단어)
        keywords = [word for word in text.split() if len(word) > 4]

        print(f"  [Keywords] Found: {keywords}")
        return {"keywords": keywords}

    # 그래프 생성
    graph = Graph()

    # 준비 노드
    def prepare(state: GraphState) -> Dict[str, Any]:
        text = state.get("input", "")
        print(f"  [Prepare] Input text: '{text}'")
        return {"text": text}

    graph.add_function_node("prepare", prepare)

    # 병렬 노드 (3개의 분석을 동시에 실행)
    parallel_node = ParallelNode(
        "multi_analysis",
        child_nodes=[
            FunctionNode("sentiment", sentiment_analysis),
            FunctionNode("entities", entity_extraction),
            FunctionNode("keywords", keyword_extraction)
        ],
        aggregate_strategy="merge"  # 모든 결과를 하나로 병합
    )
    graph.add_node(parallel_node)

    # 엣지
    graph.add_edge("prepare", "multi_analysis")
    graph.set_entry_point("prepare")

    # 시간 측정
    print("\nRunning parallel analysis...")
    start_time = time.time()

    result = await graph.run({
        "input": "Python is great for Machine Learning and AI applications"
    }, verbose=False)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Results (completed in {elapsed:.2f}s):")
    print(f"{'='*60}")
    print(f"Sentiment: {result.get('sentiment_sentiment')}")
    print(f"Entities: {result.get('entities_entities')}")
    print(f"Keywords: {result.get('keywords_keywords')}")

    # 순차 실행 시간 예상
    sequential_time = 0.5 + 0.3 + 0.4  # 1.2s
    speedup = sequential_time / elapsed
    print(f"\nSpeedup: {speedup:.2f}x (sequential would take ~{sequential_time}s)")


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_parallel_node())
    pass


# =============================================================================
# Part 5: Advanced Patterns - 복합 워크플로우
# =============================================================================

print("\n" + "="*80)
print("Part 5: Advanced Patterns - 실전 워크플로우")
print("="*80)

"""
이 섹션에서는 여러 노드 타입을 조합한 복잡한 워크플로우를 구현합니다.

시나리오:
    문서 처리 파이프라인
    1. 문서 로드
    2. 병렬로 여러 분석 수행
    3. 품질 검사 (조건부)
    4. 품질이 낮으면 개선 루프
    5. 최종 결과 생성
"""


async def demo_advanced_workflow():
    """복합 워크플로우: 문서 처리 파이프라인"""
    from llmkit import (
        Graph, GraphState, FunctionNode,
        ConditionalNode, LoopNode, ParallelNode
    )

    # 1. 초기 처리
    def load_document(state: GraphState) -> Dict[str, Any]:
        doc = state.get("document", "")
        print(f"\n[Load] Document: '{doc}'")
        return {
            "text": doc,
            "word_count": len(doc.split()),
            "improvement_count": 0
        }

    # 2. 병렬 분석
    def analyze_length(state: GraphState) -> Dict[str, Any]:
        text = state.get("text", "")
        length_score = min(len(text.split()) * 10, 100)
        print(f"  [Length Analysis] Score: {length_score}")
        return {"length_score": length_score}

    def analyze_complexity(state: GraphState) -> Dict[str, Any]:
        text = state.get("text", "")
        # 간단한 복잡도: 평균 단어 길이
        words = text.split()
        avg_len = sum(len(w) for w in words) / len(words) if words else 0
        complexity_score = min(avg_len * 15, 100)
        print(f"  [Complexity Analysis] Score: {complexity_score:.1f}")
        return {"complexity_score": complexity_score}

    # 3. 품질 평가
    def evaluate_quality(state: GraphState) -> Dict[str, Any]:
        length = state.get("length_score", 0)
        complexity = state.get("complexity_score", 0)
        overall = (length + complexity) / 2
        print(f"\n[Quality] Overall Score: {overall:.1f}")
        return {"quality_score": overall}

    # 4. 개선
    def improve_document(state: GraphState) -> Dict[str, Any]:
        text = state.get("text", "")
        count = state.get("improvement_count", 0)

        # 개선: 설명 추가
        additions = [
            "with detailed analysis",
            "including comprehensive examples",
            "featuring expert insights",
            "with practical applications"
        ]

        if count < len(additions):
            improved = f"{text} {additions[count]}"
        else:
            improved = text

        print(f"  [Improve #{count + 1}] Added enhancement")

        return {
            "text": improved,
            "word_count": len(improved.split()),
            "improvement_count": count + 1
        }

    # 5. 최종 포맷팅
    def format_output(state: GraphState) -> Dict[str, Any]:
        text = state.get("text", "")
        quality = state.get("quality_score", 0)
        improvements = state.get("improvement_count", 0)

        output = f"""
{'='*70}
PROCESSED DOCUMENT
{'='*70}
Text: {text}

Quality Score: {quality:.1f}/100
Improvements Applied: {improvements}
Final Word Count: {len(text.split())}
{'='*70}
        """.strip()

        print(f"\n{output}")
        return {"final_output": output}

    # 조건 및 종료 함수
    def needs_improvement(state: GraphState) -> bool:
        quality = state.get("quality_score", 0)
        return quality < 70

    def should_stop_improving(state: GraphState) -> bool:
        # 품질이 충분하거나 3번 개선했으면 종료
        quality = state.get("quality_score", 0)
        count = state.get("improvement_count", 0)
        return quality >= 70 or count >= 3

    # 그래프 구성
    graph = Graph()

    # 노드들
    graph.add_function_node("load", load_document)

    # 병렬 분석
    parallel_analysis = ParallelNode(
        "parallel_analysis",
        child_nodes=[
            FunctionNode("length", analyze_length),
            FunctionNode("complexity", analyze_complexity)
        ],
        aggregate_strategy="merge"
    )
    graph.add_node(parallel_analysis)

    graph.add_function_node("evaluate", evaluate_quality)

    # 조건부: 개선 필요 여부
    # 개선 루프
    improve_node = FunctionNode("improve", improve_document)
    improvement_loop = LoopNode(
        "improvement_loop",
        body_node=improve_node,
        termination_condition=should_stop_improving,
        max_iterations=5
    )

    # 재평가
    graph.add_function_node("re_evaluate", evaluate_quality)

    # 최종 출력
    graph.add_function_node("format", format_output)

    # 조건부 노드: 개선 필요시 루프, 아니면 바로 포맷팅
    conditional = ConditionalNode(
        "quality_check",
        condition=needs_improvement,
        true_node=improvement_loop,
        false_node=FunctionNode("skip", lambda s: {})
    )
    graph.add_node(conditional)

    # 엣지 연결
    graph.add_edge("load", "parallel_analysis")
    graph.add_edge("parallel_analysis", "evaluate")
    graph.add_edge("evaluate", "quality_check")

    # 조건부 이후 재평가
    # Note: 실제로는 Graph의 conditional_edge 기능을 사용하거나
    # 더 복잡한 로직이 필요합니다. 여기서는 단순화.

    # 시작점
    graph.set_entry_point("load")

    # 테스트
    test_documents = [
        "AI systems",  # 낮은 품질 -> 개선 필요
        "Artificial intelligence and machine learning are revolutionizing modern technology"  # 높은 품질
    ]

    for doc in test_documents:
        print(f"\n{'='*70}")
        print(f"Processing: '{doc}'")
        print(f"{'='*70}")

        result = await graph.run({"document": doc}, verbose=False)


# 실행
if __name__ == "__main__":
    # asyncio.run(demo_advanced_workflow())
    pass


# =============================================================================
# Part 6: Performance Analysis - 성능 분석
# =============================================================================

print("\n" + "="*80)
print("Part 6: Performance Analysis - 캐싱과 병렬화의 효과")
print("="*80)

"""
Theory:
    캐싱과 병렬화의 성능 향상을 정량적으로 측정합니다.

    Metrics:
    1. Cache Hit Rate = hits / (hits + misses)
    2. Speedup = T_sequential / T_parallel
    3. Efficiency = Speedup / num_processors
"""


async def demo_performance_analysis():
    """성능 분석 실험"""
    from llmkit import Graph, GraphState, FunctionNode

    # 비용이 큰 연산 시뮬레이션
    def expensive_computation(state: GraphState) -> Dict[str, Any]:
        x = state.get("x", 0)
        time.sleep(0.1)  # 시뮬레이션
        result = x ** 2
        return {"result": result}

    # 캐싱 있는 그래프
    graph_cached = Graph(enable_cache=True)
    graph_cached.add_function_node("compute", expensive_computation, cache=True)
    graph_cached.set_entry_point("compute")

    # 캐싱 없는 그래프
    graph_no_cache = Graph(enable_cache=False)
    graph_no_cache.add_function_node("compute", expensive_computation, cache=False)
    graph_no_cache.set_entry_point("compute")

    # 테스트: 동일한 입력 10번 반복
    test_inputs = [{"x": i % 5} for i in range(10)]  # 5개 값 반복

    # 캐싱 있음
    print("\nWith Caching:")
    start = time.time()
    for inp in test_inputs:
        await graph_cached.run(inp, verbose=False)
    cached_time = time.time() - start

    stats = graph_cached.cache.get_stats()
    print(f"Time: {cached_time:.2f}s")
    print(f"Cache Hits: {stats['hits']}")
    print(f"Cache Misses: {stats['misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.2%}")

    # 캐싱 없음
    print("\nWithout Caching:")
    start = time.time()
    for inp in test_inputs:
        await graph_no_cache.run(inp, verbose=False)
    no_cache_time = time.time() - start

    print(f"Time: {no_cache_time:.2f}s")

    # 비교
    speedup = no_cache_time / cached_time
    print(f"\nSpeedup with caching: {speedup:.2f}x")


# =============================================================================
# 전체 실행
# =============================================================================

async def run_all_demos():
    """모든 데모 실행"""
    demos = [
        ("Basic Graph", demo_basic_graph),
        ("Conditional Node", demo_conditional_node),
        ("Loop Node", demo_loop_node),
        ("Parallel Node", demo_parallel_node),
        ("Advanced Workflow", demo_advanced_workflow),
        ("Performance Analysis", demo_performance_analysis)
    ]

    for name, demo in demos:
        print("\n" + "="*80)
        print(f"Running: {name}")
        print("="*80)
        try:
            await demo()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80)
        print(f"Completed: {name}")
        print("="*80)

        # 잠깐 대기
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    print("""
이 튜토리얼을 실행하려면:

1. llmkit 설치:
   pip install -e .

2. 실행:
   python docs/tutorials/03_graph_tutorial.py

각 데모를 개별적으로 실행하려면 해당 함수의 주석을 해제하세요.
    """)

    # 전체 실행
    # asyncio.run(run_all_demos())

    # 개별 실행 예시:
    asyncio.run(demo_basic_graph())
    # asyncio.run(demo_conditional_node())
    # asyncio.run(demo_loop_node())
    # asyncio.run(demo_parallel_node())
    # asyncio.run(demo_advanced_workflow())
    # asyncio.run(demo_performance_analysis())


"""
연습 문제:

1. RetryNode 구현하기
   - 실패시 재시도하는 노드
   - Exponential backoff 적용
   - 최대 재시도 횟수 제한

2. CacheNode 개선하기
   - LRU 정책 구현
   - TTL (Time To Live) 추가
   - 캐시 크기 제한

3. ParallelNode에 timeout 추가
   - 일정 시간 내에 완료되지 않으면 취소
   - 부분 결과 반환 옵션

4. 복잡한 워크플로우 구현
   - 여러 조건부 분기
   - 중첩된 루프
   - 동적 그래프 생성

5. 성능 최적화
   - Work-Span 분석
   - Critical Path 찾기
   - 병렬화 가능한 부분 식별
"""
